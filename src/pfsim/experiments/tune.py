from __future__ import annotations

import csv
from dataclasses import asdict, replace
import itertools
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from hashlib import sha256
import time
from typing import Any, Iterable

import numpy as np
import yaml

from pfsim.config import SimConfig
from pfsim.prefetchers import NoPrefetcher, make_prefetcher
from pfsim.rl import AgentConfig, RLPrefetcher, RewardConfig
from pfsim.sim import Simulator
from pfsim.trace import generate

DEV_WORKLOADS = ("seq_stream", "stride_const", "multi_stride", "matmul_blocked",
                 "spmv_csr", "hash_lookup", "phase_mix")
TEST_WORKLOADS = ("bfs_graph", "list_walk", "ptr_chase", "random_uniform",
                  "mix_reg_irr", "stencil2d")
TUNING_TRACE_SEEDS = (100, 101)
TUNING_AGENT_SEEDS = (100, 101, 102)
TUNING_BANDWIDTHS = (3.2, 12.8)


def validate_tuning_workloads(workloads: Iterable[str]) -> tuple[str, ...]:
    names = tuple(workloads)
    invalid = sorted(set(names) - set(DEV_WORKLOADS))
    if invalid:
        raise ValueError(f"tuning is restricted to development workloads; rejected: {invalid}")
    if not names: raise ValueError("at least one development workload is required")
    return names


def geometric_mean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values: return 0.0
    return math.exp(sum(math.log(max(value, 1e-12)) for value in values) / len(values))


class _TuningContext:
    def __init__(self, config: SimConfig, records: int | None):
        self.config, self.records = config, records
        self.traces: dict[tuple[str, int, int | None], Any] = {}
        self.baselines: dict[tuple[str, int, float, int | None], Any] = {}

    def trace(self, workload: str, seed: int, records: int | None = None):
        count = self.records if records is None else records
        key = workload, seed, count
        if key not in self.traces: self.traces[key] = generate(workload, seed, count)
        return self.traces[key]

    def baseline(self, workload: str, trace_seed: int, bandwidth: float, records: int | None = None):
        count = self.records if records is None else records
        key = workload, trace_seed, bandwidth, count
        if key not in self.baselines:
            cfg = self.config.with_bandwidth(bandwidth)
            self.baselines[key] = Simulator(cfg, NoPrefetcher()).run(self.trace(workload, trace_seed, count), 0)
        return self.baselines[key]


def _write_rows(path: Path, rows: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force: raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _write_yaml(path: Path, value: dict[str, Any], force: bool) -> None:
    if path.exists() and not force: raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _checkpoint_records(path: Path, resume: bool) -> dict[int, dict[str, Any]]:
    if not resume:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return {}
    if not path.exists(): return {}
    completed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate an interrupted final append
        completed[int(record["candidate"])] = record
    return completed


def _checkpoint_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n"); handle.flush()


def _progress(label: str, done: int, total: int, started: float, enabled: bool) -> None:
    if not enabled: return
    elapsed = time.perf_counter() - started
    eta = elapsed / done * (total - done) if done else 0
    print(f"[{label}] {done}/{total} candidates, elapsed {elapsed:.1f}s, ETA {eta:.1f}s", flush=True)


def _code_hash() -> str:
    digest = sha256()
    root = Path(__file__).parents[1]
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _guard_fresh(paths: Iterable[Path], force: bool, resume: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force and not resume:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing tuning artifacts: {names}; pass --force or --resume")


def tune_baselines(config: SimConfig, *, workloads: Iterable[str] = DEV_WORKLOADS,
                   records: int | None = 1_000_000, out_dir: str | Path = "results/tune",
                   config_out: str | Path = "configs/baselines_tuned.yaml",
                   force: bool = False, resume: bool = False, progress: bool = True) -> dict[str, dict[str, Any]]:
    names = validate_tuning_workloads(workloads); context = _TuningContext(config, records)
    grids = {
        "next_line": ({"degree": degree} for degree in (1, 2, 4, 8)),
        "stride_pc": ({"degree": degree, "min_conf": confidence}
                      for degree, confidence in itertools.product((1, 2, 4, 8), (1, 2, 3))),
        "bop": ({"degree": degree, "roundmax": rounds}
                for degree, rounds in itertools.product((1, 2, 4), (50, 100, 200))),
    }
    chosen: dict[str, dict[str, Any]] = {}; rows: list[dict[str, Any]] = []
    code_hash = _code_hash()
    checkpoint_paths = [Path(out_dir) / f"{name}_{config.digest()}_code{code_hash}_n{records}_checkpoint.jsonl"
                        for name in grids]
    meta_path=Path(str(config_out)+".meta.yaml")
    _guard_fresh([Path(out_dir) / "baselines_runs.csv", Path(config_out), meta_path, *checkpoint_paths], force, resume)
    for (prefetcher, candidates), checkpoint in zip(grids.items(), checkpoint_paths):
        candidate_list = list(candidates)
        completed = _checkpoint_records(checkpoint, resume)
        started = time.perf_counter()
        best_score = -math.inf; best = None
        for candidate_id, parameters in enumerate(candidate_list):
            if candidate_id in completed:
                record = completed[candidate_id]; score = float(record["score"])
                rows.extend(record["rows"])
                if score > best_score: best_score, best = score, dict(parameters)
                _progress(prefetcher, candidate_id + 1, len(candidate_list), started, progress)
                continue
            speeds = []
            candidate_rows = []
            for workload, bandwidth in itertools.product(names, TUNING_BANDWIDTHS):
                cfg = config.with_bandwidth(bandwidth); trace = context.trace(workload, 100)
                baseline = context.baseline(workload, 100, bandwidth)
                result = Simulator(cfg, make_prefetcher(prefetcher, **parameters)).run(trace, 0)
                speedup = result.ipc / baseline.ipc; speeds.append(speedup)
                candidate_rows.append({"target": prefetcher, "candidate": candidate_id, **parameters,
                             "workload": workload, "trace_seed": 100, "agent_seed": "",
                             "bw_gbps": bandwidth, "ipc": result.ipc, "speedup": speedup})
            score = geometric_mean(speeds)
            rows.extend(candidate_rows)
            _checkpoint_append(checkpoint, {"candidate": candidate_id, "score": score,
                                             "parameters": parameters, "rows": candidate_rows})
            if score > best_score: best_score, best = score, dict(parameters)
            _progress(prefetcher, candidate_id + 1, len(candidate_list), started, progress)
        assert best is not None
        chosen[prefetcher] = best
    _write_rows(Path(out_dir) / "baselines_runs.csv", rows, force or resume)
    _write_yaml(Path(config_out), chosen, force or resume)
    _write_yaml(meta_path,{"tuning_status":"full" if records==1_000_000 and set(names)==set(DEV_WORKLOADS) else "provisional",
                           "records":records,"code_hash":code_hash},force or resume)
    return chosen


def sample_agent_configs(count: int, seed: int = 2026) -> list[AgentConfig]:
    rng = np.random.default_rng(seed); output = []
    for _ in range(count):
        c_w0 = float(rng.uniform(.1, .6)); c_w1 = max(c_w0, float(rng.uniform(.5, 1.5)))
        reward = RewardConfig(c_w0=c_w0, c_w1=c_w1, c_pol=float(rng.uniform(.2, 1.2)),
                              t_min=float(rng.uniform(0, .5)))
        output.append(AgentConfig(alpha=float(np.exp(rng.uniform(np.log(.005), np.log(.3)))),
            gamma=float(rng.uniform(0, .95)), eps_0=float(rng.uniform(.05, .3)),
            half_life=int(rng.integers(50_000, 500_001)), q0=float(rng.uniform(0, .6)),
            pc_bits=int(rng.integers(4, 11)), reward=reward,
            admission_control=False, reach_control=False))
    return output


def _agent_dict(config: AgentConfig) -> dict[str, Any]:
    value = asdict(config)
    for key in ("reaches", "patterns", "feature_drop", "reset_boundaries"): value[key] = list(value[key])
    return value


def _evaluate_agent_candidate(args):
    config, names, records, candidate, candidate_id, stage, trace_seeds = args
    context = _TuningContext(config, records); candidate_rows=[]; grouped=[]
    for workload, trace_seed, bandwidth in itertools.product(names, trace_seeds, TUNING_BANDWIDTHS):
        cfg=config.with_bandwidth(bandwidth); trace=context.trace(workload,trace_seed)
        baseline=context.baseline(workload,trace_seed,bandwidth); seed_speeds=[]
        for agent_seed in TUNING_AGENT_SEEDS:
            run_cfg=replace(candidate,miss_lat_ref=cfg.llc.latency+cfg.dram_latency+cfg.transfer_cycles)
            result=Simulator(cfg,RLPrefetcher(run_cfg)).run(trace,agent_seed)
            speedup=result.ipc/baseline.ipc; seed_speeds.append(speedup)
            candidate_rows.append({"stage":stage,"candidate":candidate_id,"workload":workload,
                "trace_seed":trace_seed,"agent_seed":agent_seed,"bw_gbps":bandwidth,
                "ipc":result.ipc,"speedup":speedup})
        grouped.append(sum(seed_speeds)/len(seed_speeds))
    return candidate_id,geometric_mean(grouped),candidate_rows


def _evaluate_refine_candidate(args):
    config, names, records, candidate, candidate_id = args
    context = _TuningContext(config, records); speeds = []; candidate_rows = []
    for workload, trace_seed, bandwidth in itertools.product(names, TUNING_TRACE_SEEDS, TUNING_BANDWIDTHS):
        trace = context.trace(workload, trace_seed)
        baseline = context.baseline(workload, trace_seed, bandwidth)
        cfg = config.with_bandwidth(bandwidth)
        for agent_seed in TUNING_AGENT_SEEDS:
            run_cfg = replace(candidate, miss_lat_ref=cfg.llc.latency + cfg.dram_latency + cfg.transfer_cycles)
            result = Simulator(cfg, RLPrefetcher(run_cfg)).run(trace, agent_seed)
            speedup = result.ipc / baseline.ipc; speeds.append(speedup)
            candidate_rows.append({"stage": "refine", "candidate": candidate_id, "workload": workload,
                         "trace_seed": trace_seed, "agent_seed": agent_seed,
                         "bw_gbps": bandwidth, "ipc": result.ipc, "speedup": speedup})
    return candidate_id, geometric_mean(speeds), candidate_rows


def tune_agent(config: SimConfig, *, workloads: Iterable[str] = DEV_WORKLOADS,
               records: int | None = 1_000_000, trials: int = 60, search_seed: int = 2026,
               refine: bool = True, refinement_records: int | None = None,
               out_dir: str | Path = "results/tune",
               config_out: str | Path = "configs/agent_tuned.yaml",
               force: bool = False, resume: bool = False, progress: bool = True,
               jobs: int = 1) -> AgentConfig:
    names = validate_tuning_workloads(workloads)
    if trials <= 0: raise ValueError("trials must be positive")
    if jobs <= 0: raise ValueError("jobs must be positive")
    candidates = sample_agent_configs(trials, search_seed)
    rows: list[dict[str, Any]] = []; scored: list[tuple[float, int, AgentConfig]] = []
    code_hash = _code_hash()
    search_checkpoint = Path(out_dir) / f"agent_search_{config.digest()}_code{code_hash}_n{records}_t{trials}_s{search_seed}.jsonl"
    refine_checkpoint = Path(out_dir) / f"agent_refine_{config.digest()}_code{code_hash}_n{refinement_records}_t{trials}_s{search_seed}.jsonl"
    _guard_fresh([Path(out_dir) / "agent_runs.csv", Path(config_out), search_checkpoint,
                  *([refine_checkpoint] if refine else [])], force, resume)
    completed_search = _checkpoint_records(search_checkpoint, resume)
    search_started = time.perf_counter()
    pending=[]
    for candidate_id, candidate in enumerate(candidates):
        if candidate_id in completed_search:
            record = completed_search[candidate_id]; rows.extend(record["rows"])
            scored.append((float(record["score"]), candidate_id, candidate))
            _progress("agent search", candidate_id + 1, len(candidates), search_started, progress)
            continue
        pending.append((config,names,records,candidate,candidate_id,"search",(100,)))
    executor = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    results = (executor.map(_evaluate_agent_candidate,pending) if executor else map(_evaluate_agent_candidate,pending))
    for done,(candidate_id,score,candidate_rows) in enumerate(results,len(completed_search)+1):
        candidate=candidates[candidate_id]; rows.extend(candidate_rows)
        scored.append((score, candidate_id, candidate))
        _checkpoint_append(search_checkpoint, {"candidate": candidate_id, "score": score,
                                                "rows": candidate_rows})
        _progress("agent search", done, len(candidates), search_started, progress)
    if executor: executor.shutdown()
    scored.sort(key=lambda item: item[0], reverse=True)
    if refine:
        completed_refine = _checkpoint_records(refine_checkpoint, resume)
        refined: list[tuple[float, int, AgentConfig]] = []
        finalists = scored[:min(5, len(scored))]; refine_started = time.perf_counter()
        pending_refine = [(config, names, refinement_records, candidate, candidate_id)
                          for _, candidate_id, candidate in finalists if candidate_id not in completed_refine]
        refine_executor = (ProcessPoolExecutor(max_workers=min(jobs, len(pending_refine)))
                           if jobs > 1 and len(pending_refine) > 1 else None)
        fresh_refine = iter(refine_executor.map(_evaluate_refine_candidate, pending_refine) if refine_executor
                            else map(_evaluate_refine_candidate, pending_refine))
        for finalist_index, (_, candidate_id, candidate) in enumerate(finalists):
            if candidate_id in completed_refine:
                record = completed_refine[candidate_id]; rows.extend(record["rows"])
                refined.append((float(record["score"]), candidate_id, candidate))
                _progress("agent refine", finalist_index + 1, len(finalists), refine_started, progress)
                continue
            _, score, candidate_rows = next(fresh_refine); rows.extend(candidate_rows)
            refined.append((score, candidate_id, candidate))
            _checkpoint_append(refine_checkpoint, {"candidate": candidate_id, "score": score,
                                                    "rows": candidate_rows})
            _progress("agent refine", finalist_index + 1, len(finalists), refine_started, progress)
        if refine_executor: refine_executor.shutdown()
        refined.sort(key=lambda item: item[0], reverse=True); best_score, _, best = refined[0]
    else:
        best_score, _, best = scored[0]
    status = ("full" if records == 1_000_000 and refinement_records == 1_000_000 and trials >= 60 and refine
              and set(names) == set(DEV_WORKLOADS) else "provisional")
    best = replace(best, tuning_status=status)
    output = _agent_dict(best)
    _write_rows(Path(out_dir) / "agent_runs.csv", rows, force or resume)
    _write_yaml(Path(config_out), output, force or resume)
    return best
