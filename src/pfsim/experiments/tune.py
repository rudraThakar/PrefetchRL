from __future__ import annotations

import csv
from dataclasses import asdict, replace
import itertools
import math
from pathlib import Path
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


def tune_baselines(config: SimConfig, *, workloads: Iterable[str] = DEV_WORKLOADS,
                   records: int | None = 1_000_000, out_dir: str | Path = "results/tune",
                   config_out: str | Path = "configs/baselines_tuned.yaml",
                   force: bool = False) -> dict[str, dict[str, Any]]:
    names = validate_tuning_workloads(workloads); context = _TuningContext(config, records)
    grids = {
        "next_line": ({"degree": degree} for degree in (1, 2, 4, 8)),
        "stride_pc": ({"degree": degree, "min_conf": confidence}
                      for degree, confidence in itertools.product((1, 2, 4, 8), (1, 2, 3))),
        "bop": ({"degree": degree, "roundmax": rounds}
                for degree, rounds in itertools.product((1, 2, 4), (50, 100, 200))),
    }
    chosen: dict[str, dict[str, Any]] = {}; rows: list[dict[str, Any]] = []
    for prefetcher, candidates in grids.items():
        best_score = -math.inf; best = None
        for candidate_id, parameters in enumerate(candidates):
            speeds = []
            for workload, bandwidth in itertools.product(names, TUNING_BANDWIDTHS):
                cfg = config.with_bandwidth(bandwidth); trace = context.trace(workload, 100)
                baseline = context.baseline(workload, 100, bandwidth)
                result = Simulator(cfg, make_prefetcher(prefetcher, **parameters)).run(trace, 0)
                speedup = result.ipc / baseline.ipc; speeds.append(speedup)
                rows.append({"target": prefetcher, "candidate": candidate_id, **parameters,
                             "workload": workload, "trace_seed": 100, "agent_seed": "",
                             "bw_gbps": bandwidth, "ipc": result.ipc, "speedup": speedup})
            score = geometric_mean(speeds)
            if score > best_score: best_score, best = score, dict(parameters)
        assert best is not None
        chosen[prefetcher] = best
    _write_rows(Path(out_dir) / "baselines_runs.csv", rows, force)
    _write_yaml(Path(config_out), chosen, force)
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
            pc_bits=int(rng.integers(4, 11)), reward=reward))
    return output


def _agent_dict(config: AgentConfig) -> dict[str, Any]:
    value = asdict(config)
    for key in ("reaches", "patterns", "feature_drop"): value[key] = list(value[key])
    return value


def tune_agent(config: SimConfig, *, workloads: Iterable[str] = DEV_WORKLOADS,
               records: int | None = 1_000_000, trials: int = 60, search_seed: int = 2026,
               refine: bool = True, refinement_records: int | None = None,
               out_dir: str | Path = "results/tune",
               config_out: str | Path = "configs/agent_tuned.yaml",
               force: bool = False) -> AgentConfig:
    names = validate_tuning_workloads(workloads)
    if trials <= 0: raise ValueError("trials must be positive")
    context = _TuningContext(config, records); candidates = sample_agent_configs(trials, search_seed)
    rows: list[dict[str, Any]] = []; scored: list[tuple[float, int, AgentConfig]] = []
    for candidate_id, candidate in enumerate(candidates):
        pair_means = []
        for workload, bandwidth in itertools.product(names, TUNING_BANDWIDTHS):
            cfg = config.with_bandwidth(bandwidth); trace = context.trace(workload, 100)
            baseline = context.baseline(workload, 100, bandwidth)
            seed_speeds = []
            for agent_seed in TUNING_AGENT_SEEDS:
                run_cfg = replace(candidate, miss_lat_ref=cfg.llc.latency + cfg.dram_latency + cfg.transfer_cycles)
                result = Simulator(cfg, RLPrefetcher(run_cfg)).run(trace, agent_seed)
                speedup = result.ipc / baseline.ipc; seed_speeds.append(speedup)
                rows.append({"stage": "search", "candidate": candidate_id, "workload": workload,
                             "trace_seed": 100, "agent_seed": agent_seed, "bw_gbps": bandwidth,
                             "ipc": result.ipc, "speedup": speedup})
            pair_means.append(sum(seed_speeds) / len(seed_speeds))
        scored.append((geometric_mean(pair_means), candidate_id, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    if refine:
        refine_context = _TuningContext(config, refinement_records)
        refined: list[tuple[float, int, AgentConfig]] = []
        for _, candidate_id, candidate in scored[:min(5, len(scored))]:
            speeds = []
            for workload, trace_seed, bandwidth in itertools.product(names, TUNING_TRACE_SEEDS, TUNING_BANDWIDTHS):
                trace = refine_context.trace(workload, trace_seed)
                baseline = refine_context.baseline(workload, trace_seed, bandwidth)
                cfg = config.with_bandwidth(bandwidth)
                for agent_seed in TUNING_AGENT_SEEDS:
                    run_cfg = replace(candidate, miss_lat_ref=cfg.llc.latency + cfg.dram_latency + cfg.transfer_cycles)
                    result = Simulator(cfg, RLPrefetcher(run_cfg)).run(trace, agent_seed)
                    speedup = result.ipc / baseline.ipc; speeds.append(speedup)
                    rows.append({"stage": "refine", "candidate": candidate_id, "workload": workload,
                                 "trace_seed": trace_seed, "agent_seed": agent_seed,
                                 "bw_gbps": bandwidth, "ipc": result.ipc, "speedup": speedup})
            refined.append((geometric_mean(speeds), candidate_id, candidate))
        refined.sort(key=lambda item: item[0], reverse=True); best_score, _, best = refined[0]
    else:
        best_score, _, best = scored[0]
    output = _agent_dict(best)
    _write_rows(Path(out_dir) / "agent_runs.csv", rows, force)
    _write_yaml(Path(config_out), output, force)
    return best
