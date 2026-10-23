from __future__ import annotations

import csv
from dataclasses import replace
import math
import statistics
from pathlib import Path
import subprocess
import time
from typing import Iterable

from pfsim.config import SimConfig
from pfsim.prefetchers import FutureLookaheadPrefetcher, NoPrefetcher, make_prefetcher
from pfsim.rl import AgentConfig, RLPrefetcher, qtable_from_agent_config
from pfsim.sim import Simulator
from pfsim.sim.metrics import comparative_metrics
from pfsim.trace import WORKLOADS, generate

RUN_FIELDS = (
    "workload", "trace_seed", "prefetcher", "variant", "agent_seed", "bw_gbps",
    "records", "instructions", "cycles", "ipc", "speedup", "l2_accesses", "l2_misses",
    "coverage", "accuracy", "late_fraction", "pollution_rate", "dram_reads",
    "overprediction", "avg_bw_util", "redundant_rate", "dropped_rate", "storage_bits",
    "mean_reward", "action_histogram", "gap_mean", "wall_seconds", "config_hash", "git_commit",
)
INTERVAL_FIELDS = ("l2_idx", "l2_accesses", "window_ipc", "window_accuracy",
                   "window_coverage_proxy", "mean_reward", "epsilon")
DEFAULT_PREFETCHERS = ("none", "next_line", "stride_pc", "bop", "qtable_small", "agent", "future_lookahead")


def _git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], check=True,
                               capture_output=True, text=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_csv(path: Path, rows: list[dict], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _run(config: SimConfig, trace, prefetcher, seed: int):
    simulator = Simulator(config, prefetcher)
    started = time.perf_counter(); result = simulator.run(trace, seed)
    return result, simulator.intervals, time.perf_counter() - started


def _summaries(rows: list[dict]) -> list[dict]:
    from .stats import bootstrap_gmean_ci
    from .tune import DEV_WORKLOADS, TEST_WORKLOADS
    output = []
    names = sorted({row["prefetcher"] for row in rows})
    workloads = sorted({row["workload"] for row in rows})
    for name in names:
        selected = [row for row in rows if row["prefetcher"] == name]
        for workload in workloads:
            group = [row for row in selected if row["workload"] == workload]
            if not group: continue
            speeds = [float(row["speedup"]) for row in group]
            low, high = bootstrap_gmean_ci({workload: speeds})
            output.append({"prefetcher": name, "workload": workload, "runs": len(group),
                           **{field: sum(float(row[field]) for row in group) / len(group)
                              for field in ("speedup", "accuracy", "coverage", "pollution_rate",
                                            "overprediction")},
                           "speedup_std": statistics.pstdev(speeds) if len(speeds) > 1 else 0.0,
                           "ci_low": low, "ci_high": high})
        groups = (("DEV", set(DEV_WORKLOADS)), ("TEST", set(TEST_WORKLOADS)), ("ALL", set(workloads)))
        for label, allowed in groups:
            by_workload = {w: [float(r["speedup"]) for r in selected if r["workload"] == w]
                           for w in workloads if w in allowed and any(r["workload"] == w for r in selected)}
            if not by_workload: continue
            means = [sum(v) / len(v) for v in by_workload.values()]
            aggregate = [r for r in selected if r["workload"] in allowed]
            low, high = bootstrap_gmean_ci(by_workload)
            output.append({"prefetcher": name, "workload": label, "runs": len(aggregate),
                "speedup": math.exp(sum(math.log(max(v, 1e-12)) for v in means) / len(means)),
                **{field: sum(float(r[field]) for r in aggregate) / len(aggregate)
                   for field in ("accuracy", "coverage", "pollution_rate", "overprediction")},
                "speedup_std": statistics.pstdev(means) if len(means) > 1 else 0.0,
                "ci_low": low, "ci_high": high})
    return output


def run_e1(config: SimConfig, out_root: str | Path = "results", *, budget: str = "quick",
           records: int | None = None, workloads: Iterable[str] | None = None,
           prefetchers: Iterable[str] = DEFAULT_PREFETCHERS, force: bool = False,
           agent_config: AgentConfig | None = None,
           baseline_config: dict[str, dict[str, Any]] | None = None) -> tuple[Path, Path]:
    if budget not in {"quick", "full"}:
        raise ValueError("budget must be 'quick' or 'full'")
    out = Path(out_root) / "E1"
    runs_path, summary_path = out / "runs.csv", out / "summary.csv"
    if not force and (runs_path.exists() or summary_path.exists()):
        raise FileExistsError(f"refusing to overwrite {out}; pass force=True")
    workload_names = tuple(workloads or WORKLOADS[:13])
    trace_seeds = (0,) if budget == "quick" else (0, 1, 2)
    requested_records = records if records is not None else (1_000_000 if budget == "quick" else None)
    commit = _git_commit(); rows: list[dict] = []
    for workload in workload_names:
        for trace_seed in trace_seeds:
            trace = generate(workload, trace_seed, requested_records)
            baseline_sim = Simulator(config, NoPrefetcher(), record_demand_lines="future_lookahead" in prefetchers)
            begin = time.perf_counter(); baseline = baseline_sim.run(trace, 0)
            baseline_wall = time.perf_counter() - begin
            for name in prefetchers:
                seeds = ((0, 1) if budget == "quick" else (0, 1, 2, 3, 4)) if name in {"agent", "qtable_small"} else (0,)
                for agent_seed in seeds:
                    variant = "default"
                    if name == "none":
                        result, intervals, wall = baseline, baseline_sim.intervals, baseline_wall
                    elif name == "future_lookahead":
                        candidates = []
                        for lookahead in (4, 8, 16, 32, 64):
                            item = _run(config, trace, FutureLookaheadPrefetcher(baseline_sim.demand_lines, lookahead), 0)
                            candidates.append((item[0].ipc, lookahead, item))
                        _, lookahead, (result, intervals, wall) = max(candidates, key=lambda item: item[0])
                        variant = f"lookahead_{lookahead}"
                    else:
                        if name == "agent":
                            pf = RLPrefetcher(replace(agent_config or AgentConfig(),
                                miss_lat_ref=config.llc.latency + config.dram_latency + config.transfer_cycles))
                        elif name == "qtable_small":
                            pf = qtable_from_agent_config(agent_config or AgentConfig(),
                                config.llc.latency + config.dram_latency + config.transfer_cycles)
                        else:
                            pf = make_prefetcher(name, **(baseline_config or {}).get(name, {}))
                        result, intervals, wall = _run(config, trace, pf, agent_seed)
                    metrics = comparative_metrics(result, baseline)
                    row = {"workload": workload, "trace_seed": trace_seed, "prefetcher": name,
                           "variant": variant, "agent_seed": agent_seed if name in {"agent", "qtable_small"} else "",
                           "bw_gbps": config.peak_bw_gbps, **metrics, "records": len(trace),
                           "wall_seconds": wall, "config_hash": config.digest(), "git_commit": commit}
                    rows.append(row)
                    key = f"{workload}_ts{trace_seed}_{name}_{variant}_as{agent_seed}"
                    _write_csv(out / "intervals" / f"{key}.csv", intervals, INTERVAL_FIELDS)
    _write_csv(runs_path, rows, RUN_FIELDS)
    _write_csv(summary_path, _summaries(rows),
               ("prefetcher", "workload", "runs", "speedup", "accuracy", "coverage",
                "pollution_rate", "overprediction", "speedup_std", "ci_low", "ci_high"))
    from .stats import paired_outcomes
    agent_values={}; baseline_values={}
    for workload in workload_names:
      for trace_seed in trace_seeds:
        agent_runs=[float(r["speedup"]) for r in rows if r["workload"]==workload and r["trace_seed"]==trace_seed and r["prefetcher"]=="agent"]
        baseline_runs=[float(r["speedup"]) for r in rows if r["workload"]==workload and r["trace_seed"]==trace_seed and r["prefetcher"] in {"next_line","stride_pc","bop"}]
        if agent_runs and baseline_runs:
            agent_values[(workload,trace_seed)]=sum(agent_runs)/len(agent_runs); baseline_values[(workload,trace_seed)]=max(baseline_runs)
    outcomes=paired_outcomes(agent_values,baseline_values)
    _write_csv(out/"paired.csv",[{"comparison":"agent_vs_best_classic",**outcomes}],
               ("comparison","win","tie","loss"))
    return runs_path, summary_path
