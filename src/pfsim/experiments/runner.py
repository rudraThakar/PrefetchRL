from __future__ import annotations

import csv
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import time
from typing import Iterable

from pfsim.config import SimConfig
from pfsim.prefetchers import FutureLookaheadPrefetcher, NoPrefetcher, make_prefetcher
from pfsim.sim import Simulator
from pfsim.sim.metrics import comparative_metrics
from pfsim.trace import WORKLOADS, generate

RUN_FIELDS = (
    "workload", "trace_seed", "prefetcher", "variant", "agent_seed", "bw_gbps",
    "records", "instructions", "cycles", "ipc", "speedup", "l2_accesses", "l2_misses",
    "coverage", "accuracy", "late_fraction", "pollution_rate", "dram_reads",
    "overprediction", "avg_bw_util", "redundant_rate", "dropped_rate", "storage_bits",
    "wall_seconds", "config_hash", "git_commit",
)
INTERVAL_FIELDS = ("l2_idx", "l2_accesses", "window_ipc", "window_accuracy",
                   "window_coverage_proxy", "mean_reward", "epsilon")
DEFAULT_PREFETCHERS = ("none", "next_line", "stride_pc", "bop", "qtable_small", "agent", "future_lookahead")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
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
    output = []
    names = sorted({row["prefetcher"] for row in rows})
    workloads = sorted({row["workload"] for row in rows})
    for name in names:
        selected = [row for row in rows if row["prefetcher"] == name]
        for workload in workloads:
            group = [row for row in selected if row["workload"] == workload]
            if not group: continue
            output.append({"prefetcher": name, "workload": workload, "runs": len(group),
                           **{field: sum(float(row[field]) for row in group) / len(group)
                              for field in ("speedup", "accuracy", "coverage", "pollution_rate",
                                            "overprediction")}})
        per_workload = []
        for workload in workloads:
            group = [row for row in selected if row["workload"] == workload]
            if group: per_workload.append(sum(float(row["speedup"]) for row in group) / len(group))
        if per_workload:
            output.append({"prefetcher": name, "workload": "ALL", "runs": len(selected),
                           "speedup": math.exp(sum(math.log(max(v, 1e-12)) for v in per_workload)
                                               / len(per_workload)),
                           **{field: sum(float(row[field]) for row in selected) / len(selected)
                              for field in ("accuracy", "coverage", "pollution_rate", "overprediction")}})
    return output


def run_e1(config: SimConfig, out_root: str | Path = "results", *, budget: str = "quick",
           records: int | None = None, workloads: Iterable[str] | None = None,
           prefetchers: Iterable[str] = DEFAULT_PREFETCHERS, force: bool = False) -> tuple[Path, Path]:
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
            baseline_sim = Simulator(config, NoPrefetcher())
            begin = time.perf_counter(); baseline = baseline_sim.run(trace, 0)
            baseline_wall = time.perf_counter() - begin
            for name in prefetchers:
                seeds = (0, 1) if name == "agent" and budget == "quick" else ((0, 1, 2, 3, 4) if name == "agent" else (0,))
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
                        result, intervals, wall = _run(config, trace, make_prefetcher(name), agent_seed)
                    metrics = comparative_metrics(result, baseline)
                    row = {"workload": workload, "trace_seed": trace_seed, "prefetcher": name,
                           "variant": variant, "agent_seed": agent_seed if name == "agent" else "",
                           "bw_gbps": config.peak_bw_gbps, **metrics, "records": len(trace),
                           "wall_seconds": wall, "config_hash": config.digest(), "git_commit": commit}
                    rows.append(row)
                    key = f"{workload}_ts{trace_seed}_{name}_{variant}_as{agent_seed}"
                    _write_csv(out / "intervals" / f"{key}.csv", intervals, INTERVAL_FIELDS)
    _write_csv(runs_path, rows, RUN_FIELDS)
    _write_csv(summary_path, _summaries(rows),
               ("prefetcher", "workload", "runs", "speedup", "accuracy", "coverage",
                "pollution_rate", "overprediction"))
    return runs_path, summary_path
