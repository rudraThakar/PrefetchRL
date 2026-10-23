from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import statistics
import time
from typing import Iterable

from pfsim.config import SimConfig
from pfsim.prefetchers import NoPrefetcher
from pfsim.rl import AgentConfig, RLPrefetcher
from pfsim.sim import Simulator
from pfsim.sim.metrics import comparative_metrics
from pfsim.trace import generate
from .runner import INTERVAL_FIELDS, RUN_FIELDS, _git_commit, _write_csv
from .tune import DEV_WORKLOADS


def recovery_times(intervals: list[dict], boundaries: list[int], phase_len: int) -> list[dict]:
    """Compute section 7.2 recovery times in L2-access coordinates."""
    output = []; first_cycle = 4 * phase_len
    for boundary in boundaries:
        if boundary < first_cycle: continue
        end = boundary + phase_len
        steady = [float(row["window_ipc"]) for row in intervals
                  if boundary + phase_len / 2 <= int(row["l2_idx"]) < end]
        if not steady: continue
        target = .9 * statistics.median(steady)
        recovered = next((int(row["l2_idx"]) for row in intervals
                          if int(row["l2_idx"]) >= boundary and float(row["window_ipc"]) >= target), None)
        horizon = min(100_000, phase_len); weighted = covered = 0.0
        previous_end = 0
        for row in intervals:
            end = int(row["l2_idx"]); start = end - int(row.get("l2_accesses", end - previous_end))
            overlap = max(0, min(end, boundary + horizon) - max(start, boundary))
            weighted += overlap * float(row["window_ipc"]); covered += overlap
            previous_end = end
        reference = statistics.median(steady)
        loss = 1 - weighted / (reference * covered) if reference > 0 and covered else ""
        output.append({"boundary": boundary, "target_ipc": target,
                       "recovery_accesses": recovered - boundary if recovered is not None else "",
                       "adaptation_loss": loss})
    return output


def _plot_learning(intervals: list[dict], boundaries: list[int], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fields = (("window_ipc", "Window IPC"), ("window_accuracy", "Window accuracy"),
              ("mean_reward", "Mean reward"))
    for ax, (field, label) in zip(axes, fields):
        ax.plot([row["l2_idx"] for row in intervals], [row[field] for row in intervals])
        for boundary in boundaries: ax.axvline(boundary, color="grey", alpha=.35, linewidth=.8)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("L2 demand accesses"); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)


def run_e4(config: SimConfig, out_root: str | Path = "results", *, records: int | None = None,
           workloads: Iterable[str] = DEV_WORKLOADS, agent_seeds=(0, 1),
           agent_config: AgentConfig | None = None, force: bool = False) -> tuple[Path, Path]:
    out = Path(out_root) / "E4"; runs_path = out / "runs.csv"; recovery_path = out / "recovery.csv"
    if not force and (runs_path.exists() or recovery_path.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    rows = []; recovery_rows = []; commit = _git_commit()
    for workload in tuple(workloads):
        trace = generate(workload, 0, records)
        record_boundaries = tuple(trace.meta.get("boundaries", []))
        base_sim = Simulator(config, NoPrefetcher(), record_boundaries=record_boundaries)
        baseline = base_sim.run(trace)
        for seed in agent_seeds:
            cfg = replace(agent_config or AgentConfig(),
                          miss_lat_ref=config.llc.latency + config.dram_latency + config.transfer_cycles)
            sim = Simulator(config, RLPrefetcher(cfg)); started = time.perf_counter(); result = sim.run(trace, seed)
            metrics = comparative_metrics(result, baseline)
            rows.append({"workload": workload, "trace_seed": 0, "prefetcher": "agent", "variant": "online",
                         "agent_seed": seed, "bw_gbps": config.peak_bw_gbps, **metrics, "records": len(trace),
                         "wall_seconds": time.perf_counter() - started, "config_hash": config.digest(),
                         "git_commit": commit})
            key = f"{workload}_seed{seed}"; _write_csv(out / "intervals" / f"{key}.csv", sim.intervals, INTERVAL_FIELDS)
            plot_boundaries = [base_sim.boundary_l2[b] for b in record_boundaries]
            _plot_learning(sim.intervals, plot_boundaries, out / f"fig_learning_{key}.png")
            if workload == "phase_mix":
                boundaries = plot_boundaries
                diffs = [b-a for a,b in zip([0]+boundaries,boundaries)]
                phase_len = max(1, int(statistics.median(diffs))) if diffs else int(trace.meta.get("phase_len",300_000))
                for item in recovery_times(sim.intervals, boundaries, phase_len):
                    recovery_rows.append({"workload": workload, "agent_seed": seed, **item})
    _write_csv(runs_path, rows, RUN_FIELDS)
    _write_csv(recovery_path, recovery_rows,
               ("workload", "agent_seed", "boundary", "target_ipc", "recovery_accesses", "adaptation_loss"))
    return runs_path, recovery_path
