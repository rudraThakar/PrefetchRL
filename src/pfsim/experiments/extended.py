from __future__ import annotations

import csv
from dataclasses import replace
import math
from pathlib import Path
import time
from typing import Iterable

from pfsim.config import SimConfig, overlay_agent_config
from pfsim.prefetchers import FutureLookaheadPrefetcher, NoPrefetcher, make_prefetcher
from pfsim.rl import AgentConfig, RLPrefetcher, qtable_from_agent_config
from pfsim.sim import Simulator
from pfsim.sim.metrics import comparative_metrics
from pfsim.trace import WORKLOADS, generate
from .runner import INTERVAL_FIELDS, RUN_FIELDS, _git_commit, _write_csv
from .stats import bootstrap_gmean_ci


def _aggregate(rows: list[dict], group_field: str, extra_field: str | None = None) -> list[dict]:
    keys = sorted({(row[group_field], row.get(extra_field) if extra_field else None) for row in rows})
    output = []
    for primary, secondary in keys:
        selected = [row for row in rows if row[group_field] == primary and
                    (extra_field is None or row.get(extra_field) == secondary)]
        workload_means = []
        for workload in sorted({row["workload"] for row in selected}):
            values = [float(row["speedup"]) for row in selected if row["workload"] == workload]
            workload_means.append(sum(values) / len(values))
        row = {group_field: primary, "workloads": len(workload_means), "runs": len(selected),
               "speedup_gmean": math.exp(sum(math.log(max(v, 1e-12)) for v in workload_means)
                                           / len(workload_means))}
        grouped = {workload: [float(item["speedup"]) for item in selected if item["workload"] == workload]
                   for workload in sorted({item["workload"] for item in selected})}
        row["ci_low"], row["ci_high"] = bootstrap_gmean_ci(grouped)
        if extra_field: row[extra_field] = secondary
        output.append(row)
    return output


def _plot_e2(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in sorted({row["prefetcher"] for row in summary}):
        points = sorted((float(row["bw_gbps"]), float(row["speedup_gmean"]))
                        for row in summary if row["prefetcher"] == name)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=name)
    ax.axhline(1, color="black", linewidth=.8); ax.set(xlabel="DRAM bandwidth (GB/s)", ylabel="Geometric-mean speedup")
    ax.legend(fontsize=8); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)


def _plot_e2_workloads(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for workload in ("seq_stream", "random_uniform", "hash_lookup", "phase_mix"):
        chosen=[r for r in rows if r["workload"]==workload]
        if not chosen: continue
        fig,ax=plt.subplots(figsize=(7,4))
        for name in sorted({r["prefetcher"] for r in chosen}):
            points=[]
            for bw in sorted({float(r["bw_gbps"]) for r in chosen}):
                values=[float(r["speedup"]) for r in chosen if r["prefetcher"]==name and float(r["bw_gbps"])==bw]
                if values: points.append((bw,sum(values)/len(values)))
            ax.plot([p[0] for p in points],[p[1] for p in points],marker="o",label=name)
        ax.axhline(1,color="black",linewidth=.8); ax.set(xlabel="DRAM bandwidth (GB/s)",ylabel="Speedup",title=workload)
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out/f"fig_speedup_bandwidth_{workload}.png",dpi=160); plt.close(fig)


def _plot_e3(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{row['variant']}\n{row['bw_gbps']}" for row in summary]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * .55), 5))
    ax.bar(range(len(labels)), [float(row["speedup_gmean"]) for row in summary])
    ax.axhline(1, color="black", linewidth=.8); ax.set_ylabel("Geometric-mean speedup")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)


def _plot_e3_workloads(rows: list[dict], out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for workload in sorted({r["workload"] for r in rows}):
      for bw in sorted({float(r["bw_gbps"]) for r in rows}):
        chosen=[r for r in rows if r["workload"]==workload and float(r["bw_gbps"])==bw]
        if not chosen: continue
        names=sorted({r["variant"] for r in chosen}); values=[]
        for name in names:
            x=[float(r["speedup"]) for r in chosen if r["variant"]==name]; values.append(sum(x)/len(x))
        fig,ax=plt.subplots(figsize=(max(8,len(names)*.55),4)); ax.bar(range(len(names)),values); ax.axhline(1,color="black",linewidth=.8)
        ax.set_xticks(range(len(names)),names,rotation=60,ha="right",fontsize=7); ax.set_ylabel("Speedup"); fig.tight_layout()
        fig.savefig(out/f"fig_ablation_{workload}_{bw}.png",dpi=160); plt.close(fig)


def run_e2(config: SimConfig, out_root: str | Path = "results", *, records: int = 1_000_000,
           workloads: Iterable[str] | None = None, bandwidths=(3.2, 6.4, 12.8, 25.6, 51.2),
           prefetchers=("none", "next_line", "stride_pc", "bop", "qtable_small", "agent", "future_lookahead"),
           agent_seeds=(0, 1, 2), force: bool = False, agent_config: AgentConfig | None = None,
           baseline_config: dict | None = None) -> tuple[Path, Path]:
    out = Path(out_root) / "E2"; runs_path = out / "runs.csv"; summary_path = out / "summary.csv"
    if not force and (runs_path.exists() or summary_path.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    rows = []; commit = _git_commit()
    for workload in tuple(workloads or WORKLOADS[:13]):
        trace = generate(workload, 0, records)
        for bandwidth in bandwidths:
            cfg = config.with_bandwidth(float(bandwidth)); base_sim = Simulator(
                cfg, NoPrefetcher(), record_demand_lines="future_lookahead" in prefetchers)
            baseline = base_sim.run(trace, 0)
            for name in prefetchers:
                seeds = agent_seeds if name in {"agent", "qtable_small"} else (0,)
                for seed in seeds:
                    start = time.perf_counter(); variant = "default"
                    if name == "none": result, sim = baseline, base_sim
                    elif name == "future_lookahead":
                        candidates = []
                        for lookahead in (4, 8, 16, 32, 64):
                            candidate_sim = Simulator(cfg, FutureLookaheadPrefetcher(base_sim.demand_lines, lookahead))
                            candidate_result = candidate_sim.run(trace)
                            candidates.append((candidate_result.ipc, lookahead, candidate_sim, candidate_result))
                        _, lookahead, sim, result = max(candidates, key=lambda item: item[0])
                        variant = f"lookahead_{lookahead}"
                    else:
                        if name == "agent":
                            pf = RLPrefetcher(replace(agent_config or AgentConfig(),
                                miss_lat_ref=cfg.llc.latency + cfg.dram_latency + cfg.transfer_cycles))
                        elif name == "qtable_small":
                            pf = qtable_from_agent_config(agent_config or AgentConfig(),
                                cfg.llc.latency + cfg.dram_latency + cfg.transfer_cycles)
                        else:
                            pf = make_prefetcher(name, **(baseline_config or {}).get(name, {}))
                        sim = Simulator(cfg, pf); result = sim.run(trace, seed)
                    metrics = comparative_metrics(result, baseline)
                    rows.append({"workload": workload, "trace_seed": 0, "prefetcher": name,
                        "variant": variant, "agent_seed": seed if name in {"agent", "qtable_small"} else "",
                        "bw_gbps": bandwidth, **metrics, "records": len(trace),
                        "wall_seconds": time.perf_counter() - start, "config_hash": cfg.digest(), "git_commit": commit})
                    _write_csv(out / "intervals" / f"{workload}_{bandwidth}_{name}_{seed}.csv", sim.intervals, INTERVAL_FIELDS)
    summary = _aggregate(rows, "prefetcher", "bw_gbps")
    _write_csv(runs_path, rows, RUN_FIELDS); _write_csv(summary_path, summary,
        ("prefetcher", "bw_gbps", "workloads", "runs", "speedup_gmean", "ci_low", "ci_high"))
    _plot_e2(summary, out / "fig_speedup_bandwidth.png")
    _plot_e2_workloads(rows, out)
    return runs_path, summary_path


E3_VARIANTS = ("full", "no_pollution", "no_bw_term", "no_graded", "no_reach", "bandit",
               "no_double_q", "no_sys_features", "no_page_features", "no_f2", "softmax",
               "eps_fixed", "compact", "qtable_small")


def run_e3(config: SimConfig, out_root: str | Path = "results", *, records: int = 1_000_000,
           workloads: Iterable[str] | None = None, variants: Iterable[str] = E3_VARIANTS,
           bandwidths=(3.2, 12.8), agent_seeds=(0, 1, 2), force: bool = False,
           full_config: AgentConfig | None = None) -> tuple[Path, Path]:
    out = Path(out_root) / "E3"; runs_path = out / "runs.csv"; summary_path = out / "summary.csv"
    if not force and (runs_path.exists() or summary_path.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    variant_names = tuple(variants); rows = []; commit = _git_commit()
    for workload in tuple(workloads or WORKLOADS[:13]):
        trace = generate(workload, 0, records)
        for bandwidth in bandwidths:
            cfg = config.with_bandwidth(float(bandwidth)); baseline = Simulator(cfg, NoPrefetcher()).run(trace)
            for variant in variant_names:
                if variant != "qtable_small":
                    path = None if variant == "full" else Path("configs/variants") / f"{variant}.yaml"
                    base = full_config or AgentConfig()
                    agent_cfg = base if variant == "full" else overlay_agent_config(base, path)
                    agent_cfg = replace(agent_cfg, miss_lat_ref=cfg.llc.latency + cfg.dram_latency + cfg.transfer_cycles)
                for seed in agent_seeds:
                    pf = (qtable_from_agent_config(full_config or AgentConfig(),
                          cfg.llc.latency + cfg.dram_latency + cfg.transfer_cycles)
                          if variant == "qtable_small" else RLPrefetcher(agent_cfg))
                    sim = Simulator(cfg, pf); start = time.perf_counter(); result = sim.run(trace, seed)
                    metrics = comparative_metrics(result, baseline)
                    rows.append({"workload": workload, "trace_seed": 0, "prefetcher": "agent",
                        "variant": variant, "agent_seed": seed, "bw_gbps": bandwidth, **metrics,
                        "records": len(trace), "wall_seconds": time.perf_counter() - start,
                        "config_hash": cfg.digest(), "git_commit": commit})
                    _write_csv(out / "intervals" / f"{workload}_{bandwidth}_{variant}_{seed}.csv", sim.intervals, INTERVAL_FIELDS)
    summary = _aggregate(rows, "variant", "bw_gbps")
    full_by_bw = {row["bw_gbps"]: row["speedup_gmean"] for row in summary if row["variant"] == "full"}
    for row in summary:
        row["delta_vs_full"] = row["speedup_gmean"] - full_by_bw.get(row["bw_gbps"], row["speedup_gmean"])
    _write_csv(runs_path, rows, RUN_FIELDS); _write_csv(summary_path, summary,
        ("variant", "bw_gbps", "workloads", "runs", "speedup_gmean", "ci_low", "ci_high", "delta_vs_full"))
    _plot_e3(summary, out / "fig_ablation_speedup.png")
    _plot_e3_workloads(rows, out)
    return runs_path, summary_path
