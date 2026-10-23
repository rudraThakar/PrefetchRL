from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
import numpy as np
from pathlib import Path
import time
from typing import Iterable

from pfsim.config import SimConfig
from pfsim.prefetchers import FutureLookaheadPrefetcher, NoPrefetcher, make_prefetcher
from pfsim.rl import AgentConfig, RLPrefetcher
from pfsim.rl.actions import decode
from pfsim.sim import Simulator
from pfsim.sim.metrics import comparative_metrics
from pfsim.trace import WORKLOADS, generate
from .runner import RUN_FIELDS, _git_commit, _write_csv
from .tune import DEV_WORKLOADS, TEST_WORKLOADS, geometric_mean


def _agent(config: SimConfig, agent_config: AgentConfig, seed: int, weights: Path | None = None,
           *, frozen: bool = False) -> RLPrefetcher:
    cfg = replace(agent_config, learn=not frozen, explore=not frozen,
                  miss_lat_ref=config.llc.latency + config.dram_latency + config.transfer_cycles)
    agent = RLPrefetcher(cfg)
    if weights is not None: agent.load(weights)
    return agent


def run_e5(config: SimConfig, out_root: str | Path = "results", *, records: int = 1_000_000,
           train_workloads: Iterable[str] = DEV_WORKLOADS,
           test_workloads: Iterable[str] = TEST_WORKLOADS, passes: int = 2,
           agent_seeds=(0, 1, 2), agent_config: AgentConfig | None = None,
           force: bool = False) -> tuple[Path, Path]:
    """Pretrain sequentially on development traces, then measure transfer."""
    out = Path(out_root) / "E5"; runs_path = out / "runs.csv"; weights = out / "pretrained_weights.npz"
    if not force and (runs_path.exists() or weights.exists()):
        raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    out.mkdir(parents=True, exist_ok=True); cfg = agent_config or AgentConfig()
    current: Path | None = None
    rng = np.random.default_rng(100)
    for pass_id in range(passes):
        order_names = list(train_workloads); rng.shuffle(order_names)
        for order, workload in enumerate(order_names):
            agent = _agent(config, cfg, 100 + pass_id, current)
            Simulator(config, agent).run(generate(workload, 100 + pass_id, records), 100 + pass_id)
            checkpoint = out / "checkpoints" / f"pass{pass_id}_{order}_{workload}.npz"
            agent.save(checkpoint); current = checkpoint
    if current is None: raise ValueError("at least one training workload and pass are required")
    final = _agent(config, cfg, 0, current); final.save(weights)
    rows = []; commit = _git_commit()
    for workload in tuple(test_workloads):
        trace = generate(workload, 0, records); baseline = Simulator(config, NoPrefetcher()).run(trace, 0)
        for mode in ("cold", "warm_learning", "warm_frozen"):
            for seed in agent_seeds:
                agent = _agent(config, cfg, seed, None if mode == "cold" else weights,
                               frozen=mode == "warm_frozen")
                started = time.perf_counter(); result = Simulator(config, agent).run(trace, seed)
                rows.append({"workload": workload, "trace_seed": 0, "prefetcher": "agent",
                    "variant": mode, "agent_seed": seed, "bw_gbps": config.peak_bw_gbps,
                    **comparative_metrics(result, baseline), "records": len(trace),
                    "wall_seconds": time.perf_counter() - started, "config_hash": config.digest(),
                    "git_commit": commit})
    _write_csv(runs_path, rows, RUN_FIELDS)
    summary = out / "summary.csv"; summary_rows = []
    for mode in ("cold", "warm_learning", "warm_frozen"):
        chosen = [r for r in rows if r["variant"] == mode]
        values = []
        for workload in sorted({r["workload"] for r in chosen}):
            group = [float(r["speedup"]) for r in chosen if r["workload"] == workload]
            values.append(sum(group) / len(group))
        summary_rows.append({"variant": mode, "workloads": len(values),
                             "speedup_gmean": geometric_mean(values)})
    _write_csv(summary, summary_rows, ("variant", "workloads", "speedup_gmean"))
    return runs_path, summary


def run_e6(config: SimConfig, out_root: str | Path = "results", *, records: int = 1_000_000,
           workloads: Iterable[str] = TEST_WORKLOADS, agent_seeds=(0, 1, 2),
           agent_config: AgentConfig | None = None, force: bool = False) -> tuple[Path, Path]:
    out = Path(out_root) / "E6"; runs_path = out / "runs.csv"; summary = out / "summary.csv"
    if not force and (runs_path.exists() or summary.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    base_cfg = agent_config or AgentConfig(); variants = {}
    for bits in (4, 6, 8, 10): variants[f"pc_bits_{bits}"] = replace(base_cfg, pc_bits=bits)
    variants.update(no_f2=replace(base_cfg, feature_drop=tuple(sorted(set(base_cfg.feature_drop) | {"f2"}))),
                    no_f3=replace(base_cfg, feature_drop=tuple(sorted(set(base_cfg.feature_drop) | {"f3"}))),
                    compact=replace(base_cfg,pc_bits=6,feature_drop=tuple(sorted(set(base_cfg.feature_drop)|{"f2"}))),
                    sparse_reach=replace(base_cfg, reaches=(1, 4, 16), low_accuracy_max_reach=1))
    rows = []; commit = _git_commit()
    for workload in tuple(workloads):
        trace = generate(workload, 0, records); baseline = Simulator(config, NoPrefetcher()).run(trace, 0)
        for name in ("next_line","stride_pc","bop"):
            pf=make_prefetcher(name); result=Simulator(config,pf).run(trace,0)
            rows.append({"workload":workload,"trace_seed":0,"prefetcher":name,"variant":name,
                "agent_seed":"","bw_gbps":config.peak_bw_gbps,**comparative_metrics(result,baseline),
                "records":len(trace),"wall_seconds":0,"config_hash":config.digest(),"git_commit":commit})
        for name, cfg in variants.items():
            for seed in agent_seeds:
                agent = _agent(config, cfg, seed); started = time.perf_counter()
                result = Simulator(config, agent).run(trace, seed)
                rows.append({"workload": workload, "trace_seed": 0, "prefetcher": "agent", "variant": name,
                    "agent_seed": seed, "bw_gbps": config.peak_bw_gbps, **comparative_metrics(result, baseline),
                    "records": len(trace), "wall_seconds": time.perf_counter() - started,
                    "config_hash": config.digest(), "git_commit": commit})
    _write_csv(runs_path, rows, RUN_FIELDS); output=[]
    for name, cfg in variants.items():
        chosen=[r for r in rows if r["variant"] == name]; per=[]
        for w in sorted({r["workload"] for r in chosen}):
            x=[float(r["speedup"]) for r in chosen if r["workload"] == w]; per.append(sum(x)/len(x))
        output.append({"variant":name,"speedup_gmean":geometric_mean(per),"storage_bits":RLPrefetcher(cfg).storage_bits()})
    for name in ("next_line","stride_pc","bop"):
        chosen=[r for r in rows if r["variant"]==name]; per=[]
        for w in sorted({r["workload"] for r in chosen}):
            x=[float(r["speedup"]) for r in chosen if r["workload"]==w]; per.append(sum(x)/len(x))
        output.append({"variant":name,"speedup_gmean":geometric_mean(per),"storage_bits":make_prefetcher(name).storage_bits()})
    _write_csv(summary,output,("variant","speedup_gmean","storage_bits")); _plot_scatter(output,out/"fig_storage_speedup.png")
    return runs_path, summary


def _plot_scatter(rows, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for row in rows: ax.scatter(row["storage_bits"],row["speedup_gmean"]); ax.annotate(row["variant"],(row["storage_bits"],row["speedup_gmean"]),fontsize=7)
    ax.set(xlabel="Storage (bits)",ylabel="Geometric-mean speedup"); fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160); plt.close(fig)


def run_e7(config: SimConfig, out_root: str | Path = "results", *, records: int = 1_000_000,
           workloads: Iterable[str] = WORKLOADS[:13], agent_seeds=(0, 1, 2),
           agent_config: AgentConfig | None = None, force: bool = False) -> tuple[Path, Path]:
    out=Path(out_root)/"E7"; hist=out/"actions.csv"; checks=out/"checks.csv"
    if not force and (hist.exists() or checks.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    rows=[]
    measure_config=replace(config,warmup_fraction=2/3)
    for workload in tuple(workloads):
        trace=generate(workload,0,records)
        for seed in agent_seeds:
            agent=_agent(measure_config,agent_config or AgentConfig(),seed); Simulator(measure_config,agent).run(trace,seed)
            for action,count in enumerate(agent.stats()["measured_action_counts"]):
                pattern,reach=decode(action); rows.append({"workload":workload,"agent_seed":seed,"action":action,
                    "pattern":pattern or "NONE","reach":reach,"count":count})
    _write_csv(hist,rows,("workload","agent_seed","action","pattern","reach","count")); output=[]
    for workload in tuple(workloads):
        selected=[r for r in rows if r["workload"]==workload]; total=sum(r["count"] for r in selected)
        none=sum(r["count"] for r in selected if r["pattern"]=="NONE")
        learned=sum(r["count"] for r in selected if r["pattern"] in {"PC_D1","PAGE_D1"})
        output.append({"workload":workload,"none_fraction":none/total if total else 0,
                       "learned_delta_fraction":learned/max(1,total-none)})
    _write_csv(checks,output,("workload","none_fraction","learned_delta_fraction"))
    return hist,checks


def run_e8(config: SimConfig, out_root: str | Path = "results", *, records: int = 1_000_000,
           workloads: Iterable[str] = TEST_WORKLOADS, agent_seeds=(0, 1, 2),
           agent_config: AgentConfig | None = None, baseline_config: dict | None = None,
           force: bool = False) -> tuple[Path, Path]:
    out=Path(out_root)/"E8"; runs_path=out/"runs.csv"; summary=out/"summary.csv"
    if not force and (runs_path.exists() or summary.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    rows=[]; report=[]; commit=_git_commit()
    for workload in tuple(workloads):
        trace=generate(workload,0,records); base_sim=Simulator(
            config,NoPrefetcher(),record_demand_lines=True); base=base_sim.run(trace,0)
        candidates=[]
        for name in ("next_line","stride_pc","bop"):
            result=Simulator(config,make_prefetcher(name,**(baseline_config or {}).get(name,{}))).run(trace,0)
            candidates.append((base.cycles/result.cycles,name,result))
        best_speed,best_name,best=max(candidates,key=lambda x:x[0])
        future_candidates=[]
        for lookahead in (4,8,16,32,64):
            pf=FutureLookaheadPrefetcher(base_sim.demand_lines,lookahead); result=Simulator(config,pf).run(trace,0)
            future_candidates.append((base.cycles/result.cycles,lookahead,result))
        future_speed,lookahead,future=max(future_candidates,key=lambda x:x[0])
        speeds=[]
        for seed in agent_seeds:
            result=Simulator(config,_agent(config,agent_config or AgentConfig(),seed)).run(trace,seed); speeds.append(base.cycles/result.cycles)
            rows.append({"workload":workload,"trace_seed":0,"prefetcher":"agent","variant":"default","agent_seed":seed,
                "bw_gbps":config.peak_bw_gbps,**comparative_metrics(result,base),"records":len(trace),"wall_seconds":0,
                "config_hash":config.digest(),"git_commit":commit})
        agent_speed=sum(speeds)/len(speeds); captured=((agent_speed-1)/(future_speed-1) if future_speed>1.02 else "")
        report.append({"workload":workload,"agent_speedup":agent_speed,"best_baseline":best_name,
            "best_baseline_speedup":best_speed,"future_lookahead":lookahead,"future_speedup":future_speed,"gain_captured":captured})
    _write_csv(runs_path,rows,RUN_FIELDS); _write_csv(summary,report,("workload","agent_speedup","best_baseline","best_baseline_speedup","future_lookahead","future_speedup","gain_captured"))
    return runs_path,summary
