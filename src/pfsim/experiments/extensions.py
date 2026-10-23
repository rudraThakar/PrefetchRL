from __future__ import annotations

import csv
from dataclasses import replace
import math
from pathlib import Path
import time
from typing import Iterable

from pfsim.config import SimConfig
from pfsim.prefetchers import FutureLookaheadPrefetcher, NoPrefetcher, make_prefetcher
from pfsim.rl import AgentConfig, RLPrefetcher, qtable_from_agent_config
from pfsim.sim import Simulator
from pfsim.sim.metrics import comparative_metrics
from pfsim.trace import generate
from pfsim.trace.format import Trace
from .learning import recovery_times
from .runner import INTERVAL_FIELDS, RUN_FIELDS, _git_commit, _write_csv
from .stats import bootstrap_gmean_ci
from .tune import TEST_WORKLOADS, geometric_mean


def _configured_agent(config: SimConfig, agent_config: AgentConfig, **changes) -> RLPrefetcher:
    return RLPrefetcher(replace(agent_config,
        miss_lat_ref=config.llc.latency + config.dram_latency + config.transfer_cycles, **changes))


def run_e10(config: SimConfig, out_root: str | Path = "results", *, records: int | None = None,
            workloads=("phase_mix", "phase_mix_fast", "phase_mix_gradual"),
            agent_seeds=(0, 1, 2), agent_config: AgentConfig | None = None,
            baseline_config: dict | None = None, force: bool = False) -> tuple[Path, Path]:
    """Compare online, frozen, cold-reset reference, and conventional phase behavior."""
    out=Path(out_root)/"E10"; runs_path=out/"runs.csv"; recovery_path=out/"recovery.csv"
    if not force and (runs_path.exists() or recovery_path.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    acfg=agent_config or AgentConfig(); rows=[]; rec=[]; commit=_git_commit()
    for workload in tuple(workloads):
        trace=generate(workload,0,records); record_boundaries=tuple(trace.meta.get("boundaries",[]))
        run_config = replace(config, stats_interval=min(config.stats_interval, 5_000)) if workload == "phase_mix_fast" else config
        base_sim=Simulator(run_config,NoPrefetcher(),record_boundaries=record_boundaries); baseline=base_sim.run(trace,0)
        boundaries=[base_sim.boundary_l2[b] for b in record_boundaries]
        record_phase_len=int(trace.meta.get("phase_len",300_000))
        phase_len=max(1,int(sum(b-a for a,b in zip([0]+boundaries,boundaries))/len(boundaries))) if boundaries else record_phase_len
        for name in ("stride_pc","bop"):
            pf=make_prefetcher(name,**(baseline_config or {}).get(name,{})); sim=Simulator(run_config,pf); result=sim.run(trace,0)
            rows.append({"workload":workload,"trace_seed":0,"prefetcher":name,"variant":"fixed","agent_seed":"",
                "bw_gbps":config.peak_bw_gbps,**comparative_metrics(result,baseline),"records":len(trace),"wall_seconds":0,
                "config_hash":config.digest(),"git_commit":commit})
        warm_end=min(len(trace),4*record_phase_len)
        warm=Trace(trace.pc[:warm_end],trace.addr[:warm_end],trace.gap[:warm_end],
                   trace.is_write[:warm_end],trace.dep[:warm_end],dict(trace.meta))
        for seed in agent_seeds:
            pretrained=_configured_agent(run_config,acfg); Simulator(run_config,pretrained).run(warm,seed)
            checkpoint=out/"checkpoints"/f"{workload}_{seed}.npz"; pretrained.save(checkpoint)
            for mode in ("online","frozen","reset"):
                if mode=="reset":
                    agent=_configured_agent(run_config,replace(acfg,reset_boundaries=tuple(b+1 for b in boundaries)))
                else:
                    changes={"learn":False,"explore":False} if mode=="frozen" else {}
                    agent=_configured_agent(run_config,acfg,**changes); agent.load(checkpoint)
                sim=Simulator(run_config,agent); started=time.perf_counter(); result=sim.run(trace,seed)
                rows.append({"workload":workload,"trace_seed":0,"prefetcher":"agent","variant":mode,"agent_seed":seed,
                    "bw_gbps":config.peak_bw_gbps,**comparative_metrics(result,baseline),"records":len(trace),
                    "wall_seconds":time.perf_counter()-started,"config_hash":config.digest(),"git_commit":commit})
                _write_csv(out/"intervals"/f"{workload}_{mode}_{seed}.csv",sim.intervals,INTERVAL_FIELDS)
                for item in recovery_times(sim.intervals,boundaries,phase_len):
                    rec.append({"workload":workload,"variant":mode,"agent_seed":seed,**item})
    _write_csv(runs_path,rows,RUN_FIELDS); _write_csv(recovery_path,rec,
        ("workload","variant","agent_seed","boundary","target_ipc","recovery_accesses","adaptation_loss"))
    return runs_path,recovery_path


def run_e11(config: SimConfig, out_root: str | Path = "results", *, records: int = 1_000_000,
            workloads: Iterable[str] = TEST_WORKLOADS, bandwidths=(3.2,6.4,12.8,25.6),
            agent_seeds=(0,1,2), agent_config: AgentConfig | None = None,
            baseline_config: dict | None = None, force: bool = False) -> tuple[Path, Path]:
    out=Path(out_root)/"E11"; runs_path=out/"runs.csv"; summary=out/"summary.csv"
    if not force and (runs_path.exists() or summary.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    acfg=agent_config or AgentConfig(); rows=[]; commit=_git_commit()
    no_sys=tuple(sorted(set(acfg.feature_drop)|{"f5","f6"}))
    variants={"full":acfg,"no_sys_features":replace(acfg,feature_drop=no_sys,
              admission_control=False,reach_control=False)}
    for theta_bw in (.6,.75,.9):
      for theta_acc in (.3,.5):
        variants[f"bw_gate_{theta_bw}_{theta_acc}"]=replace(acfg,feature_drop=no_sys,
            admission_control=False,reach_control=False,bw_gate=True,
            theta_bw=theta_bw,theta_acc=theta_acc)
    for workload in tuple(workloads):
      trace=generate(workload,0,records)
      for bw in bandwidths:
        cfg=config.with_bandwidth(float(bw)); baseline=Simulator(cfg,NoPrefetcher()).run(trace,0)
        for name in ("next_line","stride_pc"):
            result=Simulator(cfg,make_prefetcher(name,**(baseline_config or {}).get(name,{}))).run(trace,0)
            rows.append({"workload":workload,"trace_seed":0,"prefetcher":name,"variant":name,"agent_seed":"","bw_gbps":bw,
                **comparative_metrics(result,baseline),"records":len(trace),"wall_seconds":0,"config_hash":cfg.digest(),"git_commit":commit})
        for variant,agent_cfg in variants.items():
          for seed in agent_seeds:
            result=Simulator(cfg,_configured_agent(cfg,agent_cfg)).run(trace,seed)
            rows.append({"workload":workload,"trace_seed":0,"prefetcher":"agent","variant":variant,"agent_seed":seed,"bw_gbps":bw,
                **comparative_metrics(result,baseline),"records":len(trace),"wall_seconds":0,"config_hash":cfg.digest(),"git_commit":commit})
    _write_csv(runs_path,rows,RUN_FIELDS); output=[]
    for variant in sorted({r["variant"] for r in rows}):
      for bw in bandwidths:
        chosen=[r for r in rows if r["variant"]==variant and float(r["bw_gbps"])==float(bw)]; per=[]
        for w in sorted({r["workload"] for r in chosen}):
            x=[float(r["speedup"]) for r in chosen if r["workload"]==w]; per.append(sum(x)/len(x))
        if per: output.append({"variant":variant,"bw_gbps":bw,"speedup_gmean":geometric_mean(per),
                               "mean_overprediction":sum(float(r["overprediction"]) for r in chosen)/len(chosen),
                               "mean_wasted_requests":sum(float(r["useless"])+float(r["dropped"]) for r in chosen)/len(chosen)})
    _write_csv(summary,output,("variant","bw_gbps","speedup_gmean","mean_overprediction","mean_wasted_requests"))
    return runs_path,summary


def run_e12(out_root: str | Path = "results", *, force: bool = False) -> tuple[Path, Path]:
    """Build storage/adaptation trade-offs and reward-alignment statistics."""
    root=Path(out_root); out=root/"E12"; summary=out/"summary.csv"; correlation=out/"reward_correlation.csv"
    if not force and (summary.exists() or correlation.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    required=[root/"E3"/"runs.csv",root/"E6"/"summary.csv",root/"E10"/"recovery.csv",
              root/"E6"/"runs.csv",root/"E10"/"runs.csv"]
    missing=[p for p in required if not p.exists()]
    if missing: raise FileNotFoundError("E12 requires " + ", ".join(str(p) for p in missing))
    commits=set()
    for path in (required[0],required[3],required[4]):
        with path.open(newline="",encoding="utf-8") as handle:
            commits.update(row.get("git_commit","") for row in csv.DictReader(handle))
    if len(commits)!=1 or any(not commit or commit.endswith("-dirty") for commit in commits):
        raise ValueError(f"E12 requires one clean freeze commit across inputs; found {sorted(commits)}")
    with required[1].open(newline="",encoding="utf-8") as handle: storage=list(csv.DictReader(handle))
    with required[2].open(newline="",encoding="utf-8") as handle: recovery=list(csv.DictReader(handle))
    rec_by_variant={}
    for row in recovery:
        if row.get("recovery_accesses") not in (None,""):
            rec_by_variant.setdefault(row.get("variant","online"),[]).append(float(row["recovery_accesses"]))
    rows=[]
    for row in storage:
        variant=row["variant"]; rec=rec_by_variant.get(variant,rec_by_variant.get("online",[]))
        rows.append({"variant":variant,"speedup_gmean":row["speedup_gmean"],"storage_bits":row["storage_bits"],
                     "mean_recovery_accesses":sum(rec)/len(rec) if rec else ""})
    _write_csv(summary,rows,("variant","speedup_gmean","storage_bits","mean_recovery_accesses"))
    with required[0].open(newline="",encoding="utf-8") as handle: e3=list(csv.DictReader(handle))
    pairs=[(float(r["mean_reward"]),float(r["speedup"])) for r in e3
           if r.get("mean_reward") not in (None,"") and r.get("speedup") not in (None,"")]
    def ranks(values):
        order=sorted(range(len(values)),key=values.__getitem__); result=[0.0]*len(values); i=0
        while i<len(order):
            j=i+1
            while j<len(order) and values[order[j]]==values[order[i]]: j+=1
            rank=(i+j-1)/2+1
            for k in range(i,j): result[order[k]]=rank
            i=j
        return result
    value=""
    if len(pairs)>=2:
        x,y=ranks([p[0] for p in pairs]),ranks([p[1] for p in pairs]); mx=sum(x)/len(x); my=sum(y)/len(y)
        numerator=sum((a-mx)*(b-my) for a,b in zip(x,y)); denominator=math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
        value=numerator/denominator if denominator else 0.0
    _write_csv(correlation,[{"metric":"spearman_reward_speedup","value":value,"runs":len(pairs)}],("metric","value","runs"))
    _plot_e12(rows,out/"fig_tradeoff.png")
    return summary,correlation


def _plot_e12(rows, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for row in rows:
        axes[0].scatter(float(row["storage_bits"]),float(row["speedup_gmean"])); axes[0].annotate(row["variant"],(float(row["storage_bits"]),float(row["speedup_gmean"])),fontsize=7)
        if row["mean_recovery_accesses"]!="": axes[1].scatter(float(row["mean_recovery_accesses"]),float(row["speedup_gmean"])); axes[1].annotate(row["variant"],(float(row["mean_recovery_accesses"]),float(row["speedup_gmean"])),fontsize=7)
    axes[0].set(xlabel="Storage (bits)",ylabel="Geometric-mean speedup"); axes[1].set(xlabel="Mean recovery accesses",ylabel="Geometric-mean speedup")
    fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160); plt.close(fig)


def run_s1(config: SimConfig, out_root: str | Path = "results", *, records: int = 100_000,
           workloads: Iterable[str] = TEST_WORKLOADS, agent_seeds=(0,1,2),
           agent_config: AgentConfig | None = None, baseline_config: dict | None = None,
           force: bool = False) -> tuple[Path, Path]:
    out=Path(out_root)/"S1"; runs_path=out/"runs.csv"; summary=out/"summary.csv"
    if not force and (runs_path.exists() or summary.exists()): raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    variants={"default":config,"rob_128":replace(config,rob_size=128),"rob_512":replace(config,rob_size=512),
              "mshr_16":replace(config,mshr_demand=16),"mshr_64":replace(config,mshr_demand=64),
              "lat_120":replace(config,dram_latency=120),"lat_300":replace(config,dram_latency=300),
              "pf_backlog_4":replace(config,pf_backlog_factor=4)}
    rows=[]; commit=_git_commit(); acfg=agent_config or AgentConfig()
    for variant,cfg in variants.items():
      for workload in tuple(workloads):
        trace=generate(workload,0,records); base_sim=Simulator(cfg,NoPrefetcher(),record_demand_lines=True); baseline=base_sim.run(trace,0)
        for name in ("next_line","stride_pc","bop"):
            result=Simulator(cfg,make_prefetcher(name,**(baseline_config or {}).get(name,{}))).run(trace,0)
            rows.append({"workload":workload,"trace_seed":0,"prefetcher":name,"variant":variant,"agent_seed":"","bw_gbps":cfg.peak_bw_gbps,
                **comparative_metrics(result,baseline),"records":len(trace),"wall_seconds":0,"config_hash":cfg.digest(),"git_commit":commit})
        for seed in agent_seeds:
            result=Simulator(cfg,_configured_agent(cfg,acfg)).run(trace,seed)
            rows.append({"workload":workload,"trace_seed":0,"prefetcher":"agent","variant":variant,"agent_seed":seed,"bw_gbps":cfg.peak_bw_gbps,
                **comparative_metrics(result,baseline),"records":len(trace),"wall_seconds":0,"config_hash":cfg.digest(),"git_commit":commit})
            small=qtable_from_agent_config(acfg,cfg.llc.latency+cfg.dram_latency+cfg.transfer_cycles)
            small_result=Simulator(cfg,small).run(trace,seed)
            rows.append({"workload":workload,"trace_seed":0,"prefetcher":"qtable_small","variant":variant,"agent_seed":seed,"bw_gbps":cfg.peak_bw_gbps,
                **comparative_metrics(small_result,baseline),"records":len(trace),"wall_seconds":0,"config_hash":cfg.digest(),"git_commit":commit})
        future=[]
        for lookahead in (4,8,16,32,64):
            result=Simulator(cfg,FutureLookaheadPrefetcher(base_sim.demand_lines,lookahead)).run(trace,0)
            future.append((result.ipc,result))
        result=max(future,key=lambda x:x[0])[1]
        rows.append({"workload":workload,"trace_seed":0,"prefetcher":"future_lookahead","variant":variant,"agent_seed":"","bw_gbps":cfg.peak_bw_gbps,
            **comparative_metrics(result,baseline),"records":len(trace),"wall_seconds":0,"config_hash":cfg.digest(),"git_commit":commit})
    _write_csv(runs_path,rows,RUN_FIELDS); output=[]
    for variant in variants:
      variant_rows=[]
      for pf in ("next_line","stride_pc","bop","qtable_small","agent","future_lookahead"):
        chosen=[r for r in rows if r["variant"]==variant and r["prefetcher"]==pf]; per=[]
        for w in sorted({r["workload"] for r in chosen}):
            x=[float(r["speedup"]) for r in chosen if r["workload"]==w]; per.append(sum(x)/len(x))
        variant_rows.append({"variant":variant,"prefetcher":pf,"speedup_gmean":geometric_mean(per)})
      for rank,row in enumerate(sorted(variant_rows,key=lambda r:r["speedup_gmean"],reverse=True),1): row["rank"]=rank
      output.extend(variant_rows)
    default={r["prefetcher"]:r for r in output if r["variant"]=="default"}
    for row in output:
        base=default.get(row["prefetcher"],row); row["sign_changed"]=(row["speedup_gmean"]-1)*(base["speedup_gmean"]-1)<0
    _write_csv(summary,output,("variant","prefetcher","speedup_gmean","rank","sign_changed")); return runs_path,summary
