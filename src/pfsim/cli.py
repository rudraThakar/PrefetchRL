from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import yaml
import csv

from pfsim.config import load_agent_config, load_config
from pfsim.prefetchers import make_prefetcher
from pfsim.prefetchers import FutureLookaheadPrefetcher, NoPrefetcher
from pfsim.sim import Simulator
from pfsim.sim.metrics import comparative_metrics
from pfsim.trace import WORKLOADS, generate, load_trace, save_trace
from pfsim.experiments import run_e1, run_e2, run_e3, run_e4, run_e5, run_e6, run_e7, run_e8
from pfsim.experiments import run_e10, run_e11, run_e12, run_s1
from pfsim.experiments.tune import DEV_WORKLOADS, tune_agent, tune_baselines
from pfsim.rl import RLPrefetcher, qtable_from_agent_config


def _commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], check=True,
                               capture_output=True, text=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def cmd_gen(args):
    names = WORKLOADS[:13] if args.workload == "all" else (args.workload,)
    n = args.records if args.records else (100_000 if args.quick else None)
    cfg = load_config(None)
    print(f"config_hash={cfg.digest()} git_commit={_commit()}")
    for name in names:
        trace = generate(name, args.seed, n)
        path = Path(args.out) / f"{name}_s{args.seed}.npz"
        if path.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {path}; pass --force")
        save_trace(trace, path); print(path)


def cmd_run(args):
    cfg = load_config(args.config)
    if args.bw is not None: cfg = cfg.with_bandwidth(args.bw)
    trace = load_trace(args.trace)
    baseline_sim = Simulator(cfg, NoPrefetcher(), record_demand_lines=args.prefetcher == "future_lookahead")
    baseline = baseline_sim.run(trace, args.seed)
    if args.prefetcher == "future_lookahead":
        pf = FutureLookaheadPrefetcher(baseline_sim.demand_lines, args.lookahead)
    elif args.prefetcher == "agent":
        pf = RLPrefetcher(load_agent_config(args.agent_config))
        if args.load_weights: pf.load(args.load_weights)
    elif args.prefetcher == "qtable_small":
        pf = qtable_from_agent_config(load_agent_config(args.agent_config),
                                     cfg.llc.latency + cfg.dram_latency + cfg.transfer_cycles)
    else:
        pf = make_prefetcher(args.prefetcher)
    started = time.perf_counter()
    result = baseline if args.prefetcher == "none" else Simulator(cfg, pf).run(trace, args.seed)
    wall_seconds = time.perf_counter() - started
    if args.save_weights:
        if args.prefetcher != "agent": raise SystemExit("--save-weights requires --prefetcher agent")
        if Path(args.save_weights).exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {args.save_weights}; pass --force")
        pf.save(args.save_weights)
    payload = {"workload": trace.meta.get("workload", Path(args.trace).stem),
               "trace_seed": trace.meta.get("seed"), "prefetcher": args.prefetcher,
               "variant": "default", "agent_seed": args.seed, "bw_gbps": cfg.peak_bw_gbps,
               **comparative_metrics(result, baseline), "wall_seconds": wall_seconds,
               "config_hash": cfg.digest(), "git_commit": _commit()}
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(f"config_hash={cfg.digest()} git_commit={payload['git_commit']}")
    if args.out: _write(Path(args.out), encoded + "\n", args.force)
    print(encoded)


def cmd_experiment(args):
    cfg = load_config(args.config)
    agent_cfg = load_agent_config(args.agent_config)
    if agent_cfg.tuning_status != "full" and not args.allow_provisional:
        raise SystemExit(f"{args.agent_config} is {agent_cfg.tuning_status!r}, not a full tuning result; "
                         "pass --allow-provisional only for smoke tests")
    baseline_meta=Path(args.baselines_config+".meta.yaml")
    baseline_status=(yaml.safe_load(baseline_meta.read_text(encoding="utf-8")) or {}).get("tuning_status") if baseline_meta.exists() else "provisional"
    if baseline_status != "full" and not args.allow_provisional:
        raise SystemExit(f"{args.baselines_config} is {baseline_status!r}, not a full tuning result; "
                         "pass --allow-provisional only for smoke tests")
    baseline_cfg = yaml.safe_load(Path(args.baselines_config).read_text(encoding="utf-8")) or {}
    try:
        if args.experiment == "E1":
            runs, summary = run_e1(cfg, args.out, budget=args.budget, records=args.records,
                                   workloads=args.workload or None, force=args.force,
                                   agent_config=agent_cfg, baseline_config=baseline_cfg)
        elif args.experiment == "E2":
            runs, summary = run_e2(cfg, args.out, records=args.records or 1_000_000,
                                   workloads=args.workload or None, force=args.force,
                                   agent_config=agent_cfg, baseline_config=baseline_cfg)
        elif args.experiment == "E3":
            runs, summary = run_e3(cfg, args.out, records=args.records or 1_000_000,
                                   workloads=args.workload or None, force=args.force,
                                   full_config=agent_cfg)
        elif args.experiment == "E4":
            runs, summary = run_e4(cfg, args.out, records=args.records,
                                   workloads=args.workload or None or DEV_WORKLOADS,
                                   agent_config=agent_cfg, force=args.force)
        elif args.experiment == "E5":
            kwargs = {"test_workloads": args.workload} if args.workload else {}
            runs, summary = run_e5(cfg, args.out, records=args.records or 1_000_000,
                                   agent_config=agent_cfg, force=args.force, **kwargs)
        elif args.experiment == "E6":
            kwargs = {"workloads": args.workload} if args.workload else {}
            runs, summary = run_e6(cfg, args.out, records=args.records or 1_000_000,
                                   agent_config=agent_cfg, force=args.force, **kwargs)
        elif args.experiment == "E7":
            kwargs = {"workloads": args.workload} if args.workload else {}
            runs, summary = run_e7(cfg, args.out, records=args.records or 1_000_000,
                                   agent_config=agent_cfg, force=args.force, **kwargs)
        elif args.experiment == "E8":
            kwargs = {"workloads": args.workload} if args.workload else {}
            runs, summary = run_e8(cfg, args.out, records=args.records or 1_000_000,
                                   agent_config=agent_cfg, baseline_config=baseline_cfg,
                                   force=args.force, **kwargs)
        elif args.experiment == "E10":
            kwargs = {"workloads": args.workload} if args.workload else {}
            runs, summary = run_e10(cfg, args.out, records=args.records,
                                    agent_config=agent_cfg, baseline_config=baseline_cfg,
                                    force=args.force, **kwargs)
        elif args.experiment == "E11":
            kwargs = {"workloads": args.workload} if args.workload else {}
            runs, summary = run_e11(cfg, args.out, records=args.records or 1_000_000,
                                    agent_config=agent_cfg, baseline_config=baseline_cfg,
                                    force=args.force, **kwargs)
        elif args.experiment == "E12":
            runs, summary = run_e12(args.out, force=args.force)
        else:
            kwargs = {"workloads": args.workload} if args.workload else {}
            runs, summary = run_s1(cfg, args.out, records=args.records or 100_000,
                                   agent_config=agent_cfg, baseline_config=baseline_cfg,
                                   force=args.force, **kwargs)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"config_hash={cfg.digest()} git_commit={_commit()}")
    print(runs); print(summary)


def cmd_tune(args):
    if args.force and args.resume:
        raise SystemExit("choose either --force for a fresh search or --resume for an interrupted one")
    cfg = load_config(args.config)
    workloads = args.workload or DEV_WORKLOADS
    try:
        if args.target == "baselines":
            chosen = tune_baselines(cfg, workloads=workloads, records=args.records,
                out_dir=args.out, config_out=args.config_out or "configs/baselines_tuned.yaml",
                force=args.force, resume=args.resume)
            print(yaml.safe_dump(chosen, sort_keys=False).strip())
        else:
            chosen = tune_agent(cfg, workloads=workloads, records=args.records, trials=args.trials,
                search_seed=args.search_seed, refine=not args.no_refine,
                refinement_records=args.refinement_records, out_dir=args.out,
                config_out=args.config_out or "configs/agent_tuned.yaml", force=args.force, jobs=args.jobs,
                resume=args.resume)
            print(yaml.safe_dump(asdict(chosen), sort_keys=False).strip())
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"config_hash={cfg.digest()} git_commit={_commit()}")


def cmd_report(args):
    root=Path(args.results); lines=["# pfsim results report",""]
    for name in ("E1","E2","E3","E4","E5","E6","E7","E8","E10","E11","E12","S1"):
        candidates=(root/name/"summary.csv",root/name/"recovery.csv",root/name/"checks.csv")
        path=next((p for p in candidates if p.exists()),None)
        if path is None: continue
        with path.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
        lines.extend((f"## {name}","",f"Source: `{path}`; rows: {len(rows)}.",""))
    output=Path(args.out); _write(output,"\n".join(lines)+"\n",args.force); print(output)


def build_parser():
    parser = argparse.ArgumentParser(prog="pfsim")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("gen-traces")
    gen.add_argument("--out", default="traces"); gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--workload", choices=("all",) + WORKLOADS, default="all")
    gen.add_argument("--records", type=int); gen.add_argument("--quick", action="store_true")
    gen.add_argument("--force", action="store_true"); gen.set_defaults(func=cmd_gen)
    run = sub.add_parser("run")
    run.add_argument("--trace", required=True); run.add_argument("--prefetcher", default="none",
                     choices=("none", "next_line", "stride_pc", "bop", "agent", "qtable_small", "future_lookahead"))
    run.add_argument("--config", default="configs/default.yaml"); run.add_argument("--bw", type=float)
    run.add_argument("--lookahead", type=int, default=16)
    run.add_argument("--agent-config", default="configs/agent_default.yaml")
    run.add_argument("--load-weights"); run.add_argument("--save-weights")
    run.add_argument("--seed", type=int, default=0); run.add_argument("--out")
    run.add_argument("--force", action="store_true"); run.set_defaults(func=cmd_run)
    experiment = sub.add_parser("experiment")
    experiment.add_argument("experiment", choices=("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
                                                   "E10", "E11", "E12", "S1"))
    experiment.add_argument("--budget", choices=("quick", "full"), default="quick")
    experiment.add_argument("--records", type=int)
    experiment.add_argument("--workload", action="append", choices=WORKLOADS)
    experiment.add_argument("--config", default="configs/default.yaml")
    experiment.add_argument("--agent-config", default="configs/agent_tuned.yaml")
    experiment.add_argument("--baselines-config", default="configs/baselines_tuned.yaml")
    experiment.add_argument("--out", default="results")
    experiment.add_argument("--force", action="store_true")
    experiment.add_argument("--allow-provisional", action="store_true")
    experiment.set_defaults(func=cmd_experiment)
    tune = sub.add_parser("tune")
    tune.add_argument("--target", choices=("agent", "baselines"), required=True)
    tune.add_argument("--records", type=int, default=1_000_000)
    tune.add_argument("--refinement-records", type=int)
    tune.add_argument("--trials", type=int, default=60); tune.add_argument("--search-seed", type=int, default=2026)
    tune.add_argument("--no-refine", action="store_true")
    tune.add_argument("--jobs", type=int, default=1)
    tune.add_argument("--workload", action="append", choices=WORKLOADS)
    tune.add_argument("--config", default="configs/default.yaml"); tune.add_argument("--config-out")
    tune.add_argument("--out", default="results/tune"); tune.add_argument("--force", action="store_true")
    tune.add_argument("--resume", action="store_true")
    tune.set_defaults(func=cmd_tune)
    report = sub.add_parser("report")
    report.add_argument("--results", default="results"); report.add_argument("--out", default="results/REPORT.md")
    report.add_argument("--force", action="store_true"); report.set_defaults(func=cmd_report)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv); args.func(args)


if __name__ == "__main__":
    main()
