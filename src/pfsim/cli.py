from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import yaml

from pfsim.config import load_agent_config, load_config
from pfsim.prefetchers import make_prefetcher
from pfsim.prefetchers import FutureLookaheadPrefetcher, NoPrefetcher
from pfsim.sim import Simulator
from pfsim.sim.metrics import comparative_metrics
from pfsim.trace import WORKLOADS, generate, load_trace, save_trace
from pfsim.experiments import run_e1
from pfsim.experiments.tune import DEV_WORKLOADS, tune_agent, tune_baselines
from pfsim.rl import RLPrefetcher


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
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
    baseline_sim = Simulator(cfg, NoPrefetcher())
    baseline = baseline_sim.run(trace, args.seed)
    if args.prefetcher == "future_lookahead":
        pf = FutureLookaheadPrefetcher(baseline_sim.demand_lines, args.lookahead)
    elif args.prefetcher == "agent":
        pf = RLPrefetcher(load_agent_config(args.agent_config))
        if args.load_weights: pf.load(args.load_weights)
    else:
        pf = make_prefetcher(args.prefetcher)
    result = baseline if args.prefetcher == "none" else Simulator(cfg, pf).run(trace, args.seed)
    if args.save_weights:
        if args.prefetcher != "agent": raise SystemExit("--save-weights requires --prefetcher agent")
        if Path(args.save_weights).exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {args.save_weights}; pass --force")
        pf.save(args.save_weights)
    payload = {"workload": trace.meta.get("workload", Path(args.trace).stem),
               "trace_seed": trace.meta.get("seed"), "prefetcher": args.prefetcher,
               "variant": "default", "agent_seed": args.seed, "bw_gbps": cfg.peak_bw_gbps,
               **comparative_metrics(result, baseline), "config_hash": cfg.digest(), "git_commit": _commit()}
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(f"config_hash={cfg.digest()} git_commit={payload['git_commit']}")
    if args.out: _write(Path(args.out), encoded + "\n", args.force)
    print(encoded)


def cmd_experiment(args):
    if args.experiment != "E1":
        raise SystemExit("only E1 is implemented currently")
    cfg = load_config(args.config)
    try:
        runs, summary = run_e1(cfg, args.out, budget=args.budget, records=args.records,
                               workloads=args.workload or None, force=args.force)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"config_hash={cfg.digest()} git_commit={_commit()}")
    print(runs); print(summary)


def cmd_tune(args):
    cfg = load_config(args.config)
    workloads = args.workload or DEV_WORKLOADS
    try:
        if args.target == "baselines":
            chosen = tune_baselines(cfg, workloads=workloads, records=args.records,
                out_dir=args.out, config_out=args.config_out or "configs/baselines_tuned.yaml",
                force=args.force)
            print(yaml.safe_dump(chosen, sort_keys=False).strip())
        else:
            chosen = tune_agent(cfg, workloads=workloads, records=args.records, trials=args.trials,
                search_seed=args.search_seed, refine=not args.no_refine,
                refinement_records=args.refinement_records, out_dir=args.out,
                config_out=args.config_out or "configs/agent_tuned.yaml", force=args.force)
            print(yaml.safe_dump(asdict(chosen), sort_keys=False).strip())
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"config_hash={cfg.digest()} git_commit={_commit()}")


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
    experiment.add_argument("experiment", choices=("E1",))
    experiment.add_argument("--budget", choices=("quick", "full"), default="quick")
    experiment.add_argument("--records", type=int)
    experiment.add_argument("--workload", action="append", choices=WORKLOADS[:13])
    experiment.add_argument("--config", default="configs/default.yaml")
    experiment.add_argument("--out", default="results")
    experiment.add_argument("--force", action="store_true")
    experiment.set_defaults(func=cmd_experiment)
    tune = sub.add_parser("tune")
    tune.add_argument("--target", choices=("agent", "baselines"), required=True)
    tune.add_argument("--records", type=int, default=1_000_000)
    tune.add_argument("--refinement-records", type=int)
    tune.add_argument("--trials", type=int, default=60); tune.add_argument("--search-seed", type=int, default=2026)
    tune.add_argument("--no-refine", action="store_true")
    tune.add_argument("--workload", action="append", choices=WORKLOADS)
    tune.add_argument("--config", default="configs/default.yaml"); tune.add_argument("--config-out")
    tune.add_argument("--out", default="results/tune"); tune.add_argument("--force", action="store_true")
    tune.set_defaults(func=cmd_tune)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv); args.func(args)


if __name__ == "__main__":
    main()
