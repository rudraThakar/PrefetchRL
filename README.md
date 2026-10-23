# pfsim

`pfsim` is a deterministic, trace-driven single-core memory-system simulator for evaluating conventional and reinforcement-learning prefetchers. This repository is being built from the accompanying implementation specification.

Current status: the package scaffold, exact generated workload streams, cache/DRAM/core simulator, outcome and interval accounting, conventional and future-lookahead baselines, comparative metrics, the online double-Q agent, `qtable_small`, YAML ablations, checkpoint persistence, tuning pipelines, and an initial E1 runner are implemented. The mandatory timing microbenchmarks pass. The full tuning campaign has not yet been executed; E2–E12, plots, and final results are subsequent milestones.

## Setup

```bash
python -m pip install -e '.[dev]'
pytest
```

Generate a small trace and run it:

```bash
pfsim gen-traces --out traces --workload stride_const --seed 0 --records 10000
pfsim run --trace traces/stride_const_s0.npz --prefetcher stride_pc --out run.json
```

Run an ablation or save/load learned weights:

```bash
pfsim run --trace traces/stride_const_s0.npz --prefetcher agent \
  --agent-config configs/variants/no_sys_features.yaml --save-weights weights.npz --out run.json
pfsim run --trace traces/stride_const_s0.npz --prefetcher agent \
  --agent-config configs/variants/no_sys_features.yaml --load-weights weights.npz --out warm.json
```

Run a small E1 smoke experiment:

```bash
pfsim experiment E1 --budget quick --records 10000 --workload seq_stream --force
```

Omit `--records` and `--workload` for the specified one-million-record quick budget across all workloads. E1 writes `runs.csv`, `summary.csv`, and interval CSV files under `results/E1/`.

Run the specified tuning searches:

```bash
pfsim tune --target baselines --force
pfsim tune --target agent --trials 60 --force
```

Agent tuning uses only the seven development workloads, trace seeds 100/101, and agent seeds 100/101/102. Test workloads are rejected by the tuning API and CLI. For a pipeline smoke test, reduce `--records`, `--trials`, add one or more development `--workload` arguments, and use `--no-refine`; do not treat that output as a tuned production configuration.

The CLI refuses to overwrite traces and results unless `--force` is supplied. Runs print their configuration hash and Git commit.

## Scope of conclusions

The project currently follows data-strategy Option A. Results will describe the specified synthetic and kernel pattern classes only; they will not claim generalization to real software.
