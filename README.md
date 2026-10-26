# pfsim

`pfsim` is a deterministic, trace-driven single-core memory-system simulator for evaluating conventional and reinforcement-learning prefetchers. This repository is being built from the accompanying implementation specification.

Current status: the package scaffold, generated workload streams, cache/DRAM/core simulator, outcome and interval accounting, conventional and future-lookahead baselines, comparative metrics, online double-Q agent, `qtable_small`, checkpoint persistence, resumable/parallel tuning, E1–E8, E10–E12, and S1 runners are implemented. The main tuner searches only learned-policy parameters; hand-written admission/reach controls are disabled. The full tuning campaign has not yet been executed, and existing tuned values/results are provisional artifacts.

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
pfsim experiment E1 --budget quick --records 10000 --workload seq_stream --allow-provisional --force
```

Omit `--records` and `--workload` for the specified one-million-record quick budget across all workloads. E1 writes `runs.csv`, `summary.csv`, and interval CSV files under `results/E1/`.

Run a small tuning smoke check:

```bash
pfsim tune --target baselines --records 10000 --workload seq_stream \
  --out results/smoke-tune --config-out /tmp/baselines_smoke.yaml --force
pfsim tune --target agent --records 10000 --trials 2 --workload seq_stream --no-refine --jobs 2 \
  --out results/smoke-tune --config-out /tmp/agent_smoke.yaml --force
```

Tuning prints candidate progress and an ETA. If it is interrupted, continue from the completed candidate checkpoints with:

```bash
pfsim tune --target baselines --records 10000 --workload seq_stream \
  --out results/smoke-tune --config-out /tmp/baselines_smoke.yaml --resume
pfsim tune --target agent --records 10000 --trials 2 --workload seq_stream --no-refine --jobs 2 \
  --out results/smoke-tune --config-out /tmp/agent_smoke.yaml --resume
```

Use `--force` to start a fresh search and `--resume` to continue one; they cannot be combined. Tuning artifact checks happen before work begins, and checkpoint names contain a source hash. Candidate-level parallelism is controlled by `--jobs`; on a 7 GB machine start with 4 and monitor RSS.

Agent tuning uses only the seven development workloads, trace seeds 100/101, and agent seeds 100/101/102. Test workloads are rejected by the tuning API and CLI. For a pipeline smoke test, reduce `--records`, `--trials`, add one or more development `--workload` arguments, and use `--no-refine`; do not treat that output as a tuned production configuration.

After code freeze, the final campaign is:

```bash
pfsim tune --target baselines --records 1000000 --force
pfsim tune --target agent --records 1000000 --refinement-records 1000000 --trials 60 --jobs 4 --force
```

Both searches write per-candidate checkpoints. Replace `--force` with `--resume` after an interruption. Commit the resulting `configs/baselines_tuned.yaml` and `configs/agent_tuned.yaml` before running held-out evaluation.

Bandwidth and ablation experiments are available as:

```bash
pfsim experiment E2 --records 10000 --workload seq_stream --allow-provisional --force
pfsim experiment E3 --records 10000 --workload seq_stream --allow-provisional --force
pfsim experiment E4 --records 10000 --workload phase_mix --allow-provisional --force
pfsim experiment E5 --records 10000 --workload random_uniform --allow-provisional --force
pfsim experiment E6 --records 10000 --workload random_uniform --allow-provisional --force
pfsim experiment E7 --records 10000 --workload random_uniform --allow-provisional --force
pfsim experiment E8 --records 10000 --workload random_uniform --allow-provisional --force
pfsim experiment E10 --records 10000 --workload phase_mix_fast --allow-provisional --force
pfsim experiment E11 --records 10000 --workload random_uniform --allow-provisional --force
pfsim experiment S1 --records 10000 --workload seq_stream --allow-provisional --force
```

Omit the record/workload restrictions only after configurations are frozen. These commands write raw runs, summaries, interval records, and PNG figures under their respective `results/` directories.
Experiment commands load `configs/agent_tuned.yaml` and `configs/baselines_tuned.yaml` by default. Use `--agent-config` or `--baselines-config` only for an explicit alternative.
E12 is an aggregation stage and runs after E3, E6, and E10:

```bash
pfsim experiment E12 --allow-provisional --force
```

The CLI refuses to overwrite traces and results unless `--force` is supplied. Runs print their configuration hash and Git commit.

## Scope of conclusions

The project currently follows data-strategy Option A. Results will describe the specified synthetic and kernel pattern classes only; they will not claim generalization to real software.
