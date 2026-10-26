# pfsim

`pfsim` is a deterministic, trace-driven single-core memory-system simulator for evaluating conventional and reinforcement-learning prefetchers. This repository is being built from the accompanying implementation specification.

Current status: the simulator, conventional and future-lookahead baselines, online double-Q agent, checkpoint persistence, tuning pipeline, and E1–E8/E10–E12/S1 runners are implemented. The full 60-candidate M8 campaign and five-candidate refinement are complete, and the frozen agent and baseline configurations are marked `full`. The authoritative full-seed E1 is under `results/final/E1`; the remote extended E2–E12 suite is archived under `final_results/results`. The submission-ready analysis is in [FINAL_REPORT.md](FINAL_REPORT.md).

## Final headline result

The definitive E1 evaluation contains 585 million-record simulations across 13 workloads, three trace seeds, and five learned-policy seeds. The final RL prefetcher achieves **1.1909x geomean performance (19.09% over no prefetching)** with a 95% bootstrap interval of **[1.1864x, 1.1950x]**. It is 12.09% faster than next-line, 4.24% faster than BOP, and 4.35% faster than the compact Q-table policy. The tuned PC-stride baseline remains 6.13% faster overall, so the project claims useful adaptive performance rather than universal dominance.

On held-out workloads, the agent retains **1.1137x** geomean performance and is close to the strongest held-out aggregate, next-line at 1.1176x. The complete remote E2–E12 suite is archived under `final_results/`; it includes million-record bandwidth and ablation studies, transfer, storage, action, headroom, phase-adaptation, controller, and sensitivity analyses. See [FINAL_REPORT.md](FINAL_REPORT.md) for the provenance audit, tables, limitations, and conclusions.

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

The completed full tuning campaign is reproducible with:

```bash
pfsim tune --target baselines --records 1000000 --force
pfsim tune --target agent --records 1000000 --refinement-records 1000000 --trials 60 --jobs 4 --force
```

Both searches write per-candidate checkpoints. Replace `--force` with `--resume` after an interruption. Commit the resulting `configs/baselines_tuned.yaml` and `configs/agent_tuned.yaml` before running held-out evaluation.

The primary final evaluation command is:

```bash
pfsim experiment E1 --budget full --records 1000000 --out results/final --force
```

E1 is deterministic per seed and workload, so workloads may be sharded across independent processes and their raw rows merged without changing the experimental population. The submitted result contains exactly 585 unique runs.

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

Generate an index of completed result sets with:

```bash
pfsim report --results results/final --out results/final/REPORT.md --force
```

## Scope of conclusions

The project follows data-strategy Option A. Results describe the specified synthetic and kernel pattern classes only; they do not claim generalization to real software. The locally merged E1 is the definitive full-seed result. Remote E2–E12 provide the extended analysis; experiments with dirty Git provenance are identified as supplementary in the final report rather than silently treated as clean-freeze evidence.
