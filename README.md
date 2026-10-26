# RL-Based Self-Optimizing Memory Prefetcher

`pfsim` is a deterministic, trace-driven memory-system simulator for training and evaluating an online reinforcement-learning memory prefetcher. The project method is RL: memory-prefetch decisions are learned through online Double Q-learning from delayed simulator feedback. Next-line, PC-stride, BOP, and future-lookahead policies are included only as evaluation baselines.

The complete implementation, frozen tuned agent configuration, experiment runners, tests, raw-result archive, and report sources are included in this repository.

## Final result

The definitive E1 evaluation contains 585 million-record simulations across 13 workloads, three trace seeds, and five learned-policy seeds. The final RL agent achieves:

- **1.1909x geometric-mean performance**, or **19.09% over no prefetching**;
- 95% bootstrap confidence interval of **[1.1864x, 1.1950x]**;
- 12.09% improvement over next-line;
- 4.24% improvement over BOP; and
- 4.35% improvement over the compact Q-table agent.

The tuned PC-stride baseline remains 6.13% faster overall. The project therefore demonstrates useful adaptive RL behavior rather than claiming universal superiority. See [FINAL_REPORT.md](FINAL_REPORT.md) for E1 through E12 analysis, limitations, and conclusions.

## RL agent

The agent observes a six-component discrete state containing recent address deltas, hashed PC context, page-local behavior, bandwidth utilization, prefetch accuracy, queue occupancy, and pollution. It selects from a dynamically masked discrete action space containing no-prefetch plus address patterns at reaches 1, 2, 4, 8, and 16.

Learning uses additive Double Q-learning with delayed outcome rewards for useful, late, useless, redundant, dropped, and polluting prefetches. The final configuration was selected using 60 candidate configurations and refinement of the best five at one million records.

The trained deliverables are:

- [`configs/agent_tuned.yaml`](configs/agent_tuned.yaml): frozen full-tuning agent and reward configuration;
- [`configs/baselines_tuned.yaml`](configs/baselines_tuned.yaml): frozen baseline parameters;
- `final_results/results/tune/`: complete search and refinement records;
- `final_results/results/E5/pretrained_weights.npz`: experimental transferable Q-table checkpoint; and
- `final_results/configs/`: configuration snapshot archived with the remote results.

The primary agent is intentionally an online learner. It starts from the tuned optimistic initialization and adapts during each workload. The E5 checkpoint is provided for checkpoint reproduction, but it is not the default policy because the transfer experiment found that cold online learning performed better.

## Repository structure

```text
configs/                 Frozen simulator, RL-agent, baseline, and ablation configs
src/pfsim/               Simulator, RL agent, prefetchers, CLI, tuning, experiments
tests/                    Unit, accounting, behavior, CLI, and experiment tests
final_results/            Archived remote tuning and E1 to E12 result suite
report/                   Two-column LaTeX assignment report and figure
FINAL_REPORT.md           Full technical analysis and conclusions
pyproject.toml            Package metadata, dependencies, and CLI entry point
```

## Requirements

- Python 3.10 or newer
- Linux or another environment supported by NumPy, pandas, and Matplotlib
- Approximately 2 GB free memory for ordinary runs
- Multiple CPU cores are recommended for the full tuning campaign
- A GPU is not required because the simulator and tabular Q-learning implementation are CPU-bound

Runtime dependencies declared in [`pyproject.toml`](pyproject.toml):

- NumPy 1.23 or newer
- pandas 1.5 or newer
- Matplotlib 3.6 or newer
- PyYAML 6 or newer
- pytest 7 or newer for development and validation

## Installation

Clone the repository and enter it:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd "RL-based memory prefetcher"
```

Create an isolated environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'
```

Confirm that the command-line interface is available:

```bash
pfsim --help
```

## Validate the installation

Run the automated test suite:

```bash
python3 -m pytest -q
```

Generate a small trace and execute the final RL agent:

```bash
pfsim gen-traces \
  --out traces \
  --workload stride_const \
  --seed 0 \
  --records 10000

pfsim run \
  --trace traces/stride_const_s0.npz \
  --prefetcher agent \
  --agent-config configs/agent_tuned.yaml \
  --out run_agent.json \
  --force
```

Run a classical baseline for comparison:

```bash
pfsim run \
  --trace traces/stride_const_s0.npz \
  --prefetcher stride_pc \
  --out run_stride.json \
  --force
```

## Save and restore RL weights

The online Q tables can be checkpointed and restored:

```bash
pfsim run \
  --trace traces/stride_const_s0.npz \
  --prefetcher agent \
  --agent-config configs/agent_tuned.yaml \
  --save-weights weights.npz \
  --out trained_run.json \
  --force

pfsim run \
  --trace traces/stride_const_s0.npz \
  --prefetcher agent \
  --agent-config configs/agent_tuned.yaml \
  --load-weights weights.npz \
  --out restored_run.json \
  --force
```

## Quick reproduction

Use the frozen full agent and baseline configurations for a short end-to-end E1 check:

```bash
pfsim experiment E1 \
  --budget quick \
  --records 10000 \
  --workload seq_stream \
  --out results/smoke \
  --force
```

Outputs are written to:

```text
results/smoke/E1/runs.csv
results/smoke/E1/summary.csv
results/smoke/E1/paired.csv
results/smoke/E1/intervals/
```

## Reproduce full tuning

The reported configurations have already been trained and frozen. To repeat the complete tuning campaign from source, run the baseline search first:

```bash
pfsim tune \
  --target baselines \
  --records 1000000 \
  --force
```

Then run the RL-agent search and refinement:

```bash
pfsim tune \
  --target agent \
  --records 1000000 \
  --refinement-records 1000000 \
  --trials 60 \
  --jobs 8 \
  --force
```

Set `--jobs` to the number of physical CPU cores available. Replace `--force` with `--resume` after an interruption. Do not combine `--force` and `--resume`.

Agent tuning uses only the seven development workloads. It evaluates two bandwidths, agent seeds 100, 101, and 102, and refinement trace seeds 100 and 101. Held-out test workloads are not used for agent selection.

For a fast tuning-pipeline check:

```bash
pfsim tune \
  --target agent \
  --records 10000 \
  --trials 2 \
  --workload seq_stream \
  --no-refine \
  --jobs 2 \
  --out results/smoke-tune \
  --config-out /tmp/agent_smoke.yaml \
  --force
```

The smoke configuration is only a pipeline check and must not replace `configs/agent_tuned.yaml` in reported results.

## Reproduce the primary evaluation

Run the complete E1 comparison at one million records:

```bash
pfsim experiment E1 \
  --budget full \
  --records 1000000 \
  --out results/final \
  --force
```

This executes 585 simulations over all 13 workloads. It may be split by repeated `--workload` arguments across independent machines or processes because every workload and seed is deterministic and independent. Sharded CSV files must be merged with exactly one header and checked for 585 unique run keys.

## Reproduce extended experiments

The following commands use the frozen full configurations by default:

```bash
pfsim experiment E2  --records 1000000 --out results/final --force
pfsim experiment E3  --records 1000000 --out results/final --force
pfsim experiment E4  --out results/final --force
pfsim experiment E5  --records 1000000 --out results/final --force
pfsim experiment E6  --records 1000000 --out results/final --force
pfsim experiment E7  --records 1000000 --out results/final --force
pfsim experiment E8  --records 1000000 --out results/final --force
pfsim experiment E10 --records 2000000 --out results/final --force
pfsim experiment E11 --records 1000000 --out results/final --force
pfsim experiment S1  --records 100000 --out results/final --force
```

E12 aggregates E3, E6, and E10 and requires those inputs to come from one clean Git commit:

```bash
pfsim experiment E12 --out results/final --force
```

Generate a Markdown index of completed experiments:

```bash
pfsim report \
  --results results/final \
  --out results/final/REPORT.md \
  --force
```

## Reproduce the report

The two-column assignment report is located at [`report/main.tex`](report/main.tex). Upload the following structure to Overleaf and compile with pdfLaTeX:

```text
main.tex
figures/results.pdf
```

Replace the team-name and GitHub-link placeholders before submission. The detailed Markdown report is available at [FINAL_REPORT.md](FINAL_REPORT.md).

## Reproducibility safeguards

- The CLI refuses to overwrite outputs unless `--force` is supplied.
- Tuning checkpoints contain source hashes and campaign parameters.
- Interrupted tuning campaigns support `--resume`.
- Experiment outputs record the simulator configuration hash and Git commit.
- Tuned configurations are explicitly marked `tuning_status: full`.
- Random seeds are passed explicitly and NumPy generators are deterministic.
- E12 rejects mixed or dirty input commits.

## Scope and limitations

Results describe the provided synthetic and kernel-pattern traces. They do not establish generalization to arbitrary real applications or multicore systems. Future lookahead uses unavailable future information and is included only as a diagnostic upper bound. Classical prefetchers are evaluation controls; the proposed decision-making method is the Double Q-learning agent.

## Citation

If this repository is used in a report or demonstration, cite the repository URL and the included technical report. Add team names and the final GitHub link here before submission.
