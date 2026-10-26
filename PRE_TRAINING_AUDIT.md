# Pre-training audit of `pfsim`

Scope: the roadmap (`Roadmap RL-Based Self-Optimizing Memory Prefetcher (1).md`) against all code in `src/`, `configs/`, `tests/`, `results/`, `README.md` and `DECISIONS.md`, as of the working tree on 2026-10-24. No code was changed.

Method: I read the whole spec and all source files, ran the 73 tests (all pass, 6 s), and ran four small experiments of my own (throughput and memory, effect of the instruction gap on speedup, default-length L2 access counts for all 13 workloads, and short agent runs on a latency-bound variant of the workloads). Numbers quoted below come from those runs. Anything I did not verify by running is marked "by reading".

"Training" here means the tuning campaign (`pfsim tune --target agent`, 60 candidates) plus the later E1–E12 runs, because the agent learns online and has no separate training stage except E5.

## Verdict

**Do not start the tuning campaign yet.** Five problems would make the expensive run either uninformative or invalid, and each one changes something that tuned values depend on (workloads, agent behaviour, simulator semantics). Fixing any of them after tuning means tuning again.

| # | Blocker | Why it blocks |
|---|---------|---------------|
| B1 | Most workloads are DRAM-channel-bound, so every prefetcher scores speedup ≈ 1.00 | The tuning objective is nearly flat on 5 of the 7 dev workloads |
| B2 | Two kernels violate the spec's "at least 400,000 L2 demand accesses" rule | `matmul_blocked` (dev) and `stencil2d` (test) give almost no learning signal |
| B3 | Agent runs at about 7–8k records/s, tuning is serial, and memory grows without bound | The campaign is roughly 70–120 CPU-hours serial and would run out of RAM in parallel |
| B4 | Agent contains non-spec rule-based overrides and lacks several spec features | Tuned values are tied to agent code, so the agent must be final first |
| B5 | `configs/*_tuned.yaml` are smoke-test artifacts, and the tree is uncommitted | The spec's freeze commit rule cannot hold, and results record the wrong commit |

Everything else (sections 3–6) can be fixed before or alongside, but B1–B5 need a decision from you first.

---

## 1. Blockers

### B1. Streaming workloads are bandwidth-saturated, so speedup is flat at 1.00

**Where:** `trace/generators.py:98` (`gap_mean = 4`, as the spec says), `sim/simulator.py` timing model, `DECISIONS.md` items 13 and 16.

**Evidence (my run, 150k records, `none` vs best simple prefetcher, 12.8 GB/s):**

| Workload | speedup at gap_mean 4 (spec) | at 16 | at 32 | at 64 | at 128 | channel util at gap 4 |
|----------|------|------|------|------|------|------|
| seq_stream (stride_pc) | 1.00 | 1.03 | 1.12 | 1.29 | 1.48 | 1.00 |
| stride_const (stride_pc) | 1.00 | 1.02 | 1.09 | 1.24 | 1.42 | 1.00 |
| multi_stride (stride_pc) | 1.00 | 1.06 | 1.40 | 1.90 | 2.24 | 1.00 |
| phase_mix (stride_pc) | 1.00 | 1.02 | 1.11 | 1.26 | 1.45 | 1.00 |
| random_uniform (next_line) | 0.75 | 0.59 | 0.77 | 0.91 | 0.94 | 1.00 |

At gap 4 each record costs about 1.25 compute cycles but one DRAM transfer costs 20 cycles, so `none` already saturates the channel (IPC 0.250 on seq_stream, stride_const, multi_stride, phase_mix, random_uniform, mix_reg_irr). A perfectly accurate prefetcher (stride_pc on seq_stream: accuracy 1.0, 177k prefetches) still gets IPC 0.250. `DECISIONS.md` already notes that T8's "speedup above 1.2" is inconsistent with T4; the consequence is bigger than a rejected test threshold.

**Why it blocks training:**
- The tuning objective is the geometric mean of speedup over 7 dev workloads × 2 bandwidths. Five of the seven (seq_stream, stride_const, multi_stride, phase_mix, and matmul_blocked, see B2) sit at 1.00 for every candidate. Candidates are effectively ranked by `spmv_csr` and a little by `hash_lookup`.
- The agent is trained by a reward that does not see speedup, so it can learn well and the objective still cannot tell good from bad hyperparameters. The spec's own checks (E1 "large speedups for stride_pc and bop on seq_stream and stride_const", E8 gain captured, E10 recovery based on window IPC) all become uninformative.
- On `random_uniform`, `next_line` loses 25% and the agent's job is only to avoid that, so the only visible signal is a loss.

**Fix (needs your decision):** the spec fixes `gap_mean = 4` as a default but also says "when the spec is silent, choose and record in DECISIONS.md", and it is not silent on T8, so the spec itself is inconsistent. The least invasive option is a per-workload or global `gap_mean` config field (the spec says "no magic numbers", and `gap_mean` is hard-coded as `1/5` in `_finish`) and generating the evaluation traces with a larger value. From my table, `gap_mean` of about 64 makes the streams latency-bound (channel util 0.36) with speedups 1.2–1.9 and keeps the random workloads near or below 1.0, so tuning gets signal in both directions. Keep the spec value as an additional "bandwidth-bound" scenario if you want the negative result in the README. Whatever you pick, record it in `DECISIONS.md`, rerun T3/T4/T17 (they use explicit gaps, so they should be unaffected) and regenerate every cached result.

### B2. Two kernels have too few L2 accesses (spec section 4.2 MUST, test T14)

**Evidence (default length, `none`, my run):**

| Workload | records | L2 accesses (all) | spec minimum | status |
|----------|---------|-------------------|--------------|--------|
| matmul_blocked (dev) | 4,000,000 | **175,260** | 400,000 | fails |
| stencil2d (test) | 4,000,000 | **333,660** | 400,000 | fails |
| all other 11 | 2M or 4M | 1.5M–2.0M | 400,000 | pass |

`matmul_blocked` also has only 11,309 L2 misses over the measured region and IPC 3.49. At the quick budget (1,000,000 records) it produces about 44k L2 decisions in total, which is below the agent's own exploration half-life (66,891 in the current tuned config; the default is 150,000). The tuner therefore sees `matmul_blocked` as noise around 1.0.

**Why T14 did not catch it:** there is no T14 test at all (no `@pytest.mark.slow` anywhere in `tests/`).

**Fix:** either lengthen the default record count for these two generators (matmul needs roughly 9–10M records, stencil2d about 5M, so spec-faithful lengths are slow) or record a deviation and a smaller threshold in `DECISIONS.md` and change the dev workload set honestly. Add the T14 test either way. Consider whether a dev workload that gives no signal should stay in the tuning objective.

### B3. Compute budget, parallelism and memory

**Evidence (my runs):**

| Item | Measured |
|------|----------|
| `none` baseline | 60k–160k records/s (meets the spec's 100k target only on some workloads) |
| `stride_pc` | 24k–100k records/s |
| agent | **7.3k records/s on seq_stream and phase_mix, 8.5k on hash_lookup, 21.7k on spmv_csr** |
| spec target | agent no worse than 5× slower than baselines |
| memory, agent, 400k records seq_stream | peak RSS 341 MB; `sim.book` held 369,605 entries (about 0.8 KB per prefetch) |

Estimated tuning cost with the current code (per-run time at 1M records, summed over the 7 dev workloads ≈ 730 s per agent-seed per bandwidth):
- Search: 60 candidates × 7 workloads × 2 bandwidths × 3 seeds = 2,520 runs ≈ **70–75 hours serial**.
- Refinement per `README.md` (1M records): 5 × 7 × 2 trace seeds × 2 bandwidths × 3 seeds = 420 runs ≈ **12 hours**. The spec says refinement at **default trace length** (2M/4M records), which makes it 25–45 hours; the CLI default `--refinement-records` is `None` which does mean default length, but the README command overrides it to 1M. Decide which one you are running and write it down.
- `tune_agent` is fully serial (`experiments/tune.py` has no multiprocessing), and the machine has 12 cores and 7 GB RAM.

**Problems to fix before the long run:**
1. `Simulator.book` (`sim/simulator.py:86,206`) stores a dict per prefetch and never deletes it. A 4M-record trace will need several GB for one run, so parallel workers will be killed by the OOM killer. Entries can be removed once a prefetch is resolved and its pollution window (4,096 L2 accesses) has passed. `demand_lines` is another list that grows with every L2 access (needed only for `future_lookahead`; record it only when asked).
2. Profile of a 50k-record agent run: `_q` is called 5 times per decision (`rl/agent.py:126`, 1.0 s of 9.6 s), `_select` 0.75 s, `valid_mask` 0.73 s (Python loop over 36 actions, `rl/actions.py`), `_update` 0.62 s. Computing Q once per decision, caching the 36-entry validity per (pattern deltas, offset) and using one `rng.random()` draw would give a large speedup without changing results if done carefully (the spec says "without changing results").
3. Add a process pool over candidates in `tune_agent` with the existing append-only checkpoint (the checkpoint is already per complete candidate, so this is safe). With about 5 workers and a 3× faster agent the search drops from about 70 h to about 5–8 h.
4. `tune_agent` / `tune_baselines` call `_checkpoint_records(path, resume=False)` which **truncates the checkpoint immediately**, even when `--force` was not given. `_write_rows(..., force or resume)` then raises `FileExistsError` only **after the whole search has finished** because `results/tune/agent_runs.csv` already exists. Running `pfsim tune --target agent` without `--force` would therefore wipe old checkpoints and fail at the very end. Check the output paths and the `--force`/`--resume` flags at the start (`cli.py:cmd_tune`, `tune.py`).
5. A checkpoint filename contains the simulator-config hash, record count, trial count and search seed, but not the agent or workload code version. If you change the agent after a partial run and use `--resume`, stale candidates are silently reused. Use `--force` after any code change.

### B4. Agent differs from the spec, and several spec features are missing

Because tuned hyperparameters belong to a particular agent, resolve all of this first.

**4a. Rule-based overrides inside the agent (not in the spec).** `rl/agent.py:190-217`:
- `admission_control`: after 4,096 feedback events, if `acc_ema < 0.15`, the chosen action is replaced by `NONE`, except every 64th suppressed choice (probe).
- `reach_control`: after 256 feedback events, if **lifetime** accuracy `useful_feedback / feedback_count < 0.5`, all reaches above `low_accuracy_max_reach` are masked.
- Both are searched by the tuner (`sample_agent_configs`) and enabled in `agent_tuned.yaml`.

Consequences:
1. These are hand-written rules on top of the learned policy, so the central claim ("a learned policy ... backs off when bandwidth is scarce") is partly a rule. The spec says the simple-rule baseline is `bw_gate` and that it should be the thing to beat. Report this plainly, or turn both off for the main result.
2. `reach_control` uses cumulative counts from the start of the run (`feedback_count`, `useful_feedback` are never decayed). After a long accurate phase it will almost never trigger; after a long inaccurate phase it stays on forever. That defeats adaptation in `phase_mix` (E4, E10). If you keep it, use a decayed estimate (for example `ev.sys.acc_ema`, which is already an EMA).
3. When a rule replaces action `a` by `NONE`, the stored decision has action `NONE` and receives the `NONE` reward. The Q-value of the originally chosen action never gets the information that it was blocked, which biases learning and makes mean reward look better than the learned policy deserves.
4. The `no_sys_features` ablation (drop f5/f6) still has `admission_control`, which reads `acc_ema`, a system-state signal. So "no system features" is not really system-blind, and E11's comparison "full vs no_sys_features vs bw_gate" is confounded. In `extensions.py` the `threshold_gate` variant switches admission off but not reach control; `no_sys_features` leaves both on.

**4b. Features in the spec that are not implemented** (`grep` for `cross_page`, `c_stall`, `conf`, `mini` in `src/` finds nothing relevant):
- `reward.mode = perf` with `c_stall` (6.5). `SysState.stall_since_last` exists but the agent never reads it.
- `actions.cross_page` and the `-c_xp` penalty (6.9, T20).
- `features.conf` (table f7, 6.9).
- `mini` preset (16 actions, one set); `compact` exists as a variant file only.
- `spp_lite` (optional) and DQN (stretch) are deferred in `DECISIONS.md`, which is fine.

E3 and E12 in the spec depend on these. If you intend to drop them, say so in `DECISIONS.md` and the README now. If you intend to build them, do it before tuning, because they change agent code and each needs its own 60-candidate tuning budget (spec section 7.2).

**4c. Smaller agent deviations:**
- `stall_since_last` (`simulator.py:274-287`) measures only the stall of the current trace record, not "since the previous L2 demand access". Records that hit L1 in between are lost. Only matters if you implement the perf reward.
- `mean_reward` in `stats()` is `reward_sum / updates`, but `updates` is also incremented by pollution corrections that add no reward, so it is biased low. It is also a **cumulative** mean since the start (and since the last reset), so learning curves (E4, E10) are smoothed and will not show a phase change clearly. Record a per-interval mean instead.
- `qtable_small` is built with default `RewardConfig` and fixed `alpha`, `gamma`, `epsilon` (`experiments/runner.py`, `extended.py`, `rl/qtable_small.py`). The spec says "same reward and delayed-credit machinery". It ignores the tuned reward (c_w1 1.46, c_pol 1.09, t_min 0.31), so the comparison "what the richer state and action space buy" is not like-for-like.
- Tuned `pc_bits = 6` came from the smoke search. `pc_bits` is both a tuned hyperparameter and an E6 sweep variable, so the "full" agent in E3/E6 will not be the 1.52 MiB configuration quoted in the spec. State this in the README.

### B5. Tuned configs are smoke artifacts, and the tree is not frozen

- `configs/agent_tuned.yaml` was produced by a **10-candidate, 10,000-record** search (`results/tune/agent_search_0cfc1a995f207986_n10000_t10_s2026.jsonl`, 10 lines). `configs/baselines_tuned.yaml` comes from legacy checkpoints (`bop_checkpoint.jsonl`, `next_line_checkpoint.jsonl`, `stride_pc_checkpoint.jsonl`) that do not follow the current naming. `README.md` says these must not be used for claims; the experiments still load them by default (`cli.py:187-188`).
- 16 tracked files are modified and 20 are untracked (`git status`). `runner.py` writes the `HEAD` short hash into every result row, which today points at the "Configuration files" commit that contains none of the current code. `DECISIONS.md` item 9 already asks for a distinct freeze commit. Do this before the campaign: commit everything, tag it, and make the runners refuse to run (or record `-dirty`) when the tree has uncommitted changes.
- `results/E1`, `results/E2` and `results/tune/*` are smoke outputs. `.gitignore` excludes `results/`, so nothing can be confused later, but delete or move them before the real run so a stale `runs.csv` is not read by mistake (the `--force` flag makes overwrite easy, which is the risk).

---

## 2. Spec deviations in the simulator and workloads (major)

| # | Where | Spec | Code | Impact |
|---|-------|------|------|--------|
| M1 | `simulator.py:155-165` (`_install_l2`) | Victim filter records every victim of a prefetch fill that was **not an unused prefetched line** (3.4, 9.4) | A victim that was a prefetched line that had already been **used** is not recorded (`elif victim.pf: ... pop`) | Misses real pollution events: a useful line evicted by a later prefetch can never be blamed. Rewards and `pollution_rate` are biased low. Record used-prefetch victims; keep unused/resolved ones out |
| M2 | `config.py` `pf_backlog_factor` 32 vs 4 | `pf_backlog_cycles = 4 × transfer_cycles` | 32 × (documented in `DECISIONS.md` 13) | Defensible, but it means `DROPPED(bandwidth)` almost never fires (a backlog of 640 cycles at 12.8 GB/s). The reward term `c_drop` and the "bandwidth" drop reason then carry little signal, and bandwidth awareness has to come from `USELESS` weighting by `u_issue`. Say so in the README, and rerun the 4-transfer variant as an S1-style sensitivity case |
| M3 | `generators.py:_hash` | `2^22` buckets of **8 bytes**, so 8 buckets per cache line | Uses the bucket index as the line index (`lines = (base + k) % 2^22`) | Table is 256 MiB (spec: 32 MiB), and every probe `h+1..h+3` lands on a different line (spec: usually the same line, an L1 hit). Hash lookups are therefore far more miss-heavy than specified, and the `dep=1` probe chain is more expensive. Use `bucket // 8` for the line and `(bucket % 8) * 8` for the byte offset |
| M4 | `experiments/learning.py`, `extensions.py` | Recovery time and adaptation loss are measured from the first full cycle (4 phases) | Default `records=1_000_000` for E4 and E10; `phase_mix` phase is 300k, so the first full cycle ends at 1.2M. `recovery_times()` finds **no boundary** and writes an empty file | E4/E10 on `phase_mix` produce no recovery data unless run with at least 2M records (default length). The CLI default of `1_000_000` must not be used for these two experiments |
| M5 | same | `stats_interval = 50,000` L2 accesses | `phase_mix_fast` phases are 30,000 records | The window is longer than the phase. Recovery cannot be resolved, and `steady` (second half of a phase = 15k accesses) usually contains no interval end. E10 on `phase_mix_fast` needs a smaller `stats_interval` (for example 5,000) |
| M6 | `extensions.py:run_e10` | Boundaries are L2 access indices | `trace.meta["boundaries"]` are **record** indices, compared with `ev.l2_idx` in `reset_boundaries` and with `l2_idx` in `recovery_times` | Equal only if no record hits L1. Phase boundaries drift by the number of L1 hits so far. Small for `phase_mix`, but wrong in principle. Convert boundaries to L2 indices from the baseline run's demand stream |
| M7 | `simulator.py:_record_interval` | Interval at every `stats_interval` L2 accesses | Counted from the end of warmup, while `l2_idx` in the record includes warmup accesses | Interval edges are not aligned to phase boundaries. A window can straddle a boundary, which understates recovery time. Align windows to L2 indices counted from 0 |
| M8 | `simulator.py:_demand` | `expire_pending_prefetches()` before the L2 lookup (3.4); step 4 after (3.7) | After the L2 path (matches 3.7) | The spec contradicts itself; the choice is fine. Record it in `DECISIONS.md` |
| M9 | `generators.py:_phase` | Gradual variant: blend of 50,000 records "at each boundary" | Blend occupies the **last** 50,000 records of each phase and applies the next phase's pattern using the current phase's position and start | Acceptable if documented. T22 asks for the fraction of next-phase records to rise from about 0 to about 1, which holds |

Lower priority: `list_walk` and `ptr_chase` use per-record Python loops (the spec says this is not acceptable at these lengths; cost is only a few seconds); `spmv_csr` redraws the matrix on every pass (irrelevant at 4M records because one pass is about 26M); `bw_bucket_cycles` is an unused config field; BOP's `reference_latency` is fixed at 254 so S1's `lat_120`/`lat_300` do not affect it; `bfs_graph` generation takes about 9 s.

---

## 3. Experiment and statistics layer (does not block tuning, but fix before the evaluation)

`README.md` says E1–E8, E10–E12 and S1 are implemented. These are the differences from the spec.

| Experiment | Gap |
|-----------|-----|
| **E12** | `run_e12` is a stub: it dumps `str(row)` from E3/E6/E10 into a CSV and writes the literal text "requires E3 interval join" as the Spearman correlation. No Pareto plot, no cross-page comparison, no recovery panel. `runs.csv` has no `mean_reward` column, so the reward/speedup correlation cannot be computed later without rerunning. **Add `mean_reward` (and the action histogram) to the per-run record now**, otherwise E3 and E6 have to be rerun |
| **E10** | No adaptation loss (spec 7.2). `mean_reward` curves are cumulative (see 4c). Recovery uses fixed 90% of the median; fine. See M4–M6 |
| **E11** | `theta_bw`/`theta_acc` are fixed at 0.75/0.5, not tuned over {0.6, 0.75, 0.9} × {0.3, 0.5}. Wasted DRAM reads (useless + dropped) are not reported; only mean overprediction. No per-workload panels |
| **E1** | Summary has only per-workload means and one `ALL` row. The spec asks for geometric means for dev, test and all, mean ± standard deviation over seeds, and the 2,000-sample bootstrap confidence interval. `paired_outcomes` in `stats.py` (win/tie/loss within 1%) is never called. The `quick` budget runs 1M records (matmul/stencil are then near-empty, see B2) |
| **E2** | `future_lookahead` is fixed at lookahead 16 here, but swept over 5 values in E1, so the two experiments are not comparable. Only one aggregate plot; per-workload plots for seq_stream, random_uniform, hash_lookup, phase_mix are missing |
| **E3** | Extension variants (perf reward, cross_page, conf, compact, mini) are not in `E3_VARIANTS`. Per-workload bars missing. Ablation variants overlay the one tuned config, which matches the spec for the 12 listed variants |
| **E4** | Learning curves for every dev workload are saved as CSV only; a figure is produced for `phase_mix` only |
| **E5** | Dev workloads are visited in fixed order, not "random order" (spec 7.2). Each workload run restarts epsilon at `eps_0` because step count resets on load |
| **E6** | No baseline storage points, no `compact`/`mini` in the scatter |
| **E7** | Defaults to test workloads only. "Learned delta fraction" counts only `PC_D1`/`PAGE_D1`, but the spec also accepts a fixed-offset pattern that matches the stream |
| **S1** | Runs test workloads only at 100k records, omits `qtable_small` and `future_lookahead`, and writes no ranking-change report |
| **CLI** | `pfsim report` (spec 9.2) does not exist; `--budget` is honoured by E1 only; `plots.py` and `e1.py … e12.py` layout differs from 9.1 (cosmetic) |

Also: `DECISIONS.md` items 10 and `README.md` disagree on the cost of the 10k-record baseline grid (17 s vs under 1 s). Fix one of them.

---

## 4. Missing tests for spec items

73 tests pass, but these spec tests have no equivalent (checked by name and by `grep` in `tests/`):

| Spec | Missing |
|------|---------|
| T5 | Accounting invariants across **every** workload for `next_line`, `stride_pc` and the agent (only one tiny trace is checked). Also "POLLUTION never exceeds prefetch fills and never repeats for one eviction", "no unknown tag" |
| T9 | Future-lookahead on `seq_stream`, `stride_const`, `random_uniform` |
| T13 | Agent smoke: ≥ 0.8 of non-NONE actions are `PC_D1`/`PAGE_D1` on `stride_const`, ≥ 0.6 NONE on `random_uniform` (last 20% of decisions) |
| T14 | Slow test for determinism, `validate`, and ≥ 400,000 L2 accesses (would have caught B2) |
| T15 | CLI: output JSON has every key of the `runs.csv` schema. `cmd_run` omits `wall_seconds` and `workload`/`variant` come from different sources, so this would likely fail |
| T18 | Pollution accounting cases a and b, and the `counterfactual=false` case. This is the main place where M1 would have shown up |
| T19, T20, T22 | Sensitivity run; cross-page and perf reward (feature missing); phase variants for two seeds |

I recommend adding T5 (all workloads), T13, T14 and T18 before tuning, since they exercise exactly the code the reward depends on.

---

## 5. Things I checked and found consistent with the spec

So you do not need to re-check these:

- Core timing loop, demand path, DRAM channel and bandwidth-utilization windows (exact intervals, not buckets): match 3.3–3.5. T17 microbenchmarks pass.
- Outcome events: one primary outcome per prefetch, synchronous order, `USEFUL` wait cycles, `USELESS` on eviction and expiry, `useful_after_expiry`, warm-up exclusion. Counterfactual shadow array logic matches 3.6 (apart from M1).
- Demand stream is prefetcher-independent (T7 test exists and passes). Prefetchers never see future records except `future_lookahead`.
- Cache LRU and set mapping; `transfer_cycles`; `acc_ema`/`pol_ema` updates; SysState fields.
- Baselines: `next_line`, `stride_pc` (table index, confidence logic, storage formula), `bop` (offset list, RR insertion at +254 cycles, SCOREMAX/ROUNDMAX/BADSCORE, end-of-phase logic, storage formula), `future_lookahead` (indexing `l2_idx - 1 + lookahead`, `trigger_page` of the target).
- Agent: PC and page history tables, quantization `qd`, all six feature index formulas and row counts (11,058 rows in total with `pc_bits = 8`), action space (36 actions), validity masking, reward function (T10 values), double-Q update with clipped TD error, pollution correction, delayed credit via `next_state`, exploration modes, storage formula, weights save/load, ablation flags (`reward_*`, `reaches`, `patterns`, `gamma = 0`, `double_q`, `feature_drop`, `explore`, `learn`, `bw_gate`, `reset_boundaries`).
- `qtable_small`: 544 states, 11 actions, structure as specified.
- Generators: `seq_stream`, `stride_const`, `multi_stride`, `mix_reg_irr`, `random_uniform`, `phase_mix` (and the two variants, apart from M9), `matmul_blocked` loop order and site mapping, `spmv_csr` event order and array sizes, `bfs_graph` traversal, `ptr_chase` (Sattolo), `stencil2d` event order. Default lengths follow the spec. Traces are deterministic for a seed.
- Tuning protocol: trace seeds {100, 101}, agent seeds {100, 101, 102}, bandwidths {3.2, 12.8}, geometric-mean objective, test workloads rejected by `validate_tuning_workloads`, baseline grids exactly as in 5.7, random-search ranges match 6.8 (extra non-spec parameters are added, see 4a), refinement keeps the best of the top 5. Final evaluation seeds {0,1,2} and {0,…,4} do not overlap with tuning seeds.

---

## 6. Observation from my short agent runs (not a bug)

I ran the default and the smoke-tuned agent for 400k records on a variant of the workloads with gap_mean 64 (so streams are latency-bound), against tuned-style baselines:

| Workload | none→ next_line / stride_pc / bop | agent_default | agent_tuned (smoke) |
|----------|------|------|------|
| seq_stream | 1.12 / 1.29 / 1.67 | 1.36 | 1.12 (chose `P1` reach 1 for 350k of 400k decisions) |
| stride_const | 0.90 / 1.24 / 1.21 | 1.13 | 1.12 |
| multi_stride | 0.96 / 1.91 / 1.01 | 1.41 | 1.48 |
| phase_mix | 0.90 / 1.18 / 1.25 | 1.16 | 1.17 |
| random_uniform | 0.91 / 1.00 / 1.00 | 1.00 | 1.00 (accuracy 0.0, almost all NONE) |
| hash_lookup | 1.12 / 1.00 / 1.12 | 1.00 | 1.00 (about 50% of decisions suppressed by admission control) |

What this says for the real run: the agent is clearly learning (high accuracy, mostly NONE on random_uniform), but at this length it does not match `stride_pc`/`bop` on streams, and the smoke-tuned config settles on a late reach (reach 1). That is what tuning should fix, but it is also a hint that credit for timeliness may be too weak: with the default reward a late prefetch still earns `c_use × (t_min + …)` (about 0.5 for a full-latency wait with `t_min` 0.31) and a timely one 1.0, and the learning rate chosen by the smoke run (0.0096) is slow. If the full campaign still shows the agent behind `bop` on `seq_stream`, look at reach selection first (print the action histogram per reach) before changing anything else.

---

## 7. Suggested order of work

1. **Decide B1 (gap_mean) and B2 (workload lengths / dev set).** These redefine the workloads, so do them first. Regenerate all traces; discard `results/`.
2. **Decide B4 (keep or remove `admission_control` / `reach_control`; implement or drop perf reward, cross-page, conf, mini).** If you keep the controls, make `reach_control` use a decayed accuracy and make sure `no_sys_features` and `bw_gate` variants disable both.
3. **Fix simulator items M1 (victim filter), M3 (hash layout), and the `book` memory growth.** Add tests T18, T5-all-workloads, T13, T14, T15.
4. **Speed up the agent hot path and add multiprocessing to `tune_agent`.** Check that results are bit-identical before and after (use T6 plus a fixed-seed run comparison on `results` from a short run).
5. **Fix the tuning script guard rails** (B3 item 4) and decide the refinement length.
6. **Fix experiment-layer items needed for data you cannot regenerate cheaply:** add `mean_reward` and the action histogram to `runs.csv`; per-window mean reward in the interval records; `stats_interval` for `phase_mix_fast`; boundary conversion to L2 indices (M4–M7).
7. **Commit everything and tag the freeze commit; clean `results/` and the old `configs/*_tuned.yaml`.** Then run:
   ```bash
   pfsim tune --target baselines --records 1000000 --force
   pfsim tune --target agent --records 1000000 --refinement-records <decision> --trials 60 --force
   ```
8. **Before E1–E12:** commit the new tuned YAML files, then build the E12 aggregator and the remaining statistics (dev/test/all geometric means, confidence intervals, paired win/tie/loss, adaptation loss, wasted DRAM reads).

Quick pre-flight checks to run after the fixes and before step 7:
- `pytest` plus the new T5/T13/T14/T18 tests.
- A 200k-record agent run on `seq_stream`, `stride_const`, `random_uniform`, `phase_mix` at the new gap: `stride_pc` should show speedup clearly above 1.0, `random_uniform` should be at or below 1.0 for `next_line`, and the agent should move toward the stride prefetcher's speedup.
- One 1M-record agent run with peak RSS logged (should stay well under 1 GB per process).
- One 3-candidate, 1M-record, single-workload tuning run with `--workload spmv_csr --no-refine`, interrupted and resumed, to confirm checkpoint behaviour and per-candidate time.
