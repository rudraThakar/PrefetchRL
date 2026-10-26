# Implementation Spec: RL-Based Self-Optimizing Memory Prefetcher

Oct 18, 2026 · @Sahil

## 1. Overview

Build `pfsim`, a trace-driven memory-system simulator in Python, with conventional prefetchers and a reinforcement-learning (RL) prefetching agent that learns online, then evaluate them with the protocol in section 7. This document is a specification: an implementing agent should be able to build everything from it without further questions.

**Goal.** Produce an RL agent that is genuinely strong across regular, irregular and phase-changing memory behaviour, and a rigorous evaluation of why it works. Published RL prefetchers (Pythia and others, see the Literature review at the end of this document) are references for ideas, not designs to copy and not numbers to match. The agent in section 6 has its own state, action space, reward and learning rule.

**Central claim to test.** The agent is a phase-adaptive, bandwidth-aware prefetch selector: a learned policy that picks the pattern and distance from per-PC, per-page and system-state context, changes behaviour when the access pattern changes (E4, E10), backs off when DRAM bandwidth is scarce (E2, E11), and has its storage cost measured and reported honestly (E6, E12). Each part has an ablation or a simple-rule baseline that can falsify it, and negative results go in the README. Pythia stays a reference for ideas, not a target.

**In scope**

- A single-core simulator with L1D, L2 and LLC caches, a bandwidth-limited DRAM channel, and a simple out-of-order timing model.
- Synthetic and kernel-derived workload generators, and a trace file format.
- Baseline prefetchers: none, next-line, PC-stride, best-offset, and a future-informed lookahead baseline used only as a diagnostic, not a bound.
- The RL agent, its ablation variants, an optional tabular agent and an optional DQN agent.
- An experiment runner, hyperparameter tuning, statistics, plots, tests and a README with results.

**Non-goals:** cycle-accurate out-of-order modelling, cache coherence, multi-core, instruction prefetching, ChampSim integration (a trace converter is described in section 4, and section 4.6 says whether it is required), and hardware synthesis.

**Deliverables**

1. Python package `pfsim` (layout in section 9) with a CLI.
2. Trace generators and the 13 workloads in section 4.
3. All prefetchers in sections 5 and 6.
4. Experiment scripts E1 to E8, producing CSV files and PNG figures under `results/`.
5. A test suite (section 8) that passes.
6. `README.md` with setup, how to reproduce every result, the final results tables and figures, and a short discussion of what the agent learned and where it fails.
7. `DECISIONS.md` listing every place where the spec was silent and what was chosen.

**Tech stack:** Python 3.10 or newer, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `pytest`. `numba` is optional for speeding up hot loops. `torch` is needed only for the optional DQN agent. No GPU is required.

**Conventions**

| Item | Value |
| --- | --- |
| Cache line | 64 bytes; internally all addresses are line addresses (byte address >> 6) |
| Page | 4096 bytes = 64 lines; page number = line >> 6; page offset = line & 63 |
| Time unit | CPU cycles at 4.0 GHz, stored as floats |
| Randomness | Every source of randomness uses a `numpy.random.Generator` built from an explicit seed; the same config and seeds MUST give bit-identical results |
| Config | Python dataclasses loaded from YAML; no magic numbers in code, every default in this spec lives in a config field |
| Language of the spec | MUST and SHOULD are normative; tables give default values |

**Rules for the implementing agent**

- Follow the spec; when it is silent, pick the simplest option, and record it in `DECISIONS.md`.
- The simulator MUST NOT depend on which prefetcher is running, except through the interface in section 3.8. The L2 demand-access stream is independent of the prefetcher by construction, because prefetches fill L2 and LLC only. Tests rely on this.
- No prefetcher except `future_lookahead` may read future trace records. Enforce this by giving prefetchers only the event objects defined in section 3.8.
- Hyperparameter tuning MUST use only the development workloads and tuning seeds defined in section 7. Test workloads are touched only for final evaluation.
- Write tests alongside each component, in the build order of section 9, and keep them passing.
- Performance target: at least 100,000 simulated accesses per second for the baselines in pure Python, and no worse than a factor of 5 slower for the RL agent. If the target is missed, optimise the hot loops with `numba` or vectorised preprocessing without changing results.

## 2. Architecture

&#91;embedded content: architecture · simulator path, prefetcher and feedback loop\]

A demand access flows down the left column. Every L1 miss reaches L2 and becomes a `DemandEvent` for the prefetcher, which returns `PrefetchRequest` objects that fill L2 and the LLC. The simulator sends back outcome events (useful, useless, redundant, dropped, pollution), and the RL agent turns them into rewards and learns from them. Baselines ignore the outcomes. Section 3 defines every box and message.

## 3. Simulator specification

The simulator replays a trace through L1D, L2 and LLC tag arrays and a single DRAM channel, tracks time with a simple window-limited timing model, and reports prefetch outcomes back to the prefetcher. The prefetcher sits at L2: it is called on every L1 miss and its prefetches fill L2 and the LLC.

### 3.1 Configuration

Every value is a config field. Latencies are cumulative load-to-use cycles.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `freq_ghz` | 4.0 | CPU frequency |
| `width` | 4 | Non-memory instructions retired per cycle |
| `rob_size` | 256 | Instruction window for overlapping misses |
| `mshr_demand` | 32 | Maximum demand misses in flight |
| `pf_mshr` | 16 | Maximum prefetches in flight |
| `l1` | 32 KiB, 8-way, 4 cycles | L1D, LRU |
| `l2` | 256 KiB, 8-way, 14 cycles | L2, LRU, prefetcher attached here |
| `llc` | 2 MiB, 16-way, 34 cycles | LLC, LRU, non-inclusive |
| `dram_latency` | 200 | Cycles from channel service start to data ready, excluding transfer |
| `peak_bw_gbps` | 12.8 | DRAM peak bandwidth |
| `transfer_cycles` | derived | `ceil(64 / peak_bw_gbps * freq_ghz)`; 20 at the default |
| `pf_backlog_cycles` | 4 x transfer\_cycles | A prefetch is dropped if the channel is backlogged by more than this |
| `bw_window_cycles` | 4096 | Window for bandwidth utilization (64 buckets of 64 cycles) |
| `pf_same_page_only` | true | Drop prefetch candidates outside the trigger line's 4 KiB page |
| `pf_issue_latency` | 0 | Extra cycles between the demand access and prefetch issue |
| `expiry_accesses` | 1024 | A prefetch unused after this many L2 demand accesses is resolved as useless |
| `pollution_window` | 4096 | L2 demand accesses during which an eviction by a prefetch can be blamed for a miss |
| `victim_filter_size` | 4096 | Entries in the victim filter (FIFO replacement) |
| `warmup_fraction` | 0.1 | Fraction of trace records before statistics start; learning continues through warmup |
| `stats_interval` | 50000 | L2 demand accesses per interval record |
| `acc_ema_weight`, `pol_ema_weight` | 1/64 | EMA weights for system-state signals |

Three bandwidth presets used in experiments: low = 3.2 GB/s (80 transfer cycles), default = 12.8 GB/s (20), high = 51.2 GB/s (5).

### 3.2 Data structures

```python
@dataclass
class Entry:            # one cache line in L1, L2 or LLC
    line: int
    ready: float        # time the data is available at this level
    pf: bool = False    # installed by a prefetch (L2 only)
    used: bool = False  # demanded after being prefetched
    resolved: bool = False  # an outcome event was already sent to the prefetcher
    pf_id: int = -1
```

Cache API: `lookup(line) -> Entry|None` (no LRU update), `touch(line)` (make MRU), `contains(line)`, `install(line, entry) -> Entry|None` (returns the evicted victim, evicting the LRU line of a full set). Per-prefetch bookkeeping, kept in dicts keyed by `pf_id`: `tag` (opaque object from the prefetcher), `line`, `u_issue` (bandwidth utilization at issue), `issue_idx` (value of the L2 demand-access counter at issue). Also keep a FIFO `expiry_queue` of `(pf_id, issue_idx)`.

### 3.3 Core timing model

The model overlaps misses up to the ROB and MSHR limits and otherwise stalls. Loads and stores are treated alike (write-allocate, no write-back traffic, no store buffer). Each trace record carries `gap` (non-memory instructions since the previous record) and `dep` (1 if this access's address depends on the previous load's data).

```python
now = 0.0; inst = 0; last_done = 0.0
outstanding = deque()          # (inst_index, done_time) for loads that missed L1, program order
for rec in trace:
    now += (rec.gap + 1) / cfg.width
    inst += rec.gap + 1
    if rec.dep and last_done > now:
        now = last_done                                   # dependent load waits for its producer
    while outstanding and outstanding[0].done <= now:
        outstanding.popleft()                             # in-order retirement of finished loads
    while outstanding and inst - outstanding[0].inst >= cfg.rob_size:
        now = max(now, outstanding[0].done)               # window full: stall for the oldest miss
        outstanding.popleft()
        while outstanding and outstanding[0].done <= now:
            outstanding.popleft()
    inflight = [e for e in outstanding if e.done > now]
    while len(inflight) >= cfg.mshr_demand:               # too many misses in flight
        now = min(e.done for e in inflight)
        inflight = [e for e in outstanding if e.done > now]
    lat = memory.demand_access(rec, now)                  # section 3.4
    last_done = now + lat
    if lat > cfg.l1.latency:
        outstanding.append((inst, now + lat))
cycles = max([now] + [e.done for e in outstanding])
ipc = instructions / cycles
```

Because independent loads overlap, this model understates the benefit of prefetching for code with long dependent chains unless `dep` is set. Workload generators set `dep` for pointer-chasing and gather loads (section 4).

**Validation required before any speedup is reported.** This is a simplified core model, not a cycle-accurate one, and its errors can differ across workload types, so its conclusions are model-relative. Three checks are mandatory.

- Hand-computed microbenchmarks (T2 to T4 and T17). Reference outputs of the equations above at width 4, ROB 256, MSHR 32, llc 34, dram 200, 12.8 GB/s: a dependent chain (`dep = 1`) takes 254.0 cycles per record at gap 3 and 300.3 at gap 1200; independent distinct lines at gap 0 take 80.0, 20.0, 10.0 and 7.5 cycles per record at 3.2, 12.8, 25.6 and 51.2 GB/s (the last is MSHR-bound, not channel-bound); independent lines at gap 100 take 127.0, 84.7 and 42.3 cycles per record at ROB 128, 256 and 512. An implementation must match these within 2%.
- Sensitivity study S1 (T19). Re-run the E1 comparison for every prefetcher with one parameter changed at a time: ROB in {128, 512}, MSHR in {16, 64}, `dram_latency` in {120, 300}, trace seed 0, 3 agent seeds. Report whether the ranking of prefetchers and the sign of each speedup change. State in the README only conclusions that hold in every variant, and list those that do not.
- Optional external cross-check (recommended under option B). Run `next_line` and a PC-stride prefetcher in ChampSim on two or three converted traces with a matching cache and DRAM configuration, and compare the speedup ranking with `pfsim` (rank correlation). Absolute IPC is not expected to match.

### 3.4 Demand access path

```python
def demand_access(rec, now):
    L = rec.addr >> 6
    e1 = l1.lookup(L)
    if e1:
        l1.touch(L); return l1.lat + max(0.0, e1.ready - now)       # no prefetcher involvement
    l2_idx += 1                                                       # this is an L2 demand access
    expire_pending_prefetches()                                       # section 3.6
    e2 = l2.lookup(L)
    if e2:
        wait = max(0.0, e2.ready - now)
        ready = now + l2.lat + wait
        if e2.pf and not e2.used:
            e2.used = True
            if not e2.resolved: emit(USEFUL, e2.pf_id, wait_cycles=wait); e2.resolved = True
            else: stats.useful_after_expiry += 1
        l2.touch(L)
    else:
        check_pollution(L)                                            # section 3.6
        e3 = llc.lookup(L)
        if e3:
            ready = now + llc.lat + max(0.0, e3.ready - now); llc.touch(L)
        else:
            ready = dram_request(now + llc.lat, 'demand')
            llc.install(L, Entry(L, ready))
        install_l2(Entry(L, ready), prefetch_fill=False)
    l1.install(L, Entry(L, ready))
    ev = build_demand_event(rec, L, l2_hit=bool(e2), l2_prefetch_hit=bool(e2 and e2.pf))
    for req in prefetcher.on_demand(ev):
        issue_prefetch(req, now + cfg.pf_issue_latency)
    return ready - now
```

An entry's `ready` time handles hits under an outstanding miss: an access to a line still being fetched waits for it. `install_l2(entry, prefetch_fill)` evicts the LRU victim and then: if the victim is an unused prefetched line and not `resolved`, emit `USELESS(victim.pf_id, reason='evicted')`; and if this is a prefetch fill and the victim was not an unused prefetched line, insert `victim.line -> (pf_id of the new prefetch, l2_idx)` into the victim filter.

```python
def dram_request(arrival, kind):
    start = max(arrival, channel_free)
    channel_free = start + cfg.transfer_cycles
    record_busy(start, cfg.transfer_cycles)           # for bandwidth utilization
    stats.dram_reads[kind] += 1
    return start + cfg.dram_latency + cfg.transfer_cycles
```

`bw_util(now)` = busy cycles inside `[now - bw_window_cycles, now]` divided by `bw_window_cycles`, clipped to 1, computed from a ring of 64-cycle buckets.

### 3.5 Prefetch issue

```python
def issue_prefetch(req, t):       # req.line, req.tag
    L = req.line
    if cfg.pf_same_page_only and (L >> 6) != req.trigger_page: return emit(DROPPED, req.tag, reason='cross_page')
    if l2.contains(L):                                  return emit(REDUNDANT, req.tag)     # present or in flight
    if inflight_prefetches(t) >= cfg.pf_mshr:           return emit(DROPPED, req.tag, reason='mshr')
    e3 = llc.lookup(L)
    if e3:
        ready = t + llc.lat + max(0.0, e3.ready - t)    # no DRAM traffic
    else:
        if channel_free - t > cfg.pf_backlog_cycles:    return emit(DROPPED, req.tag, reason='bandwidth')
        ready = dram_request(t + llc.lat, 'prefetch')
        llc.install(L, Entry(L, ready))
    pf_id = new_id(); record(pf_id, tag=req.tag, line=L, u_issue=bw_util(t), issue_idx=l2_idx)
    install_l2(Entry(L, ready, pf=True, pf_id=pf_id), prefetch_fill=True)
    expiry_queue.append((pf_id, l2_idx)); stats.issued += 1
```

`inflight_prefetches(t)` counts L2 entries with `pf` set, not yet `used`, and `ready > t`.

### 3.6 Outcome events

Every prefetch request returned by a prefetcher ends in exactly one primary outcome, delivered to `prefetcher.on_outcome`:

- `USEFUL(tag, wait_cycles)`: a demand access hit the line after the prefetch was issued. `wait_cycles` is 0 if the data had arrived, otherwise the remaining wait (a late prefetch).
- `USELESS(tag, reason)`: the line was evicted from L2 unused, or `expiry_accesses` L2 demand accesses passed without a demand hit. At expiry the entry stays in the cache but is marked `resolved`.
- `REDUNDANT(tag)` and `DROPPED(tag, reason)`: the request never entered the memory system (reasons: `cross_page`, `mshr`, `bandwidth`).

A secondary event, `POLLUTION(tag)`, may follow later for a prefetch that already got a primary outcome. `check_pollution(L)` runs on every L2 demand miss: if `L` is in the victim filter and `l2_idx - stored_idx <= pollution_window`, emit `POLLUTION(tag of the blamed prefetch)` and delete the filter entry (so each eviction is blamed at most once). `expire_pending_prefetches()` pops `expiry_queue` entries with `issue_idx + expiry_accesses <= l2_idx` and emits `USELESS(reason='expired')` for those not yet `resolved`. Outcome events carry `u_issue`, the bandwidth utilization at issue time, because rewards use it.

**Counterfactual pollution check (default on, `pollution.counterfactual = true`).** The victim filter alone can blame a prefetch for an eviction that demand traffic would have caused anyway. Because the L2 demand stream is independent of the prefetcher, keep a demand-only shadow tag array with the same sets and ways, driven only by L2 demand accesses in order (lookup, install on miss, LRU), ignoring timing. Emit `POLLUTION(tag)` only if the real cache misses on line `L`, `L` is in the victim filter (evicted by a prefetch fill within the window), and the shadow array holds `L` at that moment. Count such events as prefetch-induced pollution; count victim-filter hits where the shadow also misses as would-have-been-evicted and report them separately (`pollution_benign`). This costs one extra tag lookup per L2 access and must be on for every run that reports pollution or trains the agent, so rewards and metrics agree. T18 checks it.

### 3.7 Order of events per trace record

1. Advance time and apply dependency, ROB and MSHR stalls (3.3).
2. L1 lookup. On a hit, stop.
3. On an L1 miss, run the L2 path: this may emit `USEFUL` or `POLLUTION`, and evictions may emit `USELESS`.
4. Expire stale prefetches (may emit `USELESS`).
5. Build the `DemandEvent` (system signals computed at this moment) and call `prefetcher.on_demand`.
6. Issue the returned requests in order; this may emit `REDUNDANT`, `DROPPED` and eviction-driven `USELESS` events.

Outcome events are delivered synchronously, in the order generated. Events from steps 3 and 4 therefore reach the prefetcher before `on_demand` is called, and events from step 6 arrive after it returns.

### 3.8 Prefetcher interface

```python
@dataclass
class SysState:
    bw_util: float        # 0..1, current DRAM utilization
    acc_ema: float        # 0..1, EMA of USEFUL (1) / USELESS (0) outcomes, starts at 0.5
    pol_ema: float        # 0..1, pollution pressure (see below)
    pf_occupancy: float   # inflight_prefetches / pf_mshr

@dataclass
class DemandEvent:
    now: float; pc: int; line: int; page: int; offset: int; is_write: bool
    l2_hit: bool; l2_prefetch_hit: bool; l2_idx: int; sys: SysState

@dataclass
class PrefetchRequest:
    line: int; tag: object; trigger_page: int

class Prefetcher(Protocol):
    def reset(self, seed: int) -> None: ...
    def on_demand(self, ev: DemandEvent) -> list[PrefetchRequest]: ...
    def on_outcome(self, out: OutcomeEvent) -> None: ...
    def storage_bits(self) -> int: ...
    def stats(self) -> dict: ...          # prefetcher-specific counters for logging
```

The simulator maintains the `SysState` signals so every prefetcher sees identical values: `acc_ema` is updated with weight `acc_ema_weight` on each `USEFUL` (value 1) and `USELESS` (value 0); `pol_ema` is multiplied by `(1 - pol_ema_weight)` on each `USEFUL` or `USELESS` and incremented by `pol_ema_weight` on each `POLLUTION`. Prefetchers with no use for a signal ignore it.

`SysState` also carries `stall_since_last` (float): the cycles the core spent blocked by ROB, MSHR or dependency stalls since the previous L2 demand access. It is zero in the default reward and is used only by `reward.mode = perf` (6.5).

### 3.9 Metrics

Counters accumulate only after the warmup fraction of the trace. Prefetches issued before warmup ends are excluded from outcome counts. The runner executes the `none` prefetcher on the same trace and config to obtain the baseline values marked with a 0 subscript.

| Metric | Definition |
| --- | --- |
| `ipc` | instructions / cycles over the measured region |
| `speedup` | `ipc / ipc0` |
| `l2_misses` | L2 demand misses |
| `coverage` | `(l2_misses0 - l2_misses) / l2_misses0` (can be negative) |
| `accuracy` | `useful / (useful + useless)` over resolved prefetches |
| `late_fraction` | share of `USEFUL` outcomes with `wait_cycles > 0` |
| `pollution_rate` | `POLLUTION` events / L2 demand accesses |
| `dram_reads` | demand plus prefetch DRAM reads |
| `overprediction` | `(dram_reads - dram_reads0) / dram_reads0` |
| `avg_bw_util` | total channel busy cycles / cycles |
| `redundant_rate`, `dropped_rate` | share of returned requests ending in `REDUNDANT` or `DROPPED` |
| `storage_bits` | `prefetcher.storage_bits()` |

In addition, every `stats_interval` L2 demand accesses the simulator appends a record `(l2_idx, window_ipc, window_accuracy, window_coverage_proxy, mean_reward, epsilon)` for learning curves; `mean_reward` and `epsilon` come from `prefetcher.stats()` and are 0 for non-learning prefetchers. Window IPC = instructions in the window / change in `now`.

## 4. Workloads and trace format

Thirteen deterministic workload generators cover streaming, strided, interleaved, blocked, gather, chained and phase-changing behaviour, and need no downloads. Seven are development workloads (for tuning) and six are test workloads (for final evaluation only).

### 4.1 Trace format

A trace is a `.npz` file named `{workload}_s{seed}.npz` with these arrays, one element per memory access, plus a `meta` JSON string (workload name, seed, parameters, record count):

| Array | Type | Meaning |
| --- | --- | --- |
| `pc` | uint64 | Program counter of the access site |
| `addr` | uint64 | Byte address |
| `gap` | uint16 | Non-memory instructions since the previous record |
| `is_write` | uint8 | 1 for a store |
| `dep` | uint8 | 1 if the address depends on the previous load's data |

Generator API: `generate(name: str, seed: int, n_records: int | None = None) -> Trace`, where `None` selects the default length (2,000,000 records for synthetic workloads, 4,000,000 for kernels). A `validate(trace)` function checks dtypes, non-zero addresses and non-negative gaps. CLI: `pfsim gen-traces --out traces/ --seed 0`.

### 4.2 Shared generation rules

- **Address layout:** array number `i` of a workload starts at byte address `(i + 1) << 32`, so arrays never overlap and are page-aligned. "Line index" below means the cache-line number inside the array, so the byte address is `base + 64 * line_index + byte_offset`.
- **Instruction gap:** `gap = rng.geometric(1 / (gap_mean + 1)) - 1`, with `gap_mean = 4` by default (about 20% of instructions are memory accesses).
- **PC sites:** each access site in a workload gets a fixed PC, `0x400000 + 0x40 * k` for site number `k`.
- **Length:** generators loop their pattern until `n_records` is reached and then truncate. With default lengths every workload MUST yield at least 400,000 L2 demand accesses under the default config; a test checks this.
- **Vectorisation:** build each trace with numpy in chunks. A Python loop per record is not acceptable at these lengths.

### 4.3 Workload list

| Workload | Split | Behaviour | `dep` |
| --- | --- | --- | --- |
| `seq_stream` | dev | Sequential lines | 0 |
| `stride_const` | dev | Constant stride of 5 lines | 0 |
| `multi_stride` | dev | Four interleaved streams with different PCs and strides | 0 |
| `matmul_blocked` | dev | Blocked dense matrix multiply | 0 |
| `spmv_csr` | dev | Sparse matrix-vector product with random gathers | gathers 1 |
| `hash_lookup` | dev | Random bucket lookups with short linear-probe chains | probes 1 |
| `phase_mix` | dev | Repeating phases: stride, random, stream, multi-stride | 0 |
| `bfs_graph` | test | Breadth-first search on a random graph | gathers 1 |
| `list_walk` | test | Linked list allocated with partial spatial locality | 1 |
| `ptr_chase` | test | Walk of a random permutation cycle | 1 |
| `random_uniform` | test | Uniform random lines | 0 |
| `mix_reg_irr` | test | Alternating regular stream and random accesses on different PCs | 0 |
| `stencil2d` | test | 5-point Jacobi sweep | 0 |

### 4.4 Generator definitions

In each definition `n` is the record index within the pattern, sites are numbered in the order listed, and "random" means drawn from the trace's seeded generator.

- **`seq_stream`:** one array of `2^20` lines (64 MiB). Record `n` reads line `n mod 2^20`, byte offset 0. One site.
- **`stride_const`:** one array of `2^20` lines. Record `n` reads line `(5 * n) mod 2^20`. One site.
- **`multi_stride`:** four arrays of `2^19` lines, one site per array, strides `+1, +3, +7, -2` lines. Records go round-robin over the four streams. Stream `j` at its `m`-th access reads line `(stride_j * m) mod 2^19`.
- **`matmul_blocked`:** `C[N][N] += A[N][N] * B[N][N]`, `N = 512`, 8-byte elements, row-major, block size 32. Loop order `ii, jj, kk` (block loops), then `i, k, j` inside the block. Per `(i, k)`: load `A[i][k]` (site 0); then per `j`: load `B[k][j]` (site 1), load `C[i][j]` (site 2), store `C[i][j]` (site 3).
- **`spmv_csr`:** `n_rows = 2^19`; row `i` has `max(1, Poisson(16))` non-zeros with column indices uniform in `[0, n_rows)`, fixed by the seed. Element sizes: `row_ptr` 4 B, `col` 4 B, `val` 8 B, `x` 8 B, `y` 8 B. Per row: load `row_ptr[i]` (site 0); per non-zero `e`: load `col[e]` (site 1), load `val[e]` (site 2), load `x[col[e]]` (site 3, `dep = 1`); then store `y[i]` (site 4). Repeat passes over the matrix.
- **`hash_lookup`:** one table of `2^22` buckets of 8 bytes. Per lookup: pick `h` uniformly at random; load bucket `h` (site 0). Then, up to 3 times, with probability 0.3 load bucket `(h + k) mod 2^22` for `k = 1, 2, 3` in turn (site 1, `dep = 1`), stopping at the first failed coin flip.
- **`phase_mix`:** one array of `2^20` lines, phases of 300,000 records each, cycling in this order. Phase A (stride 3): site 0 reads lines `(start + 3 m) mod 2^20`. Phase B (random): site 0 reads uniformly random lines. Phase C (stream): site 0 reads `(start + m) mod 2^20`. Phase D (multi-stride): four sites (0 to 3) interleave with strides `+1, +3, +7, -2`. `start` is drawn at random at the beginning of each phase, and phases reuse site 0's PC on purpose, so the agent must notice the behaviour changed.
- **Phase variants for E10** (derived from `phase_mix`, same A, B, C, D order and the same reuse of site 0's PC, not counted among the 13 workloads): `phase_mix_fast` has abrupt boundaries every 30,000 records; `phase_mix_gradual` has phases of 100,000 records with a 50,000-record blend at each boundary, where each record in the blend takes site 0's behaviour from the next phase with a probability that rises linearly from 0 to 1 and from the current phase otherwise. The generator takes `phase_len` and `transition` (`abrupt` or `gradual`) parameters and records boundary positions in `meta`.
- **`bfs_graph`:** random directed graph with `n = 2^18` vertices, out-degree `max(1, Poisson(8))`, neighbours uniform at random. CSR arrays: `row_ptr` 4 B, `col` 4 B; plus `visited` 1 B, `parent` 4 B, `queue` 4 B per vertex. Run BFS from a random source with a FIFO queue; after the queue empties, pick a new random source and reset `visited` (the reset itself generates no records). Per popped vertex `v`: load `queue[head]` (site 0), load `row_ptr[v]` and `row_ptr[v+1]` (site 1, `dep = 1`); per neighbour index `e`: load `col[e]` (site 2), load `visited[u]` (site 3, `dep = 1`); if `u` was unvisited: store `visited[u]` (site 4), store `parent[u]` (site 5), store `queue[tail]` (site 6).
- **`list_walk`:** an arena of `2^20` nodes of 64 bytes. Start at node 0. Each step loads the current node (site 0, `dep = 1` except for the first record) and moves to node `cur + 1` with probability `p_seq = 0.5`, otherwise to a uniformly random node. `p_seq` is a parameter.
- **`ptr_chase`:** a random single-cycle permutation of `2^18` nodes of 64 bytes (use Sattolo's algorithm with the trace seed). Walk the cycle repeatedly, loading each node (site 0, `dep = 1`).
- **`random_uniform`:** one array of `2^20` lines; each record reads a uniformly random line (site 0).
- **`mix_reg_irr`:** two arrays of `2^20` lines. Records strictly alternate: sequential stream read of array 0 (site 0, line `n/2 mod 2^20`), then a uniformly random read of array 1 (site 1).
- **`stencil2d`:** `N = 2048` doubles per row, arrays `a` (source) and `b` (destination), row-major. For `i` in `1..N-2`, `j` in `1..N-2`: load `a[i-1][j]` (site 0), `a[i][j-1]` (site 1), `a[i][j]` (site 2), `a[i][j+1]` (site 3), `a[i+1][j]` (site 4), then store `b[i][j]` (site 5). Repeat sweeps.

### 4.5 Converting ChampSim traces (optional under option A, required under option B)

If real traces are wanted, `trace/champsim.py` MAY convert `.champsimtrace.xz` files (the DPC-3 SPEC traces are a common source). Each instruction record is 64 bytes: `ip` (uint64), `is_branch` (uint8), `branch_taken` (uint8), `destination_registers[2]` (uint8), `source_registers[4]` (uint8), `destination_memory[2]` (uint64), `source_memory[4]` (uint64), little-endian. Emit one load record per non-zero `source_memory` entry and one store record per non-zero `destination_memory` entry, with `pc = ip`, `gap` = instructions since the previous record, `dep = 0`. Verify the struct layout against the ChampSim version that produced the traces before trusting the output. Converted traces are an extra evaluation set and are never used for tuning.

### 4.6 Data strategy: two options (decision pending)

The spec supports two ways of choosing the data. Decide before starting build step M5, because option B needs the trace converter and the real traces first. Under either option the agent learns online from the trace it is running; "training data" means the traces it runs on, plus optional pre-training on development traces in E5.

**Recommendation: choose option B whenever the real traces can be obtained and converted.** The action patterns, the PC and page features and most synthetic generators were designed together, so strong results on synthetic traces alone can reflect that match and not the learning method; held-out real software is the one kind of evidence that does not share those assumptions. Under option B, use at least six real traces from at least three benchmark suites as the test set, never tune on them, and report real and synthetic results separately as well as pooled. Under option A, say plainly in the README that conclusions are about the synthetic pattern classes and make no claim about real software.

**Option A: synthetic and kernel traces only (the spec as written)**

- **Data:** the 13 generated workloads in section 4.3. No downloads.
- **Strengths:** fully reproducible, quick to start, and every pattern has known ground truth, so ablations and behaviour checks (E7, section 7.5) are easy to interpret.
- **Weakness:** results may not carry over to real programs, so the strongest claim you can make is about these workload families, not about real software.
- **Changes to the spec:** none.

**Option B: synthetic plus real traces (stronger evidence)**

- **Data:** the 13 generated workloads plus about 8 real traces converted from ChampSim files, for example SPEC CPU2017 traces from the DPC-3 set. The ChampSim README puts the full DPC-3 set at roughly 20 GB, so download only the files you pick.
- **Selection:** run each candidate under `none` and keep traces with at least 3 LLC misses per thousand instructions, a filter used in the Pythia evaluation. Split the kept traces into about 4 development traces (for tuning) and about 4 held-out test traces, mixing regular and irregular programs.
- **Length:** the converter takes `--skip` and `--max-records` (default 4,000,000 records after skipping the first 1,000,000) so each trace has a size comparable to the generated ones.
- **Real workloads in the tables:** name them `real_<program>` with `Split` set to `dev` or `test`, and add them to section 4.3.
- **Changes to tuning (7.1):** the objective becomes the average of two geometric means, one over the synthetic development workloads and one over the real development traces, so real data counts for half.
- **Changes to experiments (7.2):** E1 to E8 include the real test traces. Report synthetic and real aggregates separately and never pool them into one headline number. E5 trains on the real development traces and evaluates on the real test traces, which is the version closest to training on real data.
- **Changes to expectations (7.5):** state no expected outcome for real traces in advance; report what happens.
- **Added test:** T16 converts a tiny hand-built ChampSim file and checks the loads, stores and `gap` values.
- **Added build step M4b:** converter plus trace selection, done when `pfsim convert` produces valid `.npz` files and the MPKI filter picks at least 8 traces.
- **Limits to state in the README:** converted traces carry `dep = 0`, so the timing model overlaps misses more than real dependent code would; and real traces have no ground-truth pattern, so E7 shows action histograms but cannot check them against a known answer.

A third option, real traces only, is not supported: it removes the controlled workloads that make the ablations interpretable.

## 5. Baseline prefetchers

Implement the classic prefetchers below behind the interface in section 3.8, tune each fairly, and add a future-informed lookahead baseline (5.6) as a diagnostic of remaining headroom. All baselines are called on every L2 demand access, return `PrefetchRequest(line, tag=None, trigger_page=ev.page)` objects, and ignore outcome events. Each reports `storage_bits()` using the formula given.

### 5.1 `none`

Returns no requests. Its runs provide `ipc0`, `l2_misses0` and `dram_reads0` for the metrics in section 3.9.

### 5.2 `next_line`

On each L2 demand access to line `L`, request `L + 1, ..., L + degree`. Parameter `degree` (default 2). Storage: 0 bits.

### 5.3 `stride_pc`

A PC-indexed stride detector with a 256-entry direct-mapped table; index `(pc >> 2) & 255`. Each entry holds `tag` (the full PC), `last_line`, `stride` and a 2-bit confidence `conf`.

```python
def on_demand(ev):
    e = table[(ev.pc >> 2) & 255]
    if e.tag != ev.pc:
        e.reset(tag=ev.pc, last_line=ev.line, stride=0, conf=0); return []
    s = ev.line - e.last_line
    e.last_line = ev.line
    if s == 0: return []
    if s == e.stride:
        e.conf = min(3, e.conf + 1)
    else:
        e.conf = max(0, e.conf - 1)
        if e.conf == 0: e.stride = s
    if e.conf >= min_conf and e.stride != 0:
        return [req(ev.line + k * e.stride) for k in range(1, degree + 1)]
    return []
```

Parameters: `degree` (default 4), `min_conf` (default 2). Storage: `256 * (16 + 32 + 16 + 2)` bits.

### 5.4 `bop` (best-offset prefetcher)

A simplified version of Michaud's best-offset prefetcher (HPCA 2016) that learns one global offset by scoring candidate offsets on how often they would have prefetched a line early enough.

- Candidate offsets: `[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60]`.
- Parameters: `SCOREMAX = 31`, `ROUNDMAX = 100`, `BADSCORE = 1`, recent-request (RR) table of 256 lines with FIFO replacement, `degree` (default 1).
- **RR insertion (our simplification):** when a demand access to line `X` occurs at time `t`, schedule `X` for insertion into RR at time `t + 254` cycles (the reference miss latency, `llc.lat + dram_latency + transfer_cycles` at the default config). At every `on_demand`, first insert all entries whose time has come. This captures the idea that an offset only counts if the prefetch would have arrived in time.
- **Learning:** keep `scores[o]` for each candidate and a pointer `test_idx`. On each `on_demand` for line `X`, test one offset `o = offsets[test_idx]`: if `X - o` is in RR and in the same page, `scores[o] += 1`; if the score reaches `SCOREMAX`, end the phase with `best = o`. Advance `test_idx`; after a full pass over the list, increment `round`; when `round == ROUNDMAX`, end the phase with the highest-scoring offset.
- **End of phase:** if `scores[best] > BADSCORE`, set the active offset `D = best`, otherwise turn prefetching off (`D = None`); reset all scores and the round counter.
- **Prefetching:** if `D` is not `None`, request `X + k * D` for `k = 1..degree`.

Storage: `256 * 16 + 26 * 5 + 64` bits.

### 5.5 `spp_lite` (optional)

A simplified signature-path prefetcher after Kim et al. (MICRO 2016): a 256-entry per-page signature table (`last_offset`, 12-bit `signature`), a 512-entry pattern table indexed by signature holding 4 `(delta, counter)` pairs, signature update `sig = ((sig << 3) ^ delta) & 0xFFF`, and look-ahead that multiplies path confidence by each delta's `counter / total` and stops below a threshold of 0.25 or at depth 6. If implemented, it is evaluated in E1 and tuned like the others.

### 5.6 `future_lookahead` (diagnostic, uses future information)

This is a diagnostic baseline, not a proven upper bound. A fixed-lookahead policy can still be late, redundant, dropped or polluting, so it is not guaranteed to beat any other prefetcher (T9 asserts only on controlled traces). Because the L2 demand-access stream does not depend on the prefetcher (section 1), run the trace once with `none` and record the sequence of L2 demand lines `S[0..M-1]`. The baseline then, at L2 access index `i`, requests `S[i + lookahead]` (tune `lookahead` over `{4, 8, 16, 32, 64}` and report the best per workload, which is why it is a lookahead sweep and not a single oracle), with `trigger_page` set to the target line's own page so the same-page rule never drops it. A true perfect-information bound would also choose which lines to skip and when to issue (for example by replaying with Belady-style replacement); this spec does not build one, so never write the words upper bound or oracle in results. Set `uses_future = True` in `stats()`, mark the rows clearly, and name the registry key `future_lookahead`.

### 5.7 Tuning the baselines

Every baseline gets a small grid, tuned with the same protocol as the agent (section 7.1): `next_line.degree` in `{1, 2, 4, 8}`; `stride_pc.degree` in `{1, 2, 4, 8}` and `min_conf` in `{1, 2, 3}`; `bop.degree` in `{1, 2, 4}` and `ROUNDMAX` in `{50, 100, 200}`. Store the chosen values in `configs/baselines_tuned.yaml`. Untuned baselines make any comparison weak.

## 6. RL agent specification

The agent makes one decision per L2 demand access: whether to prefetch, which stream pattern to follow, and how far ahead to fetch. It learns online with double Q-learning over a linear value function built from six small feature tables, and is trained by a reward that sees accuracy, graded timeliness, bandwidth pressure and cache pollution.

### 6.1 What is original here

Ideas shared with the literature (cite them in the README): prefetching as RL, page-local offsets as actions, and delayed rewards through a queue of pending prefetches. The design below differs in four ways that matter:

- **Distance is part of the action.** The action is a stream pattern plus a prefetch distance (1 to 16 steps ahead), so the agent controls timeliness directly instead of only choosing an offset.
- **Graded timeliness.** A prefetch that arrives late earns a partial reward in proportion to how much latency it still saved, not a fixed binary reward.
- **Explicit pollution credit.** Evictions caused by prefetches are tracked in a victim filter, and a later miss on an evicted line penalizes the prefetch that caused it.
- **System state is part of the state.** Bandwidth utilization, recent accuracy, prefetch occupancy and pollution pressure enter the state, so the same access pattern can get different actions under different system conditions.

It also uses a different learner: double Q-learning on a sum of feature tables, not SARSA with a max over feature vaults.

### 6.2 Decision point and history tables

Each call to `on_demand` is one decision. Before computing features, update two small history tables with the current access.

- **PC table:** 1024 entries, direct-mapped by `pc_hash = ((pc >> 2) ^ (pc >> 12)) & 1023`. An entry stores `tag` (full PC), `last_line`, and the three most recent deltas `d1, d2, d3` in lines (`d1` newest, so `d1 = line - last_line` includes the current access). A tag mismatch resets the entry to `last_line = line`, deltas 0. Deltas are clipped to `[-1023, 1023]`.
- **Page table:** 256 entries, direct-mapped by `page & 255`, storing `tag`, `last_offset`, and `pd1`, the last in-page offset delta (clipped to `[-63, 63]`). A tag mismatch resets `pd1 = 0`.

### 6.3 State

Quantize a delta `d` (in lines) into one of 17 classes:

```python
def qd(d):
    if d == 0: return 0
    a = abs(d)
    c = a if a <= 4 else 5 if a <= 8 else 6 if a <= 16 else 7 if a <= 64 else 8
    return c if d > 0 else 8 + c          # 0, 1..8 positive, 9..16 negative
```

System buckets from `ev.sys`: `bw_b = min(3, int(bw_util * 4))`; `acc_b` = 0, 1, 2, 3 for `acc_ema` below 0.25, 0.5, 0.75 or above; `q_b` = 0, 1, 2 for `pf_occupancy` below 1/3, below 2/3 or above; `pol_b` = 0, 1, 2 for `pol_ema` below 0.02, below 0.10 or above.

The state `s` is the list of six row indices below. The value of an action is the sum of one weight per feature table (section 6.6).

| Table | Index formula | Rows |
| --- | --- | --- |
| `f1` | `qd(d1) * 17 + qd(d2)` | 289 |
| `f2` | `qd(d1) * 289 + qd(d2) * 17 + qd(d3)` | 4913 |
| `f3` | `(pc_hash >> (10 - pc_bits)) * 17 + qd(d1)`, `pc_bits` default 8 | 4352 |
| `f4` | `offset * 17 + qd(pd1)` | 1088 |
| `f5` | `((bw_b * 4 + acc_b) * 3 + q_b) * 3 + pol_b` | 144 |
| `f6` | `(qd(d1) * 4 + bw_b) * 4 + acc_b` | 272 |

The tables combine local delta history (`f1`, `f2`), per-PC identity (`f3`), position inside the page (`f4`), pure system condition (`f5`) and the interaction between pattern and system condition (`f6`).

### 6.4 Actions

`NA = 36`: action 0 is `NONE`, and action `1 + 5 * p + j` means pattern `p` with reach `r_j` for `p` in the list below and `r_j` in `(1, 2, 4, 8, 16)`.

| Pattern `p` | Step `delta` in lines |
| --- | --- |
| `P1` | +1 |
| `P2` | +2 |
| `P4` | +4 |
| `M1` | -1 |
| `PC_D1` | `d1`, the PC's most recent delta |
| `PC_D2` | `d2`, the PC's previous delta (catches alternating patterns) |
| `PAGE_D1` | `pd1`, the most recent in-page delta |

The action requests the single line `line + r * delta`: reach `r` sets how far ahead of the stream the prefetch lands, which is how the agent trades timeliness against waste. An action is **valid** only if its `delta` is non-zero with absolute value at most 63 for the `D1`, `D2` and page patterns, and `line + r * delta` is in the same 4 KiB page as `line`. Invalid actions are masked out of greedy choice, random exploration and the bootstrap maximum. `NONE` is always valid.

### 6.5 Reward

Each decision produces one reward, from the outcome of its single prefetch (or immediately for `NONE`). Defaults are starting values to be tuned (section 7.1). `miss_lat_ref = llc.lat + dram_latency + transfer_cycles` (254 at the default config).

| Outcome | Reward |
| --- | --- |
| `NONE` chosen | `-c_none_miss` if this L2 access missed, else 0 |
| `USEFUL(wait)` | `c_use * (t_min + (1 - t_min) * (1 - min(1, wait / miss_lat_ref)))`: 1.0 for a timely prefetch, falling to `c_use * t_min` for one that arrived just as it was needed |
| `USELESS` | `-(c_w0 + (c_w1 - c_w0) * u_issue)`: wasted bandwidth costs more when the channel was busy at issue |
| `REDUNDANT` | `-c_red` |
| `DROPPED` (`mshr` or `bandwidth`) | `-c_drop` |
| `POLLUTION` (later) | extra `-c_pol`, applied as a correction (section 6.6) |

Defaults: `c_use = 1.0`, `t_min = 0.2`, `c_w0 = 0.3`, `c_w1 = 1.0`, `c_red = 0.05`, `c_drop = 0.10`, `c_none_miss = 0.05`, `c_pol = 0.6`.

**Performance-sensitive reward variant (`reward.mode = perf`).** The default reward scores individual prefetch outcomes, which correlates with speedup but is not equivalent to it: an agent can collect many timely-prefetch rewards while its extra traffic slows other misses, and a useful prefetch whose latency the core would have overlapped anyway earns more credit than it deserves. The variant adds a stall term. The core model reports `stall_since_last` in `SysState`: the cycles the core spent blocked on ROB, MSHR or dependency stalls since the previous L2 demand access. At each decision the agent adds `delta_perf = -c_stall * min(1, stall_since_last / miss_lat_ref)` (default `c_stall = 0.3`) to the previous decision's reward, with no change to the bootstrap target. Treat it as an experiment: run E3 with `reward.mode` in {outcome, perf} and `c_stall` in {0.15, 0.3, 0.6} and do not assume it is better. Always report mean reward next to speedup, and the Spearman correlation between per-run mean reward and speedup across all E3 variants (reward and speedup alignment, E12); a variant that raises reward but lowers speedup is a reward-hacking finding to report, not hide.

### 6.6 Learning rule

The action value is `Q(s, a) = sum over k of W_k[f_k(s), a]`, with six weight tables `W_k` of shape `(rows_k, NA)`. There are two complete sets, `A` and `B`. Greedy action selection uses `Q = (Q_A + Q_B) / 2`. Initialize every weight to `q0 / 6` (optimistic start, default `q0 = 0.2`).

For a completed transition `(s, a, r, s', valid')`, with step size `alpha` and `K = 6`:

```python
X, Y = random choice of (A, B) or (B, A)          # which set to update
if gamma > 0 and s_next is not None:
    a_star = argmax over valid' of Q_X(s', .)       # choose with X
    target = r + gamma * Q_Y(s', a_star)            # evaluate with Y
else:
    target = r
delta = clip(target - Q_X(s, a), -delta_max, delta_max)
for k in range(K):
    W_X[k][f_k(s), a] += (alpha / K) * delta
```

**Pollution correction.** When `POLLUTION(tag)` arrives for a decision still in the history ring, do one update on a randomly chosen set with `delta = -c_pol`, no bootstrap, using the same step rule. If the decision has aged out of the ring, ignore the event.

**Delayed credit.** The next state of a decision is the state at the next decision. Keep a `Decision` record `(id, s, a, valid, r, s_next, valid_next, updated)` for each decision, in a dict plus a deque of the most recent 8192 ids.

```python
def on_demand(ev):
    h = update_history(ev)                         # PC and page tables (6.2)
    s = features(ev, h); valid = valid_mask(ev.line, h)
    if self.prev: self.prev.s_next, self.prev.valid_next = s, valid; try_finalize(self.prev)
    a = select(s, valid)                           # 6.7
    d = Decision(id=next_id(), s=s, a=a, valid=valid); self.prev = d; store(d)
    if a == NONE:
        d.r = -c_none_miss if not ev.l2_hit else 0.0
        return []
    delta = resolve(pattern_of(a), h)
    return [PrefetchRequest(ev.line + reach_of(a) * delta, tag=d.id, trigger_page=ev.page)]

def on_outcome(out):
    d = lookup(out.tag)
    if d is None: return
    if out.kind == POLLUTION: pollution_update(d)
    else: d.r = reward_for(out); try_finalize(d)  # REDUNDANT, DROPPED, USEFUL, USELESS

def try_finalize(d):
    if d.r is not None and (d.s_next is not None or gamma == 0) and not d.updated:
        td_update(d); d.updated = True
```

Because events from before `on_demand` (step 3 of section 3.7) arrive first, a decision whose outcome is known may wait for the next decision to supply `s_next`. Decisions still unresolved at the end of a trace are discarded.

### 6.7 Exploration

Modes (`explore.mode`):

- `eps_decay` (default): `epsilon_t = max(eps_min, eps_0 * 0.5 ** (t / H))` with `t` the decision count, `eps_0 = 0.12`, `eps_min = 0.01`, `H = 150000`. With probability `epsilon_t` choose uniformly among valid actions, otherwise choose the greedy action; ties break uniformly at random.
- `eps_fixed`: constant `eps_min`.
- `softmax`: sample from a softmax over valid actions with temperature decaying linearly from 0.5 to 0.05 over the first `2 * H` decisions.

Optimistic initialization (`q0`) also drives early exploration.

### 6.8 Hyperparameters

| Parameter | Default | Tuning range |
| --- | --- | --- |
| `alpha` | 0.05 | log-uniform 0.005 to 0.3 |
| `gamma` | 0.6 | uniform 0 to 0.95 |
| `eps_0`, `eps_min`, `H` | 0.12, 0.01, 150000 | `eps_0` 0.05 to 0.3, `H` 50000 to 500000 |
| `q0` | 0.2 | 0 to 0.6 |
| `c_w0`, `c_w1` | 0.3, 1.0 | 0.1 to 0.6, 0.5 to 1.5 (with `c_w1 >= c_w0`) |
| `c_pol` | 0.6 | 0.2 to 1.2 |
| `t_min` | 0.2 | 0 to 0.5 |
| `c_red`, `c_drop`, `c_none_miss` | 0.05, 0.10, 0.05 | fixed unless an ablation needs them |
| `delta_max` | 2.0 | fixed |
| `pc_bits` | 8 | 4 to 10 (storage sweep, E6) |
| `double_q` | true | ablation only |

Presets: `default` as above; `compact` uses `pc_bits = 6` and drops table `f2`, for the storage comparison in E6.

### 6.9 Variants and ablation flags

Every ablation in section 7 is a config flag, not a code fork:

- `reward.pollution = false` (sets `c_pol = 0`); `reward.bw_term = false` (sets `c_w1 = c_w0` and makes `bandwidth` drops cost the same as `mshr` drops); `reward.graded = false` (sets `t_min = 1`).
- `actions.reaches = [1]` (no distance control); `actions.patterns` selects a subset.
- `agent.gamma = 0` (contextual bandit); `agent.double_q = false`.
- `features.drop = [f2, ...]` removes tables; `features.drop = [f5, f6]` removes all system state.
- `explore.mode` as in 6.7.
- `agent.learn = false` and `agent.explore = false` for frozen evaluation after warm start; `agent.save(path)` and `agent.load(path)` persist weights.
- `reward.mode = perf` and `c_stall` as in 6.5 (performance-sensitive reward).
- `actions.cross_page = true` (page-boundary-protected cross-page variant). Compared against the page-local default, which stays the controlled baseline. Delta-based patterns (`P1`, `P2`, `P4`, `M1`, `PC_D1`, `PC_D2`, `PAGE_D1`) may target lines in the adjacent page (page + 1 for positive deltas, page - 1 for negative) only when the current line is within the last 16 lines of the page for a positive delta, or the first 16 for a negative one, so the agent cannot jump pages from the middle. The request sets `trigger_page` to the target page and the simulator accounts it normally instead of dropping it as `cross_page`. An unused cross-page prefetch earns an extra penalty `-c_xp` (default 0.2) on top of the USELESS reward. Report cross-page issue count, accuracy and speedup against the page-local agent, and keep the variant only if the speedup gain justifies the added risk.
- `features.conf = true` (confidence and recency state, experimental). Add table `f7` of 48 rows indexed by (`pf_conf`, `age_b`, `d1_class`): `pf_conf` is a 2-bit saturating counter per PC-hash entry, incremented on a USEFUL outcome and decremented on a USELESS outcome whose tag carries that PC hash; `age_b` is a 4-way log bucket of L2 accesses since that PC was last seen (under 16, under 256, under 4096, more); `d1_class` is 0 for zero, 1 for small positive or negative within 4 lines, 2 otherwise. Compare against `full` under an equal tuning budget (60 random-search configs for each) and keep it only if it wins on the dev workloads and holds on test.
- `bw_gate` (simple threshold controller, E11). The agent with `features.drop = [f5, f6]` and a rule on top: when `bw_util > theta_bw` and `acc_ema < theta_acc`, force action NONE. Tune `theta_bw` in {0.6, 0.75, 0.9} and `theta_acc` in {0.3, 0.5}. This is the baseline that shows whether learned bandwidth awareness is worth more than a one-line rule.
- `agent.reset_on_boundary = true` (reset weights at each phase boundary, E10 only; the boundary comes from the generator metadata, which the agent never sees in normal runs), and `agent.learn = false` after a warm-up cycle (frozen, E10).

### 6.10 Storage accounting

`storage_bits()` reports `sum_k rows_k * NA * 16` for one weight set at 16-bit precision, plus the history tables (`1024 * (16 + 32 + 33)` for the PC table and `256 * (16 + 6 + 7 + 8)` for the page table), and prints weights and tables separately. Measured figures for the default configuration: the six tables hold 11,058 rows, so one weight set is 11,058 x 36 x 16 = 6,369,408 bits = 777.5 KiB, and double Q-learning keeps two sets, 12,738,816 bits = 1.52 MiB, before the history tables. A deployed agent would keep one set (greedy action from the averaged or single set), but the spec never claims this agent is hardware-lightweight. Pythia uses 25.5 KB and Berti about 2.55 KB, so storage is a first-class result: E6 plots speedup against storage for the full agent (about 1.52 MiB with two sets), the `compact` preset (pc\_bits 6, no f2, about 405 KiB with two sets), a `mini` preset (16 actions, 1 set, about 47.5 KiB) and `qtable_small`, next to every baseline. Claim hardware-lightness only for a configuration whose measured bits support it.

### 6.11 Optional agents for comparison

- **`qtable_small`:** a plain table. State: `(qd(d1), d1 == d2, bw_b, acc_b)`, 544 states. Actions: `NONE` plus patterns `P1, P2, P4, M1, PC_D1` with reaches 2 and 8 (11 actions). Single-table Q-learning with the same reward and delayed-credit machinery. Its job is to show what the richer state and action space buy.
- **`dqn` (stretch):** Double DQN. Input: the six feature indices, each embedded to 8 dimensions and concatenated; two hidden layers of 128 ReLU units; 36 outputs; same action masking and reward. Replay buffer of 100,000 transitions pushed when a decision is finalized; batch 64 every 4 decisions after 5,000 decisions; Adam with learning rate 1e-3; Huber loss; target network synchronised every 2,000 updates. Run on traces of at most 1,000,000 records and report wall-clock time.

## 7. Experiments and evaluation protocol

Tune everything on development workloads with tuning seeds, freeze the configurations, then run eight experiments on all workloads with separate evaluation seeds. Every experiment is a CLI command that writes raw runs, a summary table and figures.

### 7.1 Tuning protocol

- **Seeds.** Tuning uses trace seeds `{100, 101}` and agent seeds `{100, 101, 102}`. Final evaluation uses trace seeds `{0, 1, 2}` and agent seeds `{0, 1, 2, 3, 4}`. The two sets never overlap.
- **Agent search.** Random search with 60 configurations drawn from the ranges in section 6.8, each evaluated on all 7 development workloads at 1,000,000 records, trace seed 100, three agent seeds, at the default and low bandwidth presets. Objective: geometric mean speedup over all (workload, bandwidth) pairs, averaged over agent seeds.
- **Refinement.** Re-run the 5 best configurations at default trace length on trace seeds `{100, 101}` and keep the best. Write it to `configs/agent_tuned.yaml`.
- **Baselines.** Run the full grids from section 5.7 with the same objective and write `configs/baselines_tuned.yaml`.
- **Freeze.** After `configs/*_tuned.yaml` exist, no test workload is used for tuning anything. The commit that contains those files MUST precede any test-workload result.
- Save every tuning run under `results/tune/`.

### 7.2 Experiments

Two budgets: `--budget full` (default trace lengths, 3 trace seeds, 5 agent seeds) and `--budget quick` (1,000,000 records, 1 trace seed, 2 agent seeds) for development. Deterministic baselines run once per trace seed.

| ID | Question | Setup | Output |
| --- | --- | --- | --- |
| E1 | How do all prefetchers compare? | `none`, `next_line`, `stride_pc`, `bop` (plus `spp_lite` if built), `qtable_small`, tuned agent, `future_lookahead`; all 13 workloads; default bandwidth | Table of speedup, accuracy, coverage, pollution rate and overprediction per workload, with geometric means for dev, test and all |
| E2 | How does bandwidth change the ranking? | Bandwidth in `{3.2, 6.4, 12.8, 25.6, 51.2}` GB/s; all prefetchers and workloads; trace seed 0, 3 agent seeds | Line plot of geometric-mean speedup against bandwidth with confidence bands; per-workload plots for `seq_stream`, `random_uniform`, `hash_lookup`, `phase_mix` |
| E3 | Which design choices matter? | Ablation variants of section 6.9: `full`, `no_pollution`, `no_bw_term`, `no_graded`, `no_reach`, `bandit`, `no_double_q`, `no_sys_features`, `no_page_features`, `no_f2`, `softmax`, `eps_fixed`, plus `qtable_small`; all workloads at default and low bandwidth; trace seed 0, 3 agent seeds | Table of geometric-mean speedup and change against `full`, with per-workload bars |
| E4 | How fast does it learn and adapt? | `phase_mix` with interval records, and learning curves on every dev workload | Plot of window IPC, window accuracy and mean reward against L2 accesses, with phase boundaries marked; recovery times (below) |
| E5 | Does experience transfer? | Train on the dev workloads in random order for 2 passes of 1,000,000 records each, save weights; on test workloads compare cold start, warm start with learning, and warm start frozen (`learn = false`, `explore = false`) | Table of speedup per test workload for the three modes; note any negative transfer |
| E6 | What does performance cost in storage? | Sweep `pc_bits` in `{4, 6, 8, 10}`, drop `f2`, drop `f3`, `reaches = [1, 4, 16]`; compare with baseline storage | Scatter of geometric-mean test speedup against `storage_bits` |
| E7 | What did the agent learn? | Action histograms (pattern by reach) over the last third of each run, per workload | Heatmaps and a table of behaviour checks (7.5) |
| E8 | How much headroom is left? | Agent, best baseline and `future_lookahead` per workload | Table of speedup and fraction of the best future\_lookahead gain captured, defined only where that baseline's speedup exceeds 1.02; read it as a headroom indication, not a bound |
| E10 | How well does it adapt to phase changes? | phase\_mix, phase\_mix\_fast and phase\_mix\_gradual (section 4.4); agent modes online, frozen after one warm-up cycle, and reset at each boundary; plus stride\_pc and bop; default bandwidth; trace seed 0, 3 agent seeds | Window IPC and mean reward against L2 accesses with boundaries marked; recovery time and adaptation loss (below) per mode and variant |
| E11 | Does learned bandwidth awareness beat a simple rule? | full, no\_sys\_features, bw\_gate (tuned thresholds), next\_line, stride\_pc; bandwidth in {3.2, 6.4, 12.8, 25.6} GB/s; all workloads; trace seed 0, 3 agent seeds | Geometric-mean speedup and wasted DRAM reads (useless plus dropped prefetches) against bandwidth for each controller; per-workload panels for random\_uniform, hash\_lookup and seq\_stream |
| E12 | What is the speedup, storage and adaptation trade-off, and does reward track speedup? | Reuse E3, E6 and E10 runs: full, compact, mini, qtable\_small, reward.mode = perf, actions.cross\_page, features.conf, and every baseline | Pareto plot of geometric-mean test speedup against storage\_bits, with a second panel of recovery time; Spearman correlation of mean reward with speedup across variants; cross-page issue count, accuracy and speedup against the page-local agent |

E9 (optional): compare the DQN agent with the main agent on the dev workloads at 1,000,000 records.

E3 also runs the extension variants of section 6.9 (`reward.mode = perf` with its `c_stall` values, `actions.cross_page`, `features.conf`, and the `compact` and `mini` presets), each tuned with the same 60-config budget as `full`, so that no variant wins through extra tuning. E10 to E12 depend on tuned configs and E3 results, so run them after M8 and M9.

**Recovery time (E4).** For each phase boundary after the first full cycle of `phase_mix`, find the first interval after the boundary whose window IPC is at least 90% of the median window IPC of the second half of that phase. Recovery time is the number of L2 demand accesses from the boundary to that interval.

**Adaptation loss (E10).** For each boundary after the first full cycle, let `ref` be the median window IPC of the second half of the new phase and `T = 100,000` L2 demand accesses. Adaptation loss is `1 - (sum over windows in [boundary, boundary + T) of IPC_w * len_w) / (ref * T)`, the fraction of performance lost while adapting. Report recovery time and adaptation loss for three agent modes: online (learning and exploration on throughout), frozen (weights trained through one full warm-up cycle, then `learn = false` and `explore = false`), and reset (weights re-initialised at every boundary using the generator's boundary metadata, an upper-effort adaptation reference that real hardware could not use). Include `stride_pc` and `bop` as non-learning references. The point is to show when online learning pays for itself and when forgetting or exploration cost outweighs it.

### 7.3 Statistics

- Primary metric: speedup over `none` (section 3.9), aggregated by geometric mean over workloads.
- Per-workload results: mean and standard deviation over seeds.
- Aggregate confidence intervals: resample agent seeds with replacement within each workload, recompute the geometric mean, repeat 2,000 times, and report the 2.5th and 97.5th percentiles.
- Agent versus best baseline: per workload, compare paired by trace seed and report win, tie or loss, where a tie means within 1% speedup.
- Report every workload. Do not drop workloads where the agent loses.

### 7.4 Outputs

Each experiment writes `results/<id>/runs.csv`, `results/<id>/summary.csv` and PNG figures `results/<id>/fig_*.png`. `runs.csv` has one row per run:

```
workload,trace_seed,prefetcher,variant,agent_seed,bw_gbps,records,instructions,cycles,ipc,speedup,
l2_accesses,l2_misses,coverage,accuracy,late_fraction,pollution_rate,dram_reads,overprediction,
avg_bw_util,redundant_rate,dropped_rate,storage_bits,wall_seconds,config_hash,git_commit
```

Interval records go to `results/<id>/intervals/<run_key>.csv`. The README reproduces every table and figure with one documented command per experiment (`pfsim experiment E1 --budget full`, and so on).

### 7.5 Expected behaviours

These are hypotheses to test, not guarantees. If one fails, first check the simulator and the reward scale before drawing conclusions, then report the outcome honestly.

- **Regular streams.** `stride_pc` and `bop` give large speedups on `seq_stream` and `stride_const`. The agent should be in the same range after its learning phase; E7 should show it choosing `PC_D1` or the fixed-offset pattern matching the stream, with a reach above 1.
- **No-win workloads.** On `random_uniform` and `ptr_chase` no prefetcher helps. The agent should stay near 1.0 speedup (no worse than about 1% below) by choosing mostly `NONE`, while `next_line` may lose, especially at low bandwidth.
- **PC-aware behaviour.** On `multi_stride`, `mix_reg_irr` and `phase_mix`, a global-offset prefetcher such as `bop` should degrade relative to PC-aware ones; the agent should be competitive with the best baseline or ahead of it.
- **Bandwidth.** At low bandwidth aggressive baselines lose ground. `no_bw_term` should hurt the agent most at low bandwidth.
- **Pollution.** `no_pollution` should raise `pollution_rate`, most visibly on workloads whose footprint just exceeds L2.
- **Headroom.** On `seq_stream` and `stride_const` the `future_lookahead` baseline should be at or above every other prefetcher (T9). Elsewhere it is only a diagnostic: when a baseline beats it, record that as a finding (late, redundant or polluting lookahead), and inspect the simulator only if the gap also appears on those controlled traces.
- **Adaptation and bandwidth (hypotheses, not requirements).** After a boundary into a phase not seen in warm-up, the online agent should recover faster than the frozen one, and should lose less than the reset agent on boundaries that return to an earlier phase. At 3.2 GB/s on `random_uniform`, `hash_lookup` and `seq_stream`, `full` should waste fewer DRAM reads than `no_sys_features` and perform no worse than `bw_gate`. If a hypothesis fails, report it as a finding with the evidence; do not retune on the test set to rescue it.

## 8. Tests and acceptance checks

Write `pytest` tests alongside each component, using tiny configs where the answer can be computed by hand. The numeric thresholds are sanity checks chosen to catch bugs; if a correct implementation lands slightly outside one, record the measured value in `DECISIONS.md` instead of loosening the test silently. Tests marked slow run only with `pytest -m slow`.

| ID | Component | Test | Pass condition |
| --- | --- | --- | --- |
| T1 | Cache | 2-way, 4-set cache: install, touch and evict in a scripted order | Victims follow LRU exactly; `touch` makes a line MRU; set mapping matches `line mod sets` |
| T2 | Core timing | Trace of one repeated address with `gap = 3` | Steady-state IPC, computed as (gap + 1) instructions per record divided by cycles per record, within 1% of `width` (the model charges (gap + 1) / width cycles per record, so the reference run gives 3.996 at width 4) |
| T3 | Core timing | Distinct lines, `gap = 300`, independent | For gap in {300, 600}: cycles per record within 5% of `miss_lat_ref` (254). For smaller gaps latency is partly hidden (gap 100 gives about 84.7 with ROB 256); for gap 1200 the instruction term dominates ((gap + 1) / width = 300.25). Expected value in general: max((gap + 1) / width, miss\_lat\_ref) when every access depends on the previous one (`dep = 1`). Test `dep = 1` at gap 3, 300 and 1200 and require 5% agreement with that formula |
| T4 | DRAM | Distinct lines, `gap = 0`, independent, at 12.8 and 3.2 GB/s | Expected cycles per record = max(transfer\_cycles, (llc\_lat + dram\_lat + transfer\_cycles) / mshr), within 10%. This is the channel bound (80 and 20) at 3.2 and 12.8 GB/s, plus 25.6 GB/s (10), with `avg_bw_util` above 0.9. At 51.2 GB/s the 32-entry MSHR limit binds instead (about 7.5 cycles per record, utilization about 0.67); test that case against the formula and require `avg_bw_util` below 0.8, which proves the MSHR bound and not the channel is the limiter |
| T5 | Accounting | Run `next_line`, `stride_pc` and the agent on every workload (quick) | Returned requests equal `issued + redundant + dropped`; every issued prefetch gets exactly one primary outcome, except those still unresolved at the end; `POLLUTION` never exceeds prefetch fills and never repeats for one eviction; no outcome has an unknown tag |
| T6 | Determinism | Same config and seeds twice; different agent seeds | Identical metrics and trace bytes in the first case; different action sequences in the second |
| T7 | Independence | Record the L2 demand line sequence under `none`, `next_line` and the agent | The three sequences are identical |
| T8 | Baselines | `stride_pc` on `stride_const`; `next_line` on `seq_stream` (quick, default bandwidth) | Accuracy at least 0.9 and speedup above 1.2 for `stride_pc`; accuracy at least 0.9 for `next_line`. If speedup fails, inspect the timing model first |
| T9 | Future-informed lookahead (diagnostic) | All prefetchers on all workloads (quick) | Diagnostic, not a guarantee: a fixed-lookahead prefetcher can be late, redundant, dropped or polluting, so it is not a bound. Assert only on controlled traces: on `seq_stream` and `stride_const` the best-lookahead variant has accuracy at least 0.95 and speedup at least that of `stride_pc`; on `random_uniform` its speedup is within 2% of 1.0 (it must not invent benefit). On every other workload log the gap to the best baseline and print a warning, never a failure, when a baseline beats the best lookahead |
| T10 | Reward | Synthetic outcomes into `reward_for` | `USEFUL(wait=0)` gives `c_use`; `USEFUL(wait=miss_lat_ref)` gives `c_use * t_min`; `USELESS` gives `-c_w0` at `u_issue = 0` and `-c_w1` at 1; `REDUNDANT` and `DROPPED` give `-c_red` and `-c_drop` |
| T11 | Learning rule | One transition with `gamma = 0`, `double_q = false`, `r = 1`, all weights at `q0 / K` | `Q(s, a)` becomes `q0 + alpha * (1 - q0)`; no other action's value changes for that state |
| T12 | Masking | Contexts with `d1 = 0`, with the page offset near 63, and a fresh PC | `PC_D1` actions invalid when `d1 = 0`; actions leaving the page invalid; `NONE` always valid; greedy and exploratory choices never pick an invalid action |
| T13 | Agent smoke | Agent on `stride_const` and on `random_uniform` (quick) | In the last 20% of decisions: at least 0.8 of non-`NONE` actions use `PC_D1` or `PAGE_D1` on `stride_const`, and at least 0.6 of actions are `NONE` on `random_uniform` |
| T14 | Generators (slow) | Each workload at default length | Deterministic for a seed; `validate` passes; at least 400,000 L2 demand accesses |
| T15 | CLI | `pfsim run` on a tiny trace | Output JSON contains every key of the `runs.csv` schema |
| T17 | Timing validation | The microbenchmark set of section 3.3: dependent chain at gap 3 and 1200; independent distinct lines at gap 0 for 3.2, 12.8, 25.6 and 51.2 GB/s; independent lines at gap 100 for ROB 128, 256, 512 | Cycles per record within 2% of 254.0 and 300.3; 80.0, 20.0, 10.0 and 7.5; 127.0, 84.7 and 42.3 |
| T18 | Pollution accounting | Single 2-way set, scripted. Case a: demand A, demand B, prefetch P (evicts A), demand A. Case b: demand A, demand B, prefetch P (evicts A), demand C, demand A | Case a emits exactly one POLLUTION for A. Case b emits none, because the demand-only shadow array also lost A, and counts one pollution\_benign. Turning pollution.counterfactual off makes case b emit POLLUTION |
| T19 | Sensitivity | Quick E1 with ROB 128 and 512, MSHR 16 and 64, dram\_latency 120 and 300 (S1 of section 3.3) | Runs complete and write a ranking table per variant; next\_line speedup on seq\_stream is above 1.0 in every variant |
| T20 | Cross-page and perf reward | Contexts at page offsets 63, 50 and 5 with positive and negative deltas; a synthetic USELESS cross-page outcome; stall\_since\_last values 0 and 2 x miss\_lat\_ref | With actions.cross\_page = false no action leaves the page. With true, a positive-delta cross-page action is valid only at offsets 48 to 63, a negative one only at 0 to 15. The cross-page USELESS reward is the base value minus c\_xp. delta\_perf is 0 at stall 0 and -c\_stall when the stall is at least miss\_lat\_ref |
| T21 | Threshold gate | bw\_gate with theta\_bw 0.75, theta\_acc 0.5; forced SysState (bw\_util 0.95, acc\_ema 0.1) and (bw\_util 0.2, acc\_ema 0.1) | The first state always yields NONE; the second leaves the agent's choice untouched |
| T22 | Phase variants | Generate phase\_mix, phase\_mix\_fast and phase\_mix\_gradual for two seeds | Deterministic per seed; boundary positions in meta match the phase length (300,000, 30,000, 100,000); in the gradual variant the fraction of next-phase records in the blend zone rises from about 0 to about 1 |

Continuous integration SHOULD run T1 to T13 on every change.

## 9. Repository layout, CLI and build order

Build in the order below. Each milestone ends with its tests passing, so a later failure can be traced to the latest step.

### 9.1 Layout

```
pfsim/
  pyproject.toml
  README.md                 # setup, reproduction commands, results, discussion
  DECISIONS.md              # every place the spec was silent and the choice made
  configs/
    default.yaml            # simulator config (section 3.1)
    agent_default.yaml      # agent defaults (section 6.8)
    agent_tuned.yaml        # written by tuning (section 7.1)
    baselines_tuned.yaml
  src/pfsim/
    config.py               # dataclasses and YAML loading
    trace/        format.py  generators.py  champsim.py (optional)
    sim/          cache.py  dram.py  core.py  simulator.py  events.py  metrics.py
    prefetchers/  base.py  none.py  next_line.py  stride_pc.py  bop.py  spp_lite.py  future_lookahead.py
    rl/           features.py  actions.py  reward.py  agent.py  qtable_small.py  dqn.py (optional)
    experiments/  runner.py  tune.py  stats.py  plots.py  e1.py ... e12.py
    cli.py
  tests/                    # T1 to T22
  results/                  # generated, git-ignored
  traces/                   # generated, git-ignored
```

### 9.2 CLI

```
pfsim gen-traces --out traces/ --seed 0 [--quick]
pfsim run --trace traces/stride_const_s0.npz --prefetcher agent --config configs/default.yaml \
          --agent-config configs/agent_tuned.yaml --bw 12.8 --seed 0 --out run.json
pfsim tune --target agent|baselines
pfsim experiment E1 --budget quick|full
pfsim report                # rebuild all tables and figures from results/
```

Every command prints the config hash and git commit and refuses to overwrite existing results unless `--force` is given.

### 9.3 Build order

| Step | Build | Done when |
| --- | --- | --- |
| M1 | Scaffolding, config, trace format, all generators | T14 passes |
| M2 | Caches, DRAM channel, core timing, `none` prefetcher | T1 to T4 and T17 pass |
| M3 | Prefetch issue path, outcome events, victim filter and expiry, `next_line`, `stride_pc` | T5, T7 and T8 pass |
| M4 | Metrics, intervals, runner, CLI, `future_lookahead` | T6, T9 and T15 pass |
| M5 | `bop` (and `spp_lite` if time allows), baseline tuning script | Baselines tuned and `configs/baselines_tuned.yaml` written |
| M6 | RL agent: history tables, features, actions, reward, learning rule, exploration | T10 to T13 pass |
| M7 | Ablation flags, `qtable_small`, save and load | Every variant in E3 runs from config alone |
| M8 | Agent tuning (section 7.1), then freeze configs | `configs/agent_tuned.yaml` committed before any test-workload run |
| M9 | Experiments E1 to E8, statistics and plots (E10 to E12 follow in M11) | All tables and figures regenerate with one command per experiment |
| M10 | README with results and discussion, `DECISIONS.md`; optional DQN (E9) and ChampSim converter | A fresh clone reproduces the results from the README alone |
| M11 | Extensions: perf reward, cross-page and confidence variants, bw\_gate, phase variants, counterfactual pollution, sensitivity harness, experiments E10 to E12 | T18 to T22 pass; E10 to E12 regenerate with one command each |

### 9.4 Common pitfalls

- Prefetch fills MUST NOT be counted as demand misses, and prefetches MUST NOT touch L1.
- The victim filter only records victims that were not unused prefetched lines; blaming a prefetch for evicting another wasted prefetch would double-count.
- Bandwidth utilization comes from channel busy time and includes prefetch traffic. A prefetch that hits in the LLC uses no channel time.
- `ready` times apply at every level: an access to a line still being fetched waits for it. Forgetting this makes late prefetches look timely.
- Outcome events can refer to decisions that have aged out of the agent's history ring, so `on_outcome` MUST tolerate unknown tags.
- Reset all per-run state in `reset(seed)`. Leaking learned weights or history between runs silently inflates results; warm start (E5) loads weights explicitly.
- Keep hot loops free of per-access object allocation: convert trace arrays with `.tolist()` once, use plain ints and preallocated arrays for caches and tables, and profile before reaching for `numba`.
- Keep experiment code free of tuning logic. Tuning writes YAML; experiments only read it.

## What already exists

The key prior work is **Pythia** ([Bera et al., MICRO 2021](https://arxiv.org/abs/2109.12021v5), ETH Zürich/SAFARI), which already validates your core idea. It frames the prefetcher as an RL agent that observes several kinds of program context on every demand request and receives a numerical reward judging prefetch quality under the current memory bandwidth usage. It beat MLOP and Bingo by 3.4% and 3.8% in single-core runs, with larger gains in twelve-core and bandwidth-constrained setups ([ETH record](https://www.research-collection.ethz.ch/entities/publication/3e094778-824f-4b7a-9eec-0c2e3aa634f1)). The [code is open](https://science.ecosyste.ms/projects/19409) and implemented in ChampSim.

Other work worth knowing:

- **PPF** (Perceptron-based Prefetch Filter) uses a linear classifier to filter out inaccurate candidates from an underlying SPP prefetcher. It is a learned filter, not RL.
- **Peled et al.** use RL for a context-based prefetcher that approximates semantic locality.
- **Hermes and Sibyl**, from the same group as Pythia: a perceptron-based off-chip predictor and an RL data-placement policy for hybrid storage ([overview](https://arxiv.org/pdf/2603.14583v1)).
- **Survey:** [Toward Intelligent Prefetching](https://arxiv.org/pdf/2606.09955) organizes ML memory-access prediction by learning paradigm (supervised, reinforcement, unsupervised) and by online versus offline training. Start there for a map of the field.

Known limitation to discuss in your report: RL learns as the program runs, so there is a learning phase where the prefetcher is not yet effective.

## Literature review

The field has converged on one RL design, a lightweight SARSA agent with table-based Q-values, and Pythia is the reference point we learn from, not a target to beat. Everything below was read from the papers or their official pages, not from memory.

| Work | Venue | Approach | Key result | Limitation that matters to us |
| --- | --- | --- | --- | --- |
| Context prefetcher (Peled et al.) | ISCA 2015 | Contextual bandit; full cacheline address as the action; compiler hints in the state | First RL prefetcher | Myopic (ignores long-term effects), huge action space, needs software hints ([Pythia §4.5](https://arxiv.org/pdf/2109.12021)) |
| Delta-LSTM (Hashemi et al.) | ICML 2018 | LSTM that classifies the next address delta | Higher precision and recall than stride and correlation prefetchers ([survey](https://arxiv.org/pdf/2606.09955)) | Neural models are often larger than the biggest caches and too slow for a prefetcher ([Pythia §2.2](https://arxiv.org/pdf/2109.12021)) |
| PPF (Bhatia et al.) | ISCA 2019 | Perceptron filter that rejects bad SPP candidates | Pythia beats SPP+PPF by 1.02% single-core | 39.3 KB; it filters candidates but does not generate them |
| Voyager (Shi et al.) | ASPLOS 2021 | Hierarchical neural model | 19.4% higher coverage than ISB ([survey](https://arxiv.org/pdf/2606.09955)) | Same neural size and latency concern |
| [Pythia](https://arxiv.org/abs/2109.12021v5) (Bera et al.) | MICRO 2021 | SARSA, tile-coded tables, 2 features, 16 offsets, bandwidth-aware reward | 22.4% over no prefetching; beats MLOP, Bingo, SPP by 3.4%, 3.8%, 4.3% single-core; 25.5 KB | Loses to Bingo on pure streaming (libquantum); online learning phase; no explicit pollution reward level |
| [Berti](https://webdiis.unizar.es/~chus/pubs/berti/berti.html) (Navarro-Torres et al.) | MICRO 2022 | Non-RL L1D prefetcher using local (per-instruction) deltas | 2.55 KB; 3.5% over IPCP on SPEC CPU2017 and GAP | Not learning-based; a strong, tiny baseline to include |
| [RL-CoPref](https://link.springer.com/article/10.1007/s11227-024-05938-9) | J. Supercomputing 2024 | RL controller that switches multiple prefetchers on or off and sets their degree; tile coding | 35.50% IPC gain; beats individual prefetchers by 5.91-16.54% and the non-RL controller SBP by 4.64% | Controls other prefetchers instead of choosing offsets |
| [CRL-Pythia](https://arxiv.org/abs/2509.10719v1) | arXiv 2025 | Cross-core shared learning on top of Pythia | 28 KB | Unreviewed preprint; treat its claims cautiously |
| [Athena](https://arxiv.org/abs/2601.17615v1) (Bera et al.) | HPCA 2026 | SARSA agent that coordinates a prefetcher and an off-chip predictor; epoch-level reward | 5.0% over the next-best coordination technique in bandwidth-constrained single-core | Control layer; it uses Pythia as the prefetcher underneath |

**What the literature agrees on**

- **Tables, not deep networks.** Pythia's QVStore is 24 KB and its evaluation queue 1.5 KB. The survey names a fundamental tension between runtime adaptability and model capacity.
- **Actions are small sets of page-local offsets.** Pythia prunes the full range of -63 to +63 down to 16 offsets, because long action lists need more exploration and more storage.
- **The reward must see bandwidth.** A bandwidth-oblivious Pythia loses up to 4.6% on average at the lowest DRAM bandwidths and matches the full version when bandwidth is plentiful.
- **Delayed credit uses a FIFO evaluation queue.** Pythia keeps 256 recent actions and applies the SARSA update when an entry is evicted, assigning the inaccurate-prefetch reward if the line was never demanded.
- **Standard evaluation:** ChampSim, DPC-3 and CRC-2 traces, IPC speedup over no prefetching, coverage and overprediction as ratios of LLC misses, a DRAM bandwidth sweep from 150 to 9600 MTPS, and traces the agent was never tuned on.

**Gaps we can use**

- **Pollution.** Pythia's five reward levels (accurate and timely, accurate but late, loss of coverage, inaccurate, no prefetch) have no pollution term. Athena's slides list prefetcher-induced cache pollution among its signals, so adding one is a defensible extension.
- **Timeliness is binary in Pythia.** The paper notes that three timestamps per queue entry would allow a graded definition.
- **Learning phase.** The survey flags the cold-start cost of online RL, which makes a warm-start experiment worthwhile.
- **Streaming workloads.** Pythia trails Bingo-style whole-page prefetching there because of timeliness, so expect to lose on these and say why.
- **Field-level gaps.** The survey reports no standardized ML-prefetcher benchmarks and uncharacterized energy-accuracy trade-offs. Cite this to justify a small, well-documented evaluation setup.

## Sources

- [Pythia (arXiv 2109.12021), read in full](https://arxiv.org/pdf/2109.12021)
- [Pythia (ETH Research Collection)](https://www.research-collection.ethz.ch/entities/publication/3e094778-824f-4b7a-9eec-0c2e3aa634f1)
- [Pythia code and ChampSim notes](https://science.ecosyste.ms/projects/19409)
- [Toward Intelligent Prefetching survey, read in full](https://arxiv.org/pdf/2606.09955)
- [Machine Learning-Driven Intelligent Memory System Design](https://arxiv.org/pdf/2603.14583v1)
- [Berti: an Accurate Local-Delta Data Prefetcher (MICRO 2022)](https://webdiis.unizar.es/~chus/pubs/berti/berti.html)
- [RL-CoPref (J. Supercomputing 2024)](https://link.springer.com/article/10.1007/s11227-024-05938-9)
- [CRL-Pythia (arXiv 2509.10719)](https://arxiv.org/abs/2509.10719v1)
- [Athena (HPCA 2026, arXiv 2601.17615)](https://arxiv.org/abs/2601.17615v1)
- [Q-fetcher, a small tabular Q-learning prefetcher prototype](https://github.com/0xd3ba/q-fetcher)
- [ChampSim usage and DPC-3 traces](https://github.com/PiPython/ChampSim)
- [ChampSim trace generation notes](https://github.com/shubhsherl/champsim-cs422)
