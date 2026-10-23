from __future__ import annotations

from typing import Callable

import numpy as np

from .format import Trace, validate

SYNTHETIC_DEFAULT = 2_000_000
KERNEL_DEFAULT = 4_000_000
WORKLOAD_DEFAULTS = {"matmul_blocked": 10_500_000, "stencil2d": 5_500_000}
DEFAULT_GAP_MEAN = 64.0
BASE_PC = np.uint64(0x400000)
ARRAY_BASE = np.uint64(1 << 32)

WORKLOADS = (
    "seq_stream", "stride_const", "multi_stride", "matmul_blocked",
    "spmv_csr", "hash_lookup", "phase_mix", "bfs_graph", "list_walk",
    "ptr_chase", "random_uniform", "mix_reg_irr", "stencil2d",
    "phase_mix_fast", "phase_mix_gradual",
)
KERNELS = {"matmul_blocked", "spmv_csr", "bfs_graph", "stencil2d"}


def _finish(name: str, seed: int, lines: np.ndarray, sites: np.ndarray,
            rng: np.random.Generator, *, arrays: np.ndarray | None = None,
            writes: np.ndarray | None = None, dep: np.ndarray | None = None,
            byte_offsets: np.ndarray | None = None, meta: dict | None = None,
            gap_mean: float = DEFAULT_GAP_MEAN) -> Trace:
    n = len(lines)
    arrays = np.zeros(n, dtype=np.uint64) if arrays is None else arrays.astype(np.uint64)
    byte_offsets = np.zeros(n, dtype=np.uint64) if byte_offsets is None else byte_offsets.astype(np.uint64)
    addr = (arrays + 1) * ARRAY_BASE + lines.astype(np.uint64) * np.uint64(64) + byte_offsets
    pc = BASE_PC + sites.astype(np.uint64) * np.uint64(0x40)
    if gap_mean < 0: raise ValueError("gap_mean must be non-negative")
    gaps = np.minimum(rng.geometric(1 / (gap_mean + 1), n) - 1,
                      np.iinfo(np.uint16).max).astype(np.uint16)
    trace = Trace(pc, addr, gaps,
                  np.zeros(n, dtype=np.uint8) if writes is None else writes.astype(np.uint8),
                  np.zeros(n, dtype=np.uint8) if dep is None else dep.astype(np.uint8),
                  {"workload": name, "seed": seed, "record_count": n,
                   "gap_mean": gap_mean, **(meta or {})})
    validate(trace)
    return trace


def _simple(name: str, seed: int, n: int, rng: np.random.Generator, gap_mean: float) -> Trace:
    idx = np.arange(n, dtype=np.int64)
    sites = np.zeros(n, dtype=np.uint8)
    if name == "seq_stream":
        lines = idx % (1 << 20)
    elif name == "stride_const":
        lines = (5 * idx) % (1 << 20)
    elif name == "random_uniform":
        lines = rng.integers(0, 1 << 20, n, dtype=np.int64)
    elif name == "multi_stride":
        sites = (idx & 3).astype(np.uint8)
        m = idx >> 2
        strides = np.array([1, 3, 7, -2], dtype=np.int64)
        lines = (m * strides[sites]) % (1 << 19)
        return _finish(name, seed, lines, sites, rng, arrays=sites, gap_mean=gap_mean)
    elif name == "mix_reg_irr":
        sites = (idx & 1).astype(np.uint8)
        lines = np.where(sites == 0, (idx >> 1) % (1 << 20),
                         rng.integers(0, 1 << 20, n, dtype=np.int64))
        return _finish(name, seed, lines, sites, rng, arrays=sites, gap_mean=gap_mean)
    else:
        raise KeyError(name)
    return _finish(name, seed, lines, sites, rng, gap_mean=gap_mean)


def _phase(name: str, seed: int, n: int, rng: np.random.Generator,
           phase_len: int, gradual: bool, gap_mean: float) -> Trace:
    idx = np.arange(n, dtype=np.int64)
    phase = idx // phase_len
    pos = idx % phase_len
    kind = phase % 4
    starts = rng.integers(0, 1 << 20, int(phase.max(initial=0)) + 1, dtype=np.int64)
    sites = np.zeros(n, dtype=np.uint8)
    lines = np.empty(n, dtype=np.int64)
    random_values = rng.integers(0, 1 << 20, n, dtype=np.int64)

    def values(k: np.ndarray, p: np.ndarray, st: np.ndarray) -> np.ndarray:
        out = np.empty(len(k), dtype=np.int64)
        out[k == 0] = (st[k == 0] + 3 * p[k == 0]) % (1 << 20)
        out[k == 1] = random_values[k == 1]
        out[k == 2] = (st[k == 2] + p[k == 2]) % (1 << 20)
        mask = k == 3
        stride = np.array([1, 3, 7, -2], dtype=np.int64)[p[mask] & 3]
        out[mask] = (st[mask] + (p[mask] >> 2) * stride) % (1 << 20)
        return out

    starts_per = starts[phase]
    effective = kind.copy()
    if gradual:
        blend = 50_000
        in_blend = pos >= phase_len - blend
        probability = np.clip((pos - (phase_len - blend)) / blend, 0, 1)
        take_next = in_blend & (rng.random(n) < probability)
        effective[take_next] = (kind[take_next] + 1) % 4
    lines[:] = values(effective, pos, starts_per)
    multi = effective == 3
    sites[multi] = (pos[multi] & 3).astype(np.uint8)
    boundaries = list(range(phase_len, n, phase_len))
    return _finish(name, seed, lines, sites, rng,
                   meta={"phase_len": phase_len, "transition": "gradual" if gradual else "abrupt",
                         "boundaries": boundaries}, gap_mean=gap_mean)


def _list_walk(name: str, seed: int, n: int, rng: np.random.Generator,
               p_seq: float = 0.5, gap_mean: float = DEFAULT_GAP_MEAN) -> Trace:
    lines = np.empty(n, dtype=np.int64)
    cur = 0
    for start in range(0, n, 65536):
        end = min(n, start + 65536)
        sequential = rng.random(end - start) < p_seq
        random_next = rng.integers(0, 1 << 20, end - start)
        for j in range(start, end):
            lines[j] = cur
            local = j - start
            cur = (cur + 1) % (1 << 20) if sequential[local] else int(random_next[local])
    dep = np.ones(n, dtype=np.uint8)
    if n: dep[0] = 0
    return _finish(name, seed, lines, np.zeros(n, dtype=np.uint8), rng, dep=dep, gap_mean=gap_mean)


def _ptr_chase(name: str, seed: int, n: int, rng: np.random.Generator, gap_mean: float) -> Trace:
    size = 1 << 18
    perm = np.arange(size, dtype=np.int64)
    # Sattolo's algorithm: exactly one cycle.
    for i in range(size - 1, 0, -1):
        j = int(rng.integers(0, i))
        perm[i], perm[j] = perm[j], perm[i]
    take = min(n, size)
    cycle = np.empty(take, dtype=np.int64)
    cur = 0
    for i in range(take):
        cycle[i] = cur
        cur = int(perm[cur])
    lines = np.resize(cycle, n)
    return _finish(name, seed, lines, np.zeros(n, dtype=np.uint8), rng,
                   dep=np.ones(n, dtype=np.uint8), gap_mean=gap_mean)


def _hash(name: str, seed: int, n: int, rng: np.random.Generator, gap_mean: float) -> Trace:
    # Vectorized batches of one primary lookup followed by a geometrically sized probe chain.
    lines: list[np.ndarray] = []
    sites: list[np.ndarray] = []
    deps: list[np.ndarray] = []
    total = 0
    while total < n:
        count = min(n - total, 131072)
        h = rng.integers(0, 1 << 22, count, dtype=np.int64)
        probes = np.minimum(rng.geometric(0.7, count) - 1, 3)
        repeat = probes + 1
        base = np.repeat(h, repeat)
        starts = np.repeat(np.cumsum(repeat) - repeat, repeat)
        k = np.arange(len(base)) - starts
        lines.append((base + k) % (1 << 22))
        sites.append((k > 0).astype(np.uint8))
        deps.append((k > 0).astype(np.uint8))
        total += len(base)
    buckets = np.concatenate(lines)[:n]
    return _finish(name, seed, buckets // 8, np.concatenate(sites)[:n], rng,
                   dep=np.concatenate(deps)[:n], byte_offsets=(buckets % 8) * 8,
                   gap_mean=gap_mean)


def _matmul(name: str, seed: int, n: int, rng: np.random.Generator, gap_mean: float) -> Trace:
    """Vectorized decoding of the exact ii,jj,kk,i,k,j blocked loop nest."""
    idx = np.arange(n, dtype=np.int64)
    group, within = idx // 97, idx % 97
    block, local_ik = (group // 1024) % 4096, group % 1024
    ii = (block // 256) * 32
    jj = ((block // 16) % 16) * 32
    kk = (block % 16) * 32
    i, k = ii + local_ik // 32, kk + local_ik % 32
    inner = np.maximum(0, within - 1)
    j = jj + inner // 3
    sites = np.where(within == 0, 0, 1 + inner % 3).astype(np.uint8)
    elements = np.where(sites == 0, i * 512 + k,
               np.where(sites == 1, k * 512 + j, i * 512 + j))
    arrays = np.where(sites == 0, 0, np.where(sites == 1, 1, 2))
    return _finish(name, seed, elements // 8, sites, rng, arrays=arrays,
                   byte_offsets=(elements % 8) * 8, writes=(sites == 3), gap_mean=gap_mean)


def _spmv(name: str, seed: int, n: int, rng: np.random.Generator, gap_mean: float) -> Trace:
    """Generate exact CSR row event order in vectorized row batches."""
    out_lines: list[np.ndarray] = []; out_sites: list[np.ndarray] = []
    out_arrays: list[np.ndarray] = []; out_offsets: list[np.ndarray] = []
    out_dep: list[np.ndarray] = []; out_writes: list[np.ndarray] = []
    produced = 0; row = 0; edge_base = 0; n_rows = 1 << 19
    while produced < n:
        count = min(8192, n_rows - row)
        if count == 0:
            row = 0; edge_base = 0; count = min(8192, n_rows)
        rows = np.arange(row, row + count, dtype=np.int64)
        nnz = np.maximum(1, rng.poisson(16, count)).astype(np.int64)
        lengths = 2 + 3 * nnz
        total = int(lengths.sum())
        row_ids = np.repeat(rows, lengths)
        starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
        rel = np.arange(total, dtype=np.int64) - starts
        row_nnz = np.repeat(nnz, lengths)
        is_first, is_last = rel == 0, rel == (1 + 3 * row_nnz)
        q = np.maximum(0, rel - 1); middle_site = 1 + q % 3
        sites = np.where(is_first, 0, np.where(is_last, 4, middle_site)).astype(np.uint8)
        edge_prefix = np.repeat(edge_base + np.cumsum(nnz) - nnz, lengths)
        edge = edge_prefix + q // 3
        columns = rng.integers(0, n_rows, int(nnz.sum()), dtype=np.int64)
        col_for_event = columns[np.clip(edge - edge_base, 0, len(columns) - 1)]
        element = np.where(sites == 0, row_ids,
                  np.where(sites == 1, edge,
                  np.where(sites == 2, edge,
                  np.where(sites == 3, col_for_event, row_ids))))
        sizes = np.array([4, 4, 8, 8, 8], dtype=np.int64)[sites]
        out_lines.append((element * sizes) // 64); out_offsets.append((element * sizes) % 64)
        out_sites.append(sites); out_arrays.append(sites.astype(np.uint64))
        out_dep.append((sites == 3).astype(np.uint8)); out_writes.append((sites == 4).astype(np.uint8))
        produced += total; row += count; edge_base += int(nnz.sum())
    return _finish(name, seed, np.concatenate(out_lines)[:n], np.concatenate(out_sites)[:n], rng,
                   arrays=np.concatenate(out_arrays)[:n], byte_offsets=np.concatenate(out_offsets)[:n],
                   dep=np.concatenate(out_dep)[:n], writes=np.concatenate(out_writes)[:n], gap_mean=gap_mean)


def _bfs(name: str, seed: int, n_records: int, rng: np.random.Generator, gap_mean: float) -> Trace:
    """Build the specified random CSR graph and replay a real FIFO BFS."""
    n_vertices = 1 << 18
    degree = np.maximum(1, rng.poisson(8, n_vertices)).astype(np.int64)
    row_ptr = np.empty(n_vertices + 1, dtype=np.int64); row_ptr[0] = 0
    np.cumsum(degree, out=row_ptr[1:])
    neighbours = rng.integers(0, n_vertices, int(row_ptr[-1]), dtype=np.int64)
    queue = np.empty(n_vertices, dtype=np.int64)
    visited = np.zeros(n_vertices, dtype=bool)
    chunks: list[tuple[np.ndarray, ...]] = []
    lines_out: list[np.ndarray] = []; sites_out: list[np.ndarray] = []
    arrays_out: list[np.ndarray] = []; offsets_out: list[np.ndarray] = []
    writes_out: list[np.ndarray] = []; deps_out: list[np.ndarray] = []
    buffered = produced = 0

    def append_elements(elements: np.ndarray, sites: np.ndarray, arrays: np.ndarray,
                        sizes: np.ndarray, writes: np.ndarray, deps: np.ndarray) -> None:
        nonlocal buffered
        byte_addr = elements * sizes
        chunks.append((byte_addr // 64, sites, arrays, byte_addr % 64, writes, deps))
        buffered += len(elements)

    def flush() -> None:
        nonlocal buffered, produced
        if not chunks: return
        cols = [np.concatenate([chunk[i] for chunk in chunks]) for i in range(6)]
        lines_out.append(cols[0]); sites_out.append(cols[1]); arrays_out.append(cols[2])
        offsets_out.append(cols[3]); writes_out.append(cols[4]); deps_out.append(cols[5])
        produced += len(cols[0]); chunks.clear(); buffered = 0

    while produced + buffered < n_records:
        source = int(rng.integers(0, n_vertices))
        visited.fill(False); visited[source] = True
        head = 0; tail = 1; queue[0] = source
        while head < tail and produced + buffered < n_records:
            vertex = int(queue[head])
            # queue[head], row_ptr[v], row_ptr[v+1]
            append_elements(np.array([head, vertex, vertex + 1]),
                            np.array([0, 1, 1], np.uint8), np.array([4, 0, 0], np.uint64),
                            np.array([4, 4, 4]), np.zeros(3, np.uint8),
                            np.array([0, 1, 1], np.uint8))
            begin, end = int(row_ptr[vertex]), int(row_ptr[vertex + 1])
            adjacent = neighbours[begin:end]
            if len(adjacent):
                # Discovery is sequential within an adjacency list: only the first occurrence writes.
                unseen = ~visited[adjacent]
                unique_values, first = np.unique(adjacent, return_index=True)
                first_mask = np.zeros(len(adjacent), dtype=bool); first_mask[first] = True
                discovered_mask = unseen & first_mask
                discovered = adjacent[discovered_mask]
                visited[discovered] = True
                count = len(adjacent); extra = int(discovered_mask.sum()) * 3
                elements = np.empty(2 * count + extra, dtype=np.int64)
                sites = np.empty_like(elements, dtype=np.uint8)
                arrays = np.empty_like(elements, dtype=np.uint64)
                sizes = np.empty_like(elements); writes = np.zeros(len(elements), np.uint8)
                deps = np.zeros(len(elements), np.uint8)
                cursor = 0; new_tail = tail
                # Loop per edge is used only to interleave conditional stores in trace order.
                for edge_offset, neighbour in enumerate(adjacent):
                    elements[cursor:cursor + 2] = (begin + edge_offset, neighbour)
                    sites[cursor:cursor + 2] = (2, 3); arrays[cursor:cursor + 2] = (1, 2)
                    sizes[cursor:cursor + 2] = (4, 1); deps[cursor + 1] = 1; cursor += 2
                    if discovered_mask[edge_offset]:
                        elements[cursor:cursor + 3] = (neighbour, neighbour, new_tail)
                        sites[cursor:cursor + 3] = (4, 5, 6); arrays[cursor:cursor + 3] = (2, 3, 4)
                        sizes[cursor:cursor + 3] = (1, 4, 4); writes[cursor:cursor + 3] = 1
                        queue[new_tail] = neighbour; new_tail += 1; cursor += 3
                tail = new_tail
                append_elements(elements, sites, arrays, sizes, writes, deps)
            head += 1
            if buffered >= 65536: flush()
        if tail == n_vertices and head == tail:
            # The next outer iteration resets visited and chooses a fresh source.
            pass
    flush()
    return _finish(name, seed, np.concatenate(lines_out)[:n_records],
                   np.concatenate(sites_out)[:n_records], rng,
                   arrays=np.concatenate(arrays_out)[:n_records],
                   byte_offsets=np.concatenate(offsets_out)[:n_records],
                   writes=np.concatenate(writes_out)[:n_records],
                   dep=np.concatenate(deps_out)[:n_records],
                   meta={"vertices": n_vertices, "edges": int(row_ptr[-1])}, gap_mean=gap_mean)


def _stencil(name: str, seed: int, n: int, rng: np.random.Generator, gap_mean: float) -> Trace:
    idx = np.arange(n, dtype=np.int64)
    if name == "stencil2d":
        sites = (idx % 6).astype(np.uint8); cell = idx // 6
        i = 1 + (cell // 2046) % 2046; j = 1 + cell % 2046
        di = np.array([-1, 0, 0, 0, 1, 0])[sites]
        dj = np.array([0, -1, 0, 1, 0, 0])[sites]
        lines = ((i + di) * 2048 + j + dj) // 8
        return _finish(name, seed, lines, sites, rng, arrays=(sites == 5), writes=(sites == 5), gap_mean=gap_mean)
    raise KeyError(name)


def generate(name: str, seed: int, n_records: int | None = None, **params: object) -> Trace:
    if name not in WORKLOADS:
        raise ValueError(f"unknown workload {name!r}; choose from {', '.join(WORKLOADS)}")
    n = int(n_records if n_records is not None else WORKLOAD_DEFAULTS.get(
        name, KERNEL_DEFAULT if name in KERNELS else SYNTHETIC_DEFAULT))
    if n < 0:
        raise ValueError("n_records must be non-negative")
    rng = np.random.default_rng(seed)
    gap_mean = float(params.get("gap_mean", DEFAULT_GAP_MEAN))
    if name in {"seq_stream", "stride_const", "multi_stride", "random_uniform", "mix_reg_irr"}:
        return _simple(name, seed, n, rng, gap_mean)
    if name == "phase_mix":
        return _phase(name, seed, n, rng, int(params.get("phase_len", 300_000)), False, gap_mean)
    if name == "phase_mix_fast":
        return _phase(name, seed, n, rng, int(params.get("phase_len", 30_000)), False, gap_mean)
    if name == "phase_mix_gradual":
        return _phase(name, seed, n, rng, int(params.get("phase_len", 100_000)), True, gap_mean)
    if name == "list_walk":
        return _list_walk(name, seed, n, rng, float(params.get("p_seq", 0.5)), gap_mean)
    if name == "ptr_chase":
        return _ptr_chase(name, seed, n, rng, gap_mean)
    if name == "hash_lookup":
        return _hash(name, seed, n, rng, gap_mean)
    if name == "matmul_blocked":
        return _matmul(name, seed, n, rng, gap_mean)
    if name == "spmv_csr":
        return _spmv(name, seed, n, rng, gap_mean)
    if name == "bfs_graph":
        return _bfs(name, seed, n, rng, gap_mean)
    return _stencil(name, seed, n, rng, gap_mean)
