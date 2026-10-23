import numpy as np
import os
import pytest

from pfsim.trace import WORKLOADS, generate, load_trace, save_trace
from pfsim.config import SimConfig
from pfsim.prefetchers import NoPrefetcher
from pfsim.sim import Simulator


@pytest.mark.parametrize("name", WORKLOADS)
def test_generators_are_valid_and_deterministic(name, tmp_path):
    first = generate(name, 7, 256)
    second = generate(name, 7, 256)
    assert np.array_equal(first.addr, second.addr)
    assert np.array_equal(first.gap, second.gap)
    path = tmp_path / "trace.npz"
    save_trace(first, path)
    loaded = load_trace(path)
    assert loaded.meta == first.meta
    assert np.array_equal(loaded.pc, first.pc)


def test_phase_metadata():
    trace = generate("phase_mix_fast", 0, 100_000)
    assert trace.meta["phase_len"] == 30_000
    assert trace.meta["boundaries"] == [30_000, 60_000, 90_000]


@pytest.mark.parametrize("name,phase_len", (("phase_mix",300_000),("phase_mix_fast",30_000),
                                             ("phase_mix_gradual",100_000)))
@pytest.mark.parametrize("seed", (0,1))
def test_phase_variant_boundaries_for_two_seeds(name, phase_len, seed):
    trace=generate(name,seed,phase_len*3)
    assert trace.meta["phase_len"]==phase_len
    assert trace.meta["boundaries"]==[phase_len,2*phase_len]


def test_exact_matmul_prefix():
    trace = generate("matmul_blocked", 0, 100)
    # A[0][0], then B[0][j], C[0][j] load/store for j=0..31, then A[0][1].
    assert trace.pc[0] == 0x400000
    assert trace.pc[1:4].tolist() == [0x400040, 0x400080, 0x4000C0]
    assert trace.is_write[3] == 1
    assert trace.pc[97] == 0x400000
    assert trace.addr[97] == trace.addr[0] + 8


def test_spmv_event_shape():
    trace = generate("spmv_csr", 3, 1000)
    assert trace.pc[0] == 0x400000
    assert set(trace.dep.tolist()) <= {0, 1}
    assert np.all(trace.dep == (trace.pc == 0x4000C0))


def test_hash_buckets_share_cache_lines():
    trace = generate("hash_lookup", 4, 2000)
    lines = trace.addr >> 6
    probe = trace.pc == 0x400040
    assert np.any(probe)
    # Linear probes usually remain in the primary bucket's 64-byte line.
    assert np.mean(lines[1:][probe[1:]] == lines[:-1][probe[1:]]) > .7


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("PFSIM_RUN_SLOW") != "1",
                    reason="set PFSIM_RUN_SLOW=1 for default-length T14")
@pytest.mark.parametrize("name", WORKLOADS[:13])
def test_default_traces_have_enough_l2_accesses(name):
    trace = generate(name, 0)
    result = Simulator(SimConfig(), NoPrefetcher()).run(trace, 0)
    assert result.l2_accesses >= 400_000
