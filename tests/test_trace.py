import numpy as np
import pytest

from pfsim.trace import WORKLOADS, generate, load_trace, save_trace


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
