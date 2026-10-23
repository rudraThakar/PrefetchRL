from dataclasses import replace

import numpy as np
import pytest

from pfsim.config import SimConfig
from pfsim.prefetchers import NoPrefetcher, NextLinePrefetcher
from pfsim.sim import Simulator
from pfsim.trace.format import Trace


def trace(lines, gap=3, dep=0):
    n = len(lines)
    return Trace(np.full(n, 0x400000, dtype=np.uint64),
                 (np.array(lines, dtype=np.uint64) << 6) + (1 << 32),
                 np.full(n, gap, dtype=np.uint16), np.zeros(n, dtype=np.uint8),
                 np.full(n, dep, dtype=np.uint8), {"workload": "test", "seed": 0})


def test_repeated_line_reaches_width():
    cfg = replace(SimConfig(), warmup_fraction=0.1)
    result = Simulator(cfg, NoPrefetcher()).run(trace([0] * 5000), 0)
    assert abs(result.ipc - 4.0) / 4.0 < 0.01


def test_determinism_and_demand_stream_independence():
    workload = trace(range(1000), gap=4)
    cfg = replace(SimConfig(), warmup_fraction=0)
    no_sim = Simulator(cfg, NoPrefetcher()); no = no_sim.run(workload, 3)
    pf_sim = Simulator(cfg, NextLinePrefetcher()); pf = pf_sim.run(workload, 3)
    assert no_sim.demand_lines == pf_sim.demand_lines
    assert no.to_dict() == Simulator(cfg, NoPrefetcher()).run(workload, 3).to_dict()
    assert pf.returned_requests == pf.issued + pf.redundant + pf.dropped
    assert pf.issued == pf.useful + pf.useless + pf.unresolved


def test_dependent_miss_latency():
    cfg = replace(SimConfig(), warmup_fraction=0)
    result = Simulator(cfg, NoPrefetcher()).run(trace(range(1000), gap=3, dep=1), 0)
    assert abs(result.cycles / result.records - 254.0) / 254.0 < 0.02


@pytest.mark.parametrize("gap,expected", [(3, 254.0), (1200, 300.3)])
def test_t17_dependent_reference(gap, expected):
    cfg = replace(SimConfig(), warmup_fraction=0.1)
    result = Simulator(cfg, NoPrefetcher()).run(trace(range(5000), gap=gap, dep=1), 0)
    assert result.cycles / result.records == pytest.approx(expected, rel=.02)


@pytest.mark.parametrize("bandwidth,expected,max_util", [
    (3.2, 80.0, None), (12.8, 20.0, None), (25.6, 10.0, None), (51.2, 7.5, .8),
])
def test_t17_bandwidth_and_mshr_reference(bandwidth, expected, max_util):
    cfg = replace(SimConfig(), peak_bw_gbps=bandwidth, warmup_fraction=.1)
    result = Simulator(cfg, NoPrefetcher()).run(trace(range(20_000), gap=0), 0)
    assert result.cycles / result.records == pytest.approx(expected, rel=.02)
    if max_util is None:
        assert result.avg_bw_util > .9
    else:
        assert result.avg_bw_util < max_util


@pytest.mark.parametrize("rob,expected", [(128, 127.0), (256, 84.7), (512, 42.3)])
def test_t17_rob_reference(rob, expected):
    cfg = replace(SimConfig(), rob_size=rob, warmup_fraction=.1)
    result = Simulator(cfg, NoPrefetcher()).run(trace(range(10_000), gap=100), 0)
    assert result.cycles / result.records == pytest.approx(expected, rel=.02)
