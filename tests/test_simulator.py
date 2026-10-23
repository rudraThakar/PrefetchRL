from dataclasses import replace

import numpy as np
import pytest

from pfsim.config import SimConfig
from pfsim.config import CacheConfig
from pfsim.sim.cache import Entry
from pfsim.prefetchers import NoPrefetcher, NextLinePrefetcher
from pfsim.sim.metrics import comparative_metrics
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
    no_sim = Simulator(cfg, NoPrefetcher(), record_demand_lines=True); no = no_sim.run(workload, 3)
    pf_sim = Simulator(cfg, NextLinePrefetcher(), record_demand_lines=True); pf = pf_sim.run(workload, 3)
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


def test_sequential_prefetch_feedback_is_not_starved_by_admission():
    workload = trace(range(10_000), gap=4)
    cfg = replace(SimConfig(), warmup_fraction=.1)
    baseline = Simulator(cfg, NoPrefetcher()).run(workload)
    prefetched = Simulator(cfg, NextLinePrefetcher(1)).run(workload)
    metrics = comparative_metrics(prefetched, baseline)
    assert prefetched.accuracy > .95
    assert metrics["coverage"] > .95
    # Both runs transfer one line per access and are channel-bound, so speedup is near one.
    assert metrics["speedup"] == pytest.approx(1.0, rel=.02)


def test_original_four_transfer_backlog_starves_sequential_feedback():
    workload = trace(range(3000), gap=4)
    cfg = replace(SimConfig(), warmup_fraction=.1, pf_backlog_factor=4)
    result = Simulator(cfg, NextLinePrefetcher(1)).run(workload)
    assert result.accuracy == 0
    assert result.dropped_rate > .99


def test_used_prefetch_victim_is_eligible_for_pollution_blame():
    cfg = replace(SimConfig(), l2=CacheConfig(1, 2, 1), warmup_fraction=0)
    sim = Simulator(cfg, NoPrefetcher()); sim._reset(0)
    sim._install_l2(Entry(0, 0), False)
    used = Entry(8, 0, pf=True, pf_id=7, used=True, resolved=True)
    sim._install_l2(used, False); sim.l2.touch(0)
    sim._install_l2(Entry(16, 0, pf=True, pf_id=8), True, 8)
    assert sim.victims[8] == (8, 0)


def test_prefetch_bookkeeping_is_bounded():
    cfg = replace(SimConfig(), warmup_fraction=0)
    sim = Simulator(cfg, NextLinePrefetcher())
    sim.run(trace(range(30_000), gap=64), 0)
    assert len(sim.book) <= cfg.pollution_window + cfg.pf_mshr + 2


def test_counterfactual_pollution_and_benign_cases():
    cfg = replace(SimConfig(), warmup_fraction=0)
    sim = Simulator(cfg, NoPrefetcher()); sim._reset(0); sim.measure=True; sim.l2_idx=10
    sim.book[1]={"tag":None,"line":9,"u_issue":0.0,"issue_idx":1,"measured":True,"resolved":True}
    sim.victims[5]=(1,9); sim.shadow.install(5,Entry(5,0)); sim._check_pollution(5)
    assert sim.counts["pollution"] == 1
    sim.victims[6]=(1,9); sim._check_pollution(6)
    assert sim.counts["pollution_benign"] == 1 and sim.counts["pollution"] == 1
    loose = replace(cfg, pollution=replace(cfg.pollution,counterfactual=False))
    sim2=Simulator(loose,NoPrefetcher()); sim2._reset(0); sim2.measure=True; sim2.l2_idx=10
    sim2.book[1]=dict(sim.book[1]); sim2.victims[6]=(1,9); sim2._check_pollution(6)
    assert sim2.counts["pollution"] == 1
