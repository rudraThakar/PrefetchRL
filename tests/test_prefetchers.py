from pfsim.prefetchers import NextLinePrefetcher, StridePCPrefetcher
from pfsim.sim.events import DemandEvent, SysState


def event(line, pc=0x400000, idx=1):
    return DemandEvent(0, pc, line, line >> 6, line & 63, False, False, False, idx,
                       SysState(0, .5, 0, 0))


def test_next_line_degree():
    reqs = NextLinePrefetcher(2).on_demand(event(10))
    assert [r.line for r in reqs] == [11, 12]


def test_stride_detector_learns():
    pf = StridePCPrefetcher(degree=2, min_conf=2)
    outputs = []
    for i, line in enumerate((10, 15, 20, 25), 1):
        outputs = pf.on_demand(event(line, idx=i))
    assert [r.line for r in outputs] == [30, 35]

