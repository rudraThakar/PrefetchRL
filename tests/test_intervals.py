from dataclasses import replace

from pfsim.config import SimConfig
from pfsim.prefetchers import NoPrefetcher
from pfsim.sim import Simulator
from pfsim.trace import generate


def test_interval_records_include_partial_tail():
    cfg = replace(SimConfig(), warmup_fraction=0, stats_interval=100)
    sim = Simulator(cfg, NoPrefetcher())
    sim.run(generate("stride_const", 0, 250))
    assert [row["l2_accesses"] for row in sim.intervals] == [100, 100, 50]
    assert all(set(row) == {"l2_idx", "l2_accesses", "window_ipc", "window_accuracy",
                            "window_coverage_proxy", "mean_reward", "epsilon"}
               for row in sim.intervals)
