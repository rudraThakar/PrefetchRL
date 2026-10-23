from dataclasses import replace

from pfsim.prefetchers import FutureLookaheadPrefetcher, NoPrefetcher
from pfsim.sim import Simulator
from pfsim.sim.metrics import comparative_metrics
from pfsim.config import SimConfig
from pfsim.trace import generate


def test_future_lookahead_and_comparative_metrics():
    trace = generate("seq_stream", 0, 5000)
    cfg = replace(SimConfig(), warmup_fraction=0)
    base_sim = Simulator(cfg, NoPrefetcher(), record_demand_lines=True)
    baseline = base_sim.run(trace)
    future = Simulator(cfg, FutureLookaheadPrefetcher(base_sim.demand_lines, 16)).run(trace)
    metrics = comparative_metrics(future, baseline)
    # A fixed long lookahead can be late or bandwidth-dropped; it must not invent a loss here.
    assert metrics["speedup"] >= .99
    assert future.accuracy > .95
    assert metrics["coverage"] > 0


def test_future_lookahead_controlled_stride_trace():
    trace = generate("stride_const", 0, 5000)
    cfg = replace(SimConfig(), warmup_fraction=0)
    base_sim = Simulator(cfg, NoPrefetcher(), record_demand_lines=True); baseline=base_sim.run(trace)
    future=Simulator(cfg,FutureLookaheadPrefetcher(base_sim.demand_lines,16)).run(trace)
    assert future.accuracy >= .95
    assert future.ipc / baseline.ipc > 1
