from dataclasses import replace

import pytest

from pfsim.config import SimConfig
from pfsim.prefetchers import NextLinePrefetcher, StridePCPrefetcher
from pfsim.rl import AgentConfig, RLPrefetcher
from pfsim.sim import Simulator
from pfsim.trace import WORKLOADS, generate


@pytest.mark.parametrize("workload", WORKLOADS[:13])
@pytest.mark.parametrize("kind", ("next", "stride", "agent"))
def test_primary_outcome_accounting_on_every_workload(workload, kind):
    cfg = replace(SimConfig(), warmup_fraction=0)
    pf = (NextLinePrefetcher() if kind == "next" else StridePCPrefetcher()
          if kind == "stride" else RLPrefetcher(AgentConfig(pc_bits=4, explore=False)))
    result = Simulator(cfg, pf).run(generate(workload, 0, 300), 0)
    assert result.returned_requests == result.issued + result.redundant + result.dropped
    assert result.issued == result.useful + result.useless + result.unresolved
    assert result.pollution <= result.issued
