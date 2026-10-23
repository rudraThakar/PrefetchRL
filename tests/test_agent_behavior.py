from dataclasses import replace

from pfsim.config import SimConfig
from pfsim.rl import AgentConfig, RLPrefetcher
from pfsim.rl.actions import decode
from pfsim.sim import Simulator
from pfsim.trace import generate


def fractions(workload):
    agent = RLPrefetcher(AgentConfig(pc_bits=4))
    Simulator(replace(SimConfig(), warmup_fraction=.8), agent).run(generate(workload, 0, 100_000), 0)
    counts = agent.stats()["measured_action_counts"]; total = sum(counts); non_none = total - counts[0]
    learned = sum(counts[i] for i in range(1, len(counts))
                  if decode(i)[0] in {"PC_D1", "PAGE_D1"})
    return counts[0] / total, learned / non_none if non_none else 0


def test_agent_behavior_on_regular_and_random_streams():
    _, learned = fractions("stride_const")
    none, _ = fractions("random_uniform")
    assert learned >= .8
    assert none >= .6
