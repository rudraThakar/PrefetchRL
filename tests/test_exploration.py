import numpy as np
import pytest

from pfsim.rl import AgentConfig, RLPrefetcher


@pytest.mark.parametrize("mode", ["eps_decay", "eps_fixed", "softmax"])
def test_exploration_modes_respect_mask(mode):
    agent = RLPrefetcher(AgentConfig(explore_mode=mode))
    state = (0, 0, 0, 0, 0, 0)
    valid = np.zeros(36, dtype=bool); valid[[0, 7, 19]] = True
    assert {agent._select(state, valid) for _ in range(100)} <= {0, 7, 19}
