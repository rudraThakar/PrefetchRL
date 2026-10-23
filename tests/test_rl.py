import numpy as np

from pfsim.rl.actions import valid_mask
from pfsim.rl.agent import AgentConfig, RLPrefetcher, _Decision
from pfsim.rl.features import History
from pfsim.rl.reward import RewardConfig, reward_for
from pfsim.sim.events import OutcomeEvent, OutcomeKind


def test_reward_values():
    cfg = RewardConfig()
    assert reward_for(OutcomeEvent(OutcomeKind.USEFUL, 0), cfg, 254) == 1.0
    assert reward_for(OutcomeEvent(OutcomeKind.USEFUL, 0, wait_cycles=254), cfg, 254) == .2
    assert reward_for(OutcomeEvent(OutcomeKind.USELESS, 0, u_issue=0), cfg, 254) == -.3
    assert reward_for(OutcomeEvent(OutcomeKind.USELESS, 0, u_issue=1), cfg, 254) == -1.0
    assert reward_for(OutcomeEvent(OutcomeKind.REDUNDANT, 0), cfg, 254) == -.05
    assert reward_for(OutcomeEvent(OutcomeKind.DROPPED, 0), cfg, 254) == -.1


def test_action_masking():
    fresh = History(0, 0, 0, 0, 0)
    mask = valid_mask(63, fresh)
    assert mask[0]
    # All positive fixed patterns leave the page at offset 63; history patterns are zero.
    assert np.flatnonzero(mask).tolist() == [0, 16, 17, 18, 19, 20]
    for action in np.flatnonzero(mask):
        assert action == 0 or 16 <= action <= 20  # negative M1 reaches remain in-page


def test_gamma_zero_update_changes_only_selected_action():
    cfg = AgentConfig(gamma=0, double_q=False, alpha=.05, q0=.2, learn=True)
    agent = RLPrefetcher(cfg)
    state = (0, 0, 0, 0, 0, 0)
    valid = np.ones(36, dtype=bool)
    decision = _Decision(0, state, 3, valid, reward=1.0)
    before = agent._q(agent.weights_a, state).copy()
    agent._update(decision)
    after = agent._q(agent.weights_a, state)
    assert np.isclose(after[3], cfg.q0 + cfg.alpha * (1 - cfg.q0))
    assert np.allclose(np.delete(after, 3), np.delete(before, 3))
