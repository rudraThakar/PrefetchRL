import numpy as np

from pfsim.rl.actions import valid_mask
from pfsim.rl.agent import AgentConfig, RLPrefetcher, _Decision
from pfsim.rl.features import History
from pfsim.rl.reward import RewardConfig, reward_for
from pfsim.sim.events import DemandEvent, OutcomeEvent, OutcomeKind, SysState


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


def test_low_accuracy_admission_suppresses_but_periodically_probes():
    cfg = AgentConfig(explore=False, admission_control=True, admission_min_feedback=1,
                      admission_min_accuracy=.15, admission_probe_interval=3)
    agent = RLPrefetcher(cfg)
    agent.feedback_count = 1
    for table in agent.weights_a + agent.weights_b:
        table[:, 1] = 10
    ev = DemandEvent(0, 4, 10, 0, 10, False, False, False, 1,
                     SysState(.5, .05, 0, 0))
    assert [len(agent.on_demand(ev)) for _ in range(5)] == [1, 0, 0, 1, 0]
    assert agent.stats()["suppressed_count"] == 5


def test_admission_does_not_override_adequate_accuracy():
    cfg = AgentConfig(explore=False, admission_min_feedback=1)
    agent = RLPrefetcher(cfg)
    agent.feedback_count = 1
    for table in agent.weights_a + agent.weights_b:
        table[:, 1] = 10
    ev = DemandEvent(0, 4, 10, 0, 10, False, False, False, 1,
                     SysState(.5, .8, 0, 0))
    assert len(agent.on_demand(ev)) == 1


def test_reach_control_masks_long_reaches_only_after_feedback():
    cfg = AgentConfig(explore=False, admission_control=False, reach_control=True, reach_min_feedback=1,
                      reach_min_accuracy=.5, low_accuracy_max_reach=2)
    agent = RLPrefetcher(cfg); agent.feedback_count = 1
    for table in agent.weights_a + agent.weights_b:
        table[:, 2] = 5   # P1, reach 2
        table[:, 3] = 10  # P1, reach 4
    low = DemandEvent(0, 4, 10, 0, 10, False, False, False, 1,
                      SysState(.2, .3, 0, 0))
    high = DemandEvent(1, 4, 10, 0, 10, False, False, False, 2,
                       SysState(.2, .8, 0, 0))
    assert agent.on_demand(low)[0].line == 12
    agent.useful_feedback = 1
    assert agent.on_demand(high)[0].line == 14
    assert agent.stats()["reach_limited_count"] == 1


def test_bandwidth_gate_only_blocks_congested_low_accuracy_state():
    cfg = AgentConfig(explore=False, admission_control=False, reach_control=False,
                      bw_gate=True, theta_bw=.75, theta_acc=.5)
    agent = RLPrefetcher(cfg)
    for table in agent.weights_a + agent.weights_b: table[:, 1] = 10
    congested = DemandEvent(0, 4, 10, 0, 10, False, False, False, 1,
                            SysState(.95, .1, 0, 0))
    free = DemandEvent(1, 4, 10, 0, 10, False, False, False, 2,
                       SysState(.2, .1, 0, 0))
    assert agent.on_demand(congested) == []
    assert len(agent.on_demand(free)) == 1
