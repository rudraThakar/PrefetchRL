from dataclasses import replace

import numpy as np

from pfsim.config import load_agent_config, overlay_agent_config
from pfsim.rl import AgentConfig, QTableSmallPrefetcher, RLPrefetcher
from pfsim.rl.reward import RewardConfig
from pfsim.sim.events import DemandEvent, SysState


def event(line=10, idx=1):
    return DemandEvent(0, 0x400000, line, line >> 6, line & 63, False, False, False, idx,
                       SysState(0, .5, 0, 0))


def test_load_agent_yaml_and_feature_ablation(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("alpha: 0.1\nfeature_drop: [f2, f5]\nreaches: [1]\nreward:\n  c_pol: 0.9\n",
                    encoding="utf-8")
    cfg = load_agent_config(path)
    agent = RLPrefetcher(cfg)
    assert cfg.alpha == .1 and cfg.reward.c_pol == .9
    assert agent.active_features == (0, 2, 3, 5)
    assert len(agent.weights_a) == 4


def test_variant_overlay_preserves_tuned_fields(tmp_path):
    base = AgentConfig(alpha=.0123, pc_bits=6, reward=RewardConfig(c_pol=.91))
    path = tmp_path / "variant.yaml"
    path.write_text("gamma: 0.0\nreward:\n  c_w0: 0.7\n", encoding="utf-8")
    cfg = overlay_agent_config(base, path)
    assert cfg.gamma == 0
    assert cfg.alpha == .0123 and cfg.pc_bits == 6
    assert cfg.reward.c_w0 == .7 and cfg.reward.c_pol == .91


def test_checkpoint_survives_reset(tmp_path):
    cfg = AgentConfig(double_q=True, explore=False)
    agent = RLPrefetcher(cfg)
    agent.weights_a[0][0, 0] = 1.25
    agent.weights_b[0][0, 0] = -.75
    path = tmp_path / "weights.npz"
    agent.save(path)
    restored = RLPrefetcher(cfg); restored.load(path); restored.reset(99)
    assert restored.weights_a[0][0, 0] == 1.25
    assert restored.weights_b[0][0, 0] == -.75


def test_qtable_small_state_action_and_determinism():
    first = QTableSmallPrefetcher(epsilon=0); first.reset(4)
    second = QTableSmallPrefetcher(epsilon=0); second.reset(4)
    sequence_a, sequence_b = [], []
    for idx, line in enumerate((10, 15, 20, 25, 30), 1):
        sequence_a.extend(req.line for req in first.on_demand(event(line, idx)))
        sequence_b.extend(req.line for req in second.on_demand(event(line, idx)))
    assert sequence_a == sequence_b
    assert first.q.shape == (544, 11)
    assert first.storage_bits() > 0
