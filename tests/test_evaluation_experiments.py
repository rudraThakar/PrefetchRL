import csv
from dataclasses import replace

from pfsim.config import SimConfig
from pfsim.experiments.evaluation import run_e5, run_e6, run_e7, run_e8
from pfsim.rl import AgentConfig


def small_config():
    return replace(SimConfig(), warmup_fraction=0, stats_interval=50)


def small_agent():
    return AgentConfig(pc_bits=4, explore=False, admission_control=False, reach_control=False)


def test_e5_pretraining_and_transfer_smoke(tmp_path):
    runs, summary = run_e5(small_config(), tmp_path, records=80,
        train_workloads=("seq_stream",), test_workloads=("stride_const",), passes=1,
        agent_seeds=(0,), agent_config=small_agent(), force=True)
    assert runs.exists() and summary.exists()
    assert (tmp_path / "E5" / "pretrained_weights.npz").exists()
    with runs.open(newline="", encoding="utf-8") as handle:
        assert {r["variant"] for r in csv.DictReader(handle)} == {"cold", "warm_learning", "warm_frozen"}


def test_e6_storage_sweep_smoke(tmp_path):
    _, summary = run_e6(small_config(), tmp_path, records=50, workloads=("seq_stream",),
                         agent_seeds=(0,), agent_config=small_agent(), force=True)
    assert summary.exists() and (tmp_path / "E6" / "fig_storage_speedup.png").exists()


def test_e7_action_analysis_smoke(tmp_path):
    actions, checks = run_e7(small_config(), tmp_path, records=50, workloads=("seq_stream",),
                              agent_seeds=(0,), agent_config=small_agent(), force=True)
    assert actions.exists() and checks.exists()


def test_e8_headroom_smoke(tmp_path):
    runs, summary = run_e8(small_config(), tmp_path, records=50, workloads=("seq_stream",),
                            agent_seeds=(0,), agent_config=small_agent(), force=True)
    assert runs.exists() and summary.exists()
