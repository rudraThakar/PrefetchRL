from dataclasses import replace

from pfsim.config import SimConfig
from pfsim.experiments.extensions import run_e10, run_e11, run_e12, run_s1
from pfsim.rl import AgentConfig


CFG = replace(SimConfig(), warmup_fraction=0, stats_interval=25)
AGENT = AgentConfig(pc_bits=4, explore=False, admission_control=False, reach_control=False)


def test_e10_phase_modes_smoke(tmp_path):
    runs, recovery = run_e10(CFG, tmp_path, records=100, workloads=("phase_mix_fast",),
        agent_seeds=(0,), agent_config=AGENT, force=True)
    assert runs.exists() and recovery.exists()


def test_e11_controller_comparison_smoke(tmp_path):
    runs, summary = run_e11(CFG, tmp_path, records=60, workloads=("seq_stream",),
        bandwidths=(3.2,), agent_seeds=(0,), agent_config=AGENT, force=True)
    assert runs.exists() and summary.exists()


def test_s1_sensitivity_smoke(tmp_path):
    runs, summary = run_s1(CFG, tmp_path, records=40, workloads=("seq_stream",),
        agent_seeds=(0,), agent_config=AGENT, force=True)
    assert runs.exists() and summary.exists()


def test_e12_requires_and_combines_prior_outputs(tmp_path):
    for name, filename, content in (
        ("E3", "runs.csv", "variant,mean_reward,speedup,git_commit\nfull,0.2,1.1,abc123\nfull,0.1,1.0,abc123\n"),
        ("E6", "summary.csv", "variant,speedup_gmean,storage_bits\nfull,1.1,10\n"),
        ("E10", "recovery.csv", "workload,recovery_accesses\nphase_mix,20\n"),
        ("E6", "runs.csv", "git_commit\nabc123\n"),
        ("E10", "runs.csv", "git_commit\nabc123\n")):
        path = tmp_path / name / filename; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    summary, correlation = run_e12(tmp_path, force=True)
    assert summary.exists() and correlation.exists()
