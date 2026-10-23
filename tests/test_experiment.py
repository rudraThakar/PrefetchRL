import csv
from dataclasses import replace

from pfsim.config import SimConfig
from pfsim.experiments import run_e1
from pfsim.rl import AgentConfig, RLPrefetcher


def test_e1_runner_writes_required_outputs(tmp_path):
    cfg = replace(SimConfig(), warmup_fraction=0, stats_interval=100)
    runs, summary = run_e1(cfg, tmp_path, records=300, workloads=("seq_stream",),
                           prefetchers=("none", "next_line"))
    assert runs.exists() and summary.exists()
    with runs.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert set((tmp_path / "E1" / "intervals").glob("*.csv"))
    assert {"speedup", "coverage", "overprediction", "config_hash"} <= set(rows[0])


def test_e1_uses_supplied_agent_configuration(tmp_path):
    cfg = replace(SimConfig(), warmup_fraction=0)
    agent_cfg = AgentConfig(pc_bits=4, explore=False)
    runs, _ = run_e1(cfg, tmp_path, records=50, workloads=("seq_stream",),
                     prefetchers=("none", "agent"), agent_config=agent_cfg)
    with runs.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    agent_rows = [row for row in rows if row["prefetcher"] == "agent"]
    expected_bits = RLPrefetcher(agent_cfg).storage_bits()
    assert {int(row["storage_bits"]) for row in agent_rows} == {expected_bits}
