from dataclasses import asdict, replace

import pytest
import yaml

from pfsim.config import SimConfig, load_agent_config
from pfsim.experiments.tune import (sample_agent_configs, tune_agent, tune_baselines,
                                    validate_tuning_workloads)


def test_tuning_rejects_test_workloads():
    with pytest.raises(ValueError, match="development"):
        validate_tuning_workloads(("seq_stream", "random_uniform"))


def test_random_search_sampling_is_deterministic():
    first = sample_agent_configs(3, 9)
    second = sample_agent_configs(3, 9)
    assert [asdict(value) for value in first] == [asdict(value) for value in second]
    assert all(value.reward.c_w1 >= value.reward.c_w0 for value in first)
    assert all(value.low_accuracy_max_reach in value.reaches for value in first)


def test_tiny_agent_tuning_writes_loadable_config(tmp_path):
    cfg = replace(SimConfig(), warmup_fraction=0, stats_interval=100)
    output = tmp_path / "agent.yaml"
    tune_agent(cfg, workloads=("seq_stream",), records=100, trials=1, refine=False,
               out_dir=tmp_path / "results", config_out=output)
    loaded = load_agent_config(output)
    assert loaded.alpha > 0
    assert (tmp_path / "results" / "agent_runs.csv").exists()


def test_tiny_baseline_grid_writes_config(tmp_path):
    cfg = replace(SimConfig(), warmup_fraction=0)
    output = tmp_path / "baselines.yaml"
    selected = tune_baselines(cfg, workloads=("seq_stream",), records=50,
                              out_dir=tmp_path / "results", config_out=output)
    assert set(selected) == {"next_line", "stride_pc", "bop"}
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == selected


def test_baseline_tuning_resume_reuses_checkpoints(tmp_path):
    cfg = replace(SimConfig(), warmup_fraction=0)
    kwargs = dict(workloads=("seq_stream",), records=20, out_dir=tmp_path / "results",
                  config_out=tmp_path / "baselines.yaml", progress=False)
    first = tune_baselines(cfg, **kwargs)
    second = tune_baselines(cfg, **kwargs, resume=True)
    assert first == second
