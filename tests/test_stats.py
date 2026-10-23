from pfsim.experiments.stats import bootstrap_gmean_ci, geometric_mean, paired_outcomes


def test_bootstrap_is_deterministic_and_bounded():
    values = {"a": [1.0, 1.2, 1.1], "b": [.9, 1.0, 1.1]}
    first = bootstrap_gmean_ci(values, samples=200, seed=4)
    second = bootstrap_gmean_ci(values, samples=200, seed=4)
    assert first == second
    assert first[0] <= geometric_mean([1.1, 1.0]) <= first[1]


def test_paired_outcomes_uses_one_percent_tie():
    agent = {("a", 0): 1.02, ("b", 0): .995, ("c", 0): .8}
    baseline = {key: 1.0 for key in agent}
    assert paired_outcomes(agent, baseline) == {"win": 1, "tie": 1, "loss": 1}
