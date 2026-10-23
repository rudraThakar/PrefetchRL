from pfsim.rl import AgentConfig, RLPrefetcher


def test_pc_bits_changes_feature_table_and_storage():
    small = RLPrefetcher(AgentConfig(pc_bits=4))
    large = RLPrefetcher(AgentConfig(pc_bits=10))
    assert small.configured_rows[2] == 16 * 17
    assert large.configured_rows[2] == 1024 * 17
    assert small.storage_bits() < large.storage_bits()
