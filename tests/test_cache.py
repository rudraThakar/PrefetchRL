from pfsim.sim.cache import Entry, SetAssociativeCache


def test_lru_and_set_mapping():
    cache = SetAssociativeCache(1, 2, 1, line_bytes=128)  # four sets
    assert cache.install(0, Entry(0, 0)) is None
    assert cache.install(4, Entry(4, 0)) is None
    cache.touch(0)
    victim = cache.install(8, Entry(8, 0))
    assert victim.line == 4
    assert cache.contains(0) and cache.contains(8) and not cache.contains(4)

