"""Patch 9024: the memory guard. On 2026-10-12 node2 fell to 471 MB available while serving: torch's cache of freed
GPU blocks had grown 2.2 GiB in two hours (Python's heap 0.34 GiB in 28 hours), and nothing watched it. Each Spark now
reads its own MemAvailable about once a second on its engine loop:
- normal from 2.0 GiB (and back to normal only above it);
- tight below 1.5 GiB: torch's unused cache released and a full Python collection, at most every 30 s;
- critical below 1.0 GiB: also every saved GPU state dropped on both Sparks (they are on the SSD and come back on their
  next turn), at most every 5 s. Admission never stops.
Node2's figures reach node1 once a second while streams are live (MEMQ: both ranks all-gather four numbers), so /health
shows both Sparks, and a critical node2 makes rank 0 drop the states. TF_GLM_MEMORY_GUARD=0 turns it off. CPU only."""

import pytest

from tensorfold.families.glm5_next.cuda import memory_guard as G

GIB = 1 << 30


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def guard(values, clock=None, **kw):
    it = iter(values)
    return G.MemoryGuard(read=lambda: int(next(it) * GIB), clock=clock or Clock(), **kw)


def test_levels_follow_the_available_memory_with_a_way_back_only_above_two_gib():
    g = guard([3.0, 1.9, 1.4, 1.2, 0.9, 1.3, 1.8, 1.95, 2.1, 1.7])
    seen = [G.LEVELS[g.sample()] for _ in range(10)]
    assert seen == ["normal", "normal", "tight", "tight", "critical", "tight", "tight", "tight", "normal", "normal"]


def test_actions_are_due_by_level_and_rate_limited():
    clock = Clock()
    g = guard([1.4] * 3 + [0.8] * 4, clock)
    g.sample()
    assert g.due() == "tight"
    g.done("tight", {"cache_mib": 300.0})
    clock.t += 10
    g.sample()
    assert g.due() is None                            # tight: at most every 30 s
    clock.t += 25
    g.sample()
    assert g.due() == "tight"
    g.done("tight", {})
    clock.t += 1
    g.sample()                                        # 0.8 GiB: critical, its own limit (every 5 s)
    assert g.due() == "critical"
    g.done("critical", {"states": 3})
    clock.t += 2
    g.sample()
    assert g.due() is None
    clock.t += 4
    g.sample()
    assert g.due() == "critical"


def test_the_settings_turn_it_off_or_set_its_three_levels(monkeypatch):
    monkeypatch.setenv("TF_GLM_MEMORY_GUARD", "0")
    assert G.settings() is None
    monkeypatch.setenv("TF_GLM_MEMORY_GUARD", "3,2,1.5")
    assert G.settings() == (3 * GIB, 2 * GIB, int(1.5 * GIB))
    monkeypatch.delenv("TF_GLM_MEMORY_GUARD")
    assert G.settings() == (2 * GIB, int(1.5 * GIB), 1 * GIB)
    for bad in ("1,2,3", "x", "2,1.5"):
        monkeypatch.setenv("TF_GLM_MEMORY_GUARD", bad)
        with pytest.raises(ValueError):
            G.settings()


def test_reading_meminfo(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       125000000 kB\nMemFree:          100000 kB\nMemAvailable:    1048576 kB\n")
    assert G.mem_available(str(p)) == 1048576 * 1024
    p.write_text("nothing here\n")
    assert G.mem_available(str(p)) is None


def test_a_failed_read_changes_nothing_and_never_raises():
    def broken():
        raise OSError("no /proc")
    g = G.MemoryGuard(read=broken, clock=Clock())
    assert g.sample() == G.NORMAL and g.avail is None and g.due() is None


def test_health_reports_the_level_the_memory_and_the_actions():
    clock = Clock()
    g = guard([1.4], clock)
    g.sample()
    g.done("tight", {"cache_mib": 812.0, "collected": 1234, "collect_s": 0.41})
    h = g.health(reserved=5 * GIB, allocated=4 * GIB)
    assert h["level"] == "tight" and h["available_mib"] == int(1.4 * 1024)
    assert h["torch_reserved_mib"] == 5120 and h["torch_allocated_mib"] == 4096
    assert h["actions"]["tight"] == 1 and h["freed"]["cache_mib"] == 812.0 and h["last"]["collected"] == 1234


# -- the decoder's hooks (multi.py) -------------------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from tensorfold.families.glm5_next.cuda import multi as M  # noqa: E402


def decoder(rank, avail, others=(), lanes=True):
    """A MultiDecoder without __init__: its guard reading ``avail`` GiB, the other ranks' status as MEMQ gathers it."""

    d = object.__new__(M.MultiDecoder)
    d.rank, d.lanes, d.outbox, d.kept = rank, ({1: object()} if lanes else {}), [], ["s1", "s2", "s3"]
    d.w = SimpleNamespace(world=2)
    d.flushed, d.evicted, d.released = 0, [], 0
    d._flush = lambda: setattr(d, "flushed", d.flushed + 1)
    d._evict = lambda c, spill=True: d.evicted.append(c)
    clock = Clock()
    d.mem_guard = G.MemoryGuard(read=lambda: int(avail * GIB), clock=clock)
    d.mem_ranks, d.mem_ranks_at = {}, None

    def gather(mine):
        return [mine] + [list(o) for o in others] if rank == 0 else [list(others[0]), mine]
    d.g = SimpleNamespace(_gather_ints=gather)
    d._mem_release = lambda: (setattr(d, "released", d.released + 1), {"cache_mib": 1.0})[1]
    return d, clock


def test_rank_zero_gathers_every_rank_once_a_second_while_streams_are_live():
    d, clock = decoder(0, 3.0, others=[[0, 3000, 10, 8]])
    d._mem_tick()
    assert M.MultiDecoder.parse(d.outbox[0]) == [(M.MEMQ, [])] and d.flushed == 1
    assert d.mem_ranks[1] == [0, 3000, 10, 8] and d.mem_ranks[0][1] == 3072
    d._mem_tick()                                      # within the second: nothing
    assert len(d.outbox) == 1
    clock.t += 1.0
    d._mem_tick()
    assert len(d.outbox) == 2
    idle, _ = decoder(0, 3.0, others=[[0, 3000, 10, 8]], lanes=False)
    idle._mem_tick()                                   # rank 1 is not listening: no MEMQ
    assert idle.outbox == [] and idle.mem_ranks == {}


def test_a_critical_node2_makes_rank_zero_drop_every_saved_state_at_most_every_five_seconds():
    d, clock = decoder(0, 3.0, others=[[G.CRITICAL, 700, 9000, 8000]])
    d._mem_tick()
    assert d.evicted == ["s1", "s2", "s3"] and d.released == 0          # rank 0 itself is fine: no release here
    clock.t += 2.0
    d._mem_tick()
    assert len(d.evicted) == 3                                          # not again within 5 s
    clock.t += 4.0
    d._mem_tick()
    assert len(d.evicted) == 6


def test_a_tight_spark_releases_its_own_cache_and_collects():
    d, clock = decoder(1, 1.3, others=[[0, 5000, 10, 8]])
    d._mem_tick()                                       # rank 1: its own action, never a MEMQ or an eviction
    assert d.released == 1 and d.evicted == [] and d.outbox == []
    assert d.mem_guard.actions["tight"] == 1


def test_health_shows_both_sparks():
    d, _ = decoder(0, 3.0, others=[[G.TIGHT, 1400, 9000, 8000]])
    d._mem_tick()
    h = d._mem_health()
    assert h["level"] == "normal" and h["ranks"]["1"] == {"level": "tight", "available_mib": 1400,
                                                          "torch_reserved_mib": 9000, "torch_allocated_mib": 8000}
    assert h["ranks_age_s"] is not None


def test_a_guard_that_raises_never_stops_the_engine():
    d, _ = decoder(0, 3.0, others=[[0, 1, 2, 3]])

    def broken(mine):
        raise RuntimeError("collective")
    d.g = SimpleNamespace(_gather_ints=broken)
    d._mem_tick()                                       # logged, nothing raised


def test_a_rank_that_never_sampled_reads_its_memory_before_it_reports_it():
    """2026-10-14: node2's first status read 0 MiB, gathered before its own first sample."""
    d, _ = decoder(1, 7.5, others=[[0, 6000, 10, 8]])
    sent = []
    d.g = SimpleNamespace(_gather_ints=lambda mine: (sent.append(mine), [[0, 6000, 10, 8], mine])[1])
    assert d.mem_guard.avail is None
    d._mem_gather()
    assert sent[0][1] == int(7.5 * 1024)
