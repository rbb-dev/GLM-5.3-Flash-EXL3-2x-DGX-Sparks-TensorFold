"""Patch 9028 (node2's SSD store on /health), on the CPU. Each rank keeps its own store on its own Spark, and /health
showed rank 0's only (2026-10-16 01:11, du: node1 27.29 GiB, node2 20.10 GiB). The memory guard's all-gather (MEMQ,
both ranks, int32) now carries each rank's store too, in MiB, so rank 0's /health lists both under spill.ranks."""

import time
import types

from tensorfold.families.glm5_next.cuda.multi import MultiDecoder

GIB = 1 << 30


class FakeStore:
    def __init__(self, stored=27 * GIB, free=2300 * GIB, fail=False):
        self._stored, self._free, self.fail = stored, free, fail
        self.stats = {"written_bytes": 17 * GIB, "written_small_bytes": 118 * GIB, "written": 1321}
        self.index, self.have = dict.fromkeys(range(184)), dict.fromkeys(range(1261))

    def stored(self):
        if self.fail:
            raise RuntimeError("a table changing under the read")
        return self._stored

    def disk_free(self):
        return self._free


class Decoder:
    _spill_ints = MultiDecoder._spill_ints
    _mem_gather = MultiDecoder._mem_gather
    _spill_health = MultiDecoder._spill_health


def test_a_ranks_store_travels_in_mib_and_fits_the_int32_gather():
    d = Decoder()
    d.disk = FakeStore()
    row = d._spill_ints()
    assert row == [27 * 1024, 135 * 1024, 184, 1261, 1321, 2300 * 1024]
    assert all(0 <= v < 2**31 for v in row)


def test_every_rank_sends_the_same_length_with_no_store_or_a_failing_read():
    d = Decoder()
    d.disk = None
    assert d._spill_ints() == [0] * 6
    d.disk = FakeStore(fail=True)
    assert d._spill_ints() == [0] * 6               # never skip the all-gather: the other rank would wait forever


def test_the_gather_keeps_the_memory_rows_and_adds_each_ranks_store():
    d = Decoder()
    d.disk, d.mem_guard, sent = FakeStore(), None, []

    def gather(values):
        sent.append(list(values))
        return [list(values), [1, 2, 3, 4, 20 * 1024, 100 * 1024, 150, 900, 1000, 2400 * 1024]]

    d.g = types.SimpleNamespace(_gather_ints=gather)
    d._mem_gather()
    assert len(sent[0]) == 10
    assert d.mem_ranks == {0: [0, 0, 0, 0], 1: [1, 2, 3, 4]}
    assert d.spill_ranks[1] == [20 * 1024, 100 * 1024, 150, 900, 1000, 2400 * 1024]


def test_health_lists_both_ranks_stores():
    d = Decoder()
    d.disk = types.SimpleNamespace(info=lambda: {"stored_bytes": 1})
    d.spill_ranks = {0: [27648, 138240, 184, 1261, 1321, 2355200], 1: [20582, 102400, 150, 900, 1000, 2457600]}
    d.spill_ranks_at = time.monotonic() - 2.0
    info = d._spill_health()
    assert info["stored_bytes"] == 1
    assert info["ranks"]["1"] == {"stored_mib": 20582, "written_total_mib": 102400, "entries": 150, "blocks": 900,
                                  "written": 1000, "disk_free_mib": 2457600}
    assert 1.5 <= info["ranks_age_s"] <= 5.0


def test_health_before_the_first_gather_has_no_ranks():
    d = Decoder()
    d.disk = types.SimpleNamespace(info=lambda: {"stored_bytes": 1})
    assert "ranks" not in d._spill_health()
