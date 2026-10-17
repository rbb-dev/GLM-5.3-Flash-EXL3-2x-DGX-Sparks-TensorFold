"""Patch 9023: the spill tier's bookkeeping off the engine loop. A kept state used to make its own room on the engine's
loop as it was kept: its conversation's quota (old turns and forks over its share, ``_trim_own``) and the cap's or the
free-disk floor's shortfall (``_make_room``), each walking every stored state while every stream waited. Now the keep
only checks counters; the writer makes that room before the state's first byte is written, in the same order. A
conversation's bytes for its quota are counted from its own points against the blocks' reference counts, exactly what
walking every block name in the store gave. CPU only (the store's harness: spill_store_kit)."""

import random
import threading

from spill_store_kit import S, ar, ids_of, keep, store  # noqa: F401  (ar: a fixture)



def state_bytes(ar, tmp_path):
    """What one 3,000-token state of the test arena takes on disk (two blocks, its files), measured."""

    st = store(ar, tmp_path / "measure")
    keep(st, ids_of(3000, seed=99), lineage="m")
    assert st.drain(30)
    return st.stored()


def spy(monkeypatch, name):
    """The threads that ran ``BlockStore.<name>``."""

    seen = []
    real = getattr(S.BlockStore, name)

    def wrapped(self, *a, **kw):
        seen.append(threading.current_thread().name)
        return real(self, *a, **kw)
    monkeypatch.setattr(S.BlockStore, name, wrapped)
    return seen


def test_room_for_a_kept_state_is_made_by_the_writer_never_on_the_engine_loop(ar, tmp_path, monkeypatch):
    seen = spy(monkeypatch, "_make_room")
    st = store(ar, tmp_path / "a", gib=2.5 * state_bytes(ar, tmp_path) / 2**30)      # room for two states
    keys = []
    for i in range(4):                                                 # four conversations, each its own rows
        _, k = keep(st, ids_of(3000, seed=i), lineage=f"c{i}", base=i * 4096)
        keys.append(k)
        assert st.drain(30)
    assert seen and all(t.startswith("spill-writer") for t in seen)
    assert keys[0] not in st.index and keys[-1] in st.index           # the least recently used conversation went
    assert st.stored() <= st.cap


def test_a_conversation_over_its_quota_trims_its_own_turns_on_the_writer(ar, tmp_path, monkeypatch):
    seen = spy(monkeypatch, "_trim_own")
    st = store(ar, tmp_path / "a", gib=6 * state_bytes(ar, tmp_path) / 2**30, quota=0.5, turns=0)
    ids = ids_of(12000)
    for n in (2048, 4096, 6144, 8192, 10240):                          # one conversation, five turns
        keep(st, ids[:n], lineage="c")
        assert st.drain(30)
    assert seen and all(t.startswith("spill-writer") for t in seen)
    assert st.stats["quota_trims"] > 0
    assert S.point_key(__import__("numpy").asarray(ids[:10240], dtype="int64")) in st.index     # its newest stays


def test_a_state_that_room_cannot_be_made_for_is_not_written(ar, tmp_path):
    st = store(ar, tmp_path / "a", gib=1.5 * state_bytes(ar, tmp_path) / 2**30)
    _, first = keep(st, ids_of(3000, seed=1), lineage="a", base=0)
    assert st.drain(30)
    st.reading.add(first)                                              # being read: never evicted meanwhile
    _, second = keep(st, ids_of(3000, seed=2), lineage="b", base=4096)
    assert st.drain(30)
    assert first in st.index and second not in st.index and st.stats["no_space"] == 1
    assert st.stats["write_failed"] == 0                               # a refusal, not a failure


def reference_bytes(st, keys):
    """A conversation's bytes as patch 0110 counted them: every block name of every other stored point walked."""

    mine = set(keys)
    others = {b for k, pt in st.index.items() if k not in mine for b in pt.blocks}
    blocks = {b for k in mine for b in st.index[k].blocks if b not in others and b in st.have}
    inos = {st.have[b][1] for b in blocks}
    own = sum(st.inodes[i][0] for i in inos if i in st.inodes and st.inodes[i][1] <= blocks)
    return own + sum(st.index[k].small_bytes for k in mine)


def test_a_conversations_bytes_count_exactly_as_before(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    ids = ids_of(12000)
    for n in (2048, 6144, 10240):                                      # a conversation of three turns
        keep(st, ids[:n], lineage="c")
    fork = S.fork("c", "f", 6144)                                      # a fork of it from its second turn
    keep(st, ids[:6144] + ids_of(4000, seed=7), lineage=fork)
    keep(st, ids[:2048], lineage="d")                                  # the same rows under another lineage: linked
    keep(st, ids_of(5000, seed=3), lineage="e", base=8192)            # a conversation of its own
    assert st.drain(30)
    keep(st, ids_of(3000, seed=4), lineage="w", base=4096)            # one still waiting while counted
    rng = random.Random(0)
    with st.lock:
        keys = list(st.index)
        groups = [keys] + [rng.sample(keys, rng.randint(1, len(keys))) for _ in range(40)]
        groups += [[k] for k in keys]
        for g in groups:
            assert st._conv_bytes(g) == reference_bytes(st, g)
    assert st.drain(30)
