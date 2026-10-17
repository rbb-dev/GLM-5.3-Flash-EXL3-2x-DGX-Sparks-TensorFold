"""Patch 9030: the spill tier's refusals on the writer (9023 moved room-making there, after the keep). A state the
writer cannot make room for is not written, nor is any waiting state that names one of its unwritten blocks (queued
after it, so none of their bytes is written yet): each is counted as no_space, never as a failed write, and their
reserved bytes are free at once. A refused state still pending gets no rows copied when its extent goes. An error
while making room leaves the state unwritten (a failed write), never written past the cap or the floor. And the
quota's count takes out the references of the waiting states, the conversation's next turn among them. CPU only (the
store's harness: spill_store_kit); a held store lock stands for a writer busy with earlier jobs."""

import time

import numpy as np

from spill_store_kit import ALIGN, ROWS, S, SKIP, SP, Pool, align_up, ar, ids_of, snap, store  # noqa: F401


def keep_r(st, ids, lineage, base=0, x=None):
    """A state kept: (its key, what persist returned, its extent)."""

    x = Pool(ROWS).add(base, align_up(len(ids))) if x is None else x
    r = st.persist(ids, snap(ids), extent=x, lineage=lineage, skip=SKIP)
    return S.point_key(np.asarray(ids, dtype=np.int64)), r, x


def one_state(ar, tmp_path):
    """What one 3,000-token state of the test arena takes on disk, measured."""

    st = store(ar, tmp_path / "one")
    keep_r(st, ids_of(3000, seed=99), "m")
    assert st.drain(30)
    return st.stored()


def need_of(ar, ids, lineage):
    """(its blocks' bytes, all the bytes its keep asks room for) of a state of ``ids``."""

    items, _ = SP.encode(snap(ids), skip=SKIP)
    small = sum(t.numel() * t.element_size() for _, t in items) + 4096
    size = sum(S._span_bytes(ar.planes, s, e)
               for s, e, _ in S.block_hashes(np.asarray(ids, dtype=np.int64), ALIGN, lineage))
    return size, size + small + 8 * len(ids)


def until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            return False
        time.sleep(0.005)
    return True


def walked(st, keys):
    """A conversation's bytes as 0110 counted them: every block name of every other stored point walked."""

    mine = set(keys)
    others = {b for k, pt in st.index.items() if k not in mine for b in pt.blocks}
    blocks = {b for k in mine for b in st.index[k].blocks if b not in others and b in st.have}
    inos = {st.have[b][1] for b in blocks}
    own = sum(st.inodes[i][0] for i in inos if i in st.inodes and st.inodes[i][1] <= blocks)
    return own + sum(st.index[k].small_bytes for k in mine)


def test_a_refused_states_next_turn_is_refused_with_it_never_a_failed_write(ar, tmp_path):
    one = one_state(ar, tmp_path)
    ids = ids_of(4096, seed=2)
    _, need_b = need_of(ar, ids[:3000], "c")
    st = store(ar, tmp_path / "cascade", gib=(one + need_b - 1) / 2**30)      # A and B never fit together
    a, _, _ = keep_r(st, ids_of(3000, seed=1), "a", base=4096)
    assert st.drain(30)
    st.reading.add(a)                                                         # being read: never evicted meanwhile
    with st.lock:                                                             # the writer behind: the loop runs ahead
        b, _, x = keep_r(st, ids[:3000], "c")                                 # a turn boundary: no room for it
        st.pump(None)
        c, _, _ = keep_r(st, ids, "c", x=x)                                   # the prompt's end, extending it
    assert st.drain(30)
    assert b not in st.index and c not in st.index
    assert st.stats["no_space"] == 2 and st.stats["write_failed"] == 0


def test_a_refused_state_frees_its_reserved_bytes_at_once(ar, tmp_path):
    one = one_state(ar, tmp_path)
    ids_b, ids_c = ids_of(4000, seed=5), ids_of(3000, seed=6)
    _, need_c = need_of(ar, ids_c, "c")
    st = store(ar, tmp_path / "phantom", gib=(one + need_c) / 2**30)          # A and C fit; A and B do not
    a, _, _ = keep_r(st, ids_of(3000, seed=1), "a", base=8192)
    assert st.drain(30)
    st.reading.add(a)
    with st.lock:
        b, _, _ = keep_r(st, ids_b, "b")                                      # too big beside A
        c, _, _ = keep_r(st, ids_c, "c", base=4096)                           # another conversation: fits beside A
    assert st.drain(30)
    assert b not in st.index and c in st.index and st.stats["no_space"] == 1


def test_a_refused_state_gets_no_rows_copied_when_its_extent_goes(ar, tmp_path):
    one = one_state(ar, tmp_path)
    ids = ids_of(8192, seed=3)                                                # four whole blocks
    _, need_b = need_of(ar, ids, "b")
    st = store(ar, tmp_path / "release", gib=(need_b + one // 2) / 2**30)     # B alone fits the cap, not beside A
    a, _, _ = keep_r(st, ids_of(3000, seed=1), "a", base=8192)
    assert st.drain(30)
    st.reading.add(a)
    b, _, x = keep_r(st, ids, "b")
    job = st.waiting.get(b)
    assert job is not None and until(lambda: job.failed is not None)          # refused on the writer, before a pump
    st.release(x, 0)                                                          # the pool takes its extent back
    assert st.stats["released_blocks"] == 0
    assert st.drain(30)
    assert b not in st.index


def test_an_error_while_making_room_leaves_the_state_unwritten(ar, tmp_path, monkeypatch):
    one = one_state(ar, tmp_path)
    st = store(ar, tmp_path / "error", gib=1.5 * one / 2**30, quota=0.5)
    a, _, _ = keep_r(st, ids_of(3000, seed=1), "a", base=8192)
    assert st.drain(30)
    st.reading.add(a)                                                         # nothing evictable: no room for B

    def boom(self, pt, need):
        raise KeyError("a stale parent")                                      # any bug in the quota's count
    monkeypatch.setattr(S.BlockStore, "_trim_own", boom)
    b, _, _ = keep_r(st, ids_of(3000, seed=2), "b", base=4096)
    assert st.drain(30)
    assert b not in st.index and st.stored() <= st.cap and st.stats["write_failed"] == 1


def test_the_quota_count_takes_the_waiting_next_turn_out(ar, tmp_path):
    st = store(ar, tmp_path / "waiting")
    ids = ids_of(12000)
    for n in (2048, 6144, 10240):                                             # a conversation of three turns
        keep_r(st, ids[:n], "c")
    assert st.drain(30)
    with st.lock:                                                             # its next turn waits while counted
        keep_r(st, ids, "c")
        keys = list(st.index)
        first = st.index[keys[0]].blocks
        assert any(not set(w.point.blocks).isdisjoint(first) for w in st.waiting.values())
        for g in [keys] + [[k] for k in keys]:
            assert st._conv_bytes(g) == walked(st, g)
    assert st.drain(30)
