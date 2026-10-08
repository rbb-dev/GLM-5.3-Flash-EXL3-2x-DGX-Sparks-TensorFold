"""Patch 9004 in the parallel decoder (rank 0 only, nothing rank 1 must know): a request's own kept state is named in
its turn holder and takes the record its prompt continued; a finished reply's state is held back from early writes
until its record lands, never from an eviction; the record's fields are saved with the state (the key in RAM, the
body only on disk); nothing here can raise into the engine."""

import importlib
import json
import os
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def mods(monkeypatch):
    """multi and decode, importable on the CPU: decode's kernels import triton, which no test here runs (a stand-in
    module, as TensorFold's own decoder tests use; the whole suite is CPU-only, so nothing needs the real one)."""

    lang = ModuleType("triton.language")
    lang.constexpr = object
    triton = ModuleType("triton")
    triton.language, triton.jit = lang, lambda fn: fn
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", lang)
    multi = importlib.import_module("tensorfold.families.glm5_next.cuda.multi")
    decode = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    monkeypatch.setattr(decode, "take_snapshot",
                        lambda e, ids, pending, *, mtp, drafter=None: decode.Snapshot(list(ids), None, None, None, -1, -1))
    return multi, decode


def decoder(multi, rank=0, cap=16):
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.rank, dec.kept, dec.record_due, dec.next_kid = rank, [], set(), 0
    dec.g = SimpleNamespace(cache_entries=cap)
    dec.e = SimpleNamespace(use=lambda st: None)
    dec.disk, dec.lanes = None, {}
    return dec


def lane(multi, prompt, point, *, shared=(), glm=None, inherit=None):
    s = SimpleNamespace(prompt=list(prompt), glm=glm if glm is not None else {}, error=None, out=[], count=8, sid=0)
    ln = multi.Lane(s, 0, 0, SimpleNamespace(kept=[], owner=0), None, 0, [0, 0, 0, 0])
    ln.point, ln.shared, ln.key, ln.images, ln.dflash, ln.inherit = point, list(shared), list(prompt), False, False, inherit
    return ln


def record(base, tail, label="L", idvec="I", t_sha="S", turns=1):
    key = f"r4|{label}|{idvec}|{t_sha}|{turns}|{base}"
    return list(tail), [json.dumps({"tail": list(tail), "turns": []})], key


def test_keep_names_the_requests_own_state_and_hands_it_the_continued_record(mods):
    multi, _ = mods
    dec = decoder(multi)
    holder = {}
    ln = lane(multi, range(200), 192, shared=[64], glm={"turn": holder}, inherit=record(128, [7, 8, 9]))
    dec._keep(ln, 64)                                    # a shared-prefix point: neither
    assert "kid" not in holder and dec.kept[-1].turn_key is None
    dec._keep(ln, 192)                                   # the request's own state
    own = dec.kept[-1]
    assert holder["kid"] == own.kid and own.turn_key == "r4|L|I|S|1|128" and own.turn == record(128, [7, 8, 9])[1]
    assert own._turn_tail == [7, 8, 9]


def test_rank_one_never_touches_a_record(mods):
    multi, _ = mods
    dec = decoder(multi, rank=1)
    holder = {}
    dec._keep(lane(multi, range(200), 192, glm={"turn": holder}, inherit=record(128, [7])), 192)
    assert holder == {} and dec.kept[-1].turn_key is None and dec.kept[-1].turn is None


def test_a_record_is_inherited_only_when_the_prompt_continues_its_tokens(mods):
    multi, decode = mods
    dec = decoder(multi)
    hit = decode.Snapshot(list(range(64)), None, None, None, -1, -1)
    hit._turn_tail, hit.turn, hit.turn_key = record(64, [70, 71])
    assert dec._turn_inherit(hit, list(range(64)) + [70, 71, 99]) == record(64, [70, 71])
    assert dec._turn_inherit(hit, list(range(64)) + [70, 72, 99]) is None            # another branch
    bare = decode.Snapshot(list(range(64)), None, None, None, -1, -1)
    assert dec._turn_inherit(bare, list(range(80))) is None                          # no record
    loaded = decode.Snapshot(list(range(64)), None, None, None, -1, -1)              # read back from disk: no tail yet
    _, loaded.turn, loaded.turn_key = record(64, [70, 71])
    assert dec._turn_inherit(loaded, list(range(64)) + [70, 71]) == record(64, [70, 71])
    broken = decode.Snapshot(list(range(64)), None, None, None, -1, -1)
    broken.turn, broken.turn_key = ["not json"], "r4|L|I|S|1|nope"
    assert dec._turn_inherit(broken, list(range(80))) is None                        # never raises


def finishing(multi, dec, out, count=8):
    holder = {}
    ln = lane(multi, range(200), 192, glm={"turn": holder})
    dec._keep(ln, 192)
    ln.s.out, ln.s.count, ln.s.sid = out, count, ln.sid
    dec.lanes = {ln.sid: ln}
    dec._ends = lambda lane: (999,)
    dec._emit = lambda op, payload: None
    dec._finish = lambda sid: dec.lanes.pop(sid)
    dec.idle, dec.broken, dec.watchdog, dec.iteration_since = True, None, 0, None
    dec.finish([ln.s])
    return holder


def test_a_reply_the_model_ended_holds_its_state_for_its_record(mods):
    multi, _ = mods
    dec = decoder(multi)
    holder = finishing(multi, dec, [1, 2, 999])
    assert holder.get("due") and holder["kid"] in dec.record_due
    dec.release_due(holder["kid"])
    assert not dec.record_due


def test_a_cancelled_reply_holds_nothing(mods):
    multi, _ = mods
    dec = decoder(multi)
    holder = finishing(multi, dec, [1, 2], count=8)
    assert not holder.get("due") and not dec.record_due


def behind_decoder(multi, dec, written):
    dec.g.cache_entries = 3                              # near the cap: _behind writes the next victim early
    dec.pool = SimpleNamespace(rows=100, free_rows=lambda: 100)
    dec.disk = SimpleNamespace(cfg=SimpleNamespace(highwater=0.7, writers=2), generation=0,
                               stats=SimpleNamespace(deferred=0, behind=0, full=0, no_space=0),
                               has=lambda ident, key: False, pending_jobs=lambda: 0)
    dec._victims = lambda protect: list(dec.kept)
    dec._superseded = lambda c: False
    dec._spillable = lambda c: True
    dec._ident = lambda c: c.ids
    dec._key = lambda c: (c.kid, 0)
    dec._kept_keys = lambda: frozenset()
    dec._emit = lambda op, payload: None

    def spill(c, leaving=False, keep=frozenset()):
        written.append((c.kid, leaving))
        return True
    dec._spill = spill


def test_an_early_write_waits_for_the_record_and_an_eviction_does_not(mods):
    multi, _ = mods
    dec = decoder(multi)
    written = []
    behind_decoder(multi, dec, written)
    dec._keep(lane(multi, range(128), 128), 128)
    dec._keep(lane(multi, range(300, 428), 128), 128)
    first = dec.kept[0]
    dec.record_due.add(first.kid)
    dec._behind()
    assert written == []                                 # the next victim's record is due: no early write
    dec.release_due(first.kid)
    dec._behind()
    assert written == [(first.kid, False)]
    second = dec.kept[1]
    dec.record_due.add(second.kid)
    dec._settle = lambda x: None
    dec._drop(second)                                    # an eviction never waits for a record
    assert written[-1] == (second.kid, True) and second.kid not in dec.record_due


def test_the_record_fields_are_saved_with_the_state_and_only_the_key_stays_in_ram(mods):
    _, decode = mods
    spill = importlib.import_module("tensorfold.cuda.spill")
    snap = decode.Snapshot([1, 2, 3], torch.zeros(2), torch.zeros(2), None, -1, -1)
    snap._turn_tail, snap.turn, snap.turn_key = record(3, [4, 5])
    items, layer = spill.encode(snap, skip=("ids", "rows", "nbytes", "pending", "mtp_len"))
    plain = spill._plain([layer])
    assert plain["turn_key"] == "r4|L|I|S|1|3" and "turn" not in plain and "_turn_tail" not in layer["fields"]
    back = spill.decode(json.loads(json.dumps(layer)), dict(items), [f"{decode.__name__}:Snapshot"])
    assert back.turn == snap.turn and back.turn_key == snap.turn_key and back._turn_tail is None
    old = json.loads(json.dumps(layer))
    for name in ("turn", "turn_key"):
        old["fields"].pop(name)
    assert spill.decode(old, dict(items), [f"{decode.__name__}:Snapshot"]).turn_key is None   # a file from before 9004


def test_nothing_in_the_hooks_raises_into_the_engine(mods):
    multi, _ = mods
    dec = decoder(multi)
    bad = lane(multi, range(200), 192, glm={"turn": "not a dict"}, inherit=("x",))   # junk everywhere
    dec._keep(bad, 192)
    dec.lanes = {bad.sid: bad}
    bad.s.out, bad.s.sid = [999], bad.sid
    dec._ends, dec._emit, dec._finish = (lambda lane: (999,)), (lambda op, p: None), (lambda sid: dec.lanes.pop(sid))
    dec.idle, dec.broken, dec.watchdog, dec.iteration_since = True, None, 0, None
    dec.finish([bad.s])
    dec.release_due("nothing")
    assert dec.kept[-1].turn_key is None and not dec.record_due


class Pair:
    """Two ranks' collectives, one thread each: ``gather`` all-gathers, ``share`` is rank 0's broadcast."""

    def __init__(self):
        self.barrier, self.slots = threading.Barrier(2, timeout=30), [None, None]

    def _swap(self, rank, value, pick):
        if value is not None:
            self.slots[rank] = list(value)
        self.barrier.wait()
        out = pick()
        self.barrier.wait()
        return out

    def gather(self, rank):
        return lambda words: self._swap(rank, words, lambda: [list(s) for s in self.slots])

    def share(self, rank):
        return lambda flat: self._swap(rank, flat if rank == 0 else None, lambda: list(self.slots[0]))


def start_two_ranks(root):
    """Both ranks' spill tiers as an engine start makes them: one build hash, then each rank's scan and the
    cross-rank reconcile."""

    from tensorfold.cuda import spill

    cfg, pair, made, errors = spill.SpillConfig(root=str(root), gib=1.0), Pair(), [None, None], []

    def start(rank):
        try:
            made[rank] = spill.SpillStore(cfg, rank=rank, world=2, device="cpu", signature="build", quiet=True,
                                          share=pair.share(rank), gather=pair.gather(rank), weights=f"rank{rank}")
        except BaseException as exc:     # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=start, args=(rank,)) for rank in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors and all(made), errors
    return made


def store_file(store, ids, fields):
    """A stored state's file as ``SpillStore.save`` heads it (its tensors left out: the fields are what counts)."""

    from tensorfold.cuda import spill

    layer = {"class": "tensorfold.families.glm5_next.cuda.decode:Snapshot", "fields": fields}
    tokens = spill.ids_field(spill.as_ids(ids), False) if hasattr(spill, "ids_field") else json.dumps(ids)  # 9011
    info = {"format": spill.FORMAT, "compat": store.compat, "model": "", "n": len(ids), "tokens": tokens,
            "layers": json.dumps([layer]), "rows": "[]", "head": "0", "saved": "0", "crc32": spill._crc_placeholder(0)}
    header, _ = spill._header([], info)
    with open(os.path.join(store.dir, spill.key_name(spill.ids_key(ids), len(ids)) + ".safetensors"), "wb") as f:
        f.write(header)


def test_two_sparks_keep_a_stored_state_whose_record_only_rank_zero_wrote(tmp_path):
    """Rank 0's file of a state carries the record's fields; rank 1's file of the same state never does. A restart's
    scan and cross-rank reconcile keep the state on both ranks, and rank 0's index holds its key (design §3, §7)."""

    from tensorfold.cuda import spill

    ids, key = list(range(3000)), "r4|L|I|S|1|3000"
    first = start_two_ranks(tmp_path)
    store_file(first[0], ids, {"turn_key": {"v": key}, "turn": {"l": [{"v": '{"tail": [], "turns": []}'}]}})
    store_file(first[1], ids, {"turn_key": {"v": None}, "turn": {"v": None}})
    again = start_two_ranks(tmp_path)
    k = spill.ids_key(ids)
    assert k in again[0].index and k in again[1].index
    assert again[0].index[k].fields["turn_key"] == key and "turn" not in again[0].index[k].fields
    assert again[1].index[k].fields.get("turn_key") is None
