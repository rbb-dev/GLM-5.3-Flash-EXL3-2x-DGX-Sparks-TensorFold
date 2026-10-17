"""Patch 9004 in the parallel decoder (rank 0 only, nothing rank 1 must know): a request's own kept state is named in
its turn holder and takes the record its prompt continued; the spill tier writes a state when it is kept (0110), so a
record that lands later is handed to the tier for the point that state was written to (``attach_record``), and a state
read back takes the record its point carries; a record never rides in the state's small file; nothing here can raise
into the engine."""

import importlib
import json
import sys
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
    dec.rank, dec.kept, dec.next_kid = rank, [], 0
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


class Tier:
    """The spill tier as the decoder's record hooks see it: the records handed to it."""

    def __init__(self, fail=False):
        self.records, self.fail = [], fail

    def attach_record(self, key, label, payload):
        if self.fail:
            raise OSError("the disk")
        self.records.append((key, label, payload))


def test_a_landed_record_goes_to_the_point_its_state_was_written_to(mods):
    multi, decode = mods
    dec = decoder(multi)
    dec.disk = Tier()
    snap = decode.Snapshot(list(range(64)), None, None, None, -1, -1)
    snap._turn_tail, snap.turn, snap.turn_key = record(64, [7, 8])
    snap.spill_key = "k" * 64
    dec.attach_record(snap)
    assert dec.disk.records == [("k" * 64, "r4|L|I|S|1|64", snap.turn[0].encode("utf-8"))]
    unwritten = decode.Snapshot(list(range(64)), None, None, None, -1, -1)            # never written: no point
    unwritten._turn_tail, unwritten.turn, unwritten.turn_key = record(64, [7, 8])
    dec.attach_record(unwritten)
    assert len(dec.disk.records) == 1
    one = decoder(multi, rank=1)                                                      # rank 1 never has one
    one.disk = Tier()
    one.attach_record(snap)
    assert one.disk.records == []


def test_an_inherited_record_goes_to_the_tier_with_its_state(mods):
    multi, _ = mods
    dec = decoder(multi)
    dec.disk = Tier()
    dec._persist = lambda c: setattr(c, "spill_key", "p" * 64)                       # written when kept (0110)
    dec._keep(lane(multi, range(200), 192, glm={"turn": {}}, inherit=record(128, [7, 8, 9])), 192)
    assert dec.disk.records == [("p" * 64, "r4|L|I|S|1|128", record(128, [7, 8, 9])[1][0].encode("utf-8"))]
    dec._keep(lane(multi, range(300, 500), 192, glm={"turn": {}}), 192)               # nothing to inherit
    assert len(dec.disk.records) == 1


def test_a_record_never_rides_in_the_states_small_file(mods):
    multi, decode = mods
    spill = importlib.import_module("tensorfold.cuda.spill")
    assert {"turn", "turn_key", "spill_key"} <= set(multi.MultiDecoder._SPILL_SKIP)
    snap = decode.Snapshot([1, 2, 3], torch.zeros(2), torch.zeros(2), None, -1, -1)
    snap._turn_tail, snap.turn, snap.turn_key = record(3, [4, 5])
    snap.spill_key = "k" * 64
    _, layer = spill.encode(snap, skip=multi.MultiDecoder._SPILL_SKIP)
    assert not {"turn", "turn_key", "_turn_tail", "spill_key"} & set(layer["fields"])


def test_a_state_read_back_takes_the_record_its_point_carries(mods):
    multi, decode = mods
    snap = decode.Snapshot(list(range(64)), None, None, None, -1, -1)
    body = record(64, [7, 8])[1][0]
    multi.MultiDecoder._turn_loaded(snap, ("r4|L|I|S|1|64", body.encode("utf-8")))
    assert snap.turn_key == "r4|L|I|S|1|64" and snap.turn == [body] and snap._turn_tail is None
    dec = decoder(multi)
    assert dec._turn_inherit(snap, list(range(64)) + [7, 8, 9]) == record(64, [7, 8])   # it continues as before
    none = decode.Snapshot(list(range(64)), None, None, None, -1, -1)
    multi.MultiDecoder._turn_loaded(none, None)                                          # no record on its point
    assert none.turn_key is None and none.turn is None


def test_nothing_in_the_hooks_raises_into_the_engine(mods):
    multi, decode = mods
    dec = decoder(multi)
    dec.disk = Tier(fail=True)
    dec._persist = lambda c: setattr(c, "spill_key", "p" * 64)
    bad = lane(multi, range(200), 192, glm={"turn": "not a dict"}, inherit=("x",))   # junk everywhere
    dec._keep(bad, 192)
    assert dec.kept[-1].turn_key is None
    snap = decode.Snapshot(list(range(64)), None, None, None, -1, -1)
    snap._turn_tail, snap.turn, snap.turn_key = record(64, [7])
    snap.spill_key = "k" * 64
    dec.attach_record(snap)                                                           # the tier raises: lost
    dec.attach_record(SimpleNamespace())                                              # not a state at all
    multi.MultiDecoder._turn_loaded(snap, ("label", b"\xff\xfe not utf-8"))          # a damaged body
    multi.MultiDecoder._turn_loaded(snap, "not a pair")
