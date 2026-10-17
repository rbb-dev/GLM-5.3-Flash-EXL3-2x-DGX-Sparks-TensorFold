"""Patch 9004 in the spill tier's block store (patch 0110): a turn record rides on the stored state it belongs to.

The tier writes a kept state when it is kept; its reply's record lands later. ``attach_record`` hands the record to the
store's writer, which puts its bytes in the point's own file (``<key>.rec``) and its label in the point's json fields
(the json last, as every point file). A record that lands while its state still waits to be written goes out with that
state, its file before the json. A restart's scan keeps the label in RAM only when the file is there; a read checks the
file's CRC; a record goes when its state goes; a record for a state the store does not hold is dropped. Nothing here
raises: a lost record is a cache miss later, never an error.

The harness is the one tools/spill_blocks_check.py (patch 0110) uses: a real BlockStore over a real Arena on the host,
with a stand-in small state. CPU only."""

import json
import os
import zlib

import numpy as np

from spill_store_kit import ROWS, S, align_up, ar, ids_of, json_of, keep, store, Pool  # noqa: F401  (ar: a fixture)

BODY = json.dumps({"tail": [7, 8, 9], "turns": []}).encode()


def test_a_kept_state_carries_its_point_key(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    obj, key = keep(st, ids_of(3000))
    assert obj.spill_key == key                       # the request thread names the point without hashing again
    assert st.drain(30)


def test_a_record_for_a_written_state_lands_in_its_own_file_and_its_json(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    _, key = keep(st, ids_of(3000))
    assert st.drain(30) and key in st.index
    small_before = st.small_on_disk
    st.attach_record(key, "r4|L|I|S|1|2048", BODY)
    assert st.drain(30)
    with open(st._point_path(key, ".rec"), "rb") as fh:
        assert fh.read() == BODY
    f = json_of(st, key)["fields"]
    assert f[S.RECORD] == "r4|L|I|S|1|2048"
    assert f[S.RECORD + "_bytes"] == len(BODY) and f[S.RECORD + "_crc"] == zlib.crc32(BODY)
    assert st.index[key].fields[S.RECORD] == "r4|L|I|S|1|2048"          # the label in RAM, for the lookup
    assert st.read_record(key) == BODY
    assert st.small_on_disk == small_before + len(BODY)                  # the record counts toward the cap


def test_a_record_that_lands_while_its_state_still_waits_goes_out_with_it_before_its_json(ar, tmp_path, monkeypatch):
    order = []
    put = S._put

    def logged(path, data, owner, direct=False):
        order.append(os.path.basename(path).split(".")[-1].split("-")[0])
        return put(path, data, owner, direct=direct)

    monkeypatch.setattr(S, "_put", logged)
    st = store(ar, tmp_path / "a")
    _, key = keep(st, ids_of(5000))                   # several blocks: still waiting for the pumps
    assert key not in st.index
    st.attach_record(key, "r4|L|I|S|1|4096", BODY)
    assert st.drain(30) and key in st.index
    assert st.read_record(key) == BODY
    assert json_of(st, key)["fields"][S.RECORD] == "r4|L|I|S|1|4096"
    names = [n for n in order if n in ("rec", "json")]
    assert names.index("rec") < names.index("json")   # the json, the point's commit, last


def test_a_restart_keeps_the_record_and_a_read_back_returns_it(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    ids = ids_of(3000)
    _, key = keep(st, ids)
    assert st.drain(30)
    st.attach_record(key, "r4|L|I|S|1|2048", BODY)
    assert st.drain(30)
    st2 = store(ar, tmp_path / "a")                   # a restart: the scan
    assert st2.index[key].fields[S.RECORD] == "r4|L|I|S|1|2048"
    assert st2.read_record(key) == BODY
    x = Pool(ROWS).add(0, align_up(len(ids)))
    lj = st2.load_async(key, x)
    small, pt = st2.finish_load(lj)
    assert pt.key == key and lj.record == ("r4|L|I|S|1|2048", BODY)     # read beside the small state, off the engine


def test_a_missing_record_file_drops_only_the_label_at_a_restart(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    _, key = keep(st, ids_of(3000))
    assert st.drain(30)
    st.attach_record(key, "r4|L|I|S|1|2048", BODY)
    assert st.drain(30)
    os.remove(st._point_path(key, ".rec"))
    st2 = store(ar, tmp_path / "a")
    assert key in st2.index                           # the state still resumes
    assert S.RECORD not in st2.index[key].fields and st2.read_record(key) is None


def test_a_damaged_record_file_is_never_read(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    ids = ids_of(3000)
    _, key = keep(st, ids)
    assert st.drain(30)
    st.attach_record(key, "r4|L|I|S|1|2048", BODY)
    assert st.drain(30)
    path = st._point_path(key, ".rec")
    raw = bytearray(open(path, "rb").read())
    raw[3] ^= 0x40
    with open(path, "wb") as fh:
        fh.write(bytes(raw))
    st2 = store(ar, tmp_path / "a")
    assert st2.read_record(key) is None
    lj = st2.load_async(key, Pool(ROWS).add(0, align_up(len(ids))))
    st2.finish_load(lj)
    assert lj.record is None                          # the state reads back; its record does not


def test_a_record_goes_with_its_state(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    _, key = keep(st, ids_of(3000))
    assert st.drain(30)
    st.attach_record(key, "r4|L|I|S|1|2048", BODY)
    assert st.drain(30)
    small = st.small_on_disk
    with st.lock:
        st._remove(key, "pruned_conversations")
    assert not os.path.exists(st._point_path(key, ".rec"))
    assert st.small_on_disk <= small - len(BODY)


def test_a_record_file_without_its_label_goes_at_a_restart(ar, tmp_path):
    """A crash between a record's file and its json: the state stays, the unlabelled file goes (never kept uncounted)."""

    st = store(ar, tmp_path / "a")
    _, key = keep(st, ids_of(3000))
    assert st.drain(30)
    with open(st._point_path(key, ".rec"), "wb") as fh:
        fh.write(BODY)
    st2 = store(ar, tmp_path / "a")
    assert key in st2.index and S.RECORD not in st2.index[key].fields
    assert not os.path.exists(st2._point_path(key, ".rec"))


def test_a_record_for_a_state_the_store_does_not_hold_is_dropped(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    st.attach_record("0" * 64, "r4|L|I|S|1|0", BODY)
    assert st.drain(30)
    assert not os.path.exists(st._point_path("0" * 64, ".rec"))
    assert st.stats["records_unstored"] == 1


def test_attaching_never_raises(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    _, key = keep(st, ids_of(3000))
    assert st.drain(30)
    st.attach_record(key, None, BODY)                 # a bad label
    st.attach_record(key, "r4|L|I|S|1|2048", "not bytes")
    assert st.drain(30)
    assert S.RECORD not in st.index[key].fields


def test_two_sparks_keep_a_state_whose_record_only_rank_zero_holds(ar, tmp_path):
    """Both ranks write a state when it is kept; only rank 0 attaches its record. Each rank's store is its own, so a
    restart keeps the state on both ranks, and only rank 0's index holds the label."""

    ids = ids_of(3000)
    r0, r1 = store(ar, tmp_path / "rank0"), store(ar, tmp_path / "rank1", rank=1)
    _, key = keep(r0, ids)
    keep(r1, ids)
    r0.attach_record(key, "r4|L|I|S|1|2048", BODY)
    assert r0.drain(30) and r1.drain(30)
    again0, again1 = store(ar, tmp_path / "rank0"), store(ar, tmp_path / "rank1", rank=1)
    assert again0.has(key) and again1.has(key)
    assert again0.index[key].fields[S.RECORD] == "r4|L|I|S|1|2048" and S.RECORD not in again1.index[key].fields
