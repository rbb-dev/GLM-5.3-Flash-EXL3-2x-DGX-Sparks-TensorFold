"""Patch 9011 (a stored prompt's token ids as bytes, not a JSON list), on the CPU: the spill tier saved each prompt's
ids in its file's header as JSON lists of ints and read them back the same way, on the engine's scheduler thread
(every rank) for a save and at start for a scan: 270 ms to save and 143 ms to read a 1,000,000-token prompt's on an
x86 core. Now one conversion (``as_ids``) serves the key, the header field (base64 of the raw ids: int32, or int64 for
a picture prompt's keyed ids) and the index entry; reading the field back is a decode and a view."""

import random
import time

import numpy as np

from tensorfold.cuda import spill

N = 1_000_000


def best(fn, runs=3):
    out, times = None, []
    for _ in range(runs):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return out, min(times)


def test_ids_survive_the_header_exactly():
    plain = [random.randrange(154_000) for _ in range(5000)]
    keyed = plain[:100] + [-1 - (1 << 61) - 5] * 7 + plain[100:]          # a picture's rows: -1 - its 62-bit key
    for ids, wide in ((plain, False), (keyed, True)):
        text = spill.ids_field(spill.as_ids(ids), wide)
        header, _ = spill._header([], {"format": spill.FORMAT, "tokens": text})
        got = spill.ids_from_field(spill._parse_header(header)[0]["__metadata__"]["tokens"])
        assert got.dtype == np.int64 and got.tolist() == ids


def test_a_long_prompts_ids_are_saved_and_read_back_quickly():
    ids = [random.randrange(154_000) for _ in range(N)]

    def save():
        arr = spill.as_ids(ids)
        key = spill.ids_key(arr)
        header, _ = spill._header([], {"format": spill.FORMAT, "n": N, "tokens": spill.ids_field(arr, False)})
        return arr, key, header

    (arr, key, header), save_s = best(save)
    assert key == spill.ids_key(ids) and np.array_equal(arr, ids)
    got, read_s = best(lambda: spill.ids_from_field(spill._parse_header(header)[0]["__metadata__"]["tokens"]))
    assert np.array_equal(got, arr)
    assert save_s < 0.06, f"saving a 1M-token prompt's ids took {save_s * 1000:.0f} ms"
    assert read_s < 0.03, f"reading them back took {read_s * 1000:.0f} ms"


def test_files_of_the_old_format_are_never_read():
    assert spill.FORMAT != "tensorfold-cuda-spill-2"


def test_a_saved_long_prompt_is_found_exactly_after_a_restart(tmp_path):
    cfg = spill.SpillConfig(root=str(tmp_path), gib=1.0)
    store = spill.SpillStore(cfg, rank=0, world=1, device="cpu", signature="build", quiet=True, weights="w")
    ids = [random.randrange(154_000) for _ in range(N)]
    t = time.perf_counter()
    assert store.save(ids) is not None
    save_s = time.perf_counter() - t
    deadline = time.monotonic() + 30
    while store.pending_jobs() and time.monotonic() < deadline:
        time.sleep(0.05)
    again = spill.SpillStore(cfg, rank=0, world=1, device="cpu", signature="build", quiet=True, weights="w")
    k = spill.ids_key(ids)
    assert k in again.index and again.index[k].ids.tolist() == ids
    assert save_s < 0.1, f"saving a 1M-token prompt held the engine's loop {save_s * 1000:.0f} ms"
