"""Patch 9027 (restores priced by the engine), on the CPU. A restore saves the prefill of the tokens the GPU no longer
held: the stored point's length less the longest state the GPU still had (the engine reads the disk only past it,
multi.prestage). Each restore is priced once, when it happens, at the engine's own prefill speed (its last 32 finished
requests with at least 1,024 uncached tokens, the board's real-prompt filter) less the restore's own load time, so the
counter only grows. Before 9027 the board priced every restored token at one fixed speed (2026-10-16)."""

import types

import pytest

from spill_store_kit import ROWS, ar, ids_of, keep, store  # noqa: F401  (ar: the fixture)
from tensorfold.cuda import health as H
from tensorfold.families.glm5_next.cuda.pool import Pool, align_up


def restored(ar, tmp_path, n, held, price):
    st = store(ar, tmp_path / "a")
    ids = ids_of(n)
    _, key = keep(st, ids)
    assert st.drain(30)
    st2 = store(ar, tmp_path / "a")                   # a restart: read back from the disk
    st2.price_tps = price
    lj = st2.load_async(key, Pool(ROWS).add(0, align_up(n)), held=held)
    st2.finish_load(lj)
    return st2.stats


def test_a_restore_saves_only_the_tokens_the_gpu_no_longer_held(ar, tmp_path):
    s = restored(ar, tmp_path, 3000, held=1000, price=2000.0)
    assert s["loaded_tokens"] == 3000 and s["saved_tokens"] == 2000
    assert s["saved_s"] == pytest.approx(max(0.0, 2000 / 2000.0 - s["load_s"]))
    assert s["unpriced"] == 0


def test_with_no_measured_speed_yet_a_restore_is_counted_unpriced(ar, tmp_path):
    s = restored(ar, tmp_path, 3000, held=0, price=None)
    assert s["saved_tokens"] == 3000 and s["saved_s"] == 0.0 and s["unpriced"] == 1


def test_a_gpu_state_as_long_as_the_point_saves_nothing(ar, tmp_path):
    s = restored(ar, tmp_path, 3000, held=5000, price=2000.0)
    assert s["saved_tokens"] == 0 and s["saved_s"] == 0.0


def request(prompt, cached, prefill_s):
    r = H.Request(prompt, [])
    r.stats = {"prefill_s": prefill_s, "cached": cached}
    return r


def health_with_disk():
    h = H.Health()
    disk = types.SimpleNamespace(price_tps=None)
    h.app = types.SimpleNamespace(engine=types.SimpleNamespace(scheduler=types.SimpleNamespace(
        decoder=types.SimpleNamespace(disk=disk))))
    return h, disk


def test_the_price_is_the_engines_own_prefill_speed_on_real_prompts():
    h, disk = health_with_disk()
    h._fold(request(5000, 1000, 2.0))                 # 4,000 uncached tokens in 2 s
    assert disk.price_tps == pytest.approx(2000.0)
    h._fold(request(900, 0, 0.1))                     # under 1,024 uncached: not a real prompt, not counted
    assert disk.price_tps == pytest.approx(2000.0)
    h._fold(request(130000, 128000, 0.5))             # 2,000 uncached in 0.5 s
    assert disk.price_tps == pytest.approx(6000 / 2.5)


def test_the_price_follows_the_last_32_real_prompts():
    h, disk = health_with_disk()
    h._fold(request(10000, 0, 100.0))                 # one slow one, then 32 at 2,000 tok/s push it out
    for _ in range(32):
        h._fold(request(4000, 0, 2.0))
    assert disk.price_tps == pytest.approx(2000.0)


def test_health_shows_the_price_and_never_fails_without_a_disk():
    h = H.Health()
    h.app = types.SimpleNamespace()                   # no engine, no decoder, no disk: nothing to price, no error
    h._fold(request(5000, 1000, 2.0))
    assert h.snapshot(types.SimpleNamespace())["prefill_price_tps"] == 2000.0


class Prestage:
    """Just enough of MultiDecoder for ``prestage`` to pick a stored prompt past a state the GPU still holds."""

    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder as _M
    prestage = _M.prestage
    _start_load_real = _M._start_load

    def __init__(self, gpu_tokens):
        from tensorfold.cuda.spill_blocks import point_key
        import numpy as np
        self.ids = list(range(1, 3001))
        key = point_key(np.asarray(self.ids, dtype=np.int64))
        self.pt = types.SimpleNamespace(key=key, n=3000, ids=self.ids, fields={})
        self.disk = types.SimpleNamespace(stats={"disk_lookups": 0, "disk_hits": 0, "has_refused": 0},
                                          best=lambda prompt, have, grid: self.pt, has=lambda key: True)
        self.hit = types.SimpleNamespace(ids=self.ids[:gpu_tokens], extent=None) if gpu_tokens else None
        self.pool = types.SimpleNamespace(rows=1 << 20, half=False)
        self.loads, self.grid, self.next_lid, self.started = {}, 0, 7, None
        self.g = types.SimpleNamespace(_gather_ints=lambda v: [list(v), list(v)])

    def _check(self): pass
    def _need(self, n): return n
    def _resume(self, prompt): return self.hit
    def _emit(self, *a): pass
    def _flush(self): pass
    def _sync(self): pass
    def _room(self, size, keep, half=None): return 0

    def _start_load(self, hi, lo, n, base, size, lid, held=0):
        self.started = held
        return types.SimpleNamespace(ids=None)


@pytest.mark.parametrize("gpu_tokens", [0, 1200])
def test_prestage_hands_the_gpus_matched_length_to_the_read(gpu_tokens):
    d = Prestage(gpu_tokens)
    s = types.SimpleNamespace(draft=True, prompt=d.ids + [9, 9], vision=None)
    assert d.prestage(s) is not None
    assert d.started == gpu_tokens


def test_start_load_passes_held_to_the_store():
    seen = {}
    me = types.SimpleNamespace(next_lid=0, loads={},
                               pool=types.SimpleNamespace(add=lambda base, size: types.SimpleNamespace()),
                               disk=types.SimpleNamespace(load_async=lambda key, x, held=0: seen.setdefault("held", held)))
    Prestage._start_load_real(me, 1, 2, 3000, 0, 3008, 0, held=1200)
    assert seen["held"] == 1200
