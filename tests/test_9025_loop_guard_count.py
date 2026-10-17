"""Patch 9025: the loop guard's fires add up in /health (``loop_guard_total``). The guard counts per reply
(``stats["loop_guard"]``, cuda/server.py) but nothing kept the count: a thinking loop it closed left no trace. CPU only."""
from types import SimpleNamespace

from tensorfold.cuda import health as H


def test_the_loop_guards_fires_add_up_in_health():
    h = H.Health()
    for fires in (2, 0, 1):
        with h.running(10, [1, 2, 3]) as r:
            r.stats = {"loop_guard": fires} if fires else {}
    assert h.snapshot(SimpleNamespace())["loop_guard_total"] == 3


def test_health_shows_zero_before_any_fire():
    assert H.Health().snapshot(SimpleNamespace())["loop_guard_total"] == 0
