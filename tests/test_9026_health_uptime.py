"""Patch 9026 (the engine's start time on /health), on the CPU: /health carries ``started_at`` (when this engine process
loaded the family's app, seconds after it started) and ``uptime_s``, with turn records on or off. The board's SINCE BOOT
card and the engine's uptime showed a dash: TensorFold reported no start time anywhere (2026-10-16)."""

import time
import types

from tensorfold.families.glm5_next.cuda import app as app_mod


def test_records_off_still_reports_the_start(monkeypatch):
    monkeypatch.setattr(app_mod, "STARTED", time.time() - 100.0)
    body = app_mod.GlmApp.health_extra(types.SimpleNamespace(label_ns=None))
    assert body["started_at"] == round(app_mod.STARTED, 3)
    assert 99.0 <= body["uptime_s"] <= 105.0
    assert "turns" not in body


def test_records_on_keeps_the_turns_block_beside_the_start(monkeypatch):
    monkeypatch.setattr(app_mod, "STARTED", time.time() - 5.0)
    me = types.SimpleNamespace(label_ns=object(), engine=types.SimpleNamespace(multi=types.SimpleNamespace(kept=[])),
                               record_counts={"hits": 3})
    body = app_mod.GlmApp.health_extra(me)
    assert body["turns"] == {"on": True, "kept_with_record": 0, "counts": {"hits": 3}}
    assert body["started_at"] == round(app_mod.STARTED, 3) and body["uptime_s"] >= 5.0


def test_the_start_is_taken_once_when_the_module_loads():
    assert isinstance(app_mod.STARTED, float) and app_mod.STARTED <= time.time()
