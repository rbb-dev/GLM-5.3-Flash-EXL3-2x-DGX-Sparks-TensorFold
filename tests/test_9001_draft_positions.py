"""Patch 9001 (live draft counts by source and position), on the CPU: the decoder's counting, its /health entry and
the /metrics text, parsed back with prometheus_client."""

import types

import pytest
from prometheus_client.parser import text_string_to_metric_families

from tensorfold.families.glm5_next.cuda import multi
from tensorfold.server import metrics


@pytest.fixture
def decoder():
    """A MultiDecoder without __init__ (as the recipe's tools/pool_room_check.py builds one), after five rounds."""

    m = multi.MultiDecoder.__new__(multi.MultiDecoder)
    m.draft_counts = {}
    m.lanes, m.kept, m.group, m.disk = {}, [], False, None
    m.pool = types.SimpleNamespace(rows=2201600, free_rows=lambda: 1462272)
    m._count_drafts("dflash2", 7, 3)      # 7 drafted, the first 3 kept
    m._count_drafts("dflash2", 4, 4)      # 4 drafted, all kept
    m._count_drafts("dflash2", 2, 0)      # 2 drafted, none kept
    m._count_drafts("copy", 15, 15)
    m._count_drafts("none", 0, 0)
    return m


def test_counts(decoder):
    d = decoder.draft_counts
    assert d["dflash2"] == {"rounds": 3, "drafted": [3, 3, 2, 2, 1, 1, 1], "accepted": [2, 2, 2, 1]}
    assert d["copy"] == {"rounds": 1, "drafted": [1] * 15, "accepted": [1] * 15}
    assert d["none"] == {"rounds": 1, "drafted": [], "accepted": []}
    # what the dashboard relies on: a position's counts sum to the rounds' drafted and kept tokens
    assert sum(d["dflash2"]["drafted"]) == 7 + 4 + 2 and sum(d["dflash2"]["accepted"]) == 3 + 4 + 0


def test_health_is_a_copy(decoder):
    h = decoder.health()
    assert h["drafts"]["dflash2"]["accepted"] == [2, 2, 2, 1]
    h["drafts"]["dflash2"]["accepted"].append(99)              # /health's reader cannot change the counts
    assert decoder.draft_counts["dflash2"]["accepted"] == [2, 2, 2, 1]


def test_metrics_text(decoder):
    lines = []
    metrics._drafts(lines, decoder.health()["drafts"])
    for want in ('tensorfold_health:draft_rounds_total{source="dflash2"} 3',
                 'tensorfold_health:draft_position_drafted_total{source="dflash2",position="7"} 1',
                 'tensorfold_health:draft_position_accepted_total{source="dflash2",position="5"} 0',   # never kept
                 'tensorfold_health:draft_position_accepted_total{source="copy",position="15"} 1',
                 'tensorfold_health:draft_rounds_total{source="none"} 1'):
        assert want in lines
    families = {f.name: f for f in text_string_to_metric_families("\n".join(lines) + "\n")}
    samples = {(s.name, tuple(sorted(s.labels.items()))): s.value for f in families.values() for s in f.samples}
    assert samples[("tensorfold_health:draft_position_accepted_total",
                    (("position", "3"), ("source", "dflash2")))] == 2
    assert len(samples) == 3 + 2 * (7 + 15)              # rounds by source, then drafted and kept per position


def test_no_drafts_no_families():
    lines = []
    metrics._drafts(lines, {})
    metrics._drafts(lines, None)
    assert lines == []
