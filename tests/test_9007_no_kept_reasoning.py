"""Patch 9007 (Mia's in-RAM reasoning LRU cut out): the turn records of 9004/9005 keep a conversation's thinking with
its state (saved, loaded and evicted with it), so nothing else puts reasoning back into a prompt. The LRU kept the text
of tool-calling replies in RAM with a retention of its own, emptied at every restart, and keyed replies on words alone:
the same words with another picture got the first conversation's thinking."""

import importlib.util
import json

from glm53_testkit import TOOLS, first_turn, make  # noqa: F401


def test_the_reasoning_lru_is_gone():
    assert importlib.util.find_spec("tensorfold.families.glm5_next.cuda.kept_reasoning") is None


def test_dropped_reasoning_comes_back_only_from_a_record(make):
    app, engine = make()
    r1, _, follow = first_turn(app, engine)
    renumbered = json.loads(json.dumps(follow).replace(r1["calls"][0]["id"], "toolu_01"))   # no record matches these
    text = app.template.render(renumbered, tools=TOOLS, enable_thinking=True)
    assert "I will read a.py first" not in text
