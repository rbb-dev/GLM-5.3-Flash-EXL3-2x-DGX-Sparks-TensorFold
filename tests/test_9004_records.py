"""Patch 9004 end to end on the CPU (a real GlmApp over a scripted parallel engine): an eligible reply's record lands on
the request's own kept state before ``run`` returns (so before the reply's closing event), holding exactly the tokens
the model read and wrote; every other finish leaves no record; the hold is always released; nothing here can fail a
reply."""

import hashlib
import json
from array import array

import pytest

from glm53_testkit import CALL_RAW, SYSTEM, ASK, TEXT_RAW, TOOLS, ask, first_turn, make, sent  # noqa: F401
from tensorfold.families.glm5_next.cuda import turn_records as R


def own_state(engine):
    return engine.multi.kept[-1]


def T_of(state):
    key = R.parse_key(state.turn_key)
    return state.ids[:key["base"]] + R.body_tail(state.turn)


def test_an_eligible_reply_lands_on_its_own_state_with_its_exact_tokens(make):
    app, engine = make()
    p1, r1 = ask(app, engine, [SYSTEM, ASK], CALL_RAW, end="<|observation|>")
    state = own_state(engine)
    T = p1.prompt + r1["out"][:-1]                          # the prompt, then the reply minus its end token
    assert r1["stats"]["resume"]["written"] and T_of(state) == T
    key = R.parse_key(state.turn_key)
    assert key["t_sha"] == hashlib.sha256(array("i", T).tobytes()).hexdigest() and key["turns"] == 1
    assert key["base"] == len(state.ids) and R.body_table(state.turn)[-1]["prov"] == "own"
    assert state._turn_tail == R.body_tail(state.turn) and not engine.multi.record_due


def test_the_records_label_is_the_one_the_next_request_computes(make):
    from tensorfold.families.glm5_next.cuda import turn_records

    app, engine = make()
    r1, _, follow = first_turn(app, engine)
    view = turn_records.view(app.template, {"model": "m", "messages": follow, "tools": app_tools(app, engine)},
                             app.prepare({"model": "m", "messages": follow, "tools": app_tools(app, engine)}, True),
                             app.label_ns, app.template.clear_thinking)
    assert R.parse_key(own_state(engine).turn_key)["label"] == view.labels[2]


def app_tools(app, engine):
    from glm53_testkit import TOOLS
    return TOOLS


@pytest.mark.parametrize("raw,end,body,why", [
    ("No close of the think block, a call <tool_call>read", "<|observation|>", {}, "no </think>"),
    ("I will read it.<tool_call>read<arg_key>path</arg_key><arg_value>a.py</arg_value></tool_call></think>",
     "<|observation|>", {}, "a call in the think block"),
    (TEXT_RAW, None, {"max_tokens": 5}, "not ended by the model"),
    (TEXT_RAW, "<|user|>", {"draft": False}, "draft:false"),
    (TEXT_RAW, "<|user|>", {"ignore_eos": True}, "ignore_eos"),
    (TEXT_RAW, "<|user|>", {"chat_template_kwargs": {"add_generation_prompt": True}}, "chat_template_kwargs"),
    ("maybe " * 300 + "</think>ok", "<|user|>", {}, "repetitive reasoning"),
    ("thinking</think>", "<|user|>", {}, "an empty answer"),
    (TEXT_RAW + " STOP here", "<|user|>", {"stop": ["STOP"]}, "not ended by the model"),
    (CALL_RAW + "<tool_call>read<arg_key>path</arg_key><arg_value>b.py</arg_value></tool_call>", "<|observation|>",
     {"parallel_tool_calls": False}, "calls cut"),
], ids=["open think block", "call in think", "length", "draft false", "ignore_eos", "kwargs", "loop", "empty answer",
        "stop string", "one call of two"])
def test_replies_that_get_no_record(make, raw, end, body, why):
    app, engine = make()
    _, r = ask(app, engine, [SYSTEM, ASK], raw, end=end, **body)
    assert r["stats"]["resume"]["not_written"] == why
    assert all(c.turn_key is None for c in engine.multi.kept) and not engine.multi.record_due


def test_a_reply_made_of_several_engine_runs_gets_no_record(make):
    app, engine = make()
    real = engine.generate

    def twice(ids, count, sampling, feed, **kw):           # a gated reply: the engine is run more than once
        engine.request.turn["runs"] = engine.request.turn.get("runs", 0) + 1
        return real(ids, count, sampling, feed, **kw)
    engine.generate = twice
    _, r = ask(app, engine, [SYSTEM, ASK], TEXT_RAW)
    assert r["stats"]["resume"]["not_written"] == "several engine runs" and not engine.multi.record_due


def test_a_record_never_fails_the_reply_and_always_releases_the_hold(make, monkeypatch):
    app, engine = make()
    monkeypatch.setattr(R, "make_record", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    _, r = ask(app, engine, [SYSTEM, ASK], TEXT_RAW)
    assert r["finish"] == "stop" and r["content"]
    assert r["stats"]["resume"]["not_written"] == "error" and not engine.multi.record_due   # no exception text out
    assert "boom" not in json.dumps(r["stats"]) and app.record_counts["not written: error"] == 1


def test_a_state_gone_before_the_record_loses_only_the_record(make):
    app, engine = make()
    real = engine.multi.finish

    def evicted(holder, ended):                            # its state left memory while the reply ended
        real(holder, ended)
        engine.multi.kept = []
    engine.multi.finish = evicted
    _, r = ask(app, engine, [SYSTEM, ASK], TEXT_RAW)
    assert r["finish"] == "stop" and r["stats"]["resume"]["not_written"] == "state gone"


def test_a_newer_turn_puts_its_record_on_its_own_newer_state(make):
    app, engine = make()
    r1, T1, follow = first_turn(app, engine)
    first = own_state(engine)
    follow[-1] = {**follow[-1], "content": "print(1)\n" * 40}         # past the next grid point: a state of its own
    p2, r2 = ask(app, engine, follow, TEXT_RAW)
    second = own_state(engine)
    assert second is not first and T_of(second) == p2.prompt + r2["out"][:-1]
    assert R.parse_key(second.turn_key)["turns"] == 2


def test_a_turn_that_keeps_no_state_of_its_own_gets_no_record(make):
    app, engine = make()
    engine.multi.keep = lambda prompt, holder: None                    # its prompt ended short of the next grid point
    _, r = ask(app, engine, [SYSTEM, ASK], TEXT_RAW)
    assert r["finish"] == "stop" and r["stats"]["resume"]["not_written"] == "no own state"


def test_another_branchs_record_is_never_overwritten(make):
    app, engine = make()
    _, _, _ = first_turn(app, engine)
    state = own_state(engine)
    before = (state.turn_key, state.turn)
    other = R.record_key("L", "I", "S", 1, len(state.ids))
    state._turn_tail, state.turn, state.turn_key = [1, 2, 3], R.record_body([1, 2, 3], []), other
    assert R.continues(state, state.ids + [9, 9]) is False             # its T is not this request's prefix
    assert before[0] != other


def test_the_holder_reaches_the_scheduler(allocations_engine):
    eng, got = allocations_engine
    holder = {}
    eng.request.turn = holder
    eng.generate([1, 2, 3], 8, None, lambda new: False)
    assert got[0]["glm"]["turn"] is holder and holder["runs"] == 1


@pytest.fixture
def allocations_engine(monkeypatch):
    import importlib
    import sys
    import threading
    from types import ModuleType, SimpleNamespace

    lang = ModuleType("triton.language")
    lang.constexpr = object
    triton = ModuleType("triton")
    triton.language, triton.jit = lang, (lambda fn: fn)
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", lang)
    engine_mod = importlib.import_module("tensorfold.families.glm5_next.cuda.engine")
    eng = engine_mod.GlmEngine.__new__(engine_mod.GlmEngine)
    eng.limit, eng.serial_only, eng.policy = 1000, False, "0"
    eng.request = threading.local()
    eng._effective = lambda code: code
    got = []
    eng.scheduler = SimpleNamespace(submit=lambda *a, **kw: got.append(kw) or {})
    return eng, got


def test_without_the_parallel_decoder_records_stay_off_and_requests_are_served_as_today(make):
    app, engine = make(parallel=False)                   # PARALLEL=1 keeps no prompt states across requests
    p, r = ask(app, engine, [SYSTEM, ASK], TEXT_RAW)
    assert app.label_ns is None and p.resume is None and "resume" not in (r.get("stats") or {})
    assert r["finish"] == "stop" and p.prompt == app.tokenize({"model": "m", "messages": [SYSTEM, ASK],
                                                                 "tools": TOOLS})["tokens"]
