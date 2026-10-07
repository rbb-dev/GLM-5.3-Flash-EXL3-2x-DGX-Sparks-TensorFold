"""Patch 9005 end to end on the CPU: a later request, also after a restart (its states and records read back from the
spill tier's files) and from a client that drops reasoning, is served its conversation's record (exact tokens, thinking
included) plus only what is new, token for token what a fresh read of the true conversation is; every doubtful case
takes today's path."""

import json
import os

import pytest

from glm53_testkit import (ASK, CALL_RAW, SYSTEM, TEXT_RAW, TOOLS, api_body, ask, first_turn, make,  # noqa: F401
                           restart, sent)
from tensorfold.engine.exact_sampling import seed_for
from tensorfold.families.glm5_next.cuda import resume, turn_records as R


def fresh_suffix(app, messages, *, marker="<|observation|>", tools=TOOLS):
    """The tokens of today's render after its last ``marker``: found without the label code, as an independent check."""

    text = app.template.render(json.loads(json.dumps(messages)), tools=tools, enable_thinking=True, extra={})
    return app.tok.encode(text[text.rindex(marker):], add_special_tokens=False).ids


def test_after_a_restart_a_dropping_client_is_served_its_record_exactly(make, tmp_path):
    app, engine = make()
    r1, T1, follow = first_turn(app, engine)
    app2, engine2 = restart(engine, make, tmp_path)
    p2, r2 = ask(app2, engine2, follow, TEXT_RAW)
    assert r2["stats"]["resume"]["path"] == "record"
    assert engine2.seen[0] == T1 + fresh_suffix(app2, follow)          # #9: exactly the record, then what is new
    today = app2.tok.encode(p2.resume.render.real, add_special_tokens=False).ids
    assert today != engine2.seen[0]                                    # today's render lost the thinking


@pytest.mark.parametrize("api", ["anthropic", "responses"])
def test_each_api_is_served_its_record_exactly_after_a_restart(make, tmp_path, api):
    app, engine = make()
    first = api_body(api, [{"role": "user", "content": "look at the repo"}])
    engine.script.append((CALL_RAW, "<|observation|>"))
    p1 = app.prepare(first, True)
    r1 = app.run(first, True, lambda delta: True, prepared=p1)
    assert r1["stats"]["resume"]["written"]
    T1 = p1.prompt + r1["out"][:-1]
    cid, args = r1["calls"][0]["id"], r1["calls"][0]["function"]["arguments"]
    if api == "anthropic":
        turns = [{"role": "user", "content": "look at the repo"},
                 {"role": "assistant", "content": [{"type": "tool_use", "id": cid, "name": "read",
                                                    "input": json.loads(args)}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": "print(1)"}]}]
    else:
        turns = [{"role": "user", "content": "look at the repo"},
                 {"type": "function_call", "call_id": cid, "name": "read", "arguments": args},
                 {"type": "function_call_output", "call_id": cid, "output": "print(1)"}]
    app2, _ = restart(engine, make, tmp_path)
    body = api_body(api, turns)
    p2 = app2.prepare(body, True)
    assert p2.resume.hit is not None, p2.resume.reason
    assert p2.prompt == T1 + fresh_suffix(app2, body["messages"], tools=body.get("tools"))


def test_a_client_returning_the_same_reasoning_is_served_the_record(make):
    app, engine = make()
    r1, T1, follow = first_turn(app, engine)
    follow[2] = sent(r1, reasoning=r1["reasoning"])
    _, r2 = ask(app, engine, follow, TEXT_RAW)
    assert r2["stats"]["resume"]["path"] == "record" and engine.seen[-1][:len(T1)] == T1


def test_a_client_changing_the_reasoning_takes_todays_path(make):
    app, engine = make()
    r1, _, follow = first_turn(app, engine)
    follow[2] = sent(r1, reasoning="Summarised: read a.py.")
    p2, r2 = ask(app, engine, follow, TEXT_RAW)
    assert r2["stats"]["resume"]["path"] == "render" and p2.resume.reason == "reasoning differs"


def test_renumbered_call_ids_take_todays_path(make, tmp_path):
    app, engine = make()
    r1, _, follow = first_turn(app, engine)
    app2, engine2 = restart(engine, make, tmp_path)
    renumbered = json.loads(json.dumps(follow).replace(r1["calls"][0]["id"], "toolu_01"))
    p2, r2 = ask(app2, engine2, renumbered, TEXT_RAW)
    assert r2["stats"]["resume"]["path"] == "render" and p2.resume.reason == "no record"


def test_a_kwargs_plant_gets_no_lookup(make):
    app, engine = make()
    _, _, follow = first_turn(app, engine)
    plant = {"messages": [SYSTEM, {"role": "user", "content": "other"}]}
    p, _ = ask(app, engine, follow, TEXT_RAW, chat_template_kwargs=plant)
    assert p.resume == "chat_template_kwargs"


def test_two_different_records_for_one_label_and_ids_are_a_miss(make):
    app, engine = make()
    r1, _, follow = first_turn(app, engine)
    state = engine.multi.kept[-1]
    key = R.parse_key(state.turn_key)
    twin = type(state)(list(state.ids), 99, turn_key=R.record_key(key["label"], key["idvec"], "0" * 64, key["turns"],
                                                                     key["base"]), turn=state.turn)
    engine.multi.kept.append(twin)
    p2, r2 = ask(app, engine, follow, TEXT_RAW)
    assert r2["stats"]["resume"]["path"] == "render" and p2.resume.reason == "ambiguous"


def test_a_record_that_changed_under_the_lookup_is_a_miss(make, monkeypatch):
    app, engine = make()
    _, _, follow = first_turn(app, engine)
    real = resume.current_key
    monkeypatch.setattr(resume, "current_key", lambda multi, carrier: real(multi, carrier) + "x")
    p2, r2 = ask(app, engine, follow, TEXT_RAW)
    assert r2["stats"]["resume"]["path"] == "render" and p2.resume.reason == "record changed"


def test_foreign_reasoning_must_be_presented_every_time(make):
    app, engine = make()
    imported = [SYSTEM, ASK, {"role": "assistant", "content": "Earlier answer from elsewhere, long enough to count.",
                              "reasoning_content": "foreign thoughts"}, {"role": "user", "content": "go on " * 30}]
    _, r1 = ask(app, engine, imported, TEXT_RAW)
    follow = imported + [sent(r1, reasoning=r1["reasoning"]), {"role": "user", "content": "and then? " * 30}]
    _, presented = ask(app, engine, follow, TEXT_RAW)
    assert presented["stats"]["resume"]["path"] == "record"
    dropped = json.loads(json.dumps(follow))
    dropped[2].pop("reasoning_content")
    p3, _ = ask(app, engine, dropped, TEXT_RAW)
    assert p3.resume.reason == "foreign reasoning not presented"


@pytest.mark.parametrize("answer,path", [("Short.", "render"), ("x" * 80, "record")])
def test_a_dropped_text_turn_needs_enough_visible_text_to_prove_it_was_ours(make, tmp_path, answer, path):
    app, engine = make()
    _, r1 = ask(app, engine, [SYSTEM, ASK], f"thinking it over</think>{answer}")
    app2, engine2 = restart(engine, make, tmp_path)
    _, r2 = ask(app2, engine2, [SYSTEM, ASK, sent(r1), {"role": "user", "content": "next " * 40}], TEXT_RAW)
    assert r2["stats"]["resume"]["path"] == path


def test_a_record_that_would_not_fit_falls_back_and_is_never_refused(make, tmp_path):
    app, engine = make()
    thought = " ".join(f"step {i} checks item {i * 7 % 13} against {i * 11 % 17}" for i in range(150))
    _, r1 = ask(app, engine, [SYSTEM, ASK], thought + "</think>" + "y" * 80)
    assert r1["stats"]["resume"]["written"]
    follow = [SYSTEM, ASK, sent(r1), {"role": "user", "content": "next"}]
    today = len(app.tokenize({"model": "m", "messages": follow, "tools": TOOLS})["tokens"])
    small, engine2 = restart(engine, make, tmp_path, limit=today + 60)
    p2, r2 = ask(small, engine2, follow, TEXT_RAW, max_tokens=50)
    assert p2.resume.reason == "context limit" and r2["finish"] == "stop"


def test_the_seed_follows_the_context(make):
    app, engine = make()
    _, T1, follow = first_turn(app, engine)
    p2, _ = ask(app, engine, follow, TEXT_RAW, temperature=0.7)
    assert engine.sampling[-1].seed == seed_for(p2.prompt) and p2.prompt[:len(T1)] == T1


def test_tokenize_is_unchanged_and_count_tokens_counts_the_context(make, tmp_path):
    from tensorfold.server import anthropic

    app, engine = make()
    _, T1, follow = first_turn(app, engine)
    app2, _ = restart(engine, make, tmp_path)
    body = {"model": "m", "messages": follow, "tools": TOOLS}
    resumed = app2.prepare(dict(body), True).prompt
    assert resumed[:len(T1)] == T1
    assert app2.tokenize(dict(body))["tokens"] != resumed                     # /tokenize renders, never resumes
    assert anthropic.count_tokens(app2, dict(body)) == len(resumed)          # count_tokens counts the context


def test_a_request_with_its_own_clear_thinking_never_takes_a_record_made_under_the_other(make):
    app, engine = make()
    _, _, follow = first_turn(app, engine)
    p2, _ = ask(app, engine, follow, TEXT_RAW, chat_template_kwargs={"clear_thinking": True})
    assert p2.resume.hit is None


def test_a_resumed_requests_new_state_carries_the_record_until_its_own_lands(make, tmp_path):
    app, engine = make()
    r1, T1, follow = first_turn(app, engine)
    follow[-1] = {**follow[-1], "content": "print(1)\n" * 40}         # past the next grid point: a state of its own
    app2, engine2 = restart(engine, make, tmp_path)
    p2, r2 = ask(app2, engine2, follow, TEXT_RAW, max_tokens=5)        # cut by its limit: no record of its own
    assert r2["stats"]["resume"]["path"] == "record" and r2["stats"]["resume"]["not_written"]
    newest = engine2.multi.kept[-1]
    assert len(newest.ids) > len(T1) - 64 and R.parse_key(newest.turn_key)["t_sha"] == R.t_digest(T1)


def test_forks_of_one_conversation_share_its_record_and_each_resumes_its_own(make, tmp_path):
    """Subagents and edits fork a conversation. Each fork keeps a state of its own carrying the shared record until a
    record of its own lands, so after a restart the shared turn has several carriers of one ``T``: never ambiguous.
    A fork with a record of its own resumes exactly that record."""

    app, engine = make()
    _, r1 = ask(app, engine, [SYSTEM, ASK], f"thinking it over</think>{'y' * 80}")
    shared = engine.multi.kept[-1].turn_key
    base = [SYSTEM, ASK, sent(r1)]
    forks = {}
    for name, limit in (("x", 5), ("y", 5), ("w", 2048)):             # x and y are cut: no record of their own
        conv = base + [{"role": "user", "content": f"now {name} " * 40}]
        p, r = ask(app, engine, conv, f"checking {name}</think>{name * 80}", max_tokens=limit)   # 80: provably ours
        assert r["stats"]["resume"]["path"] == "record"
        forks[name] = (conv, p, r)
    assert sum(c.turn_key == shared for c in engine.multi.kept) >= 3
    app2, engine2 = restart(engine, make, tmp_path)
    _, z = ask(app2, engine2, base + [{"role": "user", "content": "now z " * 40}], TEXT_RAW)
    assert z["stats"]["resume"]["path"] == "record"
    conv, pw, rw = forks["w"]
    _, again = ask(app2, engine2, conv + [sent(rw), {"role": "user", "content": "go on " * 40}], TEXT_RAW)
    Tw = pw.prompt + rw["out"][:-1]
    assert again["stats"]["resume"]["path"] == "record" and engine2.seen[-1][:len(Tw)] == Tw


def test_an_edited_earlier_turn_never_takes_a_later_record(make):
    app, engine = make()
    _, _, follow = first_turn(app, engine)
    follow[-1] = {**follow[-1], "content": "print(1)\n" * 40}         # past the next grid point: a state of its own
    _, r2 = ask(app, engine, follow, TEXT_RAW)
    assert r2["stats"]["resume"]["written"]
    edited = json.loads(json.dumps(follow))
    edited[1]["content"] = "look at the repository"                   # the client changed its first message
    later = edited + [sent(r2), {"role": "user", "content": "next " * 40}]
    p3, r3 = ask(app, engine, later, TEXT_RAW)
    assert r3["stats"]["resume"]["path"] == "render" and p3.resume.hit is None
    assert p3.prompt == app.tokenize({"model": "m", "messages": later, "tools": TOOLS})["tokens"]


def test_an_anthropic_client_sending_its_thinking_back_is_served_the_record(make):
    """Claude Code returns each thinking block as it got it (this server signs them with an empty signature)."""

    app, engine = make()
    first = api_body("anthropic", [{"role": "user", "content": "look at the repo"}])
    engine.script.append((CALL_RAW, "<|observation|>"))
    p1 = app.prepare(first, True)
    r1 = app.run(first, True, lambda delta: True, prepared=p1)
    cid, args = r1["calls"][0]["id"], r1["calls"][0]["function"]["arguments"]
    turns = [{"role": "user", "content": "look at the repo"},
             {"role": "assistant", "content": [{"type": "thinking", "thinking": r1["reasoning"], "signature": ""},
                                               {"type": "tool_use", "id": cid, "name": "read",
                                                "input": json.loads(args)}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": "print(1)"}]}]
    p2 = app.prepare(api_body("anthropic", turns), True)
    T1 = p1.prompt + r1["out"][:-1]
    assert p2.resume.hit is not None, p2.resume.reason
    assert p2.prompt[:len(T1)] == T1


def test_with_clear_thinking_an_empty_thinking_turn_never_lets_hidden_thinking_back_in(make, tmp_path):
    """clear_thinking=1. Turn 1 thinks and is recorded; turn 2 is served from that record but thinks nothing; a later
    request hides both turns' thinking. It must be served exactly what a fresh render of the true conversation gives:
    turn 2's record describes turn 1 as the record it continued holds it (the thinking), never as the client sent it
    (dropped), or the lookup would think nothing hidden is in T. (The reviewer's case, 2026-10-08.)"""

    ct = {"clear_thinking": True}
    app, engine = make()
    _, r1 = ask(app, engine, [SYSTEM, ASK], CALL_RAW, end="<|observation|>", chat_template_kwargs=ct)
    assert r1["stats"]["resume"]["written"]
    follow = [SYSTEM, ASK, sent(r1), {"role": "tool", "tool_call_id": r1["calls"][0]["id"],
                                      "content": "print(1)\n" * 40}]
    app2, engine2 = restart(engine, make, tmp_path)
    _, r2 = ask(app2, engine2, follow, "</think>" + "a.py prints 1; nothing else in the repository needs a look. " * 2,
                chat_template_kwargs=ct)
    assert r2["stats"]["resume"]["path"] == "record" and r2["stats"]["resume"]["written"]
    later = follow + [sent(r2), {"role": "user", "content": "next question " * 20}]
    true = json.loads(json.dumps(later))
    true[2]["reasoning_content"], true[4]["reasoning_content"] = r1["reasoning"], r2["reasoning"] or ""
    fresh = app2.tokenize({"model": "m", "messages": true, "tools": TOOLS, "chat_template_kwargs": ct})["tokens"]
    ask(app2, engine2, later, "ok</think>" + "fine " * 20, chat_template_kwargs=ct)
    assert engine2.seen[-1] == fresh


def test_a_stored_record_that_cannot_be_read_falls_back_to_a_shallower_one(make, tmp_path):
    """The deepest record's file is gone (a write cut short, a disk error): that label is a miss and the lookup goes on
    to the next shallower record, which serves the request (never an error that drops every record)."""

    app, engine = make()
    r1, T1, follow = first_turn(app, engine)
    follow[-1] = {**follow[-1], "content": "print(1)\n" * 40}         # past the next grid point: a state of its own
    _, r2 = ask(app, engine, follow, TEXT_RAW)
    assert r2["stats"]["resume"]["written"]
    app2, engine2 = restart(engine, make, tmp_path)
    deepest = max(engine2.multi.disk.index.values(), key=lambda e: e.n)
    os.remove(os.path.join(engine2.multi.disk.dir, deepest.name + ".safetensors"))
    later = follow + [sent(r2), {"role": "user", "content": "next " * 40}]
    p3, r3 = ask(app2, engine2, later, TEXT_RAW)
    assert r3["stats"]["resume"]["path"] == "record" and p3.resume.hit_turn == 2 and p3.prompt[:len(T1)] == T1


def test_an_entry_the_tier_failed_to_write_is_never_a_carrier(make, tmp_path):
    app, engine = make()
    first_turn(app, engine)
    app2, engine2 = restart(engine, make, tmp_path)
    assert resume.carriers(engine2.multi)
    for e in engine2.multi.disk.index.values():
        e.bad = True
    assert all(c[0] == "kept" for c in resume.carriers(engine2.multi))
