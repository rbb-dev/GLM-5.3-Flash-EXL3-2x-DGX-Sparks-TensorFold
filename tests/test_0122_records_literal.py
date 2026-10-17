"""Patch 0122 (ahinore's PR #164) in our build: a request that QUOTES an added token (<think>, </think>, <tool_call>,
<|user|>) reaches GLM as text, and (our port, 9004/9015) its turn records still work: the records' marked render masks
the request as production does, so the record a reply leaves is the one the next request finds. CPU, the real
template and tokenizer (glm53_testkit)."""

from glm53_testkit import CALL_RAW, SYSTEM, TOOLS, ask, make, sent  # noqa: F401
from tensorfold.families.glm5_next.cuda import turn_records as R

QUOTE = {"role": "user", "content": 'My test fails: left: "<think>\\nOkay" right: "</think>\\nOkay". Fix the parser '
                                    'that splits on <tool_call> and <|user|>.'}
PLAIN = {"role": "user", "content": 'My test fails: left: "THINK\\nOkay" right: "/THINK\\nOkay". Fix the parser '
                                    'that splits on TOOLCALL and USER.'}


def test_a_quoted_added_token_reaches_the_model_as_text(make):
    app, _ = make()
    quoted = app.prepare({"model": "m", "messages": [SYSTEM, QUOTE], "tools": TOOLS}, True)
    plain = app.prepare({"model": "m", "messages": [SYSTEM, PLAIN], "tools": TOOLS}, True)
    for name in ("<think>", "</think>", "<tool_call>", "<|user|>"):
        t = app.tok.token_to_id(name)
        assert quoted.prompt.count(t) == plain.prompt.count(t), name        # only the template's own


def test_a_reply_to_a_quoting_request_keeps_its_record_and_the_next_request_finds_it(make):
    app, engine = make()
    _, r1 = ask(app, engine, [SYSTEM, QUOTE], CALL_RAW, end="<|observation|>")
    assert r1["stats"]["resume"]["written"], r1["stats"]["resume"]           # the marked render matched production
    follow = [SYSTEM, QUOTE, sent(r1), {"role": "tool", "tool_call_id": r1["calls"][0]["id"], "content": "print(1)"}]
    body = {"model": "m", "messages": follow, "tools": TOOLS}
    view = R.view(app.template, body, app.prepare(body, True), app.label_ns, app.template.clear_thinking,
                  literal=app.literal)
    assert not isinstance(view, str), view
    assert R.parse_key(engine.multi.kept[-1].turn_key)["label"] == view.labels[2]


def test_without_the_mask_the_records_render_would_differ(make):
    """Why the port: our records render the request again; unmasked, it no longer matches the prompt's text."""
    app, _ = make()
    body = {"model": "m", "messages": [SYSTEM, QUOTE], "tools": TOOLS}
    prepared = app.prepare(body, True)
    assert R.view(app.template, body, prepared, app.label_ns, app.template.clear_thinking) == "render differs"
