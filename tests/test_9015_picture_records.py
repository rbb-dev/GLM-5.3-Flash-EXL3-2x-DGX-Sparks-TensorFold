"""Patch 9015 (turn records for picture and clip conversations), on the CPU with the real GLM-5.3-Flash template,
tokenizer and picture frontend (without the GPU tower): a picture conversation resumes from its record exactly as a
text one does (in memory and after a restart, from a client that drops the thinking), a screenshot loop's turns are
served from records past its 8th picture, the same words with another picture never take another conversation's
record (counted as "pictures differ"), and a reply that writes a picture marker keeps no record."""

import base64
import copy
import json

from glm53_testkit import SYSTEM, TOOLS, api_body, ask, make, picture_part, png, restart, sent  # noqa: F401

LONG_RAW = ("I look at the picture; it is a coloured panel with a stripe.</think>"
            "The picture shows a coloured panel with a short bright stripe near its top left corner, and nothing else.")
SHOT_RAW = ("I take a screenshot to see the screen as it is now.</think>"
            "<tool_call>read<arg_key>path</arg_key><arg_value>screen.png</arg_value></tool_call>")
ASK = "what is in the picture?"


def opening(colour=(40, 90, 160), size=(280, 168)):
    return [SYSTEM, {"role": "user", "content": [picture_part(png(size, colour)), {"type": "text", "text": ASK}]}]


def reference(app, messages, tools=TOOLS):
    """Today's render of the true conversation (thinking included), with records off: what a fresh read prefills."""

    ns, app.label_ns = app.label_ns, None
    try:
        return app.prepare({"model": "glm-5.3-flash", "messages": messages, "max_tokens": 2048, "tools": tools}, True)
    finally:
        app.label_ns = ns


def test_a_picture_conversation_resumes_from_its_record_exactly(make):
    app, engine = make(vision=True)
    p1, r1 = ask(app, engine, opening(), LONG_RAW)
    assert r1["stats"]["resume"].get("written"), r1["stats"]["resume"]
    follow = opening() + [sent(r1), {"role": "user", "content": "and its colour?"}]
    p2, r2 = ask(app, engine, follow, LONG_RAW)
    assert r2["stats"]["resume"]["path"] == "record", p2.resume.reason
    truth = reference(app, opening() + [sent(r1, r1["reasoning"]), {"role": "user", "content": "and its colour?"}])
    assert p2.prompt == truth.prompt and p2.prompt is p2.vision.token_ids
    assert p2.vision.keyed_ids() == truth.vision.keyed_ids() and p2.vision.item_idents == truth.vision.item_idents


def test_after_a_restart_a_picture_conversation_resumes_from_the_disk(make, tmp_path):
    app, engine = make(vision=True)
    p1, r1 = ask(app, engine, opening(), LONG_RAW)
    app2, engine2 = restart(engine, make, tmp_path, vision=True)
    follow = opening() + [sent(r1), {"role": "user", "content": "and its colour?"}]
    p2, r2 = ask(app2, engine2, follow, LONG_RAW)
    assert r2["stats"]["resume"]["path"] == "record", p2.resume.reason
    truth = reference(app2, opening() + [sent(r1, r1["reasoning"]), {"role": "user", "content": "and its colour?"}])
    assert p2.prompt == truth.prompt and p2.vision.keyed_ids() == truth.vision.keyed_ids()


def test_a_screenshot_loop_resumes_every_turn_past_the_eighth_picture(make):
    app, engine = make(vision=True)
    msgs = [SYSTEM, {"role": "user", "content": "watch the screen and tell me when it turns red"}]
    for k in range(11):
        p, r = ask(app, engine, msgs, SHOT_RAW, end="<|observation|>")
        if k:
            assert r["stats"]["resume"]["path"] == "record", (k, p.resume.reason)
        assert r["stats"]["resume"].get("written"), (k, r["stats"]["resume"])
        screenshot = picture_part(png((1920, 1080), (20 * k, 80, 200)))
        msgs = msgs + [sent(r), {"role": "tool", "tool_call_id": r["calls"][0]["id"], "content": [screenshot]}]
    assert len(p.vision.item_rows) == 10 and set(p.vision.item_rows) == {2040}        # never re-sized


def test_the_same_words_with_another_picture_never_take_the_records_turn(make):
    app, engine = make(vision=True)
    p1, r1 = ask(app, engine, opening((40, 90, 160)), LONG_RAW)
    follow = opening((160, 90, 40)) + [sent(r1), {"role": "user", "content": "and its colour?"}]
    p2, r2 = ask(app, engine, follow, LONG_RAW)
    assert r2["stats"]["resume"]["path"] == "render" and p2.resume.reason == "pictures differ"
    assert "I look at the picture" not in app.tok.decode(p2.prompt)       # no other conversation's thinking


def test_a_reply_that_writes_a_picture_marker_keeps_no_record(make):
    app, engine = make(vision=True)
    raw = "Thinking it over.</think>A marker written out in the answer: <|image|> and the answer goes on from there."
    p1, r1 = ask(app, engine, opening(), raw)
    assert r1["stats"]["resume"].get("not_written") == "media token in the reply"


def with_reasoning(messages, reasoning):
    """The chat messages with the assistant turn's reasoning as the model wrote it (the true conversation)."""

    out = copy.deepcopy(messages)
    next(m for m in out if m.get("role") == "assistant")["reasoning_content"] = reasoning
    return out


def test_an_anthropic_screenshot_tool_result_resumes_after_a_restart(make, tmp_path):
    app, engine = make(vision=True)
    first = api_body("anthropic", [{"role": "user", "content": "show me the screen"}])
    engine.script.append((SHOT_RAW, "<|observation|>"))
    r1 = app.run(first, True, lambda delta: True, prepared=app.prepare(first, True))
    assert r1["stats"]["resume"]["written"]
    cid, args = r1["calls"][0]["id"], r1["calls"][0]["function"]["arguments"]
    shot = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png((1920, 1080), (5, 120, 60))).decode()}}
    turns = [{"role": "user", "content": "show me the screen"},
             {"role": "assistant", "content": [{"type": "tool_use", "id": cid, "name": "read", "input": json.loads(args)}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": [shot]}]}]
    app2, _ = restart(engine, make, tmp_path, vision=True)
    body = api_body("anthropic", turns)
    p2 = app2.prepare(body, True)
    assert p2.resume.hit is not None, p2.resume.reason
    truth = reference(app2, with_reasoning(body["messages"], r1["reasoning"]), body.get("tools"))
    assert p2.prompt == truth.prompt and p2.vision.keyed_ids() == truth.vision.keyed_ids()


def test_a_responses_picture_conversation_resumes(make):
    app, engine = make(vision=True)
    pic = {"type": "input_image", "image_url": picture_part(png())["image_url"]["url"]}
    opening_turn = [{"role": "user", "content": [pic, {"type": "input_text", "text": ASK}]}]
    first = api_body("responses", opening_turn)
    engine.script.append((LONG_RAW, "<|user|>"))
    r1 = app.run(first, True, lambda delta: True, prepared=app.prepare(first, True))
    assert r1["stats"]["resume"]["written"], r1["stats"]["resume"]
    turns = opening_turn + [{"role": "assistant", "content": [{"type": "output_text", "text": r1["content"]}]},
                            {"role": "user", "content": [{"type": "input_text", "text": "and its colour?"}]}]
    body = api_body("responses", turns)
    p2 = app.prepare(body, True)
    assert p2.resume.hit is not None, p2.resume.reason
    truth = reference(app, with_reasoning(body["messages"], r1["reasoning"]), body.get("tools"))
    assert p2.prompt == truth.prompt


def clip_bytes(colour, seconds=2, fps=4, size=112):
    import io

    import numpy as np
    import pytest

    av = pytest.importorskip("av")              # PyAV: TensorFold's clip decoder; without it the clip test is skipped

    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for k in range(seconds * fps):
            frame = np.full((size, size, 3), colour, dtype=np.uint8)
            frame[: 8 + 8 * k, :16] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def test_a_clip_conversation_resumes_from_its_record_exactly(make):
    app, engine = make(vision=True)
    clip = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," +
                                                       base64.b64encode(clip_bytes((30, 60, 90))).decode()}}
    start = [SYSTEM, {"role": "user", "content": [clip, {"type": "text", "text": "what happens in the clip?"}]}]
    p1, r1 = ask(app, engine, start, LONG_RAW)
    assert r1["stats"]["resume"].get("written"), r1["stats"]["resume"]
    follow = start + [sent(r1), {"role": "user", "content": "and at the end?"}]
    p2, r2 = ask(app, engine, follow, LONG_RAW)
    assert r2["stats"]["resume"]["path"] == "record", p2.resume.reason
    truth = reference(app, start + [sent(r1, r1["reasoning"]), {"role": "user", "content": "and at the end?"}])
    assert p2.prompt == truth.prompt and p2.vision.keyed_ids() == truth.vision.keyed_ids()
    assert p2.vision.item_seconds == truth.vision.item_seconds and p2.vision.item_seconds[0] is not None
