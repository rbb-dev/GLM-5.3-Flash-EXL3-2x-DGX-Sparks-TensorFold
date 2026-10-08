"""Patch 9016 (API translations), on the CPU: the Anthropic Messages route carries TensorFold's info block (the reply's
token checksum, whether a record served or was written) on its stream's message_delta and in a non-streamed message,
as the chat route does, so a client or the bench can check exactness on Claude Code's route too; and a Responses
function_call_output may carry pictures (input_image parts), rendered inside its tool response as a chat tool result's
pictures are, and resumed from records as they are."""

from glm53_testkit import api_body, make, picture_part, png, restart  # noqa: F401
from tensorfold.server.anthropic_translate import Reply

INFO = {"sha256": "abc123", "resume": {"path": "record", "written": True}}
SHOT_RAW = ("I take a screenshot to see the screen as it is now.</think>"
            "<tool_call>read<arg_key>path</arg_key><arg_value>screen.png</arg_value></tool_call>")


def test_the_anthropic_stream_ends_with_tensorfolds_info():
    sent = []
    r = Reply("glm-5.3-flash", sent.append)
    r.start()
    r.chunk({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
    r.chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 1}, "tensorfold": INFO})
    r.chunk(None)
    (delta,) = [e for e in sent if e["type"] == "message_delta"]
    assert delta["tensorfold"] == INFO and delta["delta"]["stop_reason"] == "end_turn"


def test_a_non_streamed_anthropic_message_carries_it_too():
    r = Reply("glm-5.3-flash", lambda e: None)
    message = r.completion({"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 1}, "tensorfold": INFO})
    assert message["tensorfold"] == INFO


def test_a_responses_tool_output_with_a_screenshot_resumes_from_its_record(make, tmp_path):
    app, engine = make(vision=True)
    first = api_body("responses", [{"role": "user", "content": "show me the screen"}])
    engine.script.append((SHOT_RAW, "<|observation|>"))
    r1 = app.run(first, True, lambda delta: True, prepared=app.prepare(first, True))
    assert r1["stats"]["resume"]["written"]
    cid, args = r1["calls"][0]["id"], r1["calls"][0]["function"]["arguments"]
    shot = {"type": "input_image", "image_url": picture_part(png((1920, 1080), (5, 120, 60)))["image_url"]["url"]}
    turns = [{"role": "user", "content": "show me the screen"},
             {"type": "function_call", "call_id": cid, "name": "read", "arguments": args},
             {"type": "function_call_output", "call_id": cid, "output": [shot]}]
    body = api_body("responses", turns)
    tool = next(m for m in body["messages"] if m["role"] == "tool")
    assert isinstance(tool["content"], list) and tool["content"][0]["type"] == "image_url"
    app2, _ = restart(engine, make, tmp_path, vision=True)
    p2 = app2.prepare(body, True)
    assert p2.resume.hit is not None, p2.resume.reason
