"""Patch 9002 (the Anthropic Messages route): a stream never goes silent while a tool call is being written (the chat
handler's keepalive becomes a ``ping`` event, as Anthropic's own API sends, so a client's idle timer never fires), and
``/v1/messages/count_tokens`` answers 400 on any request it cannot prepare instead of dropping the connection."""

from tensorfold.server import anthropic
from tensorfold.server.anthropic_translate import Reply


def stream():
    sent = []
    reply = Reply("glm", sent.append)
    reply.start()
    return reply, sent


def test_a_keepalive_while_a_call_is_held_is_a_ping():
    reply, sent = stream()
    reply.chunk({"choices": [{"index": 0, "delta": {}}]})            # the chat handler's keepalive (an empty delta)
    assert sent[-1] == {"type": "ping"}


def test_a_delta_with_content_is_not_a_ping():
    reply, sent = stream()
    reply.chunk({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
    reply.chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    reply.chunk({"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "choices": []})
    assert {"type": "ping"} not in sent


def test_count_tokens_answers_400_when_preparation_fails():
    class App:
        def prepare(self, chat, is_chat):
            raise TypeError("TextInputSequence must be str")           # e.g. a lone UTF-16 surrogate in a message

    status, payload = anthropic.count_reply(App(), {"messages": [{"role": "user", "content": "\ud83d"}]})
    assert status == 400 and payload["type"] == "error" and "TextInputSequence" in payload["error"]["message"]


def test_count_tokens_still_counts():
    class App:
        def prepare(self, chat, is_chat):
            return type("P", (), {"prompt": [1, 2, 3]})()

    assert anthropic.count_reply(App(), {"messages": []}) == (200, {"input_tokens": 3})
