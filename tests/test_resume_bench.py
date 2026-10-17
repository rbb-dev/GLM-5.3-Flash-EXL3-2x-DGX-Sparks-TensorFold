"""tools/resume_bench.py on the CPU: a fake engine over HTTP and fake hook scripts. What is checked is the bench's own
logic: the hooks it calls and in which order, the requests it sends (a client that drops the thinking; the fresh read
of the same prompt with "draft": false), the numbers it takes from the stream, its verdicts and its report."""

import http.server
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import resume_bench as rb  # noqa: E402


class Fake:
    """An engine that answers /v1/models, /health, /tokenize and streamed chat requests. ``policy(body)`` returns the
    reply's fields: reasoning, content, prompt_tokens, cached, sha, path, written, delay (seconds before the first
    token)."""

    def __init__(self):
        self.bodies = []
        self.log = []                            # every POST's path, in order
        self.policy = lambda body: {}
        self.health = {"ok": True, "backend": "tensorfold"}
        self.calls = []                          # a whole tool call to stream with the first turn's reply
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._json({"data": [{"id": "glm-5.3-flash"}]})
                return self._json(fake.health() if callable(fake.health) else fake.health)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.log.append(self.path)
                if self.path == "/tokenize":
                    n = sum(len(str(m.get("content") or "").split()) for m in body["messages"])
                    return self._json({"count": n})
                fake.bodies.append(body)
                pol = fake.policy(body)                  # once a request: a policy may count what it sees
                status = pol.get("status")
                if status:
                    data = json.dumps({"error": {"message": "out of memory"}}).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                p = {"reasoning": "think", "content": "an answer that is long enough", "prompt_tokens": 100,
                     "cached": 0, "sha": "s1", "path": "render", "written": True, "delay": 0.0, **pol}
                if not body.get("stream"):
                    return self._json({"choices": [{"message": {"role": "assistant", "content": p["content"]},
                                                    "finish_reason": "length"}],
                                       "usage": {"prompt_tokens": p["prompt_tokens"], "completion_tokens": 1}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def send(obj):
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    self.wfile.flush()

                send({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
                time.sleep(p["delay"])
                for word in p["reasoning"].split():
                    send({"choices": [{"index": 0, "delta": {"reasoning_content": word + " "}}]})
                for word in p["content"].split():
                    send({"choices": [{"index": 0, "delta": {"content": word + " "}}]})
                if fake.calls and not any(m["role"] == "tool" for m in body["messages"]):
                    send({"choices": [{"index": 0, "delta": {"tool_calls": fake.calls}}]})
                resume = {"path": p["path"], ("written" if p["written"] else "not_written"): p["written"] or "x"}
                send({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": p["prompt_tokens"], "completion_tokens": 7,
                                "prompt_tokens_details": {"cached_tokens": p["cached"]}},
                      "tensorfold": {"token_sha": p["sha"], "resume": resume}})
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.close()


def hooks(tmp_path, *names, fail=()):
    """Hook scripts that append their name to calls.log (a failing one exits 3 after printing why)."""

    folder = tmp_path / "hooks"
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        script = folder / name
        script.write_text(f"#!/bin/sh\necho {name} >> {tmp_path}/calls.log\n"
                          + ("echo 'no such engine' ; exit 3\n" if name in fail else "")
                          + ("echo '{\"nodes\": [{\"name\": \"spark 1\", \"gpu clock\": \"2418 MHz\"}]}'\n"
                             if name == "node-info" else ""))
        script.chmod(0o755)
    return rb.Hooks(folder)


def calls(tmp_path):
    p = tmp_path / "calls.log"
    return p.read_text().split() if p.exists() else []


def engine(fake):
    return rb.Engine(fake.url, "glm-5.3-flash", sampling=dict(rb.SAMPLING), timeout=30)


def test_a_hook_runs_and_is_timed_and_a_missing_optional_one_is_reported(tmp_path):
    h = hooks(tmp_path, "restart", "clear-buffers")
    assert h.has("restart") and not h.has("clear-kernel-cache")
    assert h.run("restart") >= 0 and calls(tmp_path) == ["restart"]


def test_a_failing_hook_stops_the_run_with_its_output(tmp_path):
    h = hooks(tmp_path, "restart", "clear-buffers", fail=("restart",))
    with pytest.raises(rb.HookError, match="no such engine"):
        h.run("restart")


def test_node_info_is_the_hooks_json(tmp_path):
    h = hooks(tmp_path, "restart", "clear-buffers", "node-info")
    assert h.node_info() == {"nodes": [{"name": "spark 1", "gpu clock": "2418 MHz"}]}
    assert hooks(tmp_path / "x", "restart").node_info() is None


def test_a_streamed_reply_gives_its_numbers(fake):
    fake.policy = lambda body: {"delay": 0.2, "prompt_tokens": 5000, "cached": 4000, "sha": "abc", "path": "record"}
    r = engine(fake).chat([{"role": "user", "content": "hi"}], max_tokens=64)
    assert 0.2 <= r.ttft < 2 and r.seconds >= r.ttft
    assert (r.prompt_tokens, r.cached_tokens, r.token_sha, r.path, r.written) == (5000, 4000, "abc", "record", True)
    assert r.reasoning.split() == ["think"] and r.content.split()[0] == "an" and r.finish == "stop"
    body = fake.bodies[-1]
    assert body["stream"] and body["temperature"] == 1.0 and body["top_p"] == 0.95
    assert body["reasoning_effort"] == "max" and body["max_tokens"] == 64 and "draft" not in body


def test_the_fresh_read_is_the_same_request_with_draft_false(fake):
    engine(fake).chat([{"role": "user", "content": "hi"}], max_tokens=8, draft=False)
    assert fake.bodies[-1]["draft"] is False


def test_a_client_that_drops_the_thinking_sends_only_the_answer():
    call = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    r = rb.Reply(content="the answer", reasoning="private thoughts", calls=[call])
    m = rb.dropped(r)
    assert m == {"role": "assistant", "content": "the answer", "tool_calls": r.calls}


def test_the_conversation_is_deterministic_and_about_its_size(fake):
    e = engine(fake)
    a, n = rb.conversation(e, 3000, seed=7)
    b, _ = rb.conversation(e, 3000, seed=7)
    c, _ = rb.conversation(e, 3000, seed=8)
    assert a == b and a != c and 2400 <= n <= 3000           # the fake counts words: within 20% under the target


def test_the_boot_is_a_cold_kernel_start_then_a_warm_one(tmp_path, fake):
    h = hooks(tmp_path, "restart", "clear-buffers", "clear-kernel-cache")
    out = rb.run_boot(engine(fake), h)
    assert calls(tmp_path) == ["clear-kernel-cache", "restart", "restart"]
    assert out["cold_kernels"] and out["starts"][0]["ready_s"] >= 0 and len(out["starts"]) == 2


def test_without_a_kernel_hook_both_starts_are_warm_and_the_report_says_so(tmp_path, fake):
    out = rb.run_boot(engine(fake), hooks(tmp_path, "restart", "clear-buffers"))
    assert not out["cold_kernels"] and calls(tmp_path) == ["restart", "restart"]


def test_resume_sends_the_turns_in_order_and_compares_the_fresh_read(tmp_path, fake):
    def policy(body):
        turn = sum(m["role"] == "assistant" for m in body["messages"]) + 1
        fresh = body.get("draft") is False
        return {"prompt_tokens": 1000 * turn, "cached": 0 if (turn == 1 or fresh) else 900 * turn,
                "sha": f"t{turn}", "path": "render" if turn == 1 else "record"}

    fake.policy = policy
    h = hooks(tmp_path, "restart", "clear-buffers")
    rows = rb.run_resume(engine(fake), h, [3000], seed=1)
    assert calls(tmp_path) == ["restart", "clear-buffers"]
    assert [i for i, b in enumerate(fake.bodies) if not b.get("stream")] == [3]   # the ready probe, after the restart
    chats = [b for b in fake.bodies if b.get("stream")]                # turn 1, its control, turn 2, turn 3, fresh
    roles = [[m["role"] for m in b["messages"]] for b in chats]
    assert roles[0] == roles[1] == ["system", "user"] and roles[2] == ["system", "user", "assistant", "user"]
    assert roles[3] == roles[4] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert chats[1]["draft"] is False and chats[4]["draft"] is False and "draft" not in chats[3]
    assert all("reasoning_content" not in m and "reasoning" not in m for b in chats for m in b["messages"])
    assert chats[3]["messages"] == chats[4]["messages"]                    # the fresh read is the same request
    row = rows[0]
    assert row["same_reply"] is True and row["ssd"]["path"] == "record" and row["ssd_share"] == pytest.approx(0.9)
    assert row["control_same"] is True and "thinking_restored" in row["ssd"] and "thinking_restored" in row["warm"]
    assert [row[k]["path"] for k in ("cold", "warm", "ssd", "fresh")] == ["render", "record", "record", "record"]


def test_a_different_fresh_reply_is_reported_as_not_the_same(tmp_path, fake):
    fake.policy = lambda body: {"sha": "other" if body.get("draft") is False else "same"}
    rows = rb.run_resume(engine(fake), hooks(tmp_path, "restart", "clear-buffers"), [3000], seed=1)
    assert rows[0]["same_reply"] is False


def test_the_load_windows_from_token_times():
    times = [i * 0.1 for i in range(100)] + [10.0 + 2.0 + i * 0.2 for i in range(50)]   # 10/s, a 2 s gap, then 5/s
    w = rb.windows(times, arrive=10.0, end=17.0, span=5.0)
    assert w["before"]["rate"] == pytest.approx(10, abs=0.5) and w["arrival"]["gap"] == pytest.approx(2.1, abs=0.05)
    assert w["arrival"]["rate"] == pytest.approx(3.0, abs=0.5) and w["after_end"]["rate"] == pytest.approx(5, abs=0.5)


def test_the_load_windows_cover_bs_whole_prefill():
    """B's whole wait for its first token: what A gets while a long prompt is read (here 2/s, the worst wait 1.5 s)."""

    times = ([i * 0.1 for i in range(100)] + [10.0 + 0.5 * i for i in range(1, 20)]
             + [21.0 + 0.5 * i for i in range(18)] + [30.0 + 0.1 * i for i in range(150)])
    w = rb.windows(times, arrive=10.0, end=40.0, span=5.0, first=30.0)
    assert w["prefill"]["rate"] == pytest.approx(1.85, abs=0.01) and w["prefill"]["gap"] == pytest.approx(1.5, abs=0.01)
    assert "prefill" not in rb.windows(times, arrive=10.0, end=40.0, span=5.0)


def test_a_load_row_keeps_as_token_times_and_the_prefill_window(fake):
    row = rb.run_load(engine(fake), [3000], seed=1, span=0.2)[0]
    assert "prefill" in row["a"] and row["a_times"] and row["a_times"] == sorted(row["a_times"])


def test_the_load_table_shows_the_prefill_window_and_a_dash_for_older_runs():
    def row(prefill):
        a = {k: {"rate": 40.0, "gap": 0.05} for k in ("before", "arrival", "after_end")}
        return {"b_tokens": 1000, "b": {"ttft": 3.0}, "b_close_s": 0.05, "a": {**a, **prefill}}

    results = {"label": "x", "settings": {"sampling": rb.SAMPLING}, "resume": [], "notes": [],
               "load": [row({"prefill": {"rate": 12.5, "gap": 0.4}}), row({})]}
    md = rb.render([results])
    assert "A while B prefills" in md and "| 12.5 (0.4 s) |" in md and "| – |" in md


def test_the_report_is_markdown_tables(tmp_path, fake):
    results = {"label": "new build", "started": "2026-10-08 10:00:00", "engine": {"model": "glm-5.3-flash"},
               "settings": {"sampling": rb.SAMPLING, "sizes": [3000]},
               "node_info": [{"when": "start", "info": {"nodes": [{"name": "spark 1", "gpu clock": "2418 MHz"}]}}],
               "boot": {"cold_kernels": True, "starts": [{"hook_s": 200.0, "ready_s": 230.5},
                                                         {"hook_s": 100.0, "ready_s": 110.0}]},
               "resume": [{"size": 3000, "cold": {"prompt_tokens": 2950, "ttft": 2.0, "cached_tokens": 0,
                                                  "path": "render", "written": True},
                           "warm": {"prompt_tokens": 3100, "ttft": 0.2, "cached_tokens": 2900, "path": "record"},
                           "ssd": {"prompt_tokens": 3200, "ttft": 0.5, "cached_tokens": 3000, "path": "record"},
                           "fresh": {"prompt_tokens": 3200, "ttft": 2.1, "cached_tokens": 0, "path": "record"},
                           "ssd_share": 0.94, "same_reply": True}],
               "load": [], "notes": ["one note"]}
    md = rb.render([results])
    assert md.startswith("# ") and "| spark 1 |" in md and "2418 MHz" in md
    assert "| 3,200 |" in md and "94%" in md and "| yes |" in md and "230.5 s" in md and "one note" in md
    for line in md.splitlines():
        if line.startswith("|"):
            assert line.endswith("|")


def test_a_run_from_the_command_line_writes_both_results(tmp_path, fake, monkeypatch):
    def policy(body):
        turn = sum(m["role"] == "assistant" for m in body["messages"]) + 1
        return {"prompt_tokens": 1000 * turn, "cached": 0 if turn == 1 or body.get("draft") is False else 950 * turn,
                "sha": f"t{turn}", "path": "render" if turn == 1 else "record"}

    fake.policy = policy
    hooks(tmp_path, "restart", "clear-buffers", "node-info")
    monkeypatch.setattr(rb, "API_URL", fake.url)
    out = tmp_path / "out"
    code = rb.main(["--hooks", str(tmp_path / "hooks"), "--sizes", "3000", "--skip", "boot,load,pictures,agent,tools",
                    "--out", str(out), "--label", "test build"])
    md = (out / "results.md").read_text()
    run = json.loads((out / "results.json").read_text())
    assert code == 0 and run["resume"][0]["same_reply"]
    assert [n["when"] for n in run["node_info"]] == ["before the tests", "end"]
    assert "## test build" in md and "| yes |" in md and run["notes"] == []


def test_the_control_fails_when_a_fresh_read_of_turn_1_differs(tmp_path, fake):
    fake.policy = lambda body: {"sha": "serial" if body.get("draft") is False else "drafted"}
    rows = rb.run_resume(engine(fake), hooks(tmp_path, "restart", "clear-buffers"), [3000], seed=1)
    assert rows[0]["control_same"] is False
    assert any("control failed" in n for n in rb.notes_for({"resume": rows}))


def test_the_first_prompt_leaves_room_for_three_turns_in_the_window():
    assert rb.room(1048576, 1048576, 8192, 3) == 1048576 - 3 * 8192 - 4096
    assert rb.room(8192, 1048576, 8192, 3) == 8192


def test_a_hung_hook_times_out(tmp_path, monkeypatch):
    folder = tmp_path / "hooks"
    folder.mkdir()
    (folder / "clear-buffers").write_text("#!/bin/sh\nsleep 5\n")
    (folder / "clear-buffers").chmod(0o755)
    monkeypatch.setitem(rb.HOOK_TIMEOUT, "clear-buffers", 0.5)
    with pytest.raises(rb.HookError, match="took more than"):
        rb.Hooks(folder).run("clear-buffers")


def test_the_long_request_is_built_before_the_stream_it_disturbs_starts(fake):
    """Building B (its /tokenize calls: seconds at 1M tokens, on the server) while A streams would disturb A in the
    very window that is A's undisturbed baseline."""

    rb.run_load(engine(fake), [3000], seed=1, span=0.2)
    first_chat = fake.log.index("/v1/chat/completions")
    assert "/tokenize" in fake.log and "/tokenize" not in fake.log[first_chat:]


# -- pictures, and the engine's own stall meter (9008-9015) ------------------------------------------------------------
def test_a_screenshot_is_a_real_png_of_a_real_screenshots_weight():
    import io

    from PIL import Image

    a, b = rb.screenshot(1), rb.screenshot(2)
    im = Image.open(io.BytesIO(a))
    assert im.size == (1920, 1080) and im.mode == "RGB" and a == rb.screenshot(1) and a != b
    assert 200_000 < len(a) < 1_500_000


def picture_policy(body):
    """A server whose reply checksum sees the pictures: the turn, the picture count and the last picture's bytes."""

    turn = sum(m["role"] == "assistant" for m in body["messages"]) + 1
    pics = sum(1 for m in body["messages"] if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") == "image_url")
    last = [p for p in body["messages"][-1]["content"] if p.get("type") == "image_url"] if isinstance(
        body["messages"][-1].get("content"), list) else []
    sha = f"t{turn}-{pics}-{hash(last[0]['image_url']['url'][-200:]) if last else 0}"
    return {"prompt_tokens": 2000 * turn, "cached": 0 if turn == 1 or body.get("draft") is False else 1900 * turn,
            "sha": sha, "path": "render" if turn == 1 or body.get("draft") is False else "record"}


def test_the_picture_loop_resumes_every_turn_and_its_controls_hold(tmp_path, fake):
    fake.policy = picture_policy
    h = hooks(tmp_path, "restart", "clear-buffers")
    row = rb.run_pictures(engine(fake), h, seed=3, turns=4, max_tokens=64)
    assert [t["path"] for t in row["turns"]] == ["render", "record", "record", "record"]
    assert row["ssd"]["path"] == "record" and row["same_reply"] and row["control_same"] and row["control_pixels"]
    assert calls(tmp_path) == ["restart", "clear-buffers"]
    first = fake.bodies[0]["messages"][-1]["content"]
    assert first[0]["type"] == "image_url" and first[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert width_of(fake.bodies[-1]) == 3840 and row["largest"]["size"] == [3840, 2160]
    assert row["largest"]["prompt_tokens"] == 2000 and "error" not in row["largest"]


def width_of(body):
    """The width in the PNG header of a request's last picture (0 without one: the restart's one-token probe)."""

    import base64
    import struct

    content = body["messages"][-1].get("content")
    pics = [p for p in content if p.get("type") == "image_url"] if isinstance(content, list) else []
    if not pics:
        return 0
    return struct.unpack(">I", base64.b64decode(pics[-1]["image_url"]["url"][22:22 + 44])[16:20])[0]


def test_the_largest_picture_failing_is_recorded_and_flagged(tmp_path, fake):
    """At 8,000 tokens a picture, a 4K screenshot is the tower's largest input: if the engine fails it, the run goes on
    and says so."""

    fake.policy = lambda body: {"status": 500} if width_of(body) == 3840 else picture_policy(body)
    row = rb.run_pictures(engine(fake), hooks(tmp_path, "restart", "clear-buffers"), seed=3, turns=2, max_tokens=64)
    assert "500" in row["largest"]["error"] and row["same_reply"]
    run = picture_run(largest={"size": [3840, 2160], "error": row["largest"]["error"]})
    assert any("3840x2160" in n and "failed" in n for n in rb.notes_for(run))
    assert "**failed**" in rb.render([run])


def test_engine_side_gaps_land_in_the_windows_they_overlap():
    events = [{"gap_s": 0.3, "ended": 1010.2, "parts": {"admit": 0.25}},
              {"gap_s": 1.2, "ended": 1015.0, "parts": {"round": 1.1}},
              {"gap_s": 0.08, "ended": 1031.0, "parts": {"reply": 0.07}}]
    w = rb.engine_windows(events, arrive=1010.0, end=1030.0, span=5.0, first=1020.0)
    assert w["before"]["gap"] == 0.0 and w["arrival"]["gap"] == 1.2 and w["arrival"]["parts"] == {"round": 1.1}
    assert w["prefill"]["gap"] == 1.2 and w["after_end"]["gap"] == 0.08


def test_the_stall_meter_is_found_wherever_health_puts_it(fake):
    fake.health = {"ok": True, "engine": {"multi": {"stalls": {"gaps": 3, "recent": [{"gap_s": 0.5, "ended": 5.0}]}}}}
    assert engine(fake).stalls()["gaps"] == 3
    fake.health = {"ok": True}
    assert engine(fake).stalls() is None


def test_the_engines_clock_is_put_on_the_benchs_own(fake):
    """The engine stamps its gaps with its own clock; the bench runs elsewhere (a workstation's clock can be seconds
    off): the watch reads the engine's "now" and moves every gap onto the bench's clock."""

    ahead = 100.0
    gap_at = time.time() + 0.05
    fake.health = lambda: {"stalls": {"now": time.time() + ahead, "gaps": 1, "over": {"0.05": 1},
                                      "recent": [{"gap_s": 0.4, "ended": gap_at + ahead, "parts": {"admit": 0.4}}]}}
    watch = rb.StallWatch(engine(fake), every=0.05)
    time.sleep(0.15)
    (event,) = watch.done()
    assert event["ended"] == pytest.approx(gap_at, abs=0.05) and watch.offset == pytest.approx(ahead, abs=0.05)


def test_gaps_from_before_the_watch_started_are_left_out(fake):
    fake.health = lambda: {"stalls": {"now": time.time(), "gaps": 2, "over": {"0.05": 2},
                                      "recent": [{"gap_s": 0.4, "ended": time.time() - 60.0, "parts": {}}]}}
    watch = rb.StallWatch(engine(fake), every=0.05)
    time.sleep(0.1)
    assert watch.done() == []


def test_a_load_row_reads_the_engines_gaps_in_its_windows(fake):
    """B's prompt is read for 1 s; the engine reports a 0.9 s pause (admit) half-way through: it lands in B's prefill
    window, and the row counts the pauses by size over the whole row."""

    events = []

    def policy(body):
        if body.get("ignore_eos"):
            return {}
        events.append({"gap_s": 0.9, "ended": time.time() + 0.5, "parts": {"admit": 0.8, "other": 0.1}})
        return {"delay": 1.0}

    fake.policy = policy
    fake.health = lambda: {"stalls": {"now": time.time(), "gaps": len(events), "worst_s": 0.9,
                                      "over": {"0.05": len(events), "0.25": len(events), "1": 0},
                                      "recent": list(events)}}
    row = rb.run_load(engine(fake), [3000], seed=1, span=0.2)[0]
    e = row["engine"]
    assert e["prefill"]["gap"] == 0.9 and e["prefill"]["parts"] == {"admit": 0.8, "other": 0.1}
    assert e["before"]["gap"] == 0.0 and e["over"] == {"0.05": 1, "0.25": 1, "1": 0} and e["missed"] == 0


def test_without_a_stall_meter_a_load_row_says_so(fake):
    row = rb.run_load(engine(fake), [3000], seed=1, span=0.2)[0]
    assert row["engine"] is None


def test_a_picture_load_row_sends_screenshots_as_request_b(fake):
    fake.policy = lambda body: {} if body.get("ignore_eos") else {"prompt_tokens": 8123}
    (row,) = rb.run_load(engine(fake), [], seed=1, span=0.2, pictures=3)
    b = next(x for x in fake.bodies if not x.get("ignore_eos"))
    parts = b["messages"][-1]["content"]
    assert row["kind"] == "pictures" and row["pictures"] == 3 and row["b_tokens"] == 8123
    assert sum(p["type"] == "image_url" for p in parts) == 3 and "/tokenize" not in fake.log


def picture_run(**over):
    turns = [{"prompt_tokens": 2700 * k, "ttft": 9.0 if k == 1 else 1.5, "cached_tokens": 0 if k == 1 else 2650 * k,
              "path": "render" if k == 1 else "record", "written": True, "pictures": k} for k in range(1, 13)]
    row = {"turns": turns, "control_same": True, "control_pixels": True, "restart": {"hook_s": 90.0, "ready_s": 120.0},
           "ssd": {"prompt_tokens": 35100, "ttft": 2.4, "cached_tokens": 32400, "path": "record"},
           "fresh": {"prompt_tokens": 35100, "ttft": 61.0, "cached_tokens": 0, "path": "render"},
           "ssd_share": 0.9231, "same_reply": True, "body_mib": 9.1, **over}
    return {"label": "x", "settings": {"sampling": rb.SAMPLING}, "resume": [], "load": [], "notes": [],
            "pictures": row}


def test_the_report_shows_the_picture_loop_turn_by_turn():
    md = rb.render([picture_run(largest={"size": [3840, 2160], "prompt_tokens": 8012, "ttft": 4.25})])
    assert "### A screenshot conversation" in md and "| 12 |" in md and "| record |" in md
    assert "9.1 MiB" in md and "92%" in md and "61.0 s" in md and "another picture" in md
    assert "3840x2160" in md and "8,012" in md and "4.25 s" in md


def test_the_report_shows_the_engines_own_pauses():
    a = {k: {"rate": 40.0, "gap": 0.05} for k in ("before", "arrival", "prefill", "after_end")}
    e = {"before": {"gap": 0.0, "parts": {}, "gaps": 0}, "arrival": {"gap": 0.31, "parts": {"admit": 0.29}, "gaps": 1},
         "prefill": {"gap": 1.2, "parts": {"round": 1.1, "other": 0.1}, "gaps": 40},
         "after_end": {"gap": 0.08, "parts": {"reply": 0.07}, "gaps": 1},
         "over": {"0.05": 42, "0.1": 41, "0.25": 40, "0.5": 39, "1": 2}, "missed": 0}
    run = {"label": "x", "settings": {"sampling": rb.SAMPLING}, "resume": [], "notes": [],
           "load": [{"kind": "text", "b_tokens": 1000, "b": {"ttft": 3.0}, "b_close_s": 0.05, "a": a, "engine": e},
                    {"kind": "pictures", "pictures": 16, "b_tokens": 43000, "b": {"ttft": 30.0}, "b_close_s": 0.05,
                     "a": a, "engine": None}]}
    md = rb.render([run])
    assert "### Inside the engine" in md and "| < 0.05 s |" in md and "| 0.31 s (admit 0.29) |" in md
    assert "| 1.20 s (round 1.1) |" in md and "| 2 |" in md and "16 screenshots" in md


def test_the_notes_flag_a_picture_turn_that_missed_and_a_checksum_blind_to_pixels():
    run = picture_run(control_pixels=False)
    run["pictures"]["turns"][4]["path"] = "render"
    notes = rb.notes_for(run)
    assert any("turn 5" in n and "not served from a record" in n for n in notes)
    assert any("pixels" in n for n in notes)
    run = picture_run(same_reply=False)
    assert any("different replies" in n for n in rb.notes_for(run))


def test_a_run_from_the_command_line_runs_the_picture_loop(tmp_path, fake, monkeypatch):
    fake.policy = picture_policy
    hooks(tmp_path, "restart", "clear-buffers")
    monkeypatch.setattr(rb, "API_URL", fake.url)
    out = tmp_path / "out"
    code = rb.main(["--hooks", str(tmp_path / "hooks"), "--skip", "boot,resume,agent,tools,load", "--pictures", "3",
                    "--out", str(out)])
    run = json.loads((out / "results.json").read_text())
    assert code == 0 and len(run["pictures"]["turns"]) == 3 and run["pictures"]["same_reply"]
    assert "### A screenshot conversation" in (out / "results.md").read_text() and run["notes"] == []


# -- an agent's conversation: the material after the first reply ------------------------------------------------------
def agent_policy(body):
    turn = sum(m["role"] == "assistant" for m in body["messages"]) + 1
    fresh = body.get("draft") is False
    return {"prompt_tokens": 1000 * turn, "cached": 0 if turn <= 2 or fresh else 950 * turn, "sha": f"t{turn}",
            "path": "render" if turn <= 2 or fresh else "record"}


def test_the_agent_conversation_puts_the_material_after_the_first_reply(tmp_path, fake):
    """As tool results and pasted files do: a client that drops the thinking changes the conversation at the first
    reply, so without turn records a server re-reads everything after it."""

    fake.policy = agent_policy
    (r,) = rb.run_agent(engine(fake), hooks(tmp_path, "restart", "clear-buffers"), [3000], seed=1)
    chats = [b["messages"] for b in fake.bodies if b.get("stream")]
    assert len(chats[0]) == 2 and len(chats[0][1]["content"]) < 300                    # turn 1: a short exchange
    assert chats[1][2] == {"role": "assistant", "content": "an answer that is long enough"}   # thinking dropped
    assert len(chats[1][3]["content"]) > 2000 and r["context_tokens"] > 2000           # the material after it
    assert [r[k]["path"] for k in ("first", "cold", "warm", "ssd")] == ["render", "render", "record", "record"]
    assert r["same_reply"] and r["ssd_share"] == 0.95 and calls(tmp_path) == ["restart", "clear-buffers"]
    assert chats[-1] == chats[-2] and fake.bodies[-1].get("draft") is False            # the fresh read: same request


def test_the_report_and_notes_show_the_agent_conversation():
    def turn(path, ttft, prompt, cached):
        return {"path": path, "ttft": ttft, "prompt_tokens": prompt, "cached_tokens": cached}

    row = {"size": 262144, "context_tokens": 256000, "first": turn("render", 0.4, 60, 0),
           "cold": turn("render", 158.0, 256100, 0), "warm": turn("record", 1.3, 256400, 256000),
           "ssd": turn("record", 1.1, 256700, 256300), "fresh": turn("render", 157.0, 256700, 0),
           "ssd_share": 0.9984, "same_reply": True}
    run = {"label": "x", "settings": {"sampling": rb.SAMPLING}, "resume": [], "load": [], "notes": [], "agent": [row]}
    md = rb.render([run])
    assert "### An agent's conversation" in md and "| 256,000 |" in md and "158.0 s" in md and "| yes |" in md
    assert rb.notes_for(run) == []
    row["ssd"]["path"], row["same_reply"] = "render", False
    notes = rb.notes_for(run)
    assert any("turn 4" in n and "not served from a record" in n for n in notes)
    assert any("different replies" in n for n in notes)


def test_a_run_from_the_command_line_runs_the_agent_step(tmp_path, fake, monkeypatch):
    fake.policy = agent_policy
    hooks(tmp_path, "restart", "clear-buffers")
    monkeypatch.setattr(rb, "API_URL", fake.url)
    out = tmp_path / "out"
    code = rb.main(["--hooks", str(tmp_path / "hooks"), "--skip", "boot,resume,pictures,load,tools",
                    "--agent-sizes", "3000", "--out", str(out)])
    run = json.loads((out / "results.json").read_text())
    md = (out / "results.md").read_text()
    assert code == 0 and run["agent"][0]["same_reply"] and "### An agent's conversation" in md


# -- an agent's tool call: the material comes back as the tool's result ------------------------------------------------
def tool_policy(body):
    turn = sum(m["role"] == "assistant" for m in body["messages"]) + 1
    fresh = body.get("draft") is False
    return {"prompt_tokens": 1000 * turn, "cached": 0 if turn <= 2 or fresh else 950 * turn, "sha": f"t{turn}",
            "path": "render" if turn <= 2 or fresh else "record"}


def test_the_tool_conversation_sends_the_material_as_the_tools_result(tmp_path, fake):
    """Turn 1 must call the tool (tool_choice "required"); its call goes back with the thinking dropped, and the
    material comes back as the tool's result: the case where reasoning put back from RAM no longer matches after a
    restart. The tools stay the same on every turn (they are part of the prompt)."""

    fake.calls = [{"index": 0, "id": "call_1", "type": "function",
                   "function": {"name": "read_file", "arguments": "{\"path\": \"report.txt\"}"}}]
    fake.policy = tool_policy
    (r,) = rb.run_tools(engine(fake), hooks(tmp_path, "restart", "clear-buffers"), [3000], seed=1)
    chats = [b for b in fake.bodies if b.get("stream")]
    assert chats[0]["tool_choice"] == "required" and all("tool_choice" not in b for b in chats[1:])
    assert all(b["tools"] == chats[0]["tools"] for b in chats)
    call, result = chats[1]["messages"][2], chats[1]["messages"][3]
    assert call["role"] == "assistant" and call["tool_calls"][0]["id"] == "call_1" and "reasoning_content" not in call
    assert result == {"role": "tool", "tool_call_id": "call_1", "content": result["content"]}
    assert len(result["content"]) > 2000 and r["context_tokens"] > 2000
    assert [r[k]["path"] for k in ("cold", "warm", "ssd")] == ["render", "record", "record"] and r["same_reply"]


def test_a_tool_turn_without_a_call_is_reported_not_measured(tmp_path, fake):
    fake.policy = tool_policy                    # the fake streams no call: the model answered instead
    (r,) = rb.run_tools(engine(fake), hooks(tmp_path, "restart", "clear-buffers"), [3000], seed=1)
    assert r["error"] == "turn 1 made no tool call" and calls(tmp_path) == []
    assert any("no tool call" in n for n in rb.notes_for({"tools": [r]}))


def test_the_report_shows_the_tool_conversation():
    def turn(path, ttft, prompt, cached):
        return {"path": path, "ttft": ttft, "prompt_tokens": prompt, "cached_tokens": cached}

    row = {"size": 262144, "context_tokens": 256000, "first": turn("render", 0.4, 500, 0),
           "cold": turn("render", 158.0, 256600, 0), "warm": turn("record", 1.3, 256900, 256500),
           "ssd": turn("record", 1.2, 257100, 256800), "fresh": turn("render", 157.0, 257100, 0),
           "ssd_share": 0.9988, "same_reply": True}
    md = rb.render([{"label": "x", "settings": {"sampling": rb.SAMPLING}, "resume": [], "load": [], "notes": [],
                     "tools": [row]}])
    assert "### An agent's tool call after a restart" in md and "| 256,000 |" in md and "| yes |" in md


def test_the_report_shows_the_engines_settings_as_it_ran_and_every_nodes_state():
    """node-info runs before the boot (the engine may not run yet) and at the end: the settings table is the end's,
    when the running engine's own arguments and settings are there, and each moment's node table is shown."""

    node = {"name": "spark 1", "GPU clock (now / max)": "2411 / 3003 MHz", "CPU temperature": "42.3-44.3 C"}
    run = {"label": "x", "settings": {"sampling": rb.SAMPLING}, "resume": [], "load": [], "notes": [],
           "node_info": [{"when": "before the tests", "info": {"nodes": [node], "engine": {
                             "image": "i", "PORT": "8001", "--parallel": "6", "KV pool (tokens)": "2,185,216",
                             "TF_GLM_FILL_BUDGET_MS": "200"}}},
                         {"when": "end", "info": {"nodes": [node], "engine": {
                             "image": "i", "PORT": "8001", "--parallel": "6", "KV pool (tokens)": "2,185,216",
                             "TF_GLM_FILL_BUDGET_MS": "200"}}}]}
    md = rb.render([run])
    assert "| --parallel | 6 |" in md and "| KV pool (tokens) | 2,185,216 |" in md
    assert "| TF_GLM_FILL_BUDGET_MS | 200 |" in md
    assert md.count("| spark 1 |") == 2 and "42.3-44.3 C" in md


# -- the comparison: two runs (the recipe, then the recipe with the patches) side by side ------------------------------
def load(kind, n, b_tokens, gap, rate):
    """A load row: B (text of ``n`` tokens, or ``n`` screenshots) and A's longest wait and rate as B arrives."""

    return {"kind": kind, ("size" if kind == "text" else "pictures"): n, "b_tokens": b_tokens, "b": {"ttft": 30.0},
            "b_close_s": 0.05, "a": {"arrival": {"gap": gap, "rate": rate}}}


def a_run(label, image, k):
    """A run's results with every step, its numbers scaled by ``k`` (the patched run resumes, the recipe's misses)."""

    def t(path, ttft, prompt, cached):
        return {"path": path, "ttft": ttft, "prompt_tokens": prompt, "cached_tokens": cached}

    def four(cold, warm, ssd, fresh):
        return {"context_tokens": 256900, "cold": t("render", cold, 257000, 0),
                "warm": t("record", warm, 257300, 257000), "ssd": t("record", ssd, 257400, 257300),
                "fresh": t("render", fresh, 257400, 0), "ssd_share": 0.99, "same_reply": True}

    node = lambda n, clk: {"name": f"spark {n}", "GPU clock (now / max)": f"{clk} / 3003 MHz",  # noqa: E731
                           "GPU temperature": "41 C", "CPU clock": "2552-3923 MHz", "CPU temperature": "41.3-43.2 C",
                           "memory available": "10.3 GiB", "kernel": "6.17.0-1029-nvidia", "driver": "580.173.02"}
    engine = {"image": image, "--parallel": "6", "--context": "1048576", "KV pool (tokens)": "2,185,216",
              "TENSORFOLD_GLM_IMAGE_TOKENS": "2048", "GPU clocks locked (nvidia-smi -lgc)": "0,2200 MHz"}
    turns = [{"pictures": n, "ttft": 2.5 if (k == 1 or n < 3) else 13.5, "path": "record",
              "cached_tokens": 0 if (k != 1 and n >= 3) or n == 1 else 2048 * (n - 1), "prompt_tokens": 2100 * n}
             for n in (1, 2, 3)]
    return {"label": label, "settings": {"sampling": rb.SAMPLING, "seed": 20261008},
            "node_info": [{"when": "before the tests", "info": {"nodes": [node(1, 2190), node(2, 2177)],
                                                                 "engine": engine}}],
            "boot": {"cold_kernels": True, "starts": [{"hook_s": 270.0, "ready_s": 275.2 if k != 1 else 274.4},
                                                      {"hook_s": 110.0, "ready_s": 117.7 if k != 1 else 117.1}]},
            "tools": [four(157.9, 1.07, 158.6 if k != 1 else 0.95, 157.0)],
            "agent": [four(157.9, 0.87, 1.05 if k != 1 else 1.33, 157.0)],
            "pictures": {"turns": turns, "ssd": t("record", 14.26 if k != 1 else 3.13, 6400, 0 if k != 1 else 6300),
                         "fresh": t("render", 12.8, 6400, 0), "same_reply": True},
            "load": [load("text", 1048576, 1021571, 1.97 if k != 1 else 0.1, 13.4 if k != 1 else 25.4),
                     load("pictures", 16, 43133, 2.62 if k != 1 else 0.23, 8.7 if k != 1 else 17.0)],
            "resume": [], "notes": []}


def test_the_comparison_has_every_table_of_the_two_runs():
    md = rb.compare(a_run("recipe v1.10", "mia", 0), a_run("v1.10 + this fork", "ours", 1))
    assert "--seed 20261008" in md and "| GPU clock / temperature | 2190 MHz / 41 C |" in md
    assert "temperature 1.0, top_p 0.95, reasoning effort max" in md
    assert "| Agent with a tool call, cold (after a restart, from the SSD) | 158.6 s | **0.95 s** |" in md
    assert "| First read, nothing cached | 157.9 s | 157.9 s |" in md and "| Plain chat, hot | 0.87 s | 0.87 s |" in md
    assert "token for token the reply of a fresh read" in md
    assert "| 3 | 3 | **13.5 s** | **0** | 2.50 s | 4,096 |" in md
    assert "| 4, cold (after a restart) | 4 | 14.3 s | 0 | **3.13 s** | 6,300 |" in md
    assert "| 1,021,571 tokens of text | 1.97 s | 13.4 tok/s | **0.10 s** | 25.4 tok/s |" in md
    assert "| 16 new screenshots | 2.62 s | 8.7 tok/s | **0.23 s** | 17.0 tok/s |" in md
    assert "| Kernels compiled at this start | 275.2 s | 274.4 s |" in md
    assert "| Kernels cached | 117.7 s | 117.1 s |" in md
    for line in md.splitlines():
        if line.startswith("|"):
            assert line.endswith("|")


def test_the_report_command_prints_the_comparison_of_two_runs(tmp_path, capsys):
    paths = []
    for name, k in (("before", 0), ("after", 1)):
        paths.append(tmp_path / f"{name}.json")
        paths[-1].write_text(json.dumps(a_run(name, name, k)))
    assert rb.main(["report", str(paths[0]), str(paths[1])]) == 0
    out = capsys.readouterr().out
    assert out.startswith("## Measured on two DGX Sparks") and "## before" in out and "## after" in out
