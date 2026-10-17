#!/usr/bin/env python3
"""Resumable conversations, measured and checked: does a conversation come back from its saved state after a restart,
exactly and quickly, even when the client drops the model's thinking (most agent clients and gateways do)?

Usage: tools/resume_bench.py --hooks DIR [--sizes 8192,65536,262144] [--load-sizes 262144] [--out DIR] [--label L]
       tools/resume_bench.py report results.json [more.json ...] > results.md

API_URL / PORT as in client.py. Needs a server started with PARALLEL above 1 and the spill tier on (SPILL_GIB).

What it does, in order (each step can be left out with --skip boot,resume,agent,tools,pictures,load):
  1. node info (the node-info hook, when there is one);
  2. boot: clear-kernel-cache (when there is one), restart, then restart again: a cold-kernel start and a warm one;
  3. resume, per size: a new conversation of that many tokens (turn 1: cold), its next turn as a client that drops the
     thinking sends it (turn 2: the state in RAM); then restart and clear-buffers; then turn 3 (the saved state read
     back from the disk) and the same request again with "draft": false, which reads the same prompt fresh and keeps
     nothing. Turn 3 must give token for token the same reply as the fresh read (tensorfold.token_sha): the saved state,
     read back from the disk, is exactly what reading that prompt gives. (That the prompt is the true conversation,
     thinking included, is what tests/test_9005_resume.py proves with the real template.) A control first: turn 1
     drafted and read fresh must also agree, or a same-reply result could not be put down to the restore;
  3b. agent, per --agent-sizes: the material comes after the model's first reply, as tool results and pasted files
     do: turn 1 a short exchange, turn 2 the material (cold), turn 3 from memory; restart and clear-buffers; turn 4 from
     the disk and the same request read fresh. A client that drops the thinking changes the conversation at turn 1's
     reply: without turn records a server matches its saved state only up to there and reads the rest again, so this
     step shows the difference records make (run it on a build without them for the "before");
  3c. tools, per --tool-sizes: as 3b, but turn 1 must call a tool (tool_choice "required") and the material comes back
     as the tool's result. A server that puts a tool call's dropped reasoning back from RAM (the recipe's patch 0036)
     serves the true conversation until it restarts; after a restart that memory is empty, the conversation no longer
     matches its saved state from that call on, and everything after it is read again;
  4. pictures: a screenshot conversation, as an agent watching a screen keeps one (--pictures turns, default 12): each
     turn adds a new 1080p screenshot and sends the conversation back without the thinking, so each must be served
     from its record, past the 8th picture too; then a restart and clear-buffers, one more turn from the disk and the
     same request read fresh (the same reply). Controls: turn 1 drafted and read fresh agree, and turn 1 with another
     picture reads differently (the reply's checksum sees the pixels);
  5. load: a reply decoding (A) while a new long request (B) arrives: A's tokens a second and largest gap before B,
     at B's arrival, over B's whole wait for its first token and after B's end (A's token times are kept), and B's
     last token to its closing event (its record is made in between); then the same with a B of --load-pictures new
     screenshots. With the engine's stall meter (/health "stalls", TensorFold patch 9008), the engine's own pauses
     between decode rounds in the same windows, and what filled them: a client sees only what the 0.4 s stream
     smoothing lets through;
  6. node info again, then results.json and results.md in --out.

Hooks are executables in --hooks DIR, the only engine-specific part (keep yours out of this repo):
  restart              required: stop the engine if it runs, start it, exit 0 once its start command returned.
                       The bench then waits for a real one-token reply (never /health alone) and times both.
  clear-buffers        required: drop the OS page cache on every node (so a saved state is read from the disk).
  clear-kernel-cache   optional: remove the engine's compiled kernels on every node (for the cold-kernel start).
  node-info            optional: print one JSON object, {"nodes": [{"name": ..., ...any keys...}, ...]}, and may add
                       "engine": {...settings...}; it is shown as tables in the report.

Sampling is Z.ai's recommendation for GLM-5.3-Flash (temperature 1.0, top_p 0.95, reasoning effort max); the server
seeds sampling from the prompt, so the same prompt gives the same reply and the exactness check holds."""
from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
import random
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
from client import API_URL, open_url, prose  # noqa: E402

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "max"}
SIZES = (8192, 65536, 262144)
LOAD_SIZES = (262144,)
SYSTEM = "You are a careful assistant. Read the material you are given, then answer in three plain sentences."
# Each answer runs to a few sentences: a client that drops the thinking returns only the answer, and an answer of
# fewer than 64 characters cannot prove it came from this server (design §3.4), so its record is (rightly) not used.
TASKS = ("In three sentences, describe what the material above is about and how it begins.",
         "In three sentences, describe how the material ends.",
         "In three sentences, describe a passage from the middle of the material.")
LOAD_PROMPT = "Write a long, detailed story about a lighthouse keeper and the town below, chapter by chapter."
AGENT_OPENING = ("I am going to send you some material in my next message. For now, reply with one sentence saying you "
                 "are ready for it.")
READ_TOOL = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a text file and return its contents.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "the file's path"}},
                   "required": ["path"]}}}]
TOOL_ASK = ("Read the file report.txt with the read_file tool, then describe in three sentences what it is about and "
            "how it begins.")
PICTURE_TASKS = ("In three sentences, describe what is on this screen and what it is for.",
                 "In three sentences, say what changed on the screen since the last one and what it means.")
READY_TIMEOUT = float(os.environ.get("READY_TIMEOUT", "3600"))   # a cold-kernel start compiles for a while
HOOK_TIMEOUT = {"restart": 3600.0}                                   # seconds; any other hook: 900
CONTEXT = 1048576                                                    # the server's window (--context)


def room(size: int, context: int, reply: int, turns: int) -> int:
    """The largest first prompt, at most ``size``, that still leaves room for ``turns`` replies of ``reply`` tokens
    (each later turn carries the earlier replies) and a margin, inside the server's window."""

    return max(1024, min(size, context - turns * reply - 4096))


def _png(width: int, height: int, raw: bytes) -> bytes:
    """An RGB PNG of ``raw`` (each row a filter byte, then 3 bytes a pixel), with the standard library alone."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    head = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def screenshot(seed: int, width: int = 1920, height: int = 1080) -> bytes:
    """A deterministic screenshot-like PNG: flat window panels, rows of text-like stripes and one photo-like region of
    noise, so it weighs what a real screenshot does (about 0.3-1 MB) and two seeds never give the same pixels."""

    rng = random.Random(seed)
    stride = 1 + 3 * width
    img = bytearray(b"\x00" + bytes([rng.randrange(200, 256)] * 3) * width) * height

    def rect(x0: int, y0: int, x1: int, y1: int, rgb: list[int]) -> None:
        line = bytes(rgb) * (x1 - x0)
        for y in range(y0, y1):
            at = y * stride + 1 + 3 * x0
            img[at:at + len(line)] = line

    for _ in range(12):                                          # window panels
        x0, y0 = rng.randrange(width - 160), rng.randrange(height - 90)
        rect(x0, y0, min(width, x0 + rng.randrange(150, 600)), min(height, y0 + rng.randrange(80, 400)),
             [rng.randrange(256) for _ in range(3)])
    for _ in range(160):                                         # text-like stripes
        x0, y0 = rng.randrange(width - 400), rng.randrange(height - 12)
        rect(x0, y0, x0 + rng.randrange(40, 400), y0 + rng.randrange(6, 12), [rng.randrange(80)] * 3)
    w, h = 480, 270                                              # a photo-like region: noise, as a picture compresses
    x0, y0 = rng.randrange(width - w), rng.randrange(height - h)
    for y in range(y0, y0 + h):
        at = y * stride + 1 + 3 * x0
        img[at:at + 3 * w] = rng.randbytes(3 * w)
    return _png(width, height, bytes(img))


def picture_part(data: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def _find(obj, key: str):
    """The value of ``key`` anywhere in a JSON object, or None."""

    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    for v in obj if isinstance(obj, list) else ():
        got = _find(v, key)
        if got is not None:
            return got
    return None


class HookError(RuntimeError):
    pass


def say(message: str) -> None:
    """A progress line, as it happens (a run takes a while: the operator should see each step)."""

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def told(label: str, r: "Reply") -> "Reply":
    say(f"  {label}: {r.prompt_tokens:,} prompt tokens ({r.cached_tokens:,} cached), TTFT {r.ttft or 0:.2f} s, "
        f"{r.completion_tokens:,} reply tokens in {r.seconds:.1f} s, path {r.path}, "
        f"{'record written' if r.written else f'no record ({r.not_written})'}, finish {r.finish}")
    return r


class Hooks:
    """The operator's executables in one folder: restart, clear-buffers, clear-kernel-cache, node-info."""

    def __init__(self, folder: str | os.PathLike) -> None:
        self.folder = Path(folder)

    def _path(self, name: str) -> Path | None:
        for candidate in (self.folder / name, self.folder / f"{name}.sh"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        return None

    def has(self, name: str) -> bool:
        return self._path(name) is not None

    def run(self, name: str, *args: str) -> float:
        """Run a hook; its seconds. A failure stops the bench with the hook's own last lines."""

        path = self._path(name)
        if path is None:
            raise HookError(f"no executable '{name}' in {self.folder}")
        t0 = time.monotonic()
        limit = HOOK_TIMEOUT.get(name, 900.0)
        try:
            r = subprocess.run([str(path), *args], capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            raise HookError(f"hook {name} took more than {limit:.0f} s") from None
        if r.returncode:
            tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-5:])
            raise HookError(f"hook {name} exited {r.returncode}: {tail}")
        return time.monotonic() - t0

    def json(self, name: str, *args: str) -> dict | None:
        """A hook that prints one JSON object (node-info): the object, None without the hook."""

        path = self._path(name)
        if path is None:
            return None
        r = subprocess.run([str(path), *args], capture_output=True, text=True, timeout=HOOK_TIMEOUT.get(name, 900.0))
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return {"error": f"{name} printed no JSON (exit {r.returncode})"}

    def node_info(self) -> dict | None:
        return self.json("node-info")


@dataclasses.dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    calls: list = dataclasses.field(default_factory=list)
    finish: str | None = None
    ttft: float | None = None            # seconds to the first token, thinking or answer
    seconds: float = 0.0                 # to the closing event
    last_token: float | None = None      # seconds to the last token
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    token_sha: str | None = None
    path: str | None = None              # tensorfold.resume.path: "record" (served from a record) or "render"
    written: bool = False                # whether this reply's record was written
    not_written: str | None = None       # why not
    token_times: list = dataclasses.field(default_factory=list)

    def summary(self) -> dict:
        d = dataclasses.asdict(self)
        for k in ("content", "reasoning", "calls", "token_times"):
            d.pop(k)
        d["decode_tps"] = (round(self.completion_tokens / (self.last_token - self.ttft), 1)
                           if self.last_token and self.ttft is not None and self.last_token > self.ttft else None)
        return d


class Engine:
    """Streamed chat requests to an OpenAI-compatible TensorFold server, with the numbers the bench needs."""

    def __init__(self, url: str, model: str | None = None, *, sampling: dict | None = None,
                 timeout: float = 7200.0) -> None:
        self.url, self.timeout = url.rstrip("/"), timeout
        self.sampling = dict(SAMPLING if sampling is None else sampling)
        self.model = model                    # None: the server's first model, asked once it answers (``main``)

    def _get(self, path: str) -> dict:
        with open_url(urllib.request.Request(self.url + path), self.timeout) as r:
            return json.load(r)

    def _post(self, path: str, body: dict, timeout: float | None = None):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def models(self) -> list[str]:
        return [m["id"] for m in self._get("/v1/models")["data"]]

    def stalls(self) -> dict | None:
        """The engine's own stall meter (TensorFold patch 9008: /health "stalls"), or None without one."""

        try:
            return _find(self._get("/health"), "stalls")
        except Exception:                        # noqa: BLE001  (a server without it, or a busy moment)
            return None

    def count(self, messages: list) -> int:
        with self._post("/tokenize", {"model": self.model, "messages": messages}) as r:
            return int(json.load(r)["count"])

    def chat(self, messages: list, *, max_tokens: int, draft: bool = True, extra: dict | None = None,
             on_open=None, stop: threading.Event | None = None) -> Reply:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
                **self.sampling, **(extra or {})}
        if not draft:
            body["draft"] = False
        r = Reply()
        t0 = time.monotonic()
        with self._post("/v1/chat/completions", body) as resp:
            if on_open is not None:
                on_open(t0)
            for raw in resp:
                if stop is not None and stop.is_set():
                    break
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                now = time.monotonic() - t0
                for choice in chunk.get("choices") or ():
                    delta = choice.get("delta") or {}
                    text = (delta.get("reasoning_content") or delta.get("reasoning") or "", delta.get("content") or "")
                    if text[0] or text[1] or delta.get("tool_calls"):
                        r.ttft = now if r.ttft is None else r.ttft
                        r.last_token = now
                        r.token_times.append(now)
                    r.reasoning += text[0]
                    r.content += text[1]
                    for call in delta.get("tool_calls") or ():
                        r.calls.append(call)
                    r.finish = choice.get("finish_reason") or r.finish
                usage = chunk.get("usage")
                if usage:
                    r.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    r.completion_tokens = int(usage.get("completion_tokens") or 0)
                    r.cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                tf = chunk.get("tensorfold")
                if tf:
                    r.token_sha = tf.get("token_sha")
                    resume = tf.get("resume") or {}
                    r.path, r.written = resume.get("path"), bool(resume.get("written"))
                    r.not_written = resume.get("not_written")
        r.seconds = time.monotonic() - t0
        return r

    def ready(self, timeout: float = READY_TIMEOUT) -> float:
        """Seconds until the server gives a one-token reply (a started server can answer /health and still fail)."""

        t0 = time.monotonic()
        while True:
            try:
                body = {"model": self.model or "default", "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1}
                with self._post("/v1/chat/completions", body, timeout=120) as resp:
                    json.load(resp)
                return time.monotonic() - t0
            except Exception:                    # noqa: BLE001  (not up yet: refused, reset, 503)
                if time.monotonic() - t0 > timeout:
                    raise RuntimeError(f"the server gave no reply within {timeout:.0f} s of its start") from None
                time.sleep(5)


def dropped(r: Reply) -> dict:
    """The assistant turn as a client that drops the thinking sends it back: the answer and its calls only."""

    m: dict = {"role": "assistant", "content": r.content.strip() or None}
    if r.calls:
        m["tool_calls"] = r.calls
    return m


def conversation(engine: Engine, size: int, seed: int) -> tuple[list, int]:
    """A system message and one user message of about ``size`` tokens (never more), deterministic in the seed:
    (messages, tokens). A new seed is a new conversation that no earlier run left in any cache."""

    def build(words: int) -> list:
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prose(words, seed) + "\n\n" + TASKS[0]}]

    n = int(size * 0.95)
    messages = build(n)
    count = engine.count(messages)
    for _ in range(4):                       # prose(t) comes out at ~0.82 t: scale toward the size, staying under it
        if 0.95 * size <= count <= size:
            break
        n = max(16, int(n * size * 0.98 / max(count, 1)))
        messages = build(n)
        count = engine.count(messages)
    while count > size:
        n = int(n * 0.97)
        messages = build(n)
        count = engine.count(messages)
    return messages, count


def ready_after(engine: Engine, hooks: Hooks) -> dict:
    say("restart hook ...")
    hook_s = hooks.run("restart")
    say(f"  restart hook returned after {hook_s:.1f} s; waiting for a one-token reply ...")
    out = {"hook_s": round(hook_s, 1), "ready_s": round(hook_s + engine.ready(), 1)}
    say(f"  first reply {out['ready_s']:.1f} s after the restart began")
    return out


def run_boot(engine: Engine, hooks: Hooks) -> dict:
    """A cold-kernel start (when the clear-kernel-cache hook exists), then a warm one."""

    cold = hooks.has("clear-kernel-cache")
    say(f"boot: {'clearing the kernel cache, then ' if cold else ''}two starts")
    if cold:
        hooks.run("clear-kernel-cache")
    return {"cold_kernels": cold, "starts": [ready_after(engine, hooks), ready_after(engine, hooks)]}


def restored(engine: Engine, messages: list, r: Reply) -> dict:
    """The reply's numbers, and how many more tokens the server read than today's render of what the client sent: the
    thinking a record put back (0 when the request took today's render)."""

    return {**r.summary(), "thinking_restored": r.prompt_tokens - engine.count(messages)}


def run_resume(engine: Engine, hooks: Hooks, sizes, *, seed: int, max_tokens: int = 8192,
               context: int = CONTEXT) -> list[dict]:
    """Per size: turn 1 cold, turn 2 warm; one restart and buffer drop for all sizes; then turn 3 from the disk and
    the same request read fresh. The first size also checks the instrument: turn 1 read fresh must match turn 1."""

    rows, threads = [], []
    for k, size in enumerate(sizes):
        first, tokens = conversation(engine, room(size, context, max_tokens, 3), seed * 1000003 + size)
        say(f"resume {size:,}: turn 1, a new {tokens:,}-token conversation (cold) ...")
        cold = told("turn 1 (cold)", engine.chat(first, max_tokens=max_tokens))
        row = {"size": size, "context_tokens": tokens, "cold": cold.summary()}
        if k == 0:
            say(f"resume {size:,}: control: turn 1 again with \"draft\": false (both cold reads must agree) ...")
            control = told("turn 1 read fresh", engine.chat(first, max_tokens=max_tokens, draft=False))
            row["control_same"] = bool(cold.token_sha) and cold.token_sha == control.token_sha
            say(f"resume {size:,}: control: drafted and fresh turn 1 agree: {'YES' if row['control_same'] else 'NO'}")
        second = first + [dropped(cold), {"role": "user", "content": TASKS[1]}]
        say(f"resume {size:,}: turn 2, thinking dropped by the client (warm: the state in memory) ...")
        warm = told("turn 2 (warm)", engine.chat(second, max_tokens=max_tokens))
        row["warm"] = restored(engine, second, warm)
        threads.append(second + [dropped(warm), {"role": "user", "content": TASKS[2]}])
        rows.append(row)
    rows_restart = ready_after(engine, hooks)
    say("clear-buffers hook (page cache dropped: saved states come back from the disk) ...")
    hooks.run("clear-buffers")
    for row, third in zip(rows, threads):
        say(f"resume {row['size']:,}: turn 3 after the restart (from the disk) ...")
        ssd = told("turn 3 (from disk)", engine.chat(third, max_tokens=max_tokens))
        say(f"resume {row['size']:,}: the same request with \"draft\": false (a fresh read of the same prompt) ...")
        fresh = told("turn 3 read fresh", engine.chat(third, max_tokens=max_tokens, draft=False))
        row["restart"] = rows_restart
        row["ssd"], row["fresh"] = restored(engine, third, ssd), fresh.summary()
        row["ssd_share"] = round(ssd.cached_tokens / ssd.prompt_tokens, 4) if ssd.prompt_tokens else None
        row["same_reply"] = bool(ssd.token_sha) and ssd.token_sha == fresh.token_sha
        say(f"resume {row['size']:,}: same reply as the fresh read: {'YES' if row['same_reply'] else 'NO'}; "
            f"{row['ssd_share'] or 0:.0%} of turn 3's prompt came from the saved state")
    return rows


def run_agent(engine: Engine, hooks: Hooks, sizes, *, seed: int, max_tokens: int = 8192,
              context: int = CONTEXT) -> list[dict]:
    """Per size: turn 1 a short exchange, turn 2 the material after the first reply (cold), turn 3 from memory; one
    restart and buffer drop for all sizes; then turn 4 from the disk and the same request read fresh."""

    rows, threads = [], []
    for size in sizes:
        material, tokens = conversation(engine, room(size, context, max_tokens, 4), seed * 1000033 + size)
        start = [material[0], {"role": "user", "content": AGENT_OPENING}]
        say(f"agent {size:,}: turn 1, a short exchange ...")
        first = told("turn 1", engine.chat(start, max_tokens=max_tokens))
        second = start + [dropped(first), material[1]]
        say(f"agent {size:,}: turn 2, the material ({tokens:,} tokens) after the first reply (cold) ...")
        cold = told("turn 2 (cold)", engine.chat(second, max_tokens=max_tokens))
        third = second + [dropped(cold), {"role": "user", "content": TASKS[1]}]
        say(f"agent {size:,}: turn 3, thinking dropped by the client (warm: the state in memory) ...")
        warm = told("turn 3 (warm)", engine.chat(third, max_tokens=max_tokens))
        rows.append({"size": size, "context_tokens": tokens, "first": first.summary(), "cold": cold.summary(),
                     "warm": warm.summary()})
        threads.append(third + [dropped(warm), {"role": "user", "content": TASKS[2]}])
    restart = ready_after(engine, hooks)
    say("clear-buffers hook (page cache dropped: saved states come back from the disk) ...")
    hooks.run("clear-buffers")
    for row, fourth in zip(rows, threads):
        say(f"agent {row['size']:,}: turn 4 after the restart (from the disk) ...")
        ssd = told("turn 4 (from disk)", engine.chat(fourth, max_tokens=max_tokens))
        say(f"agent {row['size']:,}: the same request with \"draft\": false (a fresh read of the same prompt) ...")
        fresh = told("turn 4 read fresh", engine.chat(fourth, max_tokens=max_tokens, draft=False))
        row["restart"], row["ssd"], row["fresh"] = restart, ssd.summary(), fresh.summary()
        row["ssd_share"] = round(ssd.cached_tokens / ssd.prompt_tokens, 4) if ssd.prompt_tokens else None
        row["same_reply"] = bool(ssd.token_sha) and ssd.token_sha == fresh.token_sha
        say(f"agent {row['size']:,}: same reply as the fresh read: {'YES' if row['same_reply'] else 'NO'}; "
            f"{row['ssd_share'] or 0:.0%} of turn 4's prompt came from the saved state")
    return rows


def run_tools(engine: Engine, hooks: Hooks, sizes, *, seed: int, max_tokens: int = 8192,
              context: int = CONTEXT) -> list[dict]:
    """Per size: turn 1 must call read_file; turn 2 sends the material back as the tool's result (cold); turn 3 from
    memory; one restart and buffer drop for all sizes; then turn 4 from the disk and the same request read fresh.
    Every turn lists the same tools (they are part of the prompt); only turn 1 requires a call."""

    rows, threads = [], []
    tools = {"tools": READ_TOOL}
    for size in sizes:
        material, tokens = conversation(engine, room(size, context, max_tokens, 4), seed * 1000037 + size)
        text = material[1]["content"].rsplit("\n\n", 1)[0]          # the prose, without the task
        start = [material[0], {"role": "user", "content": TOOL_ASK}]
        say(f"tools {size:,}: turn 1, the model must call read_file ...")
        first = told("turn 1 (tool call)", engine.chat(start, max_tokens=max_tokens,
                                                       extra={**tools, "tool_choice": "required"}))
        call_id = next((c.get("id") for c in first.calls if c.get("id")), None)
        if call_id is None:
            rows.append({"size": size, "context_tokens": tokens, "first": first.summary(),
                         "error": "turn 1 made no tool call"})
            say(f"tools {size:,}: turn 1 made no tool call: this size is not measured")
            continue
        second = start + [dropped(first), {"role": "tool", "tool_call_id": call_id, "content": text}]
        say(f"tools {size:,}: turn 2, the material ({tokens:,} tokens) as the tool's result (cold) ...")
        cold = told("turn 2 (cold)", engine.chat(second, max_tokens=max_tokens, extra=tools))
        third = second + [dropped(cold), {"role": "user", "content": TASKS[1]}]
        say(f"tools {size:,}: turn 3, thinking dropped by the client (warm: the state in memory) ...")
        warm = told("turn 3 (warm)", engine.chat(third, max_tokens=max_tokens, extra=tools))
        rows.append({"size": size, "context_tokens": tokens, "first": first.summary(), "cold": cold.summary(),
                     "warm": warm.summary(), "calls_after": [k for k, r in ((2, cold), (3, warm)) if r.calls]})
        threads.append((rows[-1], third + [dropped(warm), {"role": "user", "content": TASKS[2]}]))
    if not threads:
        return rows
    restart = ready_after(engine, hooks)
    say("clear-buffers hook (page cache dropped: saved states come back from the disk) ...")
    hooks.run("clear-buffers")
    for row, fourth in threads:
        say(f"tools {row['size']:,}: turn 4 after the restart (from the disk) ...")
        ssd = told("turn 4 (from disk)", engine.chat(fourth, max_tokens=max_tokens, extra=tools))
        say(f"tools {row['size']:,}: the same request with \"draft\": false (a fresh read of the same prompt) ...")
        fresh = told("turn 4 read fresh", engine.chat(fourth, max_tokens=max_tokens, draft=False, extra=tools))
        row["restart"], row["ssd"], row["fresh"] = restart, ssd.summary(), fresh.summary()
        row["ssd_share"] = round(ssd.cached_tokens / ssd.prompt_tokens, 4) if ssd.prompt_tokens else None
        row["same_reply"] = bool(ssd.token_sha) and ssd.token_sha == fresh.token_sha
        say(f"tools {row['size']:,}: same reply as the fresh read: {'YES' if row['same_reply'] else 'NO'}; "
            f"{row['ssd_share'] or 0:.0%} of turn 4's prompt came from the saved state")
    return rows


def windows(times: list[float], arrive: float, end: float, span: float = 10.0, first: float | None = None) -> dict:
    """A's tokens a second and largest gap in the ``span`` seconds before B arrives, after it arrives, and after it
    ends, and with ``first`` (B's first token) over B's whole wait: while its prompt is read (``times``: A's token
    arrival times, seconds)."""

    def window(lo: float, hi: float) -> dict:
        inside = [t for t in times if lo <= t < hi]
        earlier = [t for t in times if t < lo]
        edges = [earlier[-1] if earlier else lo] + inside + [hi]  # a gap counts from A's last token before the window
        return {"rate": round(len(inside) / (hi - lo), 2) if hi > lo else None,
                "gap": round(max(b - a for a, b in zip(edges, edges[1:])), 3)}

    out = {"before": window(arrive - span, arrive), "arrival": window(arrive, arrive + span),
           "after_end": window(end, end + span)}
    if first is not None and first > arrive:
        out["prefill"] = window(arrive, first)
    return out


def engine_windows(events: list[dict], arrive: float, end: float, span: float = 10.0,
                   first: float | None = None) -> dict:
    """The engine's own gaps (its stall meter's, wall clock) by window: a gap counts in the window its end falls in,
    as A's token gaps do; each window's worst gap, what filled it, and how many there were."""

    def window(lo: float, hi: float) -> dict:
        inside = [e for e in events if lo < e["ended"] <= hi]
        worst = max(inside, key=lambda e: e["gap_s"], default=None)
        return {"gap": worst["gap_s"] if worst else 0.0, "parts": dict(worst.get("parts") or {}) if worst else {},
                "gaps": len(inside)}

    out = {"before": window(arrive - span, arrive), "arrival": window(arrive, arrive + span),
           "after_end": window(end, end + span)}
    if first is not None and first > arrive:
        out["prefill"] = window(arrive, first)
    return out


class StallWatch:
    """The engine's stall meter, read every ``every`` seconds while a load test runs (it lists its last 32 gaps).
    The engine stamps a gap with its own clock; the bench may run on another machine, so the watch reads the engine's
    "now" once and moves every gap onto the bench's clock (``offset``: the engine's clock minus ours)."""

    def __init__(self, engine: Engine, every: float = 2.0) -> None:
        self.engine, self.every, self.events, self.stop = engine, every, {}, threading.Event()
        t0 = time.time()
        self.first = engine.stalls()
        t1 = time.time()
        self.found = self.first is not None
        self.offset = (self.first["now"] - (t0 + t1) / 2) if self.found and "now" in self.first else 0.0
        self.start = t0
        self.last = self.first
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def take(self) -> None:
        h = self.engine.stalls()
        if h is None:
            return
        self.last = h
        for e in h.get("recent") or ():
            ended = e.get("ended", 0.0) - self.offset
            if ended >= self.start:
                self.events[(e.get("ended"), e.get("gap_s"))] = {**e, "ended": ended}

    def run(self) -> None:
        while not self.stop.is_set():
            self.take()
            self.stop.wait(self.every)

    def done(self) -> list[dict]:
        self.stop.set()
        self.thread.join(30)
        self.take()
        return sorted(self.events.values(), key=lambda e: e["ended"])

    def counted(self) -> dict:
        """The meter's own counts of gaps at or past each size while the watch ran (exact: no list limit)."""

        before, after = (self.first or {}).get("over") or {}, (self.last or {}).get("over") or {}
        return {k: n - before.get(k, 0) for k, n in after.items()}


def run_pictures(engine: Engine, hooks: Hooks, *, seed: int, turns: int = 12, max_tokens: int = 8192,
                 largest: tuple[int, int] = (3840, 2160)) -> dict:
    """A screenshot conversation, as an agent watching a screen keeps one: a screenshot and a question (turn 1, cold),
    then ``turns`` - 1 more turns, each with a new 1080p screenshot and the thinking dropped (each must be served from
    a record, past the 8th picture too); a restart and a page-cache drop, one more turn from the disk, and the same
    request read fresh (the same reply). Controls: turn 1 drafted and read fresh agree; turn 1 with one other picture
    of the same size reads differently (the reply's checksum sees the pixels). Last, one ``largest`` screenshot (4K:
    laid out at the per-picture cap, the vision tower's largest input): a failure is recorded, and the run goes on."""

    def turn(k: int, text: str) -> dict:
        return {"role": "user", "content": [picture_part(screenshot(seed * 1000 + k)), {"type": "text", "text": text}]}

    messages = [{"role": "system", "content": SYSTEM}, turn(0, PICTURE_TASKS[0])]
    say("pictures: turn 1, a screenshot and a question (cold) ...")
    first = told("picture turn 1 (cold)", engine.chat(messages, max_tokens=max_tokens))
    control = told("picture turn 1 read fresh", engine.chat(messages, max_tokens=max_tokens, draft=False))
    other = [messages[0], turn(1_000_000, PICTURE_TASKS[0])]
    swapped = told("picture turn 1, another picture, read fresh", engine.chat(other, max_tokens=max_tokens,
                                                                             draft=False))
    row = {"turns": [{**first.summary(), "pictures": 1}],
           "control_same": bool(first.token_sha) and first.token_sha == control.token_sha,
           "control_pixels": bool(swapped.token_sha) and swapped.token_sha != control.token_sha}
    say(f"pictures: controls: drafted and fresh agree: {'YES' if row['control_same'] else 'NO'}; another picture "
        f"reads differently: {'YES' if row['control_pixels'] else 'NO'}")
    last = first
    for k in range(1, turns):
        messages = messages + [dropped(last), turn(k, PICTURE_TASKS[1])]
        last = told(f"picture turn {k + 1} ({k + 1} screenshots)", engine.chat(messages, max_tokens=max_tokens))
        row["turns"].append({**last.summary(), "pictures": k + 1})
    row["restart"] = ready_after(engine, hooks)
    say("clear-buffers hook (page cache dropped) ...")
    hooks.run("clear-buffers")
    messages = messages + [dropped(last), turn(turns, PICTURE_TASKS[1])]
    ssd = told("picture turn after the restart (from disk)", engine.chat(messages, max_tokens=max_tokens))
    fresh = told("the same request read fresh", engine.chat(messages, max_tokens=max_tokens, draft=False))
    row["ssd"], row["fresh"] = ssd.summary(), fresh.summary()
    row["ssd_share"] = round(ssd.cached_tokens / ssd.prompt_tokens, 4) if ssd.prompt_tokens else None
    row["same_reply"] = bool(ssd.token_sha) and ssd.token_sha == fresh.token_sha
    row["body_mib"] = round(len(json.dumps(messages)) / 2 ** 20, 1)
    say(f"pictures: after the restart, same reply as the fresh read: {'YES' if row['same_reply'] else 'NO'}; "
        f"{row['ssd_share'] or 0:.0%} from the saved state; request body {row['body_mib']} MiB")
    w, h = largest
    say(f"pictures: the largest picture, one {w}x{h} screenshot (laid out at the per-picture cap) ...")
    big = [messages[0], {"role": "user", "content": [picture_part(screenshot(seed * 1000 + 999, w, h)),
                                                     {"type": "text", "text": PICTURE_TASKS[0]}]}]
    try:
        row["largest"] = {**told(f"one {w}x{h} screenshot", engine.chat(big, max_tokens=max_tokens)).summary(),
                          "size": [w, h]}
    except Exception as exc:                     # noqa: BLE001  (recorded: the report says what the engine answered)
        row["largest"] = {"size": [w, h], "error": f"{type(exc).__name__}: {exc}"}
        say(f"  the {w}x{h} screenshot failed: {row['largest']['error']}")
    return row


def load_row(engine: Engine, b_msgs: list, b_tokens: int | None, *, span: float, kind: str, label: str) -> dict:
    """A decodes; after ``span`` s, B arrives; A runs until ``span`` s after B's end. A's tokens in the windows, and
    (with the engine's stall meter) the engine's own pauses in the same windows."""

    stop, a_reply, a_t0 = threading.Event(), {}, {}
    watch = StallWatch(engine)

    def opened(t0: float) -> None:
        a_t0.setdefault("t0", t0)
        a_t0.setdefault("wall0", time.time() - (time.monotonic() - t0))   # the bench's wall clock at A's t0

    def stream_a() -> None:
        a_reply["r"] = engine.chat([{"role": "user", "content": LOAD_PROMPT}], max_tokens=60000,
                                   extra={"ignore_eos": True}, on_open=opened, stop=stop)

    a = threading.Thread(target=stream_a, daemon=True)
    a.start()
    while "t0" not in a_t0:
        time.sleep(0.05)
    time.sleep(span + 5)                      # A's thinking and first tokens are under way
    say(f"load {label}: reply A is decoding; request B arrives now ...")
    b_arrive = time.monotonic() - a_t0["t0"]
    b = told("B", engine.chat(b_msgs, max_tokens=2048))
    b_end = time.monotonic() - a_t0["t0"]
    time.sleep(span)
    stop.set()
    a.join(120)
    events = watch.done()
    times = (a_reply.get("r") or Reply()).token_times
    b_first = b_arrive + b.ttft if b.ttft is not None else None
    row = {"kind": kind, "b_tokens": b_tokens if b_tokens is not None else b.prompt_tokens, "b": b.summary(),
           "b_close_s": round(b.seconds - b.last_token, 3) if b.last_token is not None else None,
           "a": windows(times, b_arrive, b_end, span, b_first), "a_times": [round(t, 3) for t in times],
           "engine": None}
    if watch.found:
        w0 = a_t0["wall0"]
        counted = watch.counted()
        row["engine"] = {**engine_windows(events, w0 + b_arrive, w0 + b_end, span,
                                          w0 + b_first if b_first is not None else None),
                         "over": counted, "missed": max(0, counted.get("0.05", 0) - len(events)),
                         "offset_s": round(watch.offset, 3), "events": events}
    w, e = row["a"], row["engine"]
    p = w.get("prefill")
    reading = f"while B's prompt is read {p['rate']} ({p['gap']} s), " if p else ""
    say(f"load {label}: A tokens/s (largest gap) before {w['before']['rate']} ({w['before']['gap']} s), "
        f"as B arrives {w['arrival']['rate']} ({w['arrival']['gap']} s), {reading}after B ends "
        f"{w['after_end']['rate']} ({w['after_end']['gap']} s); B close {row['b_close_s']} s")
    if e:
        say(f"load {label}: the engine's own largest pause: " + ", ".join(
            f"{k} {e[k]['gap']} s" for k in ("before", "arrival", "prefill", "after_end") if k in e))
    return row


def run_load(engine: Engine, sizes, *, seed: int, span: float = 10.0, context: int = CONTEXT,
             pictures: int = 0) -> list[dict]:
    """Per size, a long text request B arrives while reply A decodes; then (``pictures``) one B that carries that
    many new 1080p screenshots (decoded, fitted and run through the vision tower as its prompt is read)."""

    rows = []
    for size in sizes:
        # B first: its /tokenize calls take seconds on the server at 1M tokens, and A's window before B is the baseline
        b_msgs, b_tokens = conversation(engine, room(size, context, 2048, 1), seed * 7919 + size)
        rows.append({"size": size, **load_row(engine, b_msgs, b_tokens, span=span, kind="text",
                                               label=f"{size:,}")})
    if pictures:
        shots = [picture_part(screenshot(seed * 7919 + 500_000 + k)) for k in range(pictures)]
        b_msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": shots + [
            {"type": "text", "text": "In three sentences, say what these screens have in common."}]}]
        rows.append({"pictures": pictures, **load_row(engine, b_msgs, None, span=span, kind="pictures",
                                                      label=f"{pictures} screenshots")})
    return rows


# -- the report --------------------------------------------------------------------------------------------------------
def _n(x) -> str:
    return "–" if x is None else f"{x:,}"


def _s(x) -> str:
    return "–" if x is None else (f"{x:.2f} s" if x < 10 else f"{x:.1f} s")


def _table(head: list[str], rows: list[list[str]], right: bool = False) -> list[str]:
    """A Markdown table; ``right``: every column but the first right-aligned (numbers, as the recipe's README shows)."""

    out = ["| " + " | ".join(head) + " |",
           "| " + " | ".join("---:" if right and i else "---" for i in range(len(head))) + " |"]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def _b(r: dict) -> str:
    """A load row's request B: its tokens, or its screenshots and tokens."""

    if r.get("kind") == "pictures":
        return f"{r.get('pictures')} screenshots ({_n(r.get('b_tokens'))} tokens)"
    return _n(r.get("b_tokens"))


def _pause(w: dict | None) -> str:
    """An engine window's largest pause and its largest part (only pauses of 0.05 s or more are listed)."""

    if not w:
        return "–"
    if not w["gap"]:
        return "< 0.05 s"
    parts = w.get("parts") or {}
    top = max(parts, key=parts.get) if parts else None
    return f"{w['gap']:.2f} s" + (f" ({top} {parts[top]:g})" if top else "")


def _four_turns(rows: list[dict]) -> list[str]:
    """The agent and tool steps' table: turn 2 cold, turn 3 warm, turn 4 from the disk and read fresh."""

    out = []
    for r in rows:
        c, w, s, f = r["cold"], r["warm"], r["ssd"], r["fresh"]
        out.append([_n(r["context_tokens"]), _s(c["ttft"]), _s(w["ttft"]), _n(s["prompt_tokens"]), _s(s["ttft"]),
                    "–" if r["ssd_share"] is None else f"{r['ssd_share']:.0%}", _s(f["ttft"]),
                    "yes" if r["same_reply"] else "**no**", " / ".join(str(x.get("path")) for x in (c, w, s))])
    return _table(["Material", "Turn 2, cold", "Turn 3, warm", "Turn 4 prompt", "Turn 4, from disk", "Read from disk",
                   "Turn 4 read fresh", "Same reply", "Path (2 / 3 / 4)"], out) + [""]


def render(runs: list[dict]) -> str:
    """results.md for one or more runs (e.g. a build with the patches and one without)."""

    md = ["# Resumable conversations: results", ""]
    for run in runs:
        md += [f"## {run.get('label') or 'run'} ({run.get('started', '')})", "",
               f"Model `{run.get('engine', {}).get('model')}`; sampling {json.dumps(run['settings']['sampling'])}.", ""]
        for info in run.get("node_info") or ():
            nodes = (info.get("info") or {}).get("nodes") or []
            if nodes:
                keys = [k for k in nodes[0] if k != "name"]
                md += [f"Nodes ({info['when']}):", ""]
                md += _table(["node"] + keys, [[str(n.get("name"))] + [str(n.get(k, "")) for k in keys]
                                               for n in nodes]) + [""]

        engines = [(info["when"], (info.get("info") or {}).get("engine")) for info in run.get("node_info") or ()]
        engines = [(when, eng) for when, eng in engines if eng]
        if engines:                                   # the moment with the most: the end, when the engine ran
            when, eng = max(engines, key=lambda we: (len(we[1]), we[0] == "end"))
            md += [f"Engine settings ({when} of the run):", ""]
            md += _table(["setting", "value"], [[str(k), str(v)] for k, v in eng.items()]) + [""]
        boot = run.get("boot")
        if boot:
            first = "cold kernels (compiled at this start)" if boot["cold_kernels"] else "warm kernels"
            md += ["### Engine start", "", "From the restart hook to the first one-token reply.", ""]
            md += _table(["Start", "Restart hook", "First reply"],
                         [[first, _s(boot["starts"][0]["hook_s"]), _s(boot["starts"][0]["ready_s"])],
                          ["warm kernels", _s(boot["starts"][1]["hook_s"]), _s(boot["starts"][1]["ready_s"])]]) + [""]
        if run.get("resume"):
            md += ["### Resume after a restart, for a client that drops the thinking", "",
                   "Turn 1 reads a new conversation (cold); turn 2 resumes it from memory (warm); after a restart "
                   "and a page-cache drop, turn 3 resumes from the saved state on the disk, and the same request with "
                   "`\"draft\": false` reads the same prompt fresh. Time to first token (TTFT) each time.", ""]
            rows = []
            for r in run["resume"]:
                c, w, s, f = r["cold"], r["warm"], r["ssd"], r["fresh"]
                rows.append([_n(c["prompt_tokens"]), _s(c["ttft"]), _s(w["ttft"]), _n(s["prompt_tokens"]),
                             _s(s["ttft"]), "–" if r["ssd_share"] is None else f"{r['ssd_share']:.0%}",
                             _n(s.get("thinking_restored")), _s(f["ttft"]), "yes" if r["same_reply"] else "**no**",
                             " / ".join(str(x.get("path")) for x in (c, w, s, f))])
            md += _table(["Turn 1 prompt", "Turn 1, cold", "Turn 2, warm", "Turn 3 prompt", "Turn 3, from disk",
                          "Read from disk", "Thinking put back (tokens)", "Turn 3 read fresh", "Same reply",
                          "Path (1 / 2 / 3 / fresh)"], rows) + [""]
            controls = [r["control_same"] for r in run["resume"] if "control_same" in r]
            if controls:
                md += ["Control: turn 1 drafted and turn 1 read fresh gave " + ("the same reply" if all(controls)
                       else "**different replies** (a same-reply result cannot be put down to the restore)") + ". "
                       "\"Same reply\" shows the saved state read back from the disk is exactly a fresh read of the "
                       "same prompt; that the prompt is the true conversation is proven by the CPU tests.", ""]
        if run.get("agent"):
            md += ["### An agent's conversation after a restart: the material after the first reply", "",
                   "Turn 1 is a short exchange, turn 2 brings the material (cold), turn 3 resumes it from memory; "
                   "after a restart and a page-cache drop, turn 4 resumes from the saved state on the disk, and the "
                   "same request with `\"draft\": false` reads it fresh. The client drops the thinking.", ""]
            md += _four_turns(run["agent"])
        if [r for r in run.get("tools") or () if "error" not in r]:
            md += ["### An agent's tool call after a restart: the material as the tool's result", "",
                   "Turn 1 must call a tool; turn 2 sends the material back as its result (cold), turn 3 resumes from "
                   "memory; after a restart and a page-cache drop, turn 4 resumes from the disk, and the same "
                   "request with `\"draft\": false` reads it fresh. The client drops the thinking, the tool call's "
                   "too: a server that put it back from RAM no longer can after a restart.", ""]
            md += _four_turns([r for r in run["tools"] if "error" not in r])
        pics = run.get("pictures")
        if pics:
            md += ["### A screenshot conversation, resumed every turn", "",
                   "An agent watching a screen: turn 1 sends a new 1080p screenshot and a question (cold); each later "
                   "turn adds a new screenshot and sends the conversation back without the thinking, so each must be "
                   "served from its record, past the 8th picture too. After a restart and a page-cache drop, the next "
                   "turn resumes from the saved state on the disk, and the same request with `\"draft\": false` "
                   "reads it fresh.", ""]
            rows = [[str(k), str(t.get("pictures")), _n(t.get("prompt_tokens")), _n(t.get("cached_tokens")),
                     _s(t.get("ttft")), str(t.get("path"))] for k, t in enumerate(pics["turns"], start=1)]
            last = len(pics["turns"]) + 1
            for name, r in ((f"{last}, after a restart (from disk)", pics["ssd"]), (f"{last}, read fresh",
                                                                                     pics["fresh"])):
                rows.append([name, str(last), _n(r.get("prompt_tokens")), _n(r.get("cached_tokens")),
                             _s(r.get("ttft")), str(r.get("path"))])
            md += _table(["Turn", "Screenshots", "Prompt", "From a saved state", "TTFT", "Served from"], rows) + [""]
            share = "–" if pics.get("ssd_share") is None else f"{pics['ssd_share']:.0%}"
            md += [f"After the restart, {share} of the prompt came from the saved state and the reply was "
                   + ("the same as the fresh read's" if pics["same_reply"] else "**not the same as the fresh read's**")
                   + ". Controls: turn 1 drafted and read fresh gave "
                   + ("the same reply" if pics.get("control_same") else "**different replies**")
                   + "; turn 1 with another picture of the same size gave "
                   + ("a different reply (the checksum sees the pixels)" if pics.get("control_pixels")
                      else "**the same checksum** (it does not see the pixels)")
                   + f". The last request's body: {pics.get('body_mib')} MiB.", ""]
            big = pics.get("largest")
            if big:
                size = "x".join(map(str, big["size"]))
                md += [f"The largest picture, one {size} screenshot (laid out at the per-picture cap): "
                       + (f"**failed**: {big['error']}." if "error" in big else
                          f"{_n(big.get('prompt_tokens'))} prompt tokens, TTFT {_s(big.get('ttft'))}."), ""]
        if run.get("load"):
            md += ["### Serving while a long request arrives", "",
                   "Reply A decodes while request B arrives: A's tokens a second (largest gap) in the 10 s before B, "
                   "in the 10 s after B arrives, over B's whole wait for its first token (while its prompt is read), "
                   "and in the 10 s after B ends; B's last token to its closing event.", ""]
            rows = []
            for r in run["load"]:
                a = r["a"]
                rows.append([_b(r), *(f"{a[k]['rate']} ({a[k]['gap']} s)" if k in a else "–" for k in
                                      ("before", "arrival", "prefill", "after_end")),
                             _s(r["b"]["ttft"]), _s(r["b_close_s"])])
            md += _table(["B prompt", "A before", "A as B arrives", "A while B prefills", "A after B ends", "B TTFT",
                          "B close"], rows) + [""]
            if any(r.get("engine") for r in run["load"]):
                md += ["### Inside the engine: pauses between decode rounds", "",
                       "The engine's stall meter (TensorFold patch 9008) times every pause between two decode rounds "
                       "while streams decode, and what filled it: admit (taking a request in), round (prompt chunks "
                       "read with the decode round), finish, reply, other. A client sees such a pause only when it "
                       "outlasts the 0.4 s stream smoothing. Largest pause in each window (its largest part), and the "
                       "pauses over the whole row.", ""]
                rows = []
                for r in run["load"]:
                    e = r.get("engine")
                    over = (e or {}).get("over") or {}
                    rows.append([_b(r), *(_pause(e.get(k)) if e else "–" for k in
                                          ("before", "arrival", "prefill", "after_end")),
                                 _n(over.get("0.25")) if e else "–", _n(over.get("1")) if e else "–"])
                md += _table(["B prompt", "Before B", "As B arrives", "While B prefills", "After B ends",
                              "Pauses ≥ 0.25 s", "Pauses ≥ 1 s"], rows) + [""]
        if run.get("notes"):
            md += ["### Notes", ""] + [f"- {n}" for n in run["notes"]] + [""]
    return "\n".join(md)


def _four_turn_notes(tag: str, r: dict) -> list[str]:
    """The agent and tool steps' flags: turn 4 from the disk must be a record's, the fresh read's reply, and whole."""

    out = []
    if r["ssd"].get("path") != "record":
        out.append(f"{tag}: turn 4 was not served from a record (path {r['ssd'].get('path')})")
    if not r["same_reply"]:
        out.append(f"{tag}: turn 4 from the disk and the fresh read gave different replies")
    if r["ssd_share"] is not None and r["ssd_share"] < 0.95:
        out.append(f"{tag}: only {r['ssd_share']:.0%} of turn 4's prompt came from the saved state")
    return out


def compare(before: dict, after: dict) -> str:
    """Two runs side by side, ``before`` (the recipe) and ``after`` (with the patches), measured with the same settings
    and seed: the setup, the nodes before the tests, and a table for each step both ran."""

    def snapshot(run: dict) -> tuple[list, dict]:
        infos = run.get("node_info") or []
        info = next((i for i in infos if i.get("when") == "before the tests"), infos[0] if infos else {})
        got = info.get("info") or {}
        return got.get("nodes") or [], got.get("engine") or {}

    def best(old, new, text_old: str, text_new: str) -> tuple[str, str]:
        """The patched number in bold where it is under half the recipe's."""

        return text_old, (f"**{text_new}**" if old and new is not None and new < old / 2 else text_new)

    lb, la = before.get("label") or "before", after.get("label") or "after"
    (bn, be), (an, ae) = snapshot(before), snapshot(after)
    sampling = (after.get("settings") or {}).get("sampling") or {}
    pool_b, pool_a, lock = be.get("KV pool (tokens)"), ae.get("KV pool (tokens)"), ae.get(
        "GPU clocks locked (nvidia-smi -lgc)")
    setup = [f"{ae['--parallel']} requests at once" if ae.get("--parallel") else None,
             f"a {int(ae['--context']):,}-token window" if ae.get("--context") else None,
             f"{int(ae['TENSORFOLD_GLM_IMAGE_TOKENS']):,} tokens a picture"
             if ae.get("TENSORFOLD_GLM_IMAGE_TOKENS") else None,
             f"GPU clocks locked at {lock}" if lock else None,
             (f"KV pool {pool_a} tokens" if pool_b == pool_a else f"KV pool {pool_b or '?'} and {pool_a or '?'} tokens")
             if pool_b or pool_a else None]
    md = ["## Measured on two DGX Sparks", "",
          f"Two DGX Sparks: {lb} (`{be.get('image', '?')}`) against {la} (`{ae.get('image', '?')}`), one after the "
          f"other, "
          f"with the same settings and the same conversations (`tools/resume_bench.py --seed "
          f"{(after.get('settings') or {}).get('seed')}`), from another machine on the network: "
          + "; ".join(x for x in setup if x) + ". Sampling: "
          + ", ".join(f"{k.replace('reasoning_effort', 'reasoning effort')} {v}" for k, v in sampling.items()) + ".",
          ""]
    if bn or an:
        cols = [(lb, n) for n in bn] + [(la, n) for n in an]

        def cell(n: dict, a: str, b: str) -> str:
            return f"{str(n.get(a, '?')).split(' / ')[0].replace(' MHz', '')} MHz / {n.get(b, '?')}" if a.startswith(
                "GPU") else f"{n.get(a, '?')} / {n.get(b, '?')}"

        md += ["**The Sparks**, before the tests:", ""]
        md += _table([""] + [f"{label}, {n.get('name')}" for label, n in cols], [
            ["GPU clock / temperature"] + [cell(n, "GPU clock (now / max)", "GPU temperature") for _, n in cols],
            ["CPU clock / temperature"] + [cell(n, "CPU clock", "CPU temperature") for _, n in cols],
            ["memory available"] + [str(n.get("memory available", "?")) for _, n in cols],
            ["driver / kernel"] + [cell(n, "driver", "kernel") for _, n in cols]]) + [""]
    rows, same = [], []
    for step, name in (("tools", "Agent with a tool call"), ("agent", "Plain chat")):
        b, a = (before.get(step) or [None])[0], (after.get(step) or [None])[0]
        if not (b and a) or "error" in b or "error" in a:
            continue
        if not rows:
            rows.append(["First read, nothing cached", _s(b["cold"]["ttft"]), _s(a["cold"]["ttft"])])
        first = len(rows) == 1                        # the first group says what hot and cold mean, once
        for when, key in (("hot" + (" (engine running)" if first else ""), "warm"),
                          ("cold" + (" (after a restart, from the SSD)" if first else ""), "ssd")):
            rows.append([f"{name}, {when}", *best(b[key]["ttft"], a[key]["ttft"], _s(b[key]["ttft"]),
                                                  _s(a[key]["ttft"]))])
        same += [b["same_reply"], a["same_reply"]]
    if rows:
        md += ["**Picking up a conversation** (the client sends the conversation back without the model's thinking, as "
               "agent clients and gateways do). Time to first token of the next turn:", ""]
        md += _table(["Next turn", lb, la], rows, right=True) + [""]
        md += [("Every resumed reply was token for token the reply of a fresh read." if all(same) else
                "**Not every resumed reply was the reply of a fresh read**: see the full reports."), ""]
    bp, ap = before.get("pictures"), after.get("pictures")
    if bp and ap:
        rows = []
        for n, (tb, ta) in enumerate(zip(bp["turns"], ap["turns"]), start=1):
            def two(t: dict) -> list[str]:
                miss = n > 1 and not t.get("cached_tokens")
                cells = [_s(t.get("ttft")), _n(t.get("cached_tokens"))]
                return [f"**{c}**" for c in cells] if miss else cells
            rows.append([str(n), str(ta.get("pictures", n)), *two(tb), *two(ta)])
        last = len(rows) + 1
        sb, sa = bp["ssd"], ap["ssd"]
        first = best(sb.get("ttft"), sa.get("ttft"), _s(sb.get("ttft")), _s(sa.get("ttft")))
        rows.append([f"{last}, cold (after a restart)", str(last), first[0], _n(sb.get("cached_tokens")), first[1],
                     _n(sa.get("cached_tokens"))])
        md += ["**A screenshot conversation** (a new 1080p screenshot every turn, the thinking dropped by the client). "
               "Time to first token, and the prompt tokens reused from the cache:", ""]
        md += _table(["Turn", "Screenshots", f"{lb}: first token", f"{lb}: reused", f"{la}: first token",
                      f"{la}: reused"], rows, right=True) + [""]
    rows = []
    for rb_ in before.get("load") or ():
        key = (rb_.get("kind", "text"), rb_.get("size") or rb_.get("pictures"))
        ra = next((r for r in after.get("load") or () if (r.get("kind", "text"), r.get("size") or r.get("pictures"))
                   == key), None)
        if ra is None:
            continue
        wb, wa = rb_["a"]["arrival"], ra["a"]["arrival"]
        label = (f"{ra['b_tokens']:,} tokens of text" if key[0] == "text" else f"{key[1]} new screenshots")
        gaps = best(wb["gap"], wa["gap"], f"{wb['gap']:.2f} s", f"{wa['gap']:.2f} s")
        rows.append([label, gaps[0], f"{wb['rate']} tok/s", gaps[1], f"{wa['rate']} tok/s"])
    if rows:
        md += ["**Other replies while a request arrives** (one reply streams while another request arrives; the "
               "streaming reply's longest wait between two tokens, and its rate, in the 10 s after the arrival):", ""]
        md += _table(["Request arriving", f"{lb}: longest wait", f"{lb}: rate", f"{la}: longest wait", f"{la}: rate"],
                     rows, right=True) + [""]
    bb, ab = before.get("boot"), after.get("boot")
    if bb and ab:
        md += ["**Engine start** (from the start command to the first one-token reply):", ""]
        md += _table(["Start", lb, la], [
            ["Kernels compiled at this start" if bb["cold_kernels"] else "First start",
             _s(bb["starts"][0]["ready_s"]), _s(ab["starts"][0]["ready_s"])],
            ["Kernels cached", _s(bb["starts"][1]["ready_s"]), _s(ab["starts"][1]["ready_s"])]], right=True) + [""]
    return "\n".join(md)


def notes_for(run: dict) -> list[str]:
    """Plain-language flags: anything a reader would check before trusting a row."""

    out = []
    for r in run.get("resume") or ():
        tag = f"{r['size']:,}-token conversation"
        if not r["cold"].get("written"):
            out.append(f"{tag}: turn 1 left no record ({r['cold'].get('not_written')}), so turns 2-3 had none to "
                       f"resume from")
        if r["ssd"].get("path") != "record":
            out.append(f"{tag}: turn 3 was not served from a record (path {r['ssd'].get('path')})")
        if not r["same_reply"]:
            out.append(f"{tag}: turn 3 from the disk and the fresh read gave different replies")
        if r.get("control_same") is False:
            out.append(f"{tag}: the control failed: turn 1 drafted and read fresh gave different replies")
        if r["ssd_share"] is not None and r["ssd_share"] < 0.95:
            out.append(f"{tag}: only {r['ssd_share']:.0%} of turn 3's prompt came from the saved state")
    for step, name in (("agent", "agent conversation"), ("tools", "tool conversation")):
        for r in run.get(step) or ():
            tag = f"{name}, {r['context_tokens']:,} tokens of material"
            if "error" in r:
                out.append(f"{tag}: {r['error']}, so it was not measured")
                continue
            for k in r.get("calls_after") or ():
                out.append(f"{tag}: turn {k} called a tool again")
            out += _four_turn_notes(tag, r)
    pics = run.get("pictures")
    if pics:
        tag = "screenshot conversation"
        for k, t in enumerate(pics["turns"], start=1):
            if k > 1 and t.get("path") != "record":
                out.append(f"{tag}: turn {k} was not served from a record (path {t.get('path')})")
            if not t.get("written"):
                out.append(f"{tag}: turn {k} left no record ({t.get('not_written')})")
        if pics["ssd"].get("path") != "record":
            out.append(f"{tag}: the turn after the restart was not served from a record "
                       f"(path {pics['ssd'].get('path')})")
        if not pics["same_reply"]:
            out.append(f"{tag}: the turn after the restart and the fresh read gave different replies")
        if pics.get("control_same") is False:
            out.append(f"{tag}: the control failed: turn 1 drafted and read fresh gave different replies")
        big = pics.get("largest") or {}
        if "error" in big:
            out.append(f"{tag}: the {'x'.join(map(str, big['size']))} screenshot failed: {big['error']}")
        if pics.get("control_pixels") is False:
            out.append(f"{tag}: turn 1 with another picture gave the same checksum: it does not see the pixels, so a "
                       f"same-reply result says nothing about the pictures")
    for r in run.get("load") or ():
        e = r.get("engine") or {}
        if e.get("missed"):
            out.append(f"load, B {_b(r)}: the watch missed {e['missed']} of the engine's pauses (more than 32 between "
                       f"two reads): the pause counts are exact, the windows may not be")
        if abs(e.get("offset_s") or 0.0) > 0.5:
            out.append(f"load, B {_b(r)}: the engine's clock is {e['offset_s']:+.1f} s from the bench's (corrected)")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["report"]:
        runs = [json.loads(Path(p).read_text()) for p in argv[1:]]
        if len(runs) == 2:                            # before and after: side by side first, then each run in full
            print(compare(runs[0], runs[1]) + "\n")
        print(render(runs))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hooks", required=True, help="folder of the restart, clear-buffers, ... executables")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--load-sizes", default=",".join(map(str, LOAD_SIZES)))
    ap.add_argument("--skip", default="",
                    help="comma list of steps to leave out: boot, resume, agent, tools, pictures, load")
    ap.add_argument("--agent-sizes", default="262144", help="material sizes of the agent step's conversations")
    ap.add_argument("--tool-sizes", default="262144", help="material sizes of the tool step's conversations")
    ap.add_argument("--pictures", type=int, default=12, help="turns of the screenshot conversation")
    ap.add_argument("--load-pictures", type=int, default=16,
                    help="screenshots in the load test's picture request (0 leaves that row out)")
    ap.add_argument("--seed", type=int, default=int(time.time()), help="new conversations every run (default: now)")
    ap.add_argument("--label", default="", help="this run's name in the report, e.g. the build")
    ap.add_argument("--out", type=Path, default=Path("resume-bench"))
    ap.add_argument("--temperature", type=float, default=SAMPLING["temperature"])
    ap.add_argument("--top-p", type=float, default=SAMPLING["top_p"])
    ap.add_argument("--effort", default=SAMPLING["reasoning_effort"])
    ap.add_argument("--max-tokens", type=int, default=8192, help="each turn's cap (a cut turn leaves no record)")
    ap.add_argument("--context", type=int, default=CONTEXT, help="the server's window: conversations fit inside it")
    a = ap.parse_args(argv)
    hooks = Hooks(a.hooks)
    for name in ("restart", "clear-buffers"):
        if not hooks.has(name):
            ap.error(f"--hooks {a.hooks}: no executable '{name}' (see the hooks in this script's help)")
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    sampling = {"temperature": a.temperature, "top_p": a.top_p, "reasoning_effort": a.effort}
    a.out.mkdir(parents=True, exist_ok=True)
    run = {"label": a.label, "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "settings": {
        "sampling": sampling, "sizes": a.sizes, "load_sizes": a.load_sizes, "seed": a.seed, "skip": sorted(skip),
        "max_tokens": a.max_tokens, "pictures": a.pictures, "load_pictures": a.load_pictures,
        "agent_sizes": a.agent_sizes, "tool_sizes": a.tool_sizes},
        "node_info": []}
    save = lambda: (a.out / "results.json").write_text(json.dumps(run, indent=1))   # noqa: E731
    save()

    def step(name: str, fn) -> None:
        run[name] = fn()
        save()

    engine = Engine(API_URL, os.environ.get("MODEL"), sampling=sampling)
    if "boot" not in skip:
        step("boot", lambda: run_boot(engine, hooks))
    engine.ready()
    engine.model = engine.model or engine.models()[0]
    run["engine"] = {"model": engine.model, "url": API_URL}
    run["node_info"].append({"when": "before the tests", "info": hooks.node_info()})   # the engine runs: its pool too
    save()
    if "resume" not in skip:
        step("resume", lambda: run_resume(engine, hooks, [int(x) for x in a.sizes.split(",") if x], seed=a.seed,
                                          max_tokens=a.max_tokens, context=a.context))
    if "agent" not in skip:
        step("agent", lambda: run_agent(engine, hooks, [int(x) for x in a.agent_sizes.split(",") if x], seed=a.seed,
                                        max_tokens=a.max_tokens, context=a.context))
    if "tools" not in skip:
        step("tools", lambda: run_tools(engine, hooks, [int(x) for x in a.tool_sizes.split(",") if x], seed=a.seed,
                                        max_tokens=a.max_tokens, context=a.context))
    if "pictures" not in skip and a.pictures > 0:
        step("pictures", lambda: run_pictures(engine, hooks, seed=a.seed, turns=a.pictures, max_tokens=a.max_tokens))
    if "load" not in skip:
        step("load", lambda: run_load(engine, [int(x) for x in a.load_sizes.split(",") if x], seed=a.seed,
                                      context=a.context, pictures=a.load_pictures))
    run["node_info"].append({"when": "end", "info": hooks.node_info()})
    run["notes"] = notes_for(run)
    save()
    (a.out / "results.md").write_text(render([run]))
    print(f"resume_bench: {a.out / 'results.md'}")
    return 1 if any("different replies" in n for n in run["notes"]) else 0


if __name__ == "__main__":
    sys.exit(main())
