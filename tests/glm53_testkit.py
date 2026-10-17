"""The kit the turn-record tests share: a real GlmApp (the real GLM-5.3-Flash template and tokenizer, see
``glm53_files``) over a scripted engine whose ``multi`` behaves as the parallel decoder does for 9004/9005 (a request
keeps its own state at its grid point, names it in its turn holder, inherits the record its prompt continues, and
hands a landed record to the spill tier), and the requests the tests send."""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import pytest

from glm53_files import model_dir
from tensorfold.families.glm5_next.cuda.app import GlmApp


GRID = 64


@dataclass
class Kept:
    """A kept prompt state as 9004 sees it (``decode.Snapshot``'s record fields, ``MultiDecoder``'s kid)."""

    ids: list
    kid: int
    shared: bool = False
    turn_key: str | None = None
    turn: list | None = None
    _turn_tail: list | None = None
    key: list | None = None          # 9015: keyed ids (a picture prompt's rows hold -1 - their key), as MultiDecoder
    spill_key: str | None = None     # the spill tier's point it was written to (``BlockStore.persist``)


@dataclass
class Entry:
    """A stored prompt in the spill tier's index (``cuda.spill_blocks.Point``): its key, ids and json fields in RAM."""

    key: str
    n: int
    ids: Any = None                  # the ids it is stored and matched by (keyed ids with pictures)
    fields: dict = field(default_factory=dict)
    tokens: Any = None               # 9015: the plain ids of a picture prompt (None for text)


class Disk:
    """The spill tier (patch 0110) as 9004/9005 use it: each state stored under its point key, a turn record riding on
    its point (``BlockStore.attach_record``: its label in the json fields, its bytes in ``<key>.rec`` with a CRC-32,
    ``read_record`` checking it); the index survives a restart, as the tier's scan keeps it."""

    def __init__(self, folder):
        self.dir, self.index, self.lock = str(folder), {}, threading.RLock()
        os.makedirs(self.dir, exist_ok=True)

    def record_path(self, key):
        return os.path.join(self.dir, key + ".rec")

    def store(self, state):
        from tensorfold.cuda.spill_blocks import point_key

        keyed = state.key if state.key is not None else state.ids        # 9015: stored and matched by keyed ids
        key = point_key(np.asarray(keyed, dtype=np.int64))
        with self.lock:
            self.index[key] = Entry(key, len(state.ids), np.asarray(keyed, dtype=np.int64), {},
                                    tokens=list(state.ids) if state.key is not None else None)
        state.spill_key = key
        return key

    def attach_record(self, key, label, payload):
        import zlib

        from tensorfold.cuda.spill_blocks import RECORD

        with self.lock:
            e = self.index.get(key)
        if e is None:
            return
        with open(self.record_path(key), "wb") as f:
            f.write(payload)
        e.fields = {**e.fields, RECORD: label, RECORD + "_crc": zlib.crc32(payload), RECORD + "_bytes": len(payload)}

    def read_record(self, key):
        import zlib

        from tensorfold.cuda.spill_blocks import RECORD

        with self.lock:
            e = self.index.get(key)
        f = e.fields if e is not None else {}
        if RECORD not in f:
            return None
        try:
            data = open(self.record_path(key), "rb").read()
        except OSError:
            return None
        return data if len(data) == f[RECORD + "_bytes"] and zlib.crc32(data) == f[RECORD + "_crc"] else None


class Multi:
    """The parallel decoder's kept prompts as 9004 drives them (``multi.MultiDecoder``): a landed record goes to the
    spill tier (``attach_record``; ``attached`` lists them); with a ``disk``, a prompt that extends a stored one longer
    than any kept one loads it first (``prestage``), its record with it."""

    def __init__(self, disk=None):
        self.kept, self.next_kid, self.disk, self.attached = [], 0, disk, []

    def attach_record(self, c):
        self.attached.append((c.kid, c.turn_key))
        if self.disk is not None and getattr(c, "spill_key", None) and c.turn_key and c.turn:
            self.disk.attach_record(c.spill_key, c.turn_key, c.turn[0].encode("utf-8"))

    def _load(self, prompt):
        if self.disk is None:
            return
        from tensorfold.families.glm5_next.cuda.resume import read_body

        best = max((e for e in self.disk.index.values() if e.n < len(prompt) and list(e.ids) == prompt[:e.n]),
                   key=lambda e: e.n, default=None)
        if best is None or any(len(c.ids) >= best.n and prompt[:len(c.ids)] == (c.key or c.ids) for c in self.kept):
            return
        from tensorfold.cuda.spill_blocks import RECORD

        try:                                     # (the tier reads a record beside its small state; none: no record)
            body = read_body(self.disk, best) if best.fields.get(RECORD) else None
        except OSError:
            body = None
        kept = Kept(list(best.tokens) if best.tokens is not None else [int(t) for t in best.ids], self.next_kid,
                    turn_key=best.fields.get(RECORD) if body is not None else None, turn=body,
                    key=[int(t) for t in best.ids])
        kept.spill_key = best.key
        self.kept.append(kept)
        self.next_kid += 1

    def keep(self, prompt, holder, vision=None):
        plain = list(prompt)
        prompt = vision.keyed_ids() if vision is not None else plain     # 9015: kept and matched by keyed ids
        self._load(prompt)
        hits = [c for c in self.kept if len(c.ids) < len(prompt) and prompt[:len(c.ids)] == (c.key or c.ids)]
        hit = max(hits, key=lambda c: len(c.ids), default=None)
        cut = len(hit.ids) if hit is not None else 0
        point = len(prompt) if len(prompt) % GRID == 0 else (len(prompt) // GRID * GRID if len(prompt) // GRID * GRID
                                                              > cut else None)
        if point is None:
            return
        snap = Kept(plain[:point], self.next_kid, key=list(prompt[:point]))
        self.next_kid += 1
        key = getattr(hit, "turn_key", None)
        if key:                                  # the inherit rule of MultiDecoder._turn_inherit
            base, tail = int(key.rsplit("|", 1)[1]), hit._turn_tail
            if tail is None:
                from tensorfold.families.glm5_next.cuda.turn_records import body_tail
                tail = body_tail(hit.turn)
            if base <= len(hit.ids) and prompt[base:base + len(tail)] == tail:
                snap._turn_tail, snap.turn, snap.turn_key = list(tail), hit.turn, key
        self.kept = [c for c in self.kept if (c.key or c.ids) != snap.key] + [snap]
        if self.disk is not None:                # the tier writes a state when it is kept (0110)...
            self.disk.store(snap)
            if snap.turn_key is not None:        # ...and the record it took from the state it resumed from (9004)
                self.attach_record(snap)
        if isinstance(holder, dict):
            holder["kid"] = snap.kid


class Engine:
    """Replies from a script: each ``generate`` decodes the next raw text, then its end token."""

    vision = None
    concurrent = True
    supports_logprobs = False

    def __init__(self, tok, limit=1_000_000):
        self.tok, self.limit = tok, limit
        self.eos = tuple(tok.token_to_id(t) for t in ("<|endoftext|>", "<|user|>", "<|observation|>"))
        self.request = threading.local()
        self.script, self.seen, self.sampling = [], [], []
        self.multi = Multi()

    def generate(self, ids, count, sampling, feed, stop_eos=True, draft=True, **extra):
        self.seen.append(list(ids))
        self.sampling.append(sampling)
        holder = getattr(self.request, "turn", None)
        if isinstance(holder, dict):
            holder["runs"] = holder.get("runs", 0) + 1
        if self.multi is not None:
            if extra.get("vision") is not None:               # 9015: a picture prompt is kept by its keyed ids
                self.multi.keep(list(ids), holder, extra["vision"])
            else:
                self.multi.keep(list(ids), holder)
        raw, end = self.script.pop(0)
        out = self.tok.encode(raw, add_special_tokens=False).ids[:count]
        ended = bool(end) and len(out) < count
        if ended:
            out.append(self.tok.token_to_id(end))
        feed(out)
        return {"cached": 0}


TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}}}}]
SYSTEM = {"role": "system", "content": "You are terse."}
ASK = {"role": "user", "content": "look at the repo"}
CALL_RAW = ("I will read a.py first; it is the entry point.</think>"
            "<tool_call>read<arg_key>path</arg_key><arg_value>a.py</arg_value></tool_call>")
TEXT_RAW = "The file prints one. Done reading.</think>a.py prints 1; nothing else in the repository needs a look."


@pytest.fixture
def make():
    folder = model_dir()

    def app(limit=1_000_000, disk=None, parallel=True, vision=False):
        from tokenizers import Tokenizer

        engine = Engine(Tokenizer.from_file(str(folder / "tokenizer.json")), limit)
        engine.multi = Multi(disk) if parallel else None      # PARALLEL=1: no parallel decoder
        if vision:
            engine.vision = picture_frontend(folder)
        return GlmApp(engine, folder, "glm-5.3-flash", default_thinking=True, sampling={"temperature": 0.0},
                      max_tokens=4096), engine
    return app


def ask(app, engine, messages, raw, end="<|user|>", **body):
    b = {"model": "glm-5.3-flash", "messages": messages, "max_tokens": 2048, "tools": TOOLS, **body}
    engine.script.append((raw, end))
    prepared = app.prepare(b, True)
    return prepared, app.run(b, True, lambda delta: True, prepared=prepared)


def sent(result, reasoning=None):
    """The reply as a client sends it back (reasoning dropped unless given)."""

    m = {"role": "assistant", "content": result["content"] or None}
    if result["calls"]:
        m["tool_calls"] = result["calls"]
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


def first_turn(app, engine):
    p1, r1 = ask(app, engine, [SYSTEM, ASK], CALL_RAW, end="<|observation|>")
    assert r1["finish"] == "tool_calls" and r1["stats"]["resume"]["written"]
    T1 = p1.prompt + r1["out"][:-1]
    return r1, T1, [SYSTEM, ASK, sent(r1), {"role": "tool", "tool_call_id": r1["calls"][0]["id"],
                                            "content": "print(1)"}]


ANTHROPIC_TOOL = {"name": "read", "description": "Read a file", "input_schema": TOOLS[0]["function"]["parameters"]}
RESPONSES_TOOL = {"type": "function", "name": "read", "description": "Read a file",
                  "parameters": TOOLS[0]["function"]["parameters"]}


def api_body(api, turns):
    """The chat body the server builds from an Anthropic or Responses request carrying ``turns``."""

    from tensorfold.server import anthropic_translate, responses_translate
    from tensorfold.server.responses import Store

    if api == "anthropic":
        return anthropic_translate.translate({"model": "m", "max_tokens": 2048, "system": "You are terse.",
                                              "thinking": {"type": "adaptive"}, "tools": [ANTHROPIC_TOOL],
                                              "messages": turns})
    return responses_translate.translate({"model": "m", "instructions": "You are terse.", "tools": [RESPONSES_TOOL],
                                          "input": turns, "chat_template_kwargs": {"enable_thinking": True}},
                                         Store()).chat


def restart(engine, make, folder, **kw):
    """A clean stop and start: the spill tier holds every kept state, each with the record that landed on it (the
    tier writes a state when it is kept and its record when it lands); the engine's RAM is gone."""

    disk = engine.multi.disk or Disk(folder)
    for c in engine.multi.kept:
        if getattr(c, "spill_key", None) is None or c.spill_key not in disk.index:
            disk.store(c)
        if c.turn_key and c.turn:
            disk.attach_record(c.spill_key, c.turn_key, c.turn[0].encode("utf-8"))
    return make(disk=disk, **kw)


def picture_frontend(folder):
    """9015: the real GLM picture frontend (decode, fit, layout, keys) without its GPU tower, as --vision serves it."""

    from collections import OrderedDict

    from tokenizers import Tokenizer

    from tensorfold.vision import glm as G

    v = object.__new__(G.GlmVision)
    v.image_token, v.limits, v.allow_urls = 154854, G.GlmVisionLimits(), False
    v._pictures, v._canvases = OrderedDict(), OrderedDict()
    v.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    return v


def png(size=(280, 168), colour=(40, 90, 160)):
    """A picture's PNG bytes: ``colour`` with a stripe, so two colours are two different pictures."""

    import io

    from PIL import Image, ImageDraw

    im = Image.new("RGB", size, colour)
    ImageDraw.Draw(im).rectangle([10, 10, size[0] // 2, 30], fill=(255 - colour[0], 0, colour[2]))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def picture_part(data):
    import base64

    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}
