"""The kit the turn-record tests share: a real GlmApp (the real GLM-5.3-Flash template and tokenizer, see
``glm53_files``) over a scripted engine whose ``multi`` behaves as the parallel decoder does for 9004/9005 (a request
keeps its own state at its grid point, names it in its turn holder, inherits the record its prompt continues, and a
reply the model ended holds that state for its record), and the requests the tests send."""

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


@dataclass
class Entry:
    """A stored prompt in the spill tier's index (``cuda.spill.Entry``): ids and plain fields in RAM."""

    key: tuple
    n: int
    name: str
    ids: Any = None
    fields: dict = field(default_factory=dict)
    tokens: Any = None               # 9015: the plain ids (the real tier reads them from the header when it loads)


class Disk:
    """The spill tier after a restart: its index, and each state's file holding a real header (``spill._header``)
    whose ``layers`` carry the state's fields as ``spill.encode`` writes them (the record's body included)."""

    def __init__(self, folder):
        self.dir, self.index, self.lock, self.generation = str(folder), {}, threading.RLock(), 0
        os.makedirs(self.dir, exist_ok=True)

    def store(self, state):
        from tensorfold.cuda import spill

        fields = {"turn_key": {"v": state.turn_key},
                  "turn": {"l": [{"v": b} for b in state.turn]} if state.turn else {"v": None}}
        layer = {"class": "tensorfold.families.glm5_next.cuda.decode:Snapshot", "fields": fields}
        keyed = state.key if state.key is not None else state.ids        # 9015: stored and matched by keyed ids
        key = spill.ids_key(keyed)
        name = spill.key_name(key, len(state.ids))
        header, _ = spill._header([], {"format": "x", "n": len(state.ids), "tokens": json.dumps(list(state.ids)),
                                       "layers": json.dumps([layer])})
        with open(os.path.join(self.dir, name + ".safetensors"), "wb") as f:
            f.write(header)
        self.index[key] = Entry(key, len(state.ids), name, np.asarray(keyed, dtype=np.int64),
                                spill._plain([layer]), tokens=list(state.ids))
        self.generation += 1


class Multi:
    """The parallel decoder's kept prompts and record hold, as 9004 drives them (``multi.MultiDecoder``); with a
    ``disk``, a prompt that extends a stored one longer than any kept one loads it first (``prestage``)."""

    def __init__(self, disk=None):
        self.kept, self.record_due, self.next_kid, self.disk = [], set(), 0, disk

    def _load(self, prompt):
        if self.disk is None:
            return
        from tensorfold.families.glm5_next.cuda.resume import read_body

        best = max((e for e in self.disk.index.values() if e.n < len(prompt) and list(e.ids) == prompt[:e.n]),
                   key=lambda e: e.n, default=None)
        if best is None or any(len(c.ids) >= best.n and prompt[:len(c.ids)] == (c.key or c.ids) for c in self.kept):
            return
        try:
            body = read_body(self.disk, best) if best.fields.get("turn_key") else None
        except OSError:                          # the tier: a state whose file cannot be read is not loaded
            return
        self.kept.append(Kept(list(best.tokens) if best.tokens is not None else [int(t) for t in best.ids],
                              self.next_kid, turn_key=best.fields.get("turn_key"), turn=body,
                              key=[int(t) for t in best.ids]))
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
        if isinstance(holder, dict):
            holder["kid"] = snap.kid

    def finish(self, holder, ended):
        kid = holder.get("kid") if isinstance(holder, dict) else None
        if ended and kid is not None and any(c.kid == kid for c in self.kept):
            self.record_due.add(kid)
            holder["due"] = True

    def release_due(self, kid):
        self.record_due.discard(kid)


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
        if self.multi is not None:
            self.multi.finish(holder, ended)
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
    """A clean stop and start: the kept states go to the spill tier's files as the shutdown flush writes them (each
    with whatever record it carries); the engine's RAM and 0036's memory are gone."""

    disk = engine.multi.disk or Disk(folder)
    for c in engine.multi.kept:
        disk.store(c)
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
