"""Patch 9006 (prompts tokenized with Python's lock released), on the CPU: the ids are the tokenizer's own, and while a
long request is prepared (rendered, tokenized, labelled) another thread of the engine process, as the decode loop is,
keeps running. The Rust tokenizer's single-text ``encode`` holds the GIL for the whole text (measured: 981 ms for
433K tokens on an x86 workstation; preparing a 996,815-token request on a GB10 core, 2,211 ms before this patch and
73 ms after); its batch call releases it and returns the same ids."""

import random
import threading
import time
from types import SimpleNamespace

from glm53_testkit import SYSTEM, TOOLS, make  # noqa: F401
from tensorfold.cuda.unlocked import encode_ids

WORDS = "time year people way day man thing woman life child world school state family student group".split()


def prose(words: int) -> str:
    rng = random.Random(1)
    return " ".join(rng.choice(WORDS) + ("." if i % 12 == 11 else "") for i in range(words))


class Watch:
    """A thread that asks to run every millisecond: the longest it waited is what the decode loop would have waited."""

    def __init__(self):
        self.worst, self.stop = 0.0, False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        time.sleep(0.05)

    def run(self):
        last = time.perf_counter()
        while not self.stop:
            time.sleep(0.001)
            now = time.perf_counter()
            self.worst = max(self.worst, now - last - 0.001)
            last = now

    def take(self):
        worst, self.worst = self.worst, 0.0
        return worst


def test_the_ids_are_the_tokenizers_own(make):
    app, _ = make()
    for text in ("plain words", "<|user|>\nhi<|assistant|><think>ok</think>", "ünïcödé ✓ " * 50, prose(3000)):
        for special in (False, True):
            assert encode_ids(app.tok, text, special) == app.tok.encode(text, add_special_tokens=special).ids


class EncodeOnly:
    """A tokenizer with ``encode`` alone, as the stand-ins in TensorFold's own tests are (``TextTokenizer``)."""

    def __init__(self):
        self.asked = []

    def encode(self, text, **kwargs):
        self.asked.append(kwargs)
        return SimpleNamespace(ids=[ord(c) for c in text])


def test_a_tokenizer_with_encode_alone_still_gets_its_ids():
    tok = EncodeOnly()
    assert encode_ids(tok, "hi ✓", True) == [ord(c) for c in "hi ✓"]
    assert tok.asked == [{"add_special_tokens": True}]


def test_a_long_request_never_stalls_the_engines_other_threads(make):
    app, _ = make()
    body = {"model": "m", "messages": [SYSTEM, {"role": "user", "content": prose(400_000)}], "tools": TOOLS,
            "max_tokens": 64}
    watch = Watch()
    watch.take()
    prepared = app.prepare(body, True)
    stall = watch.take()
    watch.stop = True
    assert len(prepared.prompt) > 400_000
    assert stall < 0.25, f"another thread waited {stall * 1000:.0f} ms while a {len(prepared.prompt):,}-token request was prepared"


def test_a_long_picture_request_never_stalls_the_engines_other_threads():
    import torch
    from tokenizers import Tokenizer

    from glm53_files import model_dir
    from tensorfold.vision.glm import IMAGE_SPAN, GlmVision, GlmVisionLimits

    vision = object.__new__(GlmVision)          # the frontend without its GPU tower: prepare lays out and tokenizes
    vision.image_token, vision.limits = 154854, GlmVisionLimits()
    vision.tok = Tokenizer.from_file(str(model_dir() / "tokenizer.json"))
    watch = Watch()
    watch.take()
    prepared = vision.prepare("<|user|>" + IMAGE_SPAN + prose(400_000), [torch.zeros((3, 56, 56), dtype=torch.uint8)])
    stall = watch.take()
    watch.stop = True
    assert len(prepared.token_ids) > 400_000 and prepared.item_rows == [4]
    assert stall < 0.25, f"another thread waited {stall * 1000:.0f} ms while a picture request was laid out"
