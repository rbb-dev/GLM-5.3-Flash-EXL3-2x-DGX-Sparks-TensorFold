"""Patch 9013 (a picture is laid out the same way whatever comes after it), on the CPU: its canvas, rows and key depend
only on its own pixels and the server's settings, never on how many other pictures or clips the request carries, so
a conversation keeps matching its saved state as screenshots accumulate; its key also names the vision code and, for a
clip, its frames' times; the frontend's shared caches are safe across request threads; and the limits follow the model
(no picture count cap; pictures share the request's size limit)."""

import base64
import threading

import pytest
import torch

from tensorfold.vision import glm as G


def test_a_pictures_cap_never_shrinks_as_pictures_accumulate():
    limits = G.GlmVisionLimits()
    assert {limits.picture_tokens(n) for n in (1, 8, 9, 50, 500, 5000)} == {limits.image_tokens}
    assert {limits.clip_tokens(c) for c in range(1, limits.max_videos + 1)} == {limits.video_tokens}


def test_the_models_own_picture_cap_is_a_setting(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_GLM_IMAGE_TOKENS", "8000")
    limits = G.GlmVisionLimits.from_env()
    assert limits.picture_tokens(9) == limits.picture_tokens(300) == 8000


def test_an_explicit_request_budget_still_shares_it(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS", "16384")     # the old behaviour, by choice
    assert G.GlmVisionLimits.from_env().picture_tokens(16) == 1024


def test_no_picture_count_cap_and_pictures_share_the_request_size_limit(monkeypatch):
    from tensorfold.cuda import http

    monkeypatch.setenv("TENSORFOLD_MAX_BODY_MIB", "200")
    v = object.__new__(G.GlmVision)
    v.limits = G.GlmVisionLimits.from_env()
    assert v.limits.max_images >= 100_000
    assert v.image_limits.max_total_encoded_bytes == http.max_body_bytes() == 200 << 20


class Handler:
    """A request handler as the API routes use one: its path, and the status of the response it writes."""

    def __init__(self, path):
        import io

        self.path, self.headers, self.close_connection, self.wfile, self.status = path, {}, False, io.BytesIO(), None

    def send_response(self, status):
        self.status = status

    def send_header(self, *args):
        pass

    def end_headers(self):
        pass


class App:
    """An app with nothing set: what the API routes touch before they read a body (Responses keeps a store per app)."""


def test_every_route_that_takes_pictures_reads_bodies_up_to_the_setting(monkeypatch):
    """TENSORFOLD_MAX_BODY_MIB is the size limit of the Anthropic Messages and Responses routes as of the chat route
    (they read their bodies with fixed limits of 96 and 32 MiB, so a larger setting reached the chat route only)."""

    from tensorfold.server import anthropic, request_body, responses
    from tensorfold.server.errors import RequestError

    seen = []

    def reader(handler, *, limit=request_body.LIMIT):
        seen.append(limit)
        raise RequestError("stopped by the test")

    monkeypatch.setenv("TENSORFOLD_MAX_BODY_MIB", "200")
    for module in (anthropic, responses):
        monkeypatch.setattr(module, "read_body", reader)
    for module, path in ((anthropic, "/v1/messages"), (responses, "/v1/responses")):
        h = Handler(path)
        module.post(h, App())
        assert h.status == 400
    assert seen == [200 << 20, 200 << 20]                       # Anthropic Messages, Responses
    assert request_body.max_body_bytes() == 200 << 20           # the one place the setting is read


def frontend():
    """The picture frontend without its GPU tower: ``prepare`` lays out, tokenizes and keys."""

    from tokenizers import Tokenizer

    from glm53_files import model_dir

    v = object.__new__(G.GlmVision)
    v.image_token, v.limits = 154854, G.GlmVisionLimits()
    v.tok = Tokenizer.from_file(str(model_dir() / "tokenizer.json"))
    return v


def test_a_pictures_key_names_its_pixels_and_the_vision_code(monkeypatch):
    v = frontend()
    canvas = torch.randint(0, 255, (3, 56, 56), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    other = canvas.clone()
    other[0, 0, 0] ^= 1                                       # one pixel
    p = v.prepare("<|user|>" + G.IMAGE_SPAN, [canvas])
    assert p.item_keys != v.prepare("<|user|>" + G.IMAGE_SPAN, [other]).item_keys
    assert len(p.item_idents) == 1 and len(p.item_idents[0]) == 32 and p.item_keys[0] == int.from_bytes(
        p.item_idents[0][:8], "big") >> 2
    monkeypatch.setattr(G, "VISION_ID", "another vision code")
    assert v.prepare("<|user|>" + G.IMAGE_SPAN, [canvas]).item_keys != p.item_keys


def test_a_clips_key_names_its_frames_times():
    v = frontend()
    frames = torch.zeros((2, 3, 56, 56), dtype=torch.uint8)
    a = v.prepare("<|user|>" + G.VIDEO_SPAN, [], videos=[G.GlmClip(frames, (0.0,))])
    b = v.prepare("<|user|>" + G.VIDEO_SPAN, [], videos=[G.GlmClip(frames, (1.5,))])
    assert a.item_keys != b.item_keys and a.item_seconds == [(0.0,)]


def test_a_layouts_identities_follow_it_through_a_continuation_and_a_resume():
    v = frontend()
    canvases = [torch.full((3, 56, 56), k, dtype=torch.uint8) for k in (1, 2, 3)]
    p = v.prepare("<|user|>" + G.IMAGE_SPAN * 3, canvases)
    assert p.continued(p.token_ids + [5]).item_idents == p.item_idents
    assert p.from_item(1).item_idents == p.item_idents[1:] and p.from_item(1).item_seconds == p.item_seconds[1:]


def png(size, colour):
    import io

    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", size, colour).save(b, "PNG")
    return b.getvalue()


def data_url(data):
    return "data:image/png;base64," + base64.b64encode(data).decode()


def cached_frontend():
    from collections import OrderedDict

    v = frontend()
    v.allow_urls, v._pictures, v._canvases = False, OrderedDict(), OrderedDict()
    return v


def test_a_picture_fitted_again_must_have_the_same_pixels():
    from tensorfold.vision.images import ImageInputError, ImageSource

    v = cached_frontend()
    (fp,) = v.load_images([ImageSource(data_url(png((64, 64), (10, 20, 30))))])
    fp.canvas = None
    v._canvases.clear()                                  # let go, as an old canvas is
    fp.data = png((64, 64), (10, 20, 31))                 # its bytes no longer give its pixels
    with pytest.raises(ImageInputError):
        v._canvas(fp)
    assert fp not in v._pictures.values()                 # it leaves the cache; a retry reads the picture fresh


def test_the_picture_cache_is_safe_across_request_threads(monkeypatch):
    import random

    from tensorfold.vision.images import ImageSource

    monkeypatch.setattr(G, "PICTURE_CACHE_MB", 1)          # evictions all the time
    monkeypatch.setattr(G, "PICTURE_CANVASES", 2)
    v = cached_frontend()
    sources = [ImageSource(data_url(png((300, 300), (k, 255 - k, 7)))) for k in range(40)]
    errors = []

    def requests(seed):
        rng = random.Random(seed)
        for _ in range(60):
            try:
                v.features_items = v.load_images(rng.sample(sources, 6))
                for x in v.features_items:
                    v._held(x)
            except Exception as exc:                      # noqa: BLE001
                errors.append(exc)

    import sys

    switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                           # threads change hands at every chance: races show
    try:
        threads = [threading.Thread(target=requests, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(300)
    finally:
        sys.setswitchinterval(switch)
    assert not errors, errors[:3]


def test_the_first_picture_after_start_does_not_hold_up_the_engine():
    import subprocess
    import sys

    script = '''
import io, threading, time
from PIL import Image
from tensorfold.vision import glm as G
from tensorfold.vision.images import _decode, DEFAULT_LIMITS
v = object.__new__(G.GlmVision); v.limits = G.GlmVisionLimits()
v.warm_fit()                                     # what start does, before the first request
b = io.BytesIO(); Image.new("RGB", (1920, 1080), (90, 40, 200)).save(b, "PNG")
d = _decode(b.getvalue(), "auto", DEFAULT_LIMITS, DEFAULT_LIMITS.max_total_pixels)
worst, stop = [0.0], [False]
def watch():
    last = time.perf_counter()
    while not stop[0]:
        time.sleep(0.001); now = time.perf_counter(); worst[0], last = max(worst[0], now - last - 0.001), now
th = threading.Thread(target=watch, daemon=True); th.start(); time.sleep(0.05); worst[0] = 0.0
v._picture(d, 2048)
stop[0] = True
print(worst[0])
'''
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    worst = float(out.stdout.split()[-1])
    assert worst < 0.03, f"the first picture's fit held other threads up {worst * 1000:.0f} ms"


def test_a_pictures_base64_is_decoded_without_holding_up_the_engine():
    import os

    from tensorfold.vision.images import _data_bytes

    blob = os.urandom(10 << 20)
    url = "data:image/png;base64," + base64.b64encode(blob).decode()
    worst, stop = [0.0], [False]

    def watch():
        import time

        last = time.perf_counter()
        while not stop[0]:
            time.sleep(0.001)
            now = time.perf_counter()
            worst[0], last = max(worst[0], now - last - 0.001), now

    import time

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    time.sleep(0.05)
    best = 1.0
    for _ in range(3):
        worst[0] = 0.0
        assert _data_bytes(url, 11 << 20) == blob
        best = min(best, worst[0])
    stop[0] = True
    assert best < 0.006, f"decoding a 10 MiB picture held other threads up {best * 1000:.1f} ms"


def test_pictures_too_big_for_the_window_at_full_size_fall_back_to_shared_sizes():
    from tensorfold.server.prompts import prepare_images

    v = cached_frontend()
    v.tool_media = False
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(png((1680, 1680),
                 (k, 9, 9)))}} for k in range(9)] + [{"type": "text", "text": "compare"}]}]
    render = lambda template: "<|user|>" + G.IMAGE_SPAN * 9 + "compare"          # noqa: E731
    full = prepare_images(v, messages, render).vision
    assert full.item_rows == [2025] * 9                                          # each at the 2,048 cap
    fallback = prepare_images(v, messages, render, context_limit=sum(full.item_rows) - 100).vision
    assert fallback.item_rows == [1764] * 9 and len(fallback.token_ids) < sum(full.item_rows) - 100   # shared caps
