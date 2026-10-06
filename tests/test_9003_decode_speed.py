"""Patch 9003 (each finished request's decode time and decode rate on /metrics), on the CPU."""

import time
import types

from prometheus_client.parser import text_string_to_metric_families

from tensorfold.cuda import health
from tensorfold.server import metrics


def test_decode_rate():
    assert metrics.decode_rate(101, 2.0) == 50.0             # 100 tokens after the first, in 2 s
    assert metrics.decode_rate(33, 1.0) == 32.0              # the floor itself is observed
    assert metrics.decode_rate(32, 1.0) is None              # 31 decoded tokens: under RATE_MIN_TOKENS
    assert metrics.decode_rate(101, 0.0) is None and metrics.decode_rate(101, None) is None


def finished(app, tokens, decode_s):
    h = health.of(app)
    with h.running(prompt=1000, out=[]) as request:
        request.out.extend(range(tokens))
        request.first = request.started + 0.5
        request.stats = {"decode_s": decode_s}


def samples(text):
    return {(s.name, tuple(sorted(s.labels.items()))): s.value
            for f in text_string_to_metric_families(text) for s in f.samples}


def test_rendered_histograms():
    app = types.SimpleNamespace()
    finished(app, 101, 2.0)          # 50 tok/s, 2 s
    finished(app, 301, 10.0)         # 30 tok/s, 10 s
    finished(app, 10, 0.2)           # too short for a rate; its decode time still counts
    s = samples(metrics.render(app))
    rate, secs = "tensorfold:request_decode_tokens_per_second", "tensorfold:request_decode_seconds"
    assert s[(rate + "_count", ())] == 2 and s[(rate + "_sum", ())] == 80.0
    # (TensorFold writes an edge as 30, not 30.0)
    assert s[(rate + "_bucket", (("le", "30"),))] == 1 and s[(rate + "_bucket", (("le", "50"),))] == 2
    assert s[(secs + "_count", ())] == 3 and s[(secs + "_sum", ())] == 12.2
    assert s[(secs + "_bucket", (("le", "2.5"),))] == 2 and s[(secs + "_bucket", (("le", "10"),))] == 3


def test_other_histograms_keep_their_edges():
    app = types.SimpleNamespace()
    finished(app, 101, 2.0)
    text = metrics.render(app)
    assert 'tensorfold:request_latency_seconds_bucket{le="300"}' in text
    assert 'tensorfold:request_decode_tokens_per_second_bucket{le="300"}' in text
    assert 'tensorfold:time_to_first_token_seconds_bucket{le="0.01"}' in text


# Upstream's test for c9259c3 (TensorFold v0.6.5 tests/test_metrics.py,
# test_decode_seconds_come_from_the_engine_on_cuda_and_from_the_first_token_on_the_mac), unchanged but for one line:
# its assert on the vLLM-named mirror (request_decode_time_seconds), which 9003 does not take. The upstream test
# itself runs in scripts/run-tests.sh's backports suite and stops at that assert; this copy runs everything after it.
def sample(body: str, name: str) -> str:
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return line.split()[-1]
    raise AssertionError(f"{name} missing")


def bucket(body: str, name: str, le: str) -> str:
    return sample(body, f"{metrics.PREFIX}{name}_bucket{{le=\"{le}\"}}")


def test_upstream_decode_seconds_without_the_vllm_mirror():
    app = types.SimpleNamespace()
    out: list[int] = []
    with health.of(app).running(5, out) as request:
        out.extend([7, 8, 9])
        request.saw()
        request.stats = {"decode_s": 1.5}
    body = metrics.render(app)
    assert sample(body, f"{metrics.PREFIX}request_decode_seconds_sum") == "1.5"
    assert sample(body, f"{metrics.PREFIX}request_decode_seconds_count") == "1"
    assert bucket(body, "request_decode_seconds", "2.5") == "1"
    with health.of(app).running(5, []) as request:
        request.stats = {}
    assert sample(metrics.render(app), f"{metrics.PREFIX}request_decode_seconds_count") == "1", "no token, no decode"

    mac = types.SimpleNamespace()
    started = time.perf_counter()
    metrics.begin(mac, 6, started)
    metrics.tokens(3, started)
    metrics.finish_request()
    body = metrics.render(mac)
    assert sample(body, f"{metrics.PREFIX}request_decode_seconds_count") == "1"
    assert 0 <= float(sample(body, f"{metrics.PREFIX}request_decode_seconds_sum")) <= float(
        sample(body, f"{metrics.PREFIX}request_latency_seconds_sum"))
