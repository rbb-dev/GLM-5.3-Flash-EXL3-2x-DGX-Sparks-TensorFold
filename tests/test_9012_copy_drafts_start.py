"""Patch 9012 (a stream's copy drafter started from its prompt array), on the CPU: when a long prompt is in, both
ranks give the copy drafter the prompt and the first token; that took a list copy and a numpy conversion of the whole
prompt on the scheduler thread (about 30 ms at 1,000,000 tokens on an x86 core). Now rank 0 hands over the request
thread's array (9009) and rank 1 converts its list in one C pass."""

import random
import time
import types

import numpy as np

from tensorfold.families.glm5_next.cuda import multi as M

N = 1_000_000


def started(prompt, prompt_arr):
    d = object.__new__(M.MultiDecoder)
    d.g = types.SimpleNamespace(copy=types.SimpleNamespace(match=4, most=8, reply_match=0, miss_most=0))
    s = types.SimpleNamespace(prompt=prompt, count=100, started=None)
    if prompt_arr is not None:
        s.prompt_arr = prompt_arr
    lane = types.SimpleNamespace(s=s, decoding=False, policy=types.SimpleNamespace(next=lambda a, b: 3), copies=None,
                                 depth=0)
    times = []
    for _ in range(3):
        t = time.perf_counter()
        d._started(lane, 7)
        times.append(time.perf_counter() - t)
    return lane, min(times)


def test_the_copy_drafter_starts_from_the_prompt_array():
    prompt = [random.randrange(154_000) for _ in range(N)]
    for arr, limit in ((M.int_array(prompt), 0.012), (None, 0.022)):          # rank 0, then rank 1 (no array)
        lane, took = started(prompt, arr)
        assert lane.copies.tokens() == prompt + [7] and lane.decoding
        assert took < limit, f"starting the copy drafter took {took * 1000:.0f} ms"
