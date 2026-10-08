"""Patch 9009 (a long request's admission without Python-list round trips), on the CPU: rank 0's ADMIT of a
1,000,000-token prompt (``_emit``, ``_flush``: the sealed message and its wire buffer) and rank 1's reading of it
(``from_wire``, ``unseal``, ``parse``, ``_parse_admit``) each keep that rank's decode loop for at most 60 ms on an x86
core (before: about 250 ms and 140 ms), and rank 1 reads back exactly what rank 0 sent."""

import gc
import random
import time
import types

import numpy as np

from tensorfold.families.glm5_next.cuda import engine as E
from tensorfold.families.glm5_next.cuda import multi as M

N = 1_000_000


class Wire:
    """GlmEngine's transport, as both ranks' ``_share`` see it, without the all-gather (the tensors stay on the host)."""

    def __init__(self):
        self.tensor = None

    def _ring(self):
        pass

    def _share(self, values, *, want=True, as_array=False):
        self.tensor = E.to_wire(values)
        return E.from_wire(self.tensor, as_array) if want else None


def rank0():
    d = object.__new__(M.MultiDecoder)
    d.g, d.outbox, d.sent, d.idle, d.partial, d.short_round_due, d.fill_budget = Wire(), [], 0, False, None, False, 0
    d.tune = types.SimpleNamespace(async_msg=False)
    return d


def admit_parts(prompt):
    """An ADMIT as ``admit`` sends it: its head, the prompt (sent from the request thread's array), its tail."""

    head = [7, 1, 0, 1_048_576, -1, 0, 64, 1, 1, *M.pack_sampling(None), *([0] * M.SHARED_SLOTS), 0, 0, 0, 0,
            len(prompt)]
    return head, [0, 0, 0, *M.pack_keys([], [], [])]


def admit_payload(prompt):
    head, tail = admit_parts(prompt)
    return head + prompt + tail


def test_a_long_admission_is_sent_and_read_back_quickly_and_exactly():
    prompt = [random.randrange(154_000) for _ in range(N)]
    head, tail = admit_parts(prompt)
    payload, arr = head + prompt + tail, M.int_array(prompt)
    sent, read = [], []
    for _ in range(3):                           # the steady state: the first run pays fresh pages for the buffers
        d = rank0()
        t = time.perf_counter()
        d._emit(M.ADMIT, payload, trusted=True, bulk=(len(head), arr))
        d._flush()
        sent.append(time.perf_counter() - t)
        t = time.perf_counter()
        body = M.unseal(E.from_wire(d.g.tensor, True), 0)
        ((op, p),) = M.MultiDecoder.parse(body)
        a = M.MultiDecoder._parse_admit(p)
        read.append(time.perf_counter() - t)
        assert op == M.ADMIT and a["prompt"] == prompt and a["sid"] == 7 and a["size"] == 1_048_576
    assert min(sent) < 0.03, f"rank 0 spent {min(sent) * 1000:.0f} ms sending a 1M-token ADMIT"
    assert min(read) < 0.06, f"rank 1 spent {min(read) * 1000:.0f} ms reading it"


def test_a_message_rank_1_did_not_get_whole_is_still_refused():
    d = rank0()
    d._emit(M.ADMIT, admit_payload(list(range(1000))), trusted=True)
    d._flush()
    got = E.from_wire(d.g.tensor, True)
    got[50] += 1                                  # one value changed on the way
    try:
        M.unseal(got, 0)
    except RuntimeError as exc:
        assert "checksum differs" in str(exc)
    else:
        raise AssertionError("a changed message was accepted")


def test_small_messages_and_list_callers_work_as_before():
    d = rank0()
    d._emit(M.FILL, [3, 128])
    d._emit(M.TOUCH, [np.int64(5), True])         # normalized as ever: numpy ints and bools become ints
    d._flush()
    assert M.MultiDecoder.parse(M.unseal(E.from_wire(d.g.tensor), 0)) == [(M.FILL, [3, 128]), (M.TOUCH, [5, 1])]
    assert M.unseal(M.seal([1, 2, 3], 4), 4).tolist() == [1, 2, 3]


def test_the_request_thread_makes_the_prompt_array_in_short_pieces():
    """Each piece of the conversion keeps the GIL briefly (about 1 ms on an x86 core), so the engine's loop runs
    between them; one conversion of the whole list (25-50 ms) would show as one long piece. The pieces are timed
    between the list's slices (a thread waiting for the GIL would time the workstation's own load as well)."""

    class Timed(list):
        def __getitem__(self, k):
            self.marks.append(time.perf_counter())
            return super().__getitem__(k)

    prompt = Timed(random.randrange(154_000) for _ in range(N))
    longest = 1.0
    gc.disable()                                 # the conversion's own pieces only (9010 measures the collector's)
    try:
        for _ in range(3):
            prompt.marks = []
            arr = M.int_array(prompt)
            longest = min(longest, max(b - a for a, b in zip(prompt.marks, prompt.marks[1:])))
    finally:
        gc.enable()
    assert arr.dtype == np.int32 and arr.tolist() == list(prompt) and len(prompt.marks) > 8
    assert longest < 0.008, f"one piece of the conversion took {longest * 1000:.1f} ms"
