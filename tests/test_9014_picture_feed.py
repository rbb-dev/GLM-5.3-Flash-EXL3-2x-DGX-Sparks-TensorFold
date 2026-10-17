"""Patch 9014 (rank 0 encodes a picture prompt's items as the prefill reaches them), on the CPU with a fake tower: the
rows each chunk gets are exactly the rows of encoding every item at once, but only the items the coming chunk covers are
encoded (between rounds, before the chunk goes out) and held, and the prefill lets go of the ones it passed, so a long
picture conversation never holds all its rows and the tower runs a chunk's pictures at a time; a failed encode fails
that stream alone before rank 1 hears of the chunk; picture streams can be handed back and replayed."""

import types

import pytest
import torch

from tensorfold.families.glm5_next.cuda import engine as E
from tensorfold.families.glm5_next.cuda import multi as M

WIDTH = 4


class Tower:
    def __init__(self):
        self.calls = []

    def features(self, prepared):
        """Item k's rows are all k (a fake tower), in item order."""

        self.calls.append([kind for kind, _ in prepared.items])
        return torch.cat([torch.full((rows, WIDTH), float(x.tag), dtype=torch.bfloat16)
                          for (kind, x), rows in zip(prepared.items, prepared.item_rows)])


def fake_engine(tower):
    comm = types.SimpleNamespace(all_gather=lambda mine, both: both[:mine.shape[0]].copy_(mine))
    return types.SimpleNamespace(torch=torch, world=1, comm=comm, vision=tower, feed_device="cpu",
                                 w=types.SimpleNamespace(cfg=types.SimpleNamespace(hidden=WIDTH)))


def prepared(rows_each, gap=3):
    """Items of ``rows_each`` rows, each starting ``gap`` tokens after the last one ended."""

    positions, items, at = [], [], 0
    for k, rows in enumerate(rows_each):
        at += gap
        positions += list(range(at, at + rows))
        items.append(("image", types.SimpleNamespace(tag=k)))
        at += rows
    return types.SimpleNamespace(items=items, positions=positions, item_rows=list(rows_each),
                                 from_item=None, token_ids=list(range(at + gap)))


def eager(p, tower):
    rows = tower.features(p)
    return E.VisionFeed(fake_engine(tower), list(p.positions), rows)


def test_each_chunk_gets_the_rows_of_encoding_everything_at_once_while_little_is_held():
    p = prepared([5, 40, 7, 12, 30])
    lazy_tower = Tower()
    lazy = E.VisionFeed(fake_engine(lazy_tower), list(p.positions), None, prepared=p)
    full = eager(p, Tower())
    n, start, most = len(p.token_ids), 0, 0
    while start < n:
        count = min(16, n - start)
        lazy.ensure(start, count)
        most = max(most, len(lazy.held))
        a, b = lazy.chunk(start, count), full.chunk(start, count)
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        lazy.release(start + count)
        start += count
    assert most <= 2 and not lazy.held                        # never more than the items one chunk covers
    assert sum(len(c) for c in lazy_tower.calls) == 5         # each item encoded once


def test_a_chunk_whose_rows_were_not_encoded_first_fails_loudly():
    p = prepared([5, 6])
    lazy = E.VisionFeed(fake_engine(Tower()), list(p.positions), None, prepared=p)
    with pytest.raises(RuntimeError):
        lazy.chunk(0, len(p.token_ids))


def lane_with(p, tower):
    s = types.SimpleNamespace(done=False, error=None, started=None, finished=None)
    feed = E.VisionFeed(fake_engine(tower), list(p.positions), None, prepared=p)
    return types.SimpleNamespace(s=s, feed=feed, st=types.SimpleNamespace(pos=0), sid=1)


def test_a_failed_encode_fails_that_stream_before_its_chunk_goes_out():
    class Broken(Tower):
        def features(self, prepared):
            raise RuntimeError("the tower failed")

    d = object.__new__(M.MultiDecoder)
    lane = lane_with(prepared([5, 6]), Broken())
    assert d._feed_ready(lane, 10) == [lane.s]
    assert lane.s.done and "the tower failed" in str(lane.s.error)


def test_a_streams_stats_say_how_long_its_pictures_took_to_encode():
    d = object.__new__(M.MultiDecoder)
    lane = lane_with(prepared([5, 6]), Tower())
    assert d._feed_ready(lane, 20) == []
    assert lane.s.glm_lane["pictures_encoded"] == 2 and lane.s.glm_lane["encode_s"] >= 0


def test_a_picture_stream_can_be_handed_back_and_replays_with_its_pictures():
    from tensorfold.cuda.streams import Stream

    d = object.__new__(M.MultiDecoder)
    d.lanes, d.requeue, d.outbox = {}, [], []
    d._evict = lambda c: None
    d._finish = lambda sid: d.lanes.pop(sid)
    vision = prepared([5])
    s = Stream([1, 2, 3], 10, None, vision=vision)
    d.lanes[7] = types.SimpleNamespace(s=s, sid=7, order=1, constraint=None, images=True,
                                       extent=types.SimpleNamespace(kept=[]))
    assert d._give_back() == [] and d.requeue == [s]
    sched = object.__new__(M.GlmScheduler)
    sched.decoder, sched.boxes, sched.yields = d, {id(s): "box"}, 0
    sched.waiting = types.SimpleNamespace(put=lambda item: setattr(sched, "queued", item))
    sched._requeue()
    again, box = sched.queued
    assert again.vision is vision and box == "box"


def test_picture_rows_must_sit_on_the_image_token():
    assert M.picture_rows_ok([7, 154854, 154854, 9], [1, 2], 154854)
    assert not M.picture_rows_ok([7, 154854, 3, 9], [1, 2], 154854)


def test_the_guard_takes_the_request_threads_array_or_the_list():
    import numpy as np

    prompt = [7, 154854, 154854, 9]
    assert M.picture_rows_ok(np.asarray(prompt, dtype=np.int32), [1, 2], 154854)
    assert M.picture_rows_ok(prompt, [], 154854)
