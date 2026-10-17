"""The spill tier's block store (patch 0110) on the host, as tools/spill_blocks_check.py builds it: a real BlockStore
over a real Arena (a latents plane, div 1, and a pooled-key plane, div 4 with 2 pad rows) with a stand-in small state.
Shared by the tests of 9004 (records on stored states) and 9023 (the store's bookkeeping off the engine loop)."""

import json
import os

import numpy as np
import pytest
import torch

from tensorfold.cuda import spill as SP
from tensorfold.cuda import spill_blocks as S
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane, Pool, align_up

ROWS = 8 * ALIGN
SKIP = ("ids", "shared", "lineage", "spill_key")      # what the decoder's _SPILL_SKIP leaves out of a stand-in


class Snap:
    """A stand-in for decode.Snapshot: a small state the tier encodes."""


CLASSES = [f"{Snap.__module__}:{Snap.__qualname__}"]


def snap(ids):
    s = Snap()
    s.ids, s.shared, s.lineage = list(ids), False, ""
    s.rec, s.conv = torch.arange(64, dtype=torch.float32) + len(ids), torch.ones(8)
    s.tail, s.head, s.drafter_rows = None, None, None
    return s


def arena(seed=0):
    g = torch.Generator().manual_seed(seed)
    lat = torch.randint(-2**31, 2**31 - 1, (ROWS, 4), generator=g, dtype=torch.int32)
    pk = torch.randint(-2**15, 2**15 - 1, (ROWS // 4 + 2, 8), generator=g, dtype=torch.int16)
    return Arena(ROWS, [Plane(lat, 1), Plane(pk, 4, 2)])


def store(ar, root, rank=0, gib=1.0, **kw):
    cfg = SP.SpillConfig(root=str(root), gib=gib, min_tokens=1, min_free_gib=0.0)
    os.makedirs(cfg.root, exist_ok=True)
    st = S.BlockStore(cfg, planes=ar.planes, block=ALIGN, rank=rank, device="cpu", classes=CLASSES, quiet=True, **kw)
    st.ring()                                         # as the decoder does at start (multi._spill_store)
    return st


def ids_of(n, seed=0):
    return [int(v) for v in np.random.default_rng(seed).integers(1, 150000, n)]


def keep(st, ids, lineage="s0", base=0):
    """A state of ``ids`` kept at ``base`` and handed to the tier (not yet drained): (its object, its key)."""

    obj = snap(ids)
    st.persist(ids, obj, extent=Pool(ROWS).add(base, align_up(len(ids))), lineage=lineage, skip=SKIP)
    return obj, S.point_key(np.asarray(ids, dtype=np.int64))


def json_of(st, key):
    with open(st._point_path(key, ".json")) as fh:
        return json.load(fh)


@pytest.fixture
def ar(monkeypatch):
    # the test arena's blocks are ~40 KB: the default 256 MiB ring would be thousands of slots, each zeroed (as
    # tools/spill_blocks_check.py does, a ring of 1 MiB and the fewest read-ahead slots)
    monkeypatch.setenv("TF_GLM_SPILL_STAGE_MIB", "1")
    monkeypatch.setenv("TF_GLM_SPILL_READ_BLOCKS", "8")
    return arena()
