"""Patch 9029 (TODO 4.11: the SSD store survives builds that cannot change a saved state), on the CPU. A stored state is
named by its exact token ids, so these files are NEUTRAL (they leave the store's folder name alone):
- files that only decide which ids arrive: tokenizing, templates, resume markers, request order;
- files that pick which id comes next: sampling, log probabilities;
- memory accounting: the gate and the startup estimates (any layout they cause is in the store's signature: the
  planes' shapes, the grid);
- debug modes, off unless set: draft dumps, the offline simulator, the profiler;
- the spill tier's own code, whose on-disk bytes are pinned below. A change to what it writes for the same input fails
  here until FORMAT (spill_blocks.py) is bumped, which renames the store, and the pin is updated.
Kernels, the forward pass, the drafter, the tensor-parallel split, weight loading, the tuning of the concurrent rounds
and the engine loop (multi.py) still start a new store."""

import hashlib
import os
from pathlib import Path

import tensorfold
from spill_store_kit import ar, ids_of, keep, store  # noqa: F401  (ar: the fixture)
from tensorfold.cuda import spill
from tensorfold.cuda import spill_blocks as S
from test_9017_spill_keeps import sources

NEW = ("cuda/literal_tokens.py", "cuda/chat_template.py", "cuda/markers.py", "families/glm5_next/prompts.py",
       "cuda/turns.py", "cuda/sampling.py", "cuda/logprobs.py", "cuda/memory_gate.py", "cuda/capacity.py",
       "families/glm5_next/cuda/draft_dump.py", "families/glm5_next/cuda/draft_sim.py",
       "families/glm5_next/cuda/prof.py", "cuda/spill.py", "cuda/spill_blocks.py")
STILL = ("families/glm5_next/cuda/forward.py", "families/glm5_next/cuda/decode.py", "families/glm5_next/cuda/multi.py",
         "families/glm5_next/cuda/multi_tune.py", "families/glm5_next/cuda/drafter_choice.py", "cuda/shares.py",
         "cuda/direct_read.py", "families/glm5_next/cuda/kda.cu", "cuda/kernels/qmm.cu")
# sha256 over every file the tier writes for one fixed point (its path, then its bytes' sha256), taken 2026-10-16 from
# FORMAT tensorfold-cuda-spill-blocks-1 (fork de2180f: Mia's 0088 with wojo's 0110 and our 9001-9025)
PINNED = "4c6775c80fddc8e661379c3d0c415db966f02825a88d7f76e357f1a0839ace41"


def test_every_neutral_name_is_a_file():
    pkg = Path(tensorfold.__file__).parent
    assert [n for n in sorted(spill.NEUTRAL) if not (pkg / n).is_file()] == []   # a renamed file would rejoin unseen


def test_the_new_neutral_files_keep_the_store(tmp_path):
    assert set(NEW) <= spill.NEUTRAL
    root = sources(tmp_path)
    family = root / "families/glm5_next"
    before = spill.sources_digest(root, family)
    for name in NEW:
        (root / name).write_text((root / name).read_text() + "\n# a change that cannot touch a saved state\n")
    assert spill.sources_digest(root, family) == before


def test_the_maths_the_layout_and_the_engine_loop_still_start_a_new_store(tmp_path):
    assert not set(STILL) & spill.NEUTRAL
    root = sources(tmp_path)
    family = root / "families/glm5_next"
    for name in STILL:
        before = spill.sources_digest(root, family)
        (root / name).write_text((root / name).read_text() + ("\n// x\n" if name.endswith(".cu") else "\n# x\n"))
        assert spill.sources_digest(root, family) != before, name


def written(st):
    """Every file under the store's folder: its path there and a digest of its bytes, in path order."""

    files = sorted(os.path.relpath(os.path.join(d, f), st.top) for d, _, fs in os.walk(st.top) for f in fs)
    return [(f, hashlib.sha256(open(os.path.join(st.top, f), "rb").read()).hexdigest()) for f in files]


def test_the_bytes_the_tier_writes_for_one_fixed_point_are_pinned(ar, tmp_path):
    st = store(ar, tmp_path / "a")
    keep(st, ids_of(3000))
    assert st.drain(30)
    files = written(st)
    whole = hashlib.sha256("".join(f"{name}\0{digest}\n" for name, digest in files).encode()).hexdigest()
    assert S.FORMAT == "tensorfold-cuda-spill-blocks-1"
    assert whole == PINNED, ("the bytes the tier writes changed: bump FORMAT in spill_blocks.py (it renames the store),"
                             f" then this pin. Now {whole}, per file: {files}")
