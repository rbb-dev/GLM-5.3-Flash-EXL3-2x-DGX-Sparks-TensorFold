"""Patch 9017 (the spill tier keeps what it can), on the CPU: a build that changes only files which never change what
a stored state holds (request handling, labels and records, the stall meter, the collector's schedule, HTTP) keeps
the tier (each new build emptied it before); any other source file, a new one included, still starts a new tier; and a
request's own state is held from the moment it is kept (not only once its reply ends), so an early write never takes
it to the disk without the record its reply is about to make."""

import shutil
import types

from tensorfold.cuda import spill
from tensorfold.families.glm5_next.cuda import multi as M


def sources(tmp_path):
    """A copy of the two source trees the identity reads (the family's and tensorfold/cuda)."""

    import tensorfold

    pkg = __import__("pathlib").Path(tensorfold.__file__).parent
    root = tmp_path / "tensorfold"
    for part in ("families/glm5_next", "cuda"):
        shutil.copytree(pkg / part, root / part, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_a_build_that_changes_only_serving_code_keeps_the_tier(tmp_path):
    root = sources(tmp_path)
    family = root / "families/glm5_next"
    before = spill.sources_digest(root, family)
    for neutral in ("families/glm5_next/cuda/app.py", "families/glm5_next/cuda/resume.py", "cuda/http.py",
                    "cuda/server.py", "families/glm5_next/cuda/stall_meter.py"):
        (root / neutral).write_text((root / neutral).read_text() + "\n# a serving change\n")
    assert spill.sources_digest(root, family) == before


def test_a_change_to_the_maths_or_a_new_file_starts_a_new_tier(tmp_path):
    root = sources(tmp_path)
    family = root / "families/glm5_next"
    before = spill.sources_digest(root, family)
    (root / "families/glm5_next/cuda/forward.py").write_text(
        (root / "families/glm5_next/cuda/forward.py").read_text() + "\n# a maths change\n")
    changed = spill.sources_digest(root, family)
    assert changed != before
    (root / "cuda/brand_new_kernel.py").write_text("X = 1\n")
    assert spill.sources_digest(root, family) != changed


def test_a_requests_own_state_is_held_from_the_moment_it_is_kept():
    d = object.__new__(M.MultiDecoder)
    d.rank, d.record_due = 0, set()
    holder = {}
    lane = types.SimpleNamespace(s=types.SimpleNamespace(glm={"turn": holder}), point=64, inherit=None)
    d._turn_kept(lane, types.SimpleNamespace(kid=7, shared=False, ids=[0] * 64))
    assert holder["kid"] == 7 and 7 in d.record_due
    d._turn_kept(lane, types.SimpleNamespace(kid=9, shared=False, ids=[0] * 64))     # a replay keeps a new state
    assert holder["kid"] == 9 and d.record_due == {9}
