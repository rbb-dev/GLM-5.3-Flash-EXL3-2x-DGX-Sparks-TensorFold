"""Patch 9017 (the spill tier keeps what it can), on the CPU: a build that changes only files which never change what
a stored state holds (request handling, labels and records, the collector's schedule, HTTP) keeps the tier (each new
build emptied it before); any other source file, a new one included, still starts a new tier. The block store (0110)
names its folder by the same build id (multi._spill_store)."""

import shutil

from tensorfold.cuda import spill


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
                    "cuda/server.py", "families/glm5_next/cuda/gc_policy.py", "families/glm5_next/cuda/labels.py"):
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
