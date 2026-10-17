"""Shared by this fork's tests. The CPU test environment has no triton. multi.py now imports patch 0118's shard.py,
which imports triton at module level (its two kernels never run in a CPU test). So shard is imported once here, with a
stand-in triton module as TensorFold's own decoder tests use one (tests/test_cuda_geometry.py), and the stand-in is
taken away again: torch's own code must still find no triton (with one, torch._dynamo looks for a real one's
attributes)."""

import importlib
import sys
from types import ModuleType

try:
    import triton  # noqa: F401
except ImportError:
    _lang = ModuleType("triton.language")
    _lang.constexpr = object
    _stub = ModuleType("triton")
    _stub.language = _lang
    _stub.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    _stub.cdiv = lambda a, b: (a + b - 1) // b
    sys.modules["triton"], sys.modules["triton.language"] = _stub, _lang
    try:
        importlib.import_module("tensorfold.families.glm5_next.cuda.shard")
    except ModuleNotFoundError as exc:            # (a tree without patch 0118: nothing to import)
        if exc.name != "tensorfold.families.glm5_next.cuda.shard":
            raise
    finally:
        del sys.modules["triton"], sys.modules["triton.language"]
