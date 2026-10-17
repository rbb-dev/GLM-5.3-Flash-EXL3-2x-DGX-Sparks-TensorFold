"""Patch 0123 (rwl4's PR #158, his 0110): a literal </tool_call> inside a GLM argument value stays in the argument.
Runs his CPU contract (tests/tool_envelope_contract.py: the real App.run and reply parser, every boundary fragmented)."""
import pathlib
import runpy


def test_tool_envelope_contract():
    runpy.run_path(str(pathlib.Path(__file__).with_name("tool_envelope_contract.py")), run_name="contract")
