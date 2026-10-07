"""The GLM-5.3-Flash files the label, record and resume tests render with (tokenizer.json, chat_template.jinja,
tokenizer_config.json), from the folder TF_GLM_TEST_MODEL_DIR names: a checkpoint snapshot, or a copy of those files.
Without them those tests are skipped, never faked: they prove what the real template does."""

import os
from pathlib import Path

import pytest

NEEDED = ("tokenizer.json", "chat_template.jinja", "tokenizer_config.json")


def model_dir() -> Path:
    folder = Path(os.environ.get("TF_GLM_TEST_MODEL_DIR", "") or "/nonexistent")
    if not all((folder / name).is_file() for name in NEEDED):
        pytest.skip("set TF_GLM_TEST_MODEL_DIR to a GLM-5.3-Flash snapshot (tokenizer.json, chat_template.jinja, "
                    "tokenizer_config.json)")
    return folder
