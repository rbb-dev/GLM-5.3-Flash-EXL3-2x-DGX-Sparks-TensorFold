# About this fork

This is a fork of [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
Mia's recipe for GLM-5.3-Flash on two DGX Sparks with [TensorFold](https://github.com/ashhart/TensorFold). The branch
`mgllm` is the recipe's `main` plus the patches below. Every other file is the recipe's, unchanged. As the recipe's
NOTICE asks, this file states what the fork adds.

## What the `mgllm` branch adds

| Patch | What it does | Origin | Notes |
|---|---|---|---|
| `9001-mgllm-draft-positions` | Live draft counts by source (dflash2, copy, none) and by draft position on `/health` and `/metrics` | this fork | Counting only: drafts, verification and replies are unchanged. Test: `tests/test_9001_draft_positions.py` |
| `9002-mgllm-anthropic-fixes` | Fixes to the recipe's Anthropic Messages API (its `0084`): a `ping` event while a tool call is held, a 400 from `count_tokens` for a request it cannot prepare (503 at capacity), a tool result's pictures kept inside it in order, and request bodies up to the chat route's limit | this fork | Test: `tests/test_9002_stream_and_count.py` |
| `9003-mgllm-decode-speed` | Each finished request's decode time and decode rate as histograms on `/metrics` | backport of TensorFold c9259c3, plus a rate histogram | Metrics only. Test: `tests/test_9003_decode_speed.py` |
| `9004-mgllm-turn-records` | Each finished reply's exact tokens, thinking included, recorded on the kept prompt state it ends on, so the spill tier (the recipe's 0088) saves, reads back and evicts them with that state | this fork | Changes nothing a request is served (9005 uses the records). Needs `--parallel` above 1. Tests: `tests/test_9004_*.py` |
| `9005-mgllm-resume-by-label` | A request whose visible conversation continues a record is served the record's exact tokens plus only what is new, so a client that drops the thinking still resumes its saved state, also after a restart | this fork | Anything doubtful is a miss: today's prompt, never a refusal. Test: `tests/test_9005_resume.py` |
| `9006-mgllm-unlocked-tokenize` | Prompts tokenized with Python's lock released: text prompts, picture prompts and a resumed request's new part alike, so a long request's tokenization no longer pauses every other request's stream | this fork | The same token ids. Test: `tests/test_9006_unlocked_tokenize.py` |

Each patch file begins with a header that explains it in full, credits included.

## Numbering

- The recipe numbers its patches from `0001` upwards. This fork's own patches start at `9001`. A new recipe release
  therefore never touches them (they are different files), and in filename order they apply after every recipe
  patch.

## Updating from the recipe

On each recipe release, this branch's own commits are replayed on the recipe's new `main`, so the history stays the
recipe's followed by one commit for each of this fork's patches:

```bash
git fetch upstream
git rebase --onto upstream/main <the recipe commit this branch was built on> mgllm
```

If a recipe change touches the same code as a `9xxx` patch, that patch is refreshed in its own commit:

1. Apply everything to TensorFold v0.6.0 in filename order.
2. Resolve the conflict.
3. Export the patch again.

## Using it

Use it the same way as the recipe. The patches apply to TensorFold v0.6.0 `src/` with `patch -p0`, in filename order,
as the recipe's own image build applies them. If you start from the recipe's published image, apply the `9xxx`
patches on top.

The tests in `tests/` are pytest unit tests for this fork's patches. They run on the CPU against a TensorFold v0.6.0
source tree with all patches applied, with that tree's `src/` on `PYTHONPATH`. They need `pytest`, `prometheus_client`
and TensorFold's own Python dependencies (a CPU build of `torch` is enough). The tests that render real GLM-5.3-Flash prompts also read its `tokenizer.json`, `chat_template.jinja` and `tokenizer_config.json` from the folder `TF_GLM_TEST_MODEL_DIR` names (a checkpoint snapshot); without it they are skipped.

## License

Apache-2.0, like the recipe (see `LICENSE` and `NOTICE`). `9003` back-ports a TensorFold commit whose authors are
credited in its patch header.
