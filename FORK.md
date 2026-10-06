# About this fork

This is a fork of [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
Mia's recipe for GLM-5.3-Flash on two DGX Sparks with [TensorFold](https://github.com/ashhart/TensorFold). The branch
`mgllm` is the recipe's `main` plus the patches below. Every other file is the recipe's, unchanged. As the recipe's
NOTICE asks, this file states what the fork adds.

## What the `mgllm` branch adds

| Patch | What it does | Origin | Notes |
|---|---|---|---|
| `0084-glm-spill-tier` | Writes kept prompt states to local disk and reads them back (`SPILL_GIB`, off by default) | @wojo, [MiaAI-Lab PR #78](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/78) | Carried exactly as published, with its author's commit. Drop it here once the recipe merges it |
| `9001-mgllm-draft-positions` | Live draft counts by source (dflash2, copy, none) and by draft position on `/health` and `/metrics` | this fork | Counting only: drafts, verification and replies are unchanged. Test: `tests/test_9001_draft_positions.py` |
| `9002-mgllm-anthropic-messages` | Anthropic Messages API on the CUDA server (`POST /v1/messages`, `/v1/messages/count_tokens`) | backport of TensorFold v0.6.3-v0.6.5 (ashhart/TensorFold 8cca6e3 and follow-ups) | Drop it once the recipe moves past TensorFold v0.6.0 or carries an equivalent (see recipe PR #90) |
| `9003-mgllm-decode-speed` | Each finished request's decode time and decode rate as histograms on `/metrics` | backport of TensorFold c9259c3, plus a rate histogram | Metrics only. Test: `tests/test_9003_decode_speed.py` |

Each patch file begins with a header that explains it in full, credits included.

## Numbering

- The recipe numbers its patches from `0001` upwards. This fork's own patches start at `9001`. Merging the recipe's
  `main` therefore never touches them (they are different files), and in filename order they apply after every
  recipe patch.
- A third-party patch that has been proposed to the recipe but not merged keeps its published number (`0084`).

## Updating from the recipe

```bash
git fetch upstream
git checkout mgllm
git merge upstream/main
```

If a recipe change touches the same code as a `9xxx` patch, that patch may need refreshing:

1. Apply everything to TensorFold v0.6.0 in filename order.
2. Resolve the conflict.
3. Export the patch again.

## Using it

Use it the same way as the recipe. The patches apply to TensorFold v0.6.0 `src/` with `patch -p0`, in filename order,
as the recipe's own image build applies them. If you start from the recipe's published image (patches up to `0083`),
apply `0084` and the `9xxx` patches on top.

The tests in `tests/` are pytest unit tests for this fork's patches. They run on the CPU against a TensorFold v0.6.0
source tree with all patches applied, with that tree's `src/` on `PYTHONPATH`. They need `pytest`, `prometheus_client`
and TensorFold's own Python dependencies (a CPU build of `torch` is enough).

## License

Apache-2.0, like the recipe (see `LICENSE` and `NOTICE`). `0084` is @wojo's work under its own header. `9002` and
`9003` back-port TensorFold commits whose authors are credited in each patch header.
