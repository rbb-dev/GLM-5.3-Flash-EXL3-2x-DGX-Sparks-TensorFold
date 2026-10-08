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
| `9007-mgllm-no-kept-reasoning` | Mia's in-RAM reasoning memory (0036's KeptReasoning) cut out: reasoning a client dropped comes back only through a turn record (9004/9005), saved, read back and evicted with its prompt state, after restarts too | this fork | The memory was a second cache with a retention of its own, and its signature ignored pictures: the same words with another picture got the first conversation's thinking. Test: `tests/test_9007_no_kept_reasoning.py` |
| `9008-mgllm-stall-meter` | The engine's own stall meter on `/health` (`"stalls"`): every pause between two decode rounds while streams decode, what filled it (admit, round, finish, reply), and the engine's clock to place them by | this fork | Measuring only. A client sees such a pause only when it outlasts the 0.4 s stream smoothing. Test: `tests/test_9008_stall_meter.py` |
| `9009-mgllm-lean-admit` | A long request's admission without Python-list round trips on the scheduler threads | this fork | The same values on the wire but for the message checksum (CRC-32). Test: `tests/test_9009_lean_admit.py` |
| `9010-mgllm-gc-policy` | Python's garbage collection on the engine's own schedule: a full collection only when idle (or every 15 minutes), never in the middle of streams | this fork | `TF_GLM_GC=0` leaves Python's own collection on. Test: `tests/test_9010_gc_policy.py` |
| `9011-mgllm-spill-ids` | The spill tier stores a prompt's token ids as bytes, not a JSON list | this fork | File format `tensorfold-cuda-spill-3`: files of the old format are not read. Test: `tests/test_9011_spill_ids.py` |
| `9012-mgllm-copy-drafts-start` | A stream's copy drafter starts from the prompt array the request thread made | this fork | Test: `tests/test_9012_copy_drafts_start.py` |
| `9013-mgllm-picture-layout` | A picture is laid out the same way whatever comes after it (its size no longer depends on how many pictures the request carries), with an identity of its exact pixels; no picture count limit (the model has none), a request-size setting instead (`TENSORFOLD_MAX_BODY_MIB`, default 96) on every route that takes pictures | this fork | Past 8 pictures each new one re-sized every earlier one, so a screenshot conversation lost its saved state on every turn. The picture caches are locked, base64 is decoded in pieces, the first resize runs at start. Test: `tests/test_9013_picture_layout.py` |
| `9014-mgllm-picture-feed` | Pictures run through the vision tower as the prompt's read reaches them, between rounds, one chunk's pictures at a time, and are let go once read | this fork | A failed encode fails that request alone; a picture request can be handed back when the pool is full. Test: `tests/test_9014_picture_feed.py` |
| `9015-mgllm-picture-records` | Turn records (9004/9005) for picture and clip conversations: each picture's identity is part of the label at its place, and a resume lays out the record's tokens with the request's own pictures | this fork | The same words with another picture never share a record ("pictures differ"). A screenshot loop resumes every turn, past its 8th picture and after a restart too. Test: `tests/test_9015_picture_records.py` |
| `9016-mgllm-api-info-and-pictures` | The Anthropic Messages route carries TensorFold's info block (token checksum, record path); a Responses `function_call_output` may carry pictures | this fork | Additive fields; such pictures were refused with 400. Test: `tests/test_9016_api_info_and_pictures.py` |
| `9017-mgllm-spill-keeps` | The spill tier keeps its saved conversations across builds that change only how requests are served; a request's kept state is held from the moment it is kept | this fork | Files that change what a stored state holds still start it empty (`spill.NEUTRAL` lists the ones that do not). Test: `tests/test_9017_spill_keeps.py` |

Each patch file begins with a header that explains it in full, credits included.

## Measured on two DGX Sparks

Two DGX Sparks: v1.10 (`tensorfold-glm53:v0.6.0-mia-a1897d591f70`) against v1.10 + fork (`tensorfold-glm53:v0.6.0-mgllm-67e2b6640113`), one after the other, with the same settings and the same conversations (`tools/resume_bench.py --seed 20261008`), from another machine on the network: 6 requests at once; a 1,048,576-token window; 2,048 tokens a picture; GPU clocks locked at 0,2200 MHz; KV pool 2,109,440 and 2,158,592 tokens. Sampling: temperature 1.0, top_p 0.95, reasoning effort max.

**The Sparks**, before the tests:

|  | v1.10, spark 1 | v1.10, spark 2 | v1.10 + fork, spark 1 | v1.10 + fork, spark 2 |
| --- | --- | --- | --- | --- |
| GPU clock / temperature | 2190 MHz / 52 C | 2177 MHz / 53 C | 2190 MHz / 53 C | 2177 MHz / 53 C |
| CPU clock / temperature | 2668-3915 MHz / 54.3-56.9 C | 2654-3898 MHz / 56.4-59.0 C | 2651-3919 MHz / 56.7-58.9 C | 2678-3900 MHz / 55.7-58.5 C |
| memory available | 11.2 GiB | 13.0 GiB | 11.0 GiB | 12.5 GiB |
| driver / kernel | 580.173.02 / 6.17.0-1029-nvidia | 580.173.02 / 6.17.0-1029-nvidia | 580.173.02 / 6.17.0-1029-nvidia | 580.173.02 / 6.17.0-1029-nvidia |

**Picking up a conversation** (the client sends the conversation back without the model's thinking, as agent clients and gateways do). Time to first token of the next turn:

| Next turn | v1.10 | v1.10 + fork |
| --- | ---: | ---: |
| First read, nothing cached | 150.7 s | 150.8 s |
| Agent with a tool call, hot (engine running) | 1.16 s | 1.13 s |
| Agent with a tool call, cold (after a restart, from the SSD) | 150.2 s | **1.15 s** |
| Plain chat, hot | 1.04 s | 0.97 s |
| Plain chat, cold | 1.43 s | 0.93 s |

Every resumed reply was token for token the reply of a fresh read.

**A screenshot conversation** (a new 1080p screenshot every turn, the thinking dropped by the client). Time to first token, and the prompt tokens reused from the cache:

| Turn | Screenshots | v1.10: first token | v1.10: reused | v1.10 + fork: first token | v1.10 + fork: reused |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2.75 s | 0 | 1.89 s | 0 |
| 2 | 2 | 3.06 s | 2,048 | 3.01 s | 2,048 |
| 3 | 3 | 2.70 s | 4,224 | 2.33 s | 4,416 |
| 4 | 4 | 2.73 s | 6,400 | 2.77 s | 6,784 |
| 5 | 5 | 2.85 s | 8,576 | 2.42 s | 9,152 |
| 6 | 6 | 2.86 s | 10,752 | 2.92 s | 11,392 |
| 7 | 7 | 2.74 s | 12,928 | 2.73 s | 13,696 |
| 8 | 8 | 2.90 s | 15,104 | 2.86 s | 16,000 |
| 9 | 9 | **13.9 s** | **0** | 2.43 s | 18,304 |
| 10 | 10 | **14.0 s** | **0** | 2.85 s | 20,544 |
| 11 | 11 | **14.1 s** | **0** | 2.47 s | 22,848 |
| 12 | 12 | **13.7 s** | **0** | 2.92 s | 25,088 |
| 13, cold (after a restart) | 13 | 14.5 s | 0 | **3.15 s** | 27,392 |

From the 9th screenshot, TensorFold shares one picture budget across the request, so every earlier screenshot shrinks and nothing matches the cache. With this fork, each picture keeps its size, and every turn resumes.

**Other replies while a request arrives** (one reply streams while another request arrives; the streaming reply's longest wait between two tokens, and its rate, in the 10 s after the arrival):

| Request arriving | v1.10: longest wait | v1.10: rate | v1.10 + fork: longest wait | v1.10 + fork: rate |
| --- | ---: | ---: | ---: | ---: |
| 1,021,598 tokens of text | 1.91 s | 8.0 tok/s | **0.18 s** | 21.7 tok/s |
| 16 new screenshots | 2.53 s | 13.9 tok/s | **0.20 s** | 16.2 tok/s |

The recipe lays those 16 screenshots out at half size (16,205 tokens, against 32,717 with this fork, each at its full size).

**Engine start** (from the start command to the first one-token reply):

| Start | v1.10 | v1.10 + fork |
| --- | ---: | ---: |
| Kernels compiled at this start | 297.4 s | 305.9 s |
| Kernels cached | 126.9 s | 126.1 s |

The full reports, every step and the settings of both runs: [`results/2026-10-08/`](results/2026-10-08/)

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
and TensorFold's own Python dependencies (a CPU build of `torch` is enough). The tests that render real GLM-5.3-Flash
prompts also read its `tokenizer.json`, `chat_template.jinja` and `tokenizer_config.json` from the folder
`TF_GLM_TEST_MODEL_DIR` names (a checkpoint snapshot); without it they are skipped. The picture tests also need
`torchvision`, and the clip test PyAV (without it, it is skipped).

## Measuring it on your own setup

`tools/resume_bench.py` runs every test above against a running server and writes `results.json` and `results.md`: the
engine's start with and without its compiled kernels; a conversation picked up hot and cold, plain and with a tool call,
each checked token for token against a fresh read; the screenshot conversation; and replies while a long request or many
screenshots arrive. What depends on your engine is a few executables of yours (`restart`, `clear-buffers`, optionally
`clear-kernel-cache` and `node-info`: `tools/resume_bench.py --help` describes them). Run it once on the recipe and once
with these patches, with the same `--seed`, and `tools/resume_bench.py report before.json after.json` prints both side
by side. Its own CPU tests: `tests/test_resume_bench.py`.

## License

Apache-2.0, like the recipe (see `LICENSE` and `NOTICE`). `9003` back-ports a TensorFold commit whose authors are
credited in its patch header.
