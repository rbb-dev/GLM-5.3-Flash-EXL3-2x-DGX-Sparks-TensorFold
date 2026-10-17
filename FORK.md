# About this fork

This is a fork of [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
Mia's recipe for GLM-5.3-Flash on two DGX Sparks with [TensorFold](https://github.com/ashhart/TensorFold). Its `main` is
the recipe's `main` plus one commit: the patches below, their tests and checks, and their credits. Every other file is
the recipe's. As the recipe's NOTICE asks, this file states what the fork adds.

Status (2026-10-18): patches 9001-9025 serve production on two DGX Sparks (since 2026-10-15); 9026-9029 pass their
CPU tests and have not yet run on the Sparks.

## Patches from open pull requests to the recipe

Not in the recipe's `main` yet. Each patch file keeps its author's own header; `CREDITS.md` credits them in full.

| Patch | What it does | Author | Pull request |
|---|---|---|---|
| `0110-glm-spill-blocks` | The spill tier in blocks shared by later turns, written when a state is kept, and one prefill of a shared prefix (`SPILL_GIB` with `PARALLEL` above 1, `PREFIX_WAIT`) | [wojo](https://github.com/wojo), with the block layout of [BadAd84](https://github.com/BadAd84)'s #118 | [#124](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/124) |
| `0111-glm-chat-continued` | A conversation keeps its id across a rewind or a resume from disk | [wojo](https://github.com/wojo) | [#128](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/128) |
| `0118-glm-kv-shard` | KV sharding (`TF_GLM_KV_SHARD=1`): each conversation's latent cache stored once, on the Spark that owns it; one core with the tp4 branch (#152's head 43383a8) | [wojo](https://github.com/wojo) | [#152](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/152) |
| `0119-cuda-budget-parked-pages` | The start-up memory budget counts the pages parked on the kernel's per-CPU free lists | [wojo](https://github.com/wojo) | [#149](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/149) |
| `0122-glm-literal-added-tokens` | A quoted added token in a request (`<think>`, `<tool_call>`, `<\|user\|>`) stays text | [ahinore](https://github.com/ahinore) | [#164](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/164) |
| `0123-glm-tool-envelope-values` | A literal `</tool_call>` inside a tool argument's value stays in the argument | [rwl4](https://github.com/rwl4) | [#158](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/158) |

## This fork's patches

| Patch | What it does | Test |
|---|---|---|
| `9001-mgllm-draft-positions` | Live draft counts by source (dflash2, copy, none) and by draft position on `/health` and `/metrics` | `test_9001_draft_positions.py` |
| `9002-mgllm-anthropic-fixes` | Fixes to the recipe's Anthropic Messages API (its `0084`): a `ping` while a tool call is held, `count_tokens` answering 400 (503 at capacity), a tool result's pictures kept inside it in order, request bodies up to the chat route's limit | `test_9002_stream_and_count.py` |
| `9003-mgllm-decode-speed` | Each finished request's decode time and decode rate as histograms on `/metrics` | `test_9003_decode_speed.py` |
| `9004-mgllm-turn-records` | Each finished reply's exact tokens, thinking included, recorded on the kept prompt state it ends on and saved with it by the spill tier (`0110`) | `test_9004_*.py` |
| `9005-mgllm-resume-by-label` | A request whose visible conversation continues a record is served the record's exact tokens plus only what is new, after a restart too | `test_9005_resume.py` |
| `9006-mgllm-unlocked-tokenize` | Prompts tokenized with Python's lock released where the recipe's `0099` does not reach | `test_9006_unlocked_tokenize.py` |
| `9007-mgllm-no-kept-reasoning` | The in-RAM reasoning memory (`0036`'s KeptReasoning) cut out: dropped reasoning comes back only through a turn record | `test_9007_no_kept_reasoning.py` |
| `9009-mgllm-lean-admit` | A long request's admission without Python-list round trips between the ranks | `test_9009_lean_admit.py` |
| `9010-mgllm-gc-policy` | Python's garbage collection on the engine's own schedule: a full collection when idle, never in the middle of streams (`TF_GLM_GC=0` restores Python's own) | `test_9010_gc_policy.py` |
| `9012-mgllm-copy-drafts-start` | A stream's copy drafter starts from the prompt array the request thread made | `test_9012_copy_drafts_start.py` |
| `9013-mgllm-picture-layout` | A picture is laid out the same way whatever comes after it; no picture count limit, a request-size setting instead (`TENSORFOLD_MAX_BODY_MIB`) | `test_9013_picture_layout.py` |
| `9014-mgllm-picture-feed` | Pictures run through the vision tower as the prompt's read reaches them, one chunk's pictures at a time | `test_9014_picture_feed.py` |
| `9015-mgllm-picture-records` | Turn records for picture and clip conversations | `test_9015_picture_records.py` |
| `9016-mgllm-api-info-and-pictures` | The Anthropic route carries TensorFold's info block; a Responses `function_call_output` may carry pictures | `test_9016_api_info_and_pictures.py` |
| `9017-mgllm-spill-keeps` | The spill tier keeps its store across builds that change only how requests are served (`spill.NEUTRAL`) | `test_9017_spill_keeps.py` |
| `9023-mgllm-spill-bookkeeping` | The spill tier's bookkeeping moved off the engine loop | `test_9023_spill_bookkeeping.py` |
| `9024-mgllm-memory-guard` | A memory guard for the Sparks' shared GPU and CPU memory: every rank's free memory watched; when it runs tight, cached memory released; when critical, the kept GPU states dropped | `test_9024_memory_guard.py` |
| `9025-mgllm-loop-guard-count` | `/health` counts the loop guard's fires (`loop_guard_total`) | `test_9025_loop_guard_count.py` |
| `9026-mgllm-health-uptime` | `/health` carries the engine's start time and uptime | `test_9026_health_uptime.py` |
| `9027-mgllm-spill-saved-seconds` | The spill tier counts what its restores saved: each restore priced once, at the engine's own prefill speed, for the tokens the GPU no longer held | `test_9027_spill_saved_seconds.py` |
| `9028-mgllm-spill-both-ranks` | Rank 0's `/health` lists every rank's spill store, not only its own | `test_9028_spill_both_ranks.py` |
| `9029-mgllm-spill-store-identity` | Builds that cannot change a saved state keep the spill store: tokenizing, templates, sampling, memory accounting, debug modes and the tier's own code join `spill.NEUTRAL`; the bytes the tier writes are pinned by a test | `test_9029_store_identity.py` |
| `9030-mgllm-spill-refusal` | The spill tier's refusals on the writer (after 9023): a state with no room is not written, nor is any waiting state counting on its blocks, each counted as no_space (never a failed write); their reserved bytes freed at once; no rows copied for a refused state; an error while making room leaves the state unwritten | `test_9030_spill_refusal.py` |

Each patch file begins with a header that explains it in full. The patches apply in filename order, as the recipe's do.

## Tests

`tests/` runs on a CPU, against TensorFold with the patches applied (`PYTHONPATH=<patched source>:tests`). The label
tests also read the checkpoint's tokenizer files (`TF_GLM_TEST_MODEL_DIR`). `tools/` holds the patches' stand-alone
checks.
`tools/kv_shard_check.py` is wojo's (#152 43383a8), its stand-ins taking this fork's interfaces: an op queued in chunks with `_emit(..., trusted=, bulk=)` (9009), ops after XMOVE (9024), a decoder's `_turn_inherit` (9004). On his stack the changes do nothing: 334 of 334 pass there and on this fork.
