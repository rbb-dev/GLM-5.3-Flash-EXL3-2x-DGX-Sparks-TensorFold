# Resumable conversations: results

## v1.10 (2026-10-08 22:46:53)

Model `glm-5.3-flash`; sampling {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "max"}.

Nodes (before the tests):

| node | time | GPU clock (now / max) | GPU temperature | GPU power | CPU clock | CPU temperature | memory | memory available | kernel | driver | disk free | CPU governor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spark 1 | 2026-10-08 22:53:59 AEDT | 2190 / 3003 MHz | 52 C | 13.89 W | 2668-3915 MHz | 54.3-56.9 C | 121.6 GiB | 11.2 GiB | 6.17.0-1029-nvidia | 580.173.02 | 2372G | performance |
| spark 2 | 2026-10-08 22:54:00 AEDT | 2177 / 3003 MHz | 53 C | 11.27 W | 2654-3898 MHz | 56.4-59.0 C | 121.6 GiB | 13.0 GiB | 6.17.0-1029-nvidia | 580.173.02 | 2371G | performance |

Nodes (end):

| node | time | GPU clock (now / max) | GPU temperature | GPU power | CPU clock | CPU temperature | memory | memory available | kernel | driver | disk free | CPU governor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spark 1 | 2026-10-08 23:50:07 AEDT | 2190 / 3003 MHz | 58 C | 11.58 W | 2686-3884 MHz | 61.7-67.9 C | 121.6 GiB | 10.3 GiB | 6.17.0-1029-nvidia | 580.173.02 | 2360G | performance |
| spark 2 | 2026-10-08 23:50:08 AEDT | 2190 / 3003 MHz | 58 C | 12.00 W | 2525-3896 MHz | 61.9-65.1 C | 121.6 GiB | 11.3 GiB | 6.17.0-1029-nvidia | 580.173.02 | 2360G | performance |

Engine settings (end of the run):

| setting | value |
| --- | --- |
| image | tensorfold-glm53:v0.6.0-mia-a1897d591f70 |
| PORT | 8001 |
| SPILL_GIB | 256 |
| GPU clocks locked (nvidia-smi -lgc) | 0,2200 MHz |
| TENSORFOLD_GLM_IMAGE_TOKENS | 2048 |
| --parallel | 6 |
| --context | 1048576 |
| --max-tokens | 1048576 |
| --spill-gib | 256 |
| --spill-highwater | 0.70 |
| --spill-min-tokens | 2048 |
| --vision | on |
| --thinking | on |
| TF_GLM_FILL_BUDGET_MS | 200 |
| TF_GLM_PREFILL_OVERLAP | 2 |
| TF_GLM_COPY_DRAFTS | 1 |
| TF_GLM_SHARED_PREFIX | 1 |
| TF_GLM_KDA_CHUNKED | 1 |
| TF_GLM_MULTI_PREFILL | 1 |
| TF_ROCE_WAIT_S | 300 |
| TF_GLM_MULTI_WINDOW | 64 |
| TF_GLM_WIDE_GRAPHS | 16 |
| TF_GLM_EFFORT_TAIL | 0 |
| TF_GLM_DISPLAY_KV_MIB | 0 |
| TF_GLM_MULTI_LONE | 0 |
| TF_GLM_MAX_QUEUED |  |
| TENSORFOLD_NUCLEUS_UNION | 1 |
| TF_GLM_STREAM_SMOOTH | 1 |
| TF_GLM_EXL3_DEC_ORDER | 0 |
| TF_GLM_CLEAR_THINKING | 0 |
| TF_GLM_DENSE | q4 |
| TF_SPILL_FLUSH_S | 60 |
| TF_GLM_COPY_MAX | 15 |
| TF_GLM_KEPT_BYTES_GIB | 0 |
| TF_GLM_EXL3_LOADS | nc |
| TF_GLM_DISPLAY_KV_BACKEND | drm |
| TF_GLM_DFLASH_POLICY | fnc7:0.3 |
| TF_GLM_FILL_DRAFTS | 1 |
| TF_GLM_HC_SPLIT | 1 |
| TF_ROCE_MAX_KB | 1024 |
| TENSORFOLD_NO_UPDATE_CHECK | 1 |
| TF_GLM_COMM | roce |
| TF_GLM_KV | fp8 |
| TF_GLM_CACHE_GIB | 12.5 |
| TF_GLM_KEEP_PER_CHAT | 0 |
| TF_GLM_L2PF | 1 |
| TF_GLM_STREAM_SMOOTH_MS | 400 |
| TF_GLM_CACHE_ENTRIES | 16 |
| TF_GLM_CACHE_SHARE_PCT | 0 |
| TF_GLM_COPY_REPLY_MATCH | 16 |
| TF_GLM_MTP | auto |
| TENSORFOLD_MEMORY_RESERVE_GIB | 17.7 |
| KV pool (tokens) | 2,189,312 |

### Engine start

From the restart hook to the first one-token reply.

| Start | Restart hook | First reply |
| --- | --- | --- |
| cold kernels (compiled at this start) | 297.3 s | 297.4 s |
| warm kernels | 126.8 s | 126.9 s |

### Resume after a restart, for a client that drops the thinking

Turn 1 reads a new conversation (cold); turn 2 resumes it from memory (warm); after a restart and a page-cache drop, turn 3 resumes from the saved state on the disk, and the same request with `"draft": false` reads the same prompt fresh. Time to first token (TTFT) each time.

| Turn 1 prompt | Turn 1, cold | Turn 2, warm | Turn 3 prompt | Turn 3, from disk | Read from disk | Thinking put back (tokens) | Turn 3 read fresh | Same reply | Path (1 / 2 / 3 / fresh) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8,007 | 5.58 s | 0.49 s | 8,269 | 0.90 s | 98% | 0 | 4.61 s | yes | None / None / None / None |
| 64,211 | 33.9 s | 0.60 s | 64,472 | 0.74 s | 100% | 0 | 33.4 s | yes | None / None / None / None |
| 256,869 | 150.6 s | 1.06 s | 257,084 | 1.42 s | 100% | 0 | 149.7 s | yes | None / None / None / None |

Control: turn 1 drafted and turn 1 read fresh gave the same reply. "Same reply" shows the saved state read back from the disk is exactly a fresh read of the same prompt; that the prompt is the true conversation is proven by the CPU tests.

### An agent's conversation after a restart: the material after the first reply

Turn 1 is a short exchange, turn 2 brings the material (cold), turn 3 resumes it from memory; after a restart and a page-cache drop, turn 4 resumes from the saved state on the disk, and the same request with `"draft": false` reads it fresh. The client drops the thinking.

| Material | Turn 2, cold | Turn 3, warm | Turn 4 prompt | Turn 4, from disk | Read from disk | Turn 4 read fresh | Same reply | Path (2 / 3 / 4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256,903 | 150.9 s | 1.04 s | 257,173 | 1.43 s | 100% | 147.6 s | yes | None / None / None |

### An agent's tool call after a restart: the material as the tool's result

Turn 1 must call a tool; turn 2 sends the material back as its result (cold), turn 3 resumes from memory; after a restart and a page-cache drop, turn 4 resumes from the disk, and the same request with `"draft": false` reads it fresh. The client drops the thinking, the tool call's too: a server that put it back from RAM no longer can after a restart.

| Material | Turn 2, cold | Turn 3, warm | Turn 4 prompt | Turn 4, from disk | Read from disk | Turn 4 read fresh | Same reply | Path (2 / 3 / 4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256,867 | 150.7 s | 1.16 s | 257,197 | 150.2 s | 0% | 147.5 s | yes | None / None / None |

### A screenshot conversation, resumed every turn

An agent watching a screen: turn 1 sends a new 1080p screenshot and a question (cold); each later turn adds a new screenshot and sends the conversation back without the thinking, so each must be served from its record, past the 8th picture too. After a restart and a page-cache drop, the next turn resumes from the saved state on the disk, and the same request with `"draft": false` reads it fresh.

| Turn | Screenshots | Prompt | From a saved state | TTFT | Served from |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 2,091 | 0 | 2.75 s | None |
| 2 | 2 | 4,279 | 2,048 | 3.06 s | None |
| 3 | 3 | 6,458 | 4,224 | 2.70 s | None |
| 4 | 4 | 8,626 | 6,400 | 2.73 s | None |
| 5 | 5 | 10,793 | 8,576 | 2.85 s | None |
| 6 | 6 | 12,985 | 10,752 | 2.86 s | None |
| 7 | 7 | 15,162 | 12,928 | 2.74 s | None |
| 8 | 8 | 17,350 | 15,104 | 2.90 s | None |
| 9 | 9 | 17,286 | 0 | 13.9 s | None |
| 10 | 10 | 17,484 | 0 | 14.0 s | None |
| 11 | 11 | 17,675 | 0 | 14.1 s | None |
| 12 | 12 | 17,067 | 0 | 13.7 s | None |
| 13, after a restart (from disk) | 13 | 17,536 | 0 | 14.5 s | None |
| 13, read fresh | 13 | 17,536 | 0 | 13.2 s | None |

After the restart, 0% of the prompt came from the saved state and the reply was the same as the fresh read's. Controls: turn 1 drafted and read fresh gave the same reply; turn 1 with another picture of the same size gave a different reply (the checksum sees the pixels). The last request's body: 7.0 MiB.

The largest picture, one 3840x2160 screenshot (laid out at the per-picture cap): 2,091 prompt tokens, TTFT 1.91 s.

### Serving while a long request arrives

Reply A decodes while request B arrives: A's tokens a second (largest gap) in the 10 s before B, in the 10 s after B arrives, over B's whole wait for its first token (while its prompt is read), and in the 10 s after B ends; B's last token to its closing event.

| B prompt | A before | A as B arrives | A while B prefills | A after B ends | B TTFT | B close |
| --- | --- | --- | --- | --- | --- | --- |
| 1,021,598 | 44.2 (0.056 s) | 8.0 (1.912 s) | 8.18 (1.912 s) | 43.9 (0.056 s) | 1161.2 s | 0.05 s |
| 16 screenshots (16,205 tokens) | 44.0 (0.055 s) | 13.9 (2.528 s) | 12.05 (2.528 s) | 46.6 (0.055 s) | 21.2 s | 0.05 s |

### Notes

- 8,192-token conversation: turn 1 left no record (None), so turns 2-3 had none to resume from
- 8,192-token conversation: turn 3 was not served from a record (path None)
- 65,536-token conversation: turn 1 left no record (None), so turns 2-3 had none to resume from
- 65,536-token conversation: turn 3 was not served from a record (path None)
- 262,144-token conversation: turn 1 left no record (None), so turns 2-3 had none to resume from
- 262,144-token conversation: turn 3 was not served from a record (path None)
- agent conversation, 256,903 tokens of material: turn 4 was not served from a record (path None)
- tool conversation, 256,867 tokens of material: turn 4 was not served from a record (path None)
- tool conversation, 256,867 tokens of material: only 0% of turn 4's prompt came from the saved state
- screenshot conversation: turn 1 left no record (None)
- screenshot conversation: turn 2 was not served from a record (path None)
- screenshot conversation: turn 2 left no record (None)
- screenshot conversation: turn 3 was not served from a record (path None)
- screenshot conversation: turn 3 left no record (None)
- screenshot conversation: turn 4 was not served from a record (path None)
- screenshot conversation: turn 4 left no record (None)
- screenshot conversation: turn 5 was not served from a record (path None)
- screenshot conversation: turn 5 left no record (None)
- screenshot conversation: turn 6 was not served from a record (path None)
- screenshot conversation: turn 6 left no record (None)
- screenshot conversation: turn 7 was not served from a record (path None)
- screenshot conversation: turn 7 left no record (None)
- screenshot conversation: turn 8 was not served from a record (path None)
- screenshot conversation: turn 8 left no record (None)
- screenshot conversation: turn 9 was not served from a record (path None)
- screenshot conversation: turn 9 left no record (None)
- screenshot conversation: turn 10 was not served from a record (path None)
- screenshot conversation: turn 10 left no record (None)
- screenshot conversation: turn 11 was not served from a record (path None)
- screenshot conversation: turn 11 left no record (None)
- screenshot conversation: turn 12 was not served from a record (path None)
- screenshot conversation: turn 12 left no record (None)
- screenshot conversation: the turn after the restart was not served from a record (path None)
