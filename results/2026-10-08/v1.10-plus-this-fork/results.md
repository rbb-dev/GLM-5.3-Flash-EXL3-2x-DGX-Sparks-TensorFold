# Resumable conversations: results

## v1.10 + fork (2026-10-08 23:50:42)

Model `glm-5.3-flash`; sampling {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "max"}.

Nodes (before the tests):

| node | time | GPU clock (now / max) | GPU temperature | GPU power | CPU clock | CPU temperature | memory | memory available | kernel | driver | disk free | CPU governor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spark 1 | 2026-10-08 23:57:56 AEDT | 2190 / 3003 MHz | 53 C | 14.60 W | 2651-3919 MHz | 56.7-58.9 C | 121.6 GiB | 11.0 GiB | 6.17.0-1029-nvidia | 580.173.02 | 2365G | performance |
| spark 2 | 2026-10-08 23:57:56 AEDT | 2177 / 3003 MHz | 53 C | 11.26 W | 2678-3900 MHz | 55.7-58.5 C | 121.6 GiB | 12.5 GiB | 6.17.0-1029-nvidia | 580.173.02 | 2365G | performance |

Nodes (end):

| node | time | GPU clock (now / max) | GPU temperature | GPU power | CPU clock | CPU temperature | memory | memory available | kernel | driver | disk free | CPU governor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spark 1 | 2026-10-09 00:51:00 AEDT | 2190 / 3003 MHz | 58 C | 11.55 W | 2679-3933 MHz | 62.1-68.5 C | 121.6 GiB | 10.2 GiB | 6.17.0-1029-nvidia | 580.173.02 | 2356G | performance |
| spark 2 | 2026-10-09 00:51:01 AEDT | 2190 / 3003 MHz | 57 C | 11.92 W | 2684-3900 MHz | 61.5-65.1 C | 121.6 GiB | 11.3 GiB | 6.17.0-1029-nvidia | 580.173.02 | 2356G | performance |

Engine settings (end of the run):

| setting | value |
| --- | --- |
| image | tensorfold-glm53:v0.6.0-mgllm-67e2b6640113 |
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
| TENSORFOLD_NUCLEUS_UNION | 1 |
| TF_GLM_COPY_DRAFTS | 1 |
| TF_GLM_MTP | auto |
| TF_GLM_SHARED_PREFIX | 1 |
| TF_GLM_EXL3_LOADS | nc |
| TF_GLM_EFFORT_TAIL | 0 |
| TF_GLM_COMM | roce |
| TF_GLM_KDA_CHUNKED | 1 |
| TF_SPILL_FLUSH_S | 60 |
| TF_GLM_FILL_DRAFTS | 1 |
| TF_ROCE_MAX_KB | 1024 |
| TF_GLM_EXL3_DEC_ORDER | 0 |
| TF_GLM_DENSE | q4 |
| TF_GLM_L2PF | 1 |
| TENSORFOLD_MEMORY_RESERVE_GIB | 17.7 |
| TF_ROCE_WAIT_S | 300 |
| TF_GLM_STREAM_SMOOTH_MS | 400 |
| TF_GLM_CACHE_ENTRIES | 16 |
| TF_GLM_DISPLAY_KV_BACKEND | drm |
| TF_GLM_MULTI_PREFILL | 1 |
| TF_GLM_CLEAR_THINKING | 0 |
| TF_GLM_KV | fp8 |
| TF_GLM_STREAM_SMOOTH | 1 |
| TF_GLM_COPY_REPLY_MATCH | 16 |
| TF_GLM_FILL_BUDGET_MS | 200 |
| TF_GLM_PREFILL_OVERLAP | 2 |
| TF_GLM_WIDE_GRAPHS | 16 |
| TF_GLM_MULTI_LONE | 0 |
| TF_GLM_MAX_QUEUED |  |
| TF_GLM_DFLASH_POLICY | fnc7:0.3 |
| TF_GLM_CACHE_SHARE_PCT | 0 |
| TF_GLM_DISPLAY_KV_MIB | 0 |
| TF_GLM_CACHE_GIB | 12.5 |
| TF_GLM_HC_SPLIT | 1 |
| TF_GLM_KEEP_PER_CHAT | 0 |
| TF_GLM_COPY_MAX | 15 |
| TF_GLM_KEPT_BYTES_GIB | 0 |
| TF_GLM_MULTI_WINDOW | 64 |
| TENSORFOLD_NO_UPDATE_CHECK | 1 |
| KV pool (tokens) | 2,181,120 |

### Engine start

From the restart hook to the first one-token reply.

| Start | Restart hook | First reply |
| --- | --- | --- |
| cold kernels (compiled at this start) | 305.7 s | 305.9 s |
| warm kernels | 126.0 s | 126.1 s |

### Resume after a restart, for a client that drops the thinking

Turn 1 reads a new conversation (cold); turn 2 resumes it from memory (warm); after a restart and a page-cache drop, turn 3 resumes from the saved state on the disk, and the same request with `"draft": false` reads the same prompt fresh. Time to first token (TTFT) each time.

| Turn 1 prompt | Turn 1, cold | Turn 2, warm | Turn 3 prompt | Turn 3, from disk | Read from disk | Thinking put back (tokens) | Turn 3 read fresh | Same reply | Path (1 / 2 / 3 / fresh) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8,007 | 7.06 s | 0.75 s | 8,434 | 0.66 s | 97% | 182 | 4.86 s | yes | render / record / record / record |
| 64,211 | 33.8 s | 0.79 s | 64,953 | 0.86 s | 100% | 497 | 33.4 s | yes | render / record / record / record |
| 256,869 | 149.9 s | 0.96 s | 257,395 | 1.47 s | 100% | 319 | 146.9 s | yes | render / record / record / record |

Control: turn 1 drafted and turn 1 read fresh gave the same reply. "Same reply" shows the saved state read back from the disk is exactly a fresh read of the same prompt; that the prompt is the true conversation is proven by the CPU tests.

### An agent's conversation after a restart: the material after the first reply

Turn 1 is a short exchange, turn 2 brings the material (cold), turn 3 resumes it from memory; after a restart and a page-cache drop, turn 4 resumes from the saved state on the disk, and the same request with `"draft": false` reads it fresh. The client drops the thinking.

| Material | Turn 2, cold | Turn 3, warm | Turn 4 prompt | Turn 4, from disk | Read from disk | Turn 4 read fresh | Same reply | Path (2 / 3 / 4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256,903 | 149.5 s | 0.97 s | 257,205 | 0.93 s | 100% | 147.3 s | yes | render / record / record |

### An agent's tool call after a restart: the material as the tool's result

Turn 1 must call a tool; turn 2 sends the material back as its result (cold), turn 3 resumes from memory; after a restart and a page-cache drop, turn 4 resumes from the disk, and the same request with `"draft": false` reads it fresh. The client drops the thinking, the tool call's too: a server that put it back from RAM no longer can after a restart.

| Material | Turn 2, cold | Turn 3, warm | Turn 4 prompt | Turn 4, from disk | Read from disk | Turn 4 read fresh | Same reply | Path (2 / 3 / 4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256,867 | 150.8 s | 1.13 s | 257,425 | 1.15 s | 100% | 147.0 s | yes | record / record / record |

### A screenshot conversation, resumed every turn

An agent watching a screen: turn 1 sends a new 1080p screenshot and a question (cold); each later turn adds a new screenshot and sends the conversation back without the thinking, so each must be served from its record, past the 8th picture too. After a restart and a page-cache drop, the next turn resumes from the saved state on the disk, and the same request with `"draft": false` reads it fresh.

| Turn | Screenshots | Prompt | From a saved state | TTFT | Served from |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 2,091 | 0 | 1.89 s | render |
| 2 | 2 | 4,419 | 2,048 | 3.01 s | record |
| 3 | 3 | 6,847 | 4,416 | 2.33 s | record |
| 4 | 4 | 9,173 | 6,784 | 2.77 s | record |
| 5 | 5 | 11,451 | 9,152 | 2.42 s | record |
| 6 | 6 | 13,749 | 11,392 | 2.92 s | record |
| 7 | 7 | 16,032 | 13,696 | 2.73 s | record |
| 8 | 8 | 18,311 | 16,000 | 2.86 s | record |
| 9 | 9 | 20,594 | 18,304 | 2.43 s | record |
| 10 | 10 | 22,855 | 20,544 | 2.85 s | record |
| 11 | 11 | 25,129 | 22,848 | 2.47 s | record |
| 12 | 12 | 27,408 | 25,088 | 2.92 s | record |
| 13, after a restart (from disk) | 13 | 29,695 | 27,392 | 3.15 s | record |
| 13, read fresh | 13 | 29,695 | 0 | 22.6 s | record |

After the restart, 92% of the prompt came from the saved state and the reply was the same as the fresh read's. Controls: turn 1 drafted and read fresh gave the same reply; turn 1 with another picture of the same size gave a different reply (the checksum sees the pixels). The last request's body: 7.0 MiB.

The largest picture, one 3840x2160 screenshot (laid out at the per-picture cap): 2,091 prompt tokens, TTFT 1.95 s.

### Serving while a long request arrives

Reply A decodes while request B arrives: A's tokens a second (largest gap) in the 10 s before B, in the 10 s after B arrives, over B's whole wait for its first token (while its prompt is read), and in the 10 s after B ends; B's last token to its closing event.

| B prompt | A before | A as B arrives | A while B prefills | A after B ends | B TTFT | B close |
| --- | --- | --- | --- | --- | --- | --- |
| 1,021,598 | 44.0 (0.06 s) | 21.7 (0.182 s) | 8.32 (1.292 s) | 45.1 (0.052 s) | 1150.3 s | 0.09 s |
| 16 screenshots (32,717 tokens) | 44.1 (0.062 s) | 16.2 (0.2 s) | 9.06 (0.4 s) | 44.8 (0.055 s) | 39.8 s | 0.05 s |

### Inside the engine: pauses between decode rounds

The engine's stall meter (TensorFold patch 9008) times every pause between two decode rounds while streams decode, and what filled it: admit (taking a request in), round (prompt chunks read with the decode round), finish, reply, other. A client sees such a pause only when it outlasts the 0.4 s stream smoothing. Largest pause in each window (its largest part), and the pauses over the whole row.

| B prompt | Before B | As B arrives | While B prefills | After B ends | Pauses ≥ 0.25 s | Pauses ≥ 1 s |
| --- | --- | --- | --- | --- | --- | --- |
| 1,021,598 | < 0.05 s | 0.46 s (round 0.418) | 1.98 s (round 1.972) | < 0.05 s | 3 | 1 |
| 16 screenshots (32,717 tokens) | < 0.05 s | 0.61 s (round 0.613) | 0.99 s (round 0.985) | < 0.05 s | 17 | 0 |
