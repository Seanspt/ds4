# Hidden-State Export Design

Status: **design accepted, Phase 1 in progress**
Last updated: 2026-07-22

This document is the design and the living milestone log for the
hidden-state export feature: pushing the hidden states of arbitrary layers
to a remote node, to enable (a) independent deployment of the DSpark draft
model and (b) online training / activation harvesting.

## 1. Goals and non-goals

Goals:

- Tap the hidden state at **arbitrary layer indices**, in decode and prefill,
  with zero measurable cost when no tap is configured.
- Export taps over TCP to a subscriber that is **not** a pipeline worker —
  a training sink or a standalone draft process.
- Phase 2: run the DSpark draft model as a separate process/machine, with
  speculative output **bit-identical** to the in-process draft (greedy).

Non-goals:

- No encryption/authentication on the export channel (same trust model as
  the distributed protocol: trusted machines, trusted networks only).
- No attention internals (K/V, scores) — layer outputs only.
- No WAN support for the remote draft loop (one synchronous RTT per decode
  step; LAN/Thunderbolt only).

## 2. Key facts from the codebase

- Inter-layer hidden state is HC-expanded: `hc_dim = n_hc * n_embd` f32 per
  token (Flash: 4*4096 = 16384; GLM: plain `n_embd`, `n_hc=0`). Buffers:
  `cur_hc` / `batch_cur_hc` in `ds4_gpu_graph` (ds4.c:14869, 15049),
  ping-ponged with `after_ffn_hc` after each layer.
- The lm head consumes `output_norm` (`n_embd` f32): HC weighted sum +
  RMS norm, `metal_graph_encode_output_head()` (ds4.c:23962).
- A capture mechanism already exists for DSpark:
  `metal_graph_dspark_capture_decode_layer()` / `_prefill_rows()`
  (ds4.c:25868, 25938) record mean-over-HC rows (`n_embd`) of configured
  target layers (Flash: 40,41,42) into `g->dspark_target_hidden`, as a
  byproduct of the normal forward pass. This proves taps are free.
- Metal buffers are `MTLResourceStorageModeShared`; readback is a `memcpy`
  (`ds4_gpu_tensor_read`, ds4_metal.m:8003).
- Decode readbacks today: **only logits**. Hidden readback happens only in
  distributed layer slices (`ds4_session_eval_layer_slice`, ds4.c:57702,
  57805) and debug paths.
- `ds4.h` exposes distributed low-level entry points that already pass raw
  hidden states as `float *input_hc / output_hc`
  (`ds4_session_eval_layer_slice`, `ds4_session_eval_output_head_from_hc`,
  ds4.h:429-477). Dimension queries: `ds4_engine_hidden_f32_values()`,
  `ds4_engine_embd_dim()`.
- Wire skeleton to copy: `ds4_distributed.c` — 12-byte big-endian frame
  header `{magic, type, bytes}` (ds4_distributed.c:91-95), message-type
  table, activation payload with f32/f16/fp8 encoding
  (`dist_write_activation_payload`, ds4_distributed.c:959), reconnect
  loops, `prefix_hash` consistency checks, HELLO model-validation.
- Per-decode-step data dependency of the DSpark draft: (a) committed token
  id + position, (b) mean-over-HC captures of the target layers
  (`target_layer_count * n_embd` f32), (c) the main model's token embedding
  table. Draft-side private KV (`dspark_raw_cache`) rollback on partial
  accept is already solved locally by the dist-dspark worker (rollback +
  replay, ds4_distributed.c:7736-7795).

## 3. Architecture

Three independent layers, each separately testable:

```
engine tap (ds4.c, zero-cost when idle)
   -> narrow public API (ds4.h)
      -> export service (ds4_hsexport.c/.h, TCP)
         -> consumers: train sink (Phase 1) / remote draft (Phase 2)
```

### 3.1 Engine tap table

Generalize the DSpark capture into a tap table on `ds4_gpu_graph`:

- `tap_layers[]`: arbitrary layer set, not limited to the DSpark targets.
- Per-tap output format:
  - `RAW_HC` — the full `hc_dim` f32 row (training use case),
  - `MEAN_HC` — mean-over-HC `n_embd` f32 row (draft use case; reuses the
    existing `ds4_gpu_hc_weighted_sum_tensor` compression path).
- Hook points: the same call sites as the existing DSpark capture
  (decode per-layer, prefill batch rows, verified-suffix rows). No tap
  configured => no extra work, no readback.
- CPU reference path implements the same taps with plain memcpy. CUDA/ROCm
  use the generic `ds4_gpu_tensor_read` readback; no backend-specific
  optimization in Phase 1.

### 3.2 Public API (ds4.h, kept narrow)

```c
#define DS4_TAP_FORMAT_RAW_HC  0   /* hc_dim f32 per row (n_embd for GLM) */
#define DS4_TAP_FORMAT_MEAN_HC 1   /* n_embd f32 per row */

/* Replace the session's tap set. n_layers == 0 disables tapping. */
int ds4_session_set_hidden_taps(ds4_session *s, const int *layers,
                                int n_layers, int format);

/* Rows captured by the most recent eval for one tapped layer.
 * Decode: 1 row. Prefill/batch: one row per submitted token, in position
 * order. Returns row count, or <0 on error. */
int ds4_session_read_tap(const ds4_session *s, int layer,
                         float *out, int max_rows);
```

Callers see position-indexed rows only; no tensor internals cross ds4.h.

### 3.3 Export service (`ds4_hsexport.c/.h`)

Dedicated TCP channel; does **not** reuse the distributed WORK frame
ecology (the publisher is not necessarily a pipeline hop, and consumers
are not workers). Framing copies the distributed conventions: 12-byte
big-endian header `{ u32 magic "DS4X" (0x44533458), u32 type, u32 bytes }`,
then the payload. Phase 1 message types: `HELLO=1`, `ERROR=2`,
`HIDDEN_BATCH=3`; mode is `STREAM=1` only (DRAFT is Phase 2). Protocol
constants and the fixed records live in `ds4_hsexport.h`; the reference
subscriber is `tests/hidden_export_client.py`.

Finalized wire layout (all fixed-record fields big-endian):

- `HELLO` (subscriber -> publisher): `ds4_hsexport_hello_fixed`
  (6 x u32: model_id, n_layers, format, mode, ctx_size, model_name_len),
  then `u32 layer_ids[n_layers]`, then `model_name[model_name_len]`
  (no NUL). Two forms: an *exact* request must match the active model id,
  model name, format (RAW_HC), mode, and layer set (ctx_size 0 means
  "don't care"); or a *wildcard* request (`model_id == 0`,
  `n_layers == 0`, `model_name_len == 0` together) adopts whatever the
  publisher has active — that is what the train sink and the reference
  client use by default. A partially wildcarded request is rejected like
  any mismatch: the publisher replies `ERROR` and closes.
- `HELLO` (publisher -> subscriber, the ack): on a successful handshake
  the publisher answers with the same record shape carrying the *active*
  configuration (model id, layer set, format, mode, ctx of the first
  attached session, model name), so wildcard subscribers learn the model
  identity before the first batch. The subscription is registered before
  the ack is written, so a batch emitted right after the ack can never
  race into the no-subscriber drop path.
- `HIDDEN_BATCH` (publisher -> subscriber): `ds4_hsexport_batch_fixed`
  (11 x u32 = 40 bytes: session_id hi/lo, pos0, n_tokens, n_layers, format,
  bits, row_values, prefix_hash hi/lo, reserved), then
  `u32 layer_ids[n_layers]`, then `i32 token_ids[n_tokens]`, then the
  payload: `n_layers * n_tokens * row_values` elements of `bits/8` bytes,
  layer-major (per layer, `n_tokens` consecutive rows). `token_ids` let the
  subscriber rebuild the hash chain locally. `prefix_hash` is the FNV-1a
  token hash (same INIT/PRIME as the distributed protocol,
  `ds4_dist_token_hash_update_span`) of the whole session timeline
  *including* this batch: `tokens[0, pos0 + n_tokens)`.
- `ERROR`: UTF-8 message; fatal for the connection. Sent on handshake
  rejection and on desync (position gap or tap row mismatch) — no silent
  recovery, because a gap would corrupt a training stream.

Stream semantics:

- **One subscriber per publisher** (Phase 1); extra connections wait in the
  listen backlog. One active tap layer set per publisher, fixed by the
  first `ds4_hsexport_attach`.
- **(Re)connect re-anchors.** The first emit after a connect adopts the
  live session frontier as the stream anchor: the subscriber must adopt
  that first batch's `prefix_hash` as its chain anchor and can verify every
  later batch incrementally from `token_ids`.
- **Append-only assumption.** Sessions are expected to grow monotonically.
  A prompt rewrite/KV truncation moves the frontier backwards, which the
  exporter reports as a position-gap desync (ERROR + close); the subscriber
  reconnects and re-anchors at the new frontier.
- **Backpressure, never drops.** Emitted batches queue in a bounded byte
  budget (default 256 MiB). With a subscriber connected and the budget
  exhausted, `ds4_hsexport_emit` blocks the caller (training must not lose
  data); an empty queue always accepts one batch, even oversized, so a
  single large prefill chunk cannot deadlock. With no subscriber, emit
  advances the frontier without queueing.
- **Desync drains first.** When a position gap or tap row mismatch poisons
  the stream, the sender first delivers every batch still queued (they were
  accepted while the stream was consistent), then the `ERROR` frame, then
  closes — a training consumer keeps all valid data up to the gap.
- **Shutdown is prompt.** `ds4_hsexport_stop` wakes the accept loop with a
  dummy self-connection (a `shutdown()` on the listen socket does not wake
  a blocked `accept()` on macOS) and shuts down the subscriber socket so a
  blocked sender thread exits instead of hanging.
- **Capture vs wire width.** Taps always capture f32 RAW_HC;
  `--hidden-export-format f32|f16` selects only the wire payload width
  (`bits`), using the same f32->f16 helper as the distributed payload.
- Prefill streams chunk-aligned: when export is active the server/CLI drive
  `ds4_session_sync` one engine prefill chunk at a time, so each internal
  eval's tap rows (a tap holds only the most recent eval) are emitted
  exactly once. Decode emits one batch per token.

### 3.4 Consumer semantics

**Train sink (landed in M4):** pure stream, no replies, no rollback.
`ds4_hsexport_sink_run` connects with a wildcard HELLO, validates the ack,
and verifies every `HIDDEN_BATCH` before appending it to the dump:

- shape fields (`n_layers`, layer ids, `format`, `bits`, `row_values`)
  must be constant across the stream and match the ack;
- per session, `pos0` must continue the previous batch (sessions may
  interleave arbitrarily; each is tracked independently);
- the FNV-1a `prefix_hash` chain must extend from the previous batch over
  this batch's `token_ids`. The first batch of a session (or the first
  after a reconnect) is the chain anchor and is adopted unverified — the
  publisher re-anchors there by design.

Any violation, an `ERROR` frame, or a truncated record is fatal with a
nonzero exit (desync is never silent for a training consumer). SIGINT/
SIGTERM stop the loop at the next batch boundary and still leave a valid,
self-describing dump file behind.

Dump layout (all fixed-record fields big-endian; payload rows keep their
on-wire little-endian element order, so a dump is a byte-exact archive of
the stream):

- Header: 12 x u32 — magic `"DS4H"` (0x44533448), version 1, model_id,
  n_layers, format, bits, row_values, ctx_size, model_name_len, then 3
  reserved u32 (zero) — followed by `u32 layer_ids[n_layers]` and
  `model_name[model_name_len]` (no NUL). Written when the first batch
  arrives, because `bits`/`row_values` are learned from the wire, not the
  handshake.
- Records: `u32 record_bytes` (payload byte count that follows), then the
  verbatim `HIDDEN_BATCH` payload bytes — one record per batch, in arrival
  order. A reader can re-parse every record with
  `ds4_hsexport_parse_batch`.

The sink never loads a model and uses plain `read`/`write` I/O for the
dump (no extra VM mappings), same rule as the KV cache files.

**Remote draft (Phase 2, the hard part is the verify loop, not the push):**

- Per decode step: publisher (target node) sends `DRAFT_REQ
  { session_id, pos, committed_token_id, captures }`; draft node runs the
  DSpark stage chain and replies `DRAFT_RESP { draft_len, draft_tokens[] }`.
  One synchronous RTT on the critical path per step — acceptable on
  Thunderbolt/10GbE (~50-100 us), pointless across WAN. Documented limit.
- Verify/rollback: target batch-verifies; the resolution is sent as
  `DRAFT_RESOLVE { keep_count }`; the draft node mirrors the local
  dist-dspark worker behavior: roll back its private KV frontier and
  replay the accepted prefix (ds4_distributed.c:7736-7795 as reference).
  Frame-level consistency via `prefix_hash`.
- The draft node needs the main model's embedding table and the shared
  output head. Phase 2 starts with the zero-tooling option: mmap the main
  GGUF on the draft node but bind only embedding/head tensors (the support
  GGUF carries the draft weights). Baking them into the support GGUF via
  gguf-tools is a later optimization.
- Correctness gate: remote-draft output must be **token-identical** to
  in-process draft output (greedy). Acceptance reuses the
  `tests/dist_dspark_acceptance.sh` approach; no sampling-based checks.

### 3.5 CLI sketch

Publisher side (landed in M3 on `ds4` and `ds4-server`):

```
--hidden-export-listen HOST PORT     # also: off when absent
--hidden-export-layers 40,41,42
--hidden-export-format f32|f16       # f32 default; draft taps MEAN_HC
```

M3 wiring limits (startup refuses these combinations with a clear error):

- `ds4`: interactive REPL only (one-shot `-p`/`--prompt-file` generation is
  not wired); not with `--mtp`/`--dspark`/`--glm-mtp`.
- `ds4-server`: not with `--batched-session` or speculative decoding.
- GLM sessions, tensor-parallel leaders, and distributed coordinators are
  rejected by `ds4_session_set_hidden_taps` (attach fails -> startup
  error).

Consumer side (train sink landed in M4; remote draft still Phase 2):

```
./ds4 --role train-sink --hidden-source HOST PORT --dump hiddens.bin
./ds4 --role draft --hidden-source HOST PORT --mtp SUPPORT.gguf --dspark   # Phase 2
```

The sink loads no model (`-m` is ignored), handshakes
wildcard, and refuses `-p`/`--prompt-file` and `--hidden-export-*` like a
`--role none` CLI refuses distributed options; it exits cleanly on
SIGINT/SIGTERM. An explicit (non-wildcard) handshake is available in
`tests/hidden_export_client.py` (`--model-id/--layers/--model-name`).

### 3.6 Precision policy

- Training/raw taps: f32 by default (do not silently degrade training
  data); f16 only when explicitly requested.
- Draft taps: MEAN_HC f32 on the wire initially. Whether f16 captures
  degrade draft acceptance must be measured locally first (capture ->
  quantize f16 -> feed in-process draft -> compare acceptance rate)
  before f16 becomes a wire option for DRAFT_REQ.

## 4. Correctness gates

Per project rules (AGENTS.md, CONTRIBUTING.md):

- Warning-free `make`; `make test` green.
- Tap disabled => Metal default path speed unchanged (ds4-bench before/
  after on the same machine).
- Tap values validated against a golden fixture: captured hidden rows feed
  the known-sane logits comparison (extend `ds4_test --local-golden-vectors`
  style checks).
- Phase 2 acceptance: remote draft == local draft, token for token, plus
  the dist-dspark acceptance fixture adapted to the two-process setup.
- Sibling backends: tap code must compile and be inert on CUDA/ROCm; the
  SSD-streaming and distributed paths must be unaffected (tap storage is
  per-session, orthogonal to expert streaming and pipeline slicing).
  Distributed + remote draft combination: taps attach on the hop that owns
  the tapped layers (for DSpark targets: the final-hop worker), never
  duplicated on the coordinator.

### 4.1 M4 acceptance checklist (needs a model machine)

The model-free gates (`make`, `make test` including `tests/test_hsexport`,
`hidden_export_client.py --self-test`) run anywhere. The rest needs a GGUF:

- `./ds4_test --tap-golden-vectors` — tapped vs untapped session must give
  byte-identical logits and greedy tokens (zero-drift rule), and the tap
  rows must match the golden fixture bitwise (first run writes
  `tests/test-vectors/tap-golden.bin`; rerun compares).
- End-to-end, three-way on one stream: `ds4-server --hidden-export-listen`
  publishing, `tests/hidden_export_client.py --verify-hash` and
  `./ds4 --role train-sink --dump out.bin` subscribing one at a time must
  agree — client hash chain green, sink exit 0, and a re-parse of
  `out.bin` records (`ds4_hsexport_parse_batch`) must reproduce the same
  per-batch `prefix_hash` sequence.
- Zero overhead with taps off: `ds4-bench` sweep (CONTRIBUTING.md
  canonical) identical with and without `--hidden-export-listen` configured
  but no subscriber; with a subscriber, prefill/gen t/s drop is explained
  by the capture readback alone. In the engine every tap hook short-
  circuits on `tap_count == 0`, so the no-export path is untouched by
  construction.
- `make test` green on the model machine.

## 5. Risks and open questions

- HC format for training: RAW_HC rows are 16384 f32/token/layer on Flash;
  a full 43-layer stream is ~2.9 MB/token f32. Fine on LAN, but consumers
  should subscribe to the layers they need.
- Whether training wants post-norm `output_norm` instead of raw layer
  outputs is left to the consumer; taps are pre-norm layer outputs.
- Runtime reconfiguration of the tap set: supported by re-calling
  `ds4_session_set_hidden_taps`; subscribers that disagree with the active
  set are rejected at HELLO (no per-subscriber tap sets in Phase 1 — one
  active tap configuration per session).
- f16 acceptance-rate impact on draft: open, must be measured before use.

## 6. Milestone log

| Milestone | Status | Date | Notes |
|---|---|---|---|
| M1: design doc landed | done | 2026-07-22 | this file |
| M2: engine tap table + ds4.h API | done | 2026-07-25 | tap table on ds4_gpu_graph (RAW_HC/MEAN_HC, DS4_MAX_TAP_LAYERS=8), GPU capture + host readback on single-session decode/prefill/SSD-streaming/layer-slice paths, CPU reference path, ds4_session_set_hidden_taps/read_tap. GLM sessions, TP, and distributed coordinators rejected (GLM tap = follow-up). Batched-session and speculative-verify paths not wired yet (captures there land with M3). Warning-free make; model-independent tests green; no model on this machine, so ds4_test/golden vectors not run |
| M3: ds4_hsexport stream export + CLI | done | 2026-07-25 | DS4X framing, HELLO validation, HIDDEN_BATCH (final layout in 3.3), single-subscriber sender thread with bounded backpressure queue, reconnect re-anchor, desync poisoning (ERROR+close). --hidden-export-listen/layers/format on ds4 (REPL only) and ds4-server; startup refuses batched-session and speculative decoding with export; chunk-aligned prefill sync so no tap rows are skipped. tests/hidden_export_client.py reference subscriber + --self-test. Warning-free make; model-independent tests green (ds4_agent_test, test_layer_pack, test_gpu_args, test_engine_mgpu_placement, test_q4k_dot); no model on this machine, so no end-to-end stream verification yet |
| M4: train-sink dump + tests | done | 2026-07-25 | Wildcard HELLO + HELLO-ACK (3.3), `./ds4 --role train-sink --hidden-source HOST PORT --dump FILE` model-free consumer with shape/position/hash-chain validation and the finalized dump layout (3.4), SIGINT/SIGTERM clean exit. tests/test_hsexport.c: 9 loopback cases (handshakes, f32/f16 roundtrip with pinned FNV/f16 vectors, desync gap, backpressure, oversized batch, re-anchor, sink dump) wired into `make test`. ds4_test --tap-golden-vectors added (runs on a model machine, 4.1). Serve loop now drains queued batches before the desync ERROR. M3 bugs fixed en route: stop() hung in accept() with no subscriber (dummy-connect wake), reference client computed the first batch's hash from INIT instead of adopting the anchor, sub_fd set after the ack raced the first emit. Warning-free make; model-free tests green (test_hsexport, ds4_agent_test, test_layer_pack, test_gpu_args, test_engine_mgpu_placement, q4k-dot-test, client --self-test); model gates in 4.1 not run here (no model on this machine) |
| P2: remote draft loop | pending | | DRAFT_REQ/RESP/RESOLVE, KV rollback mirror, token-identical acceptance |
