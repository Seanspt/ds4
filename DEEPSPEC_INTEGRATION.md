# DeepSpec Training Integration Design

Status: **design proposal**
Last updated: 2026-07-26

How to train a draft model for DeepSeek V4 Flash with the DeepSpec
framework (`/Volumes/datadisk/projects/DeepSpec`, clean upstream fork of
`deepseek-ai`'s DSpark/DFlash/EAGLE3 trainer), using **ds4 as the hidden
-state provider** instead of DeepSpec's HF-transformers target-cache
pipeline.

Related: `HIDDEN_EXPORT.md` (Phase 1 tap/export machinery this design
builds on).

## 1. Why ds4 instead of the stock pipeline

DeepSpec's stock cache builder (`scripts/data/prepare_target_cache.py`)
runs the target model with HF transformers in bf16 and captures layer
outputs with forward hooks. For DeepSeek V4 Flash that is impractical:
the model only exists as ds4-specific GGUFs, there is no HF checkpoint,
and ds4's Metal graph inference is orders of magnitude faster than an HF
reference path would be. ds4 already has the Phase 1 tap machinery that
captures arbitrary layer outputs during normal inference.

## 2. The DeepSpec data contract (what training consumes)

A target-cache directory (format v2, `deepspec/data/target_cache_dataset.py`):

```
manifest.json     # {version:2, num_samples, num_shards, target_layer_ids (asc),
                  #  hidden_dtype:"bfloat16", hidden_size:H, shards[], ...}
samples.idx       # N x 56B records "<QIIQQQQQ":
                  # (sample_id, shard_id, seq_len, off_input_ids, off_attn,
                  #  off_loss_mask, off_target_hidden, off_target_last_hidden)
shard-%05d.bin    # per-sample 5 byte-ranges, concatenated
```

Per sample (`T` = seq_len, `L` = len(target_layer_ids), `H` = hidden_size):

| field | shape | dtype | semantics |
|---|---|---|---|
| `input_ids` | `[T]` | int32 | full rendered conversation, target tokenizer |
| `attention_mask` | `[T]` | uint8 | written, ignored by the reader |
| `loss_mask` | `[T]` | uint8 | 1 on the assistant span; samples with <14 kept |
| `target_hidden_states` | `[T, L*H]` | bf16 | **raw decoder-layer outputs** (post-residual, pre-next-norm), layer ids ascending, concat on last dim |
| `target_last_hidden_states` | `[T, H]` | bf16 | hidden **after the final RMSNorm** |

Key semantics:

- Positions are **not shifted** in the cache; alignment happens in the
  trainer (DSpark: anchor `a`, slot `k` predicts token `a+1+k`; target
  logits for the L1/confidence losses come from
  `lm_head(target_last_hidden_states)`).
- The draft consumes `fc(target_hidden_states)` (`Linear(L*H -> H)`) as
  attention K/V context; Q comes from noisy block embeddings.
- `validate_train_cache` enforces cache `target_layer_ids` /
  `hidden_size` / target-name == draft config.

## 3. Contract mapping: DeepSeek V4 Flash via ds4

### 3.1 What "one layer's hidden" means for a hyper-connections model

Flash layers produce HC-expanded state (`hc_dim = 4 x 4096`). DeepSpec's
contract wants one `H`-wide vector per layer. The natural — and already
battle-tested — definition is **mean-over-HC** (`n_embd = 4096`): this is
exactly what the released Flash DSpark draft consumes in ds4
(`metal_graph_dspark_capture_hc`, ds4.c:25820) and what Phase 1 exports
as `DS4_TAP_FORMAT_MEAN_HC`. So for the cache contract:

- `hidden_size = 4096`, one MEAN_HC row per tapped layer.
- ds4's capture point is the layer output after the cur/after_ffn swap —
  post-residual, pre-next-norm — matching DeepSpec's "raw layer output"
  semantics.
- Any layer set is valid (the released draft uses 40,41,42; a fresh
  DeepSpec config would more likely spread, e.g. `[1, 11, 21, 31, 41]`
  of 43 layers). Note DeepSpec forbids the *last* layer only because of
  an HF `output_hidden_states` quirk; ds4 captures raw outputs so the
  last layer is fine — but if DeepSpec's HF-based evaluator is ever used
  against the same cache, avoid layer 42 or accept the semantic mismatch.

### 3.2 `target_last_hidden_states`

The final-norm hidden exists in ds4 as `output_norm` (n_embd, post
`output_norm` RMS norm, pre lm head). It is tapped through sentinel layer
ids in `ds4_session_set_hidden_taps` (landed, see the D2 milestone row):

```
DS4_TAP_LAYER_EMBED      -1   /* embedding row (mean-over-HC == plain embd) */
DS4_TAP_LAYER_FINAL_NORM -2   /* output_norm row, n_embd f32 */
```

mirroring DeepSpec's own `-1` = embedding convention. Both are `n_embd`
wide regardless of the session's tap format, and both are session-local
only (the export wire format keeps one uniform row width, so the export
channel rejects negative layer ids).

### 3.3 Tokens and loss_mask

`input_ids` are ds4's own token ids — the cache must be built from
ds4-rendered conversations, never re-tokenized elsewhere. `loss_mask`
needs the assistant span: ds4's renderer knows the assistant start marker
and the end-of-turn token, so the cache generator computes the mask from
token spans (same role as DeepSpec's `parser.py`, but for the DS4 chat
template — DeepSpec only ships qwen/gemma4 parsers, this part is ours).

### 3.4 dtype

Cache stores bf16; ds4 taps are f32. The writer casts f32 -> bf16 on
output (round-to-nearest-even), little-endian.

## 4. Pipeline design

Three stages, each independently testable:

### Stage A — response regeneration (text only)

Same role as DeepSpec's `generate_train_data.py`, but with ds4-server
running Flash (OpenAI-compatible API already exists): replay each dataset
conversation's prompt, sample the assistant response, write
`regen.jsonl`. This guarantees the training text distribution matches
ds4's own template, tokenizer, and generation behavior.

### Stage B — target-cache generation (new ds4 mode)

**Implemented** (D3, see the milestone row): `--role cache-gen`, offline,
no network:

```
./ds4 --role cache-gen -m ds4flash.gguf \
      --dataset regen.jsonl --cache-out cache_dir \
      --tap-layers 1,11,21,31,41 --target-name /path/to/DeepSeek-V4-Flash-DSpark \
      --max-length 4096 --min-loss-tokens 14
```

Per conversation:

1. Render + tokenize with ds4's template; truncate at max_length; compute
   `loss_mask` from the assistant span.
2. Create a session, configure taps (chosen layers as MEAN_HC +
   `DS4_TAP_LAYER_FINAL_NORM`), prefill the whole sequence in chunks —
   **teacher forcing only, no generation**: prefill already produces tap
   rows for every position (`ds4_session_read_tap` returns n_tokens rows
   per layer).
3. Write the sample straight into target-cache v2 format: `manifest.json`
   (`hidden_size=4096`, `target_layer_ids`, `hidden_dtype="bfloat16"`,
   `target_model_name_or_path="ds4/deepseek-v4-flash"`), `samples.idx`,
   `shard-%05d.bin` (64 GiB shard cap). All pieces are local — the DS4X
   network channel is *not* involved (it remains the streaming/online
   path).

This is the key efficiency point: HF pipeline cost = one full target
forward per sample; ds4 cache-gen = one prefill per sample at Metal
speed, with taps captured as a byproduct.

A C-side writer is preferred over a Python converter (no 2x storage for
an intermediate DS4H dump, reuses `ds4_session_read_tap` directly). The
v2 format is implemented in `ds4_cache_writer.c` from
`deepspec/data/target_cache_dataset.py:351-386, 580-602` as the
reference; `tests/test_cache_writer.c` + `tests/validate_target_cache.py`
prove the contract with DeepSpec's own `CacheDataset` (bit-exact). One
format correction vs an earlier draft of this doc: the index record
`"<QIIQQQQQ"` is **56 bytes**, and the manifest's `index_record_size` is
56 — DeepSpec's `INDEX_RECORD_SIZE` is the source of truth.

Model-machine acceptance for D3 (run 2026-07-26 on the M4 Max 128 GB
model machine, IQ2XXS chat-v2 GGUF — see the "D2/D3 model-machine
acceptance" milestone row):

1. Small real `regen.jsonl` through the command above: 5 lines (4 valid +
   1 two-assistant-turn sample) → 4 written (286 tokens, 220 loss tokens),
   1 dropped and counted ("more than one assistant turn"), 13.41 MiB in
   shards, 3.3 s wall (88 tok/s; tiny-sample figure, dominated by per-run
   setup). Two bugs found and fixed en route: chunking used the raw
   `ds4_engine_prefill_chunk` option (0 when unset → empty-prompt failure;
   now the session's resolved `ds4_session_prefill_cap`), and the tap-row
   base assumed prefix reuse where sync actually does a from-scratch
   rebuild for a shorter/divergent prompt (rows rebased to 0).
2. `tests/validate_target_cache.py` reads the produced cache with
   DeepSpec's own `CacheDataset` (the script now auto-branches: the
   synthetic "/fake/..." manifest keeps the bit-exact contract, anything
   else gets generic checks): 4 samples, shapes/dtypes per the v2
   contract, layers [1,11,21,31,41], hidden 4096, loss_mask 0/1 with
   sum >= 14, all hidden values finite. Sanity numbers: loss_mask is one
   contiguous tail run per sample (loss 41/28/64/87 tokens);
   `target_last_hidden_states` row L2 16.8..22.9 (mean ~21); same-position
   cosine layer 1 vs 11 mean 0.41, layer 31 vs 41 mean 0.55. The "feeds
   lm_head to consistent logits" spot check is covered more strongly on
   the ds4 side by `ds4_test --tap-golden-vectors` head consistency (the
   tapped RAW_HC frontier row through the output head reproduces the
   session logits bitwise).
3. `validate_train_cache` against the D5 draft config (layer ids, hidden
   size, target name): still pending — needs the draft model config, runs
   with D5 on the 5090.

### Stage C — DeepSpec training (on the 5090)

Training host: the RTX 5090 machine. The HF assets already exist there at
`~/models/deepseek-ai/DeepSeek-V4-Flash-DSpark/` — the official DSpark HF
repo, which supplies config, tokenizer, the shared `embed_tokens` /
`lm_head`, and the reference `mtp.*` draft weights. **No GGUF->HF
conversion project is needed**; before training, verify the repo contains
everything `base_trainer.py:269-280` loads (frozen embed/lm_head) plus a
tokenizer.

New config `config/dspark/dspark_ds4_flash.py`:

- `model.target_layer_ids` = the Stage B layer set, `hidden_size = 4096`,
  `target_model_name_or_path` = the 5090 HF path above. cache-gen must
  write the *same* string into the manifest (`validate_train_cache`
  compares it), so cache-gen takes a `--target-name` option.
- Draft architecture: DeepSpec's DSpark draft is a generic small decoder
  (`deepspec/modeling/dspark/qwen3/modeling.py`) whose only
  target-coupled parts are the **frozen `embed_tokens` and `lm_head`
  copies** and the mask/noise token id. Reusing the Qwen3 draft module
  with DS4 vocab/embed sizes is likely enough (it is a self-contained
  decoder), but expect a small `deepspec/modeling/dspark/ds4flash/` port
  if rope/norm details differ. The released Flash DSpark GGUF in ds4
  (`dspark.*` metadata: block_size, noise_token_id, target_layer_ids) is
  the ground-truth reference for the intended draft hyper-params — match
  them unless deliberately training a different draft.

The cache itself is produced on the Mac by Stage B and transferred to the
5090 (manifest + idx + shards are plain files; rsync is fine — plan the
transfer against the ~48 KB/token estimate in section 6).

### Stage D — trained draft back into ds4

`gguf-tools/deepseek4-quantize.c` already converts an HF DSpark
checkpoint into the support GGUF used by `ds4 --mtp`:

```sh
gguf-tools/deepseek4-quantize \
  --hf /path/to/converted-checkpoint \
  --dspark-support \
  --out DeepSeek-V4-Flash-DSpark-support.gguf
```

with `--dspark-manifest` to inspect tensor coverage first, and
`--dspark-block-size` / `--dspark-target-layers` / `--dspark-markov-rank`
/ `--dspark-noise-token-id` overrides for non-default drafts (see
`gguf-tools/README.md`, "Convert A DSpark Support Checkpoint").

The one remaining glue piece: the converter expects the **official repo
layout** (draft tensors under `mtp.0` / `mtp.1` / `mtp.2`), while
DeepSpec saves its own module names via `save_pretrained`. A small
repack script (DeepSpec checkpoint -> `mtp.*` safetensors layout +
index) closes that gap; `--dspark-manifest` reports any unmapped tensor
before a full write.

## 5. What this design deliberately does not do

- The **offline cache-gen path (Stage B) is always built first**, even when
  the goal is online training: it produces the fixed held-out eval cache,
  and it is the correctness reference for the streaming path (section 8).
- No shift in the cache (DeepSpec aligns internally).
- No HF-transformers forward of Flash anywhere in the path.

## 6. Storage and throughput estimates

- Per token: `L` layers x 4096 x 2 B + 4096 x 2 B (last hidden); for
  `L=5`: ~48 KB/token. A 1B-token corpus => ~48 TB — plan the corpus
  size accordingly (DeepSpec's own default config is ~38 TB).
- Cost per sample = one ds4 prefill of <= 4096 tokens (Metal), plus bf16
  cast + write. No generation, no draft, no network.

## 7. Risks and open questions

- **HF assets**: resolved — the official repo exists on the 5090 at
  `~/models/deepseek-ai/DeepSeek-V4-Flash-DSpark/`; verify at training
  time that it carries the frozen embed/lm_head and tokenizer the
  trainer loads.
- **Template/parser ownership**: the DS4 chat template's assistant-span
  detection must be implemented in cache-gen and must exactly match what
  ds4-server rendered in Stage A (use the same renderer for both).
- **bf16 rounding**: f32 -> bf16 cast loses precision vs an all-bf16 HF
  pipeline; draft training tolerates this in principle (DeepSpec stores
  bf16 itself), but the acceptance-rate impact should be sanity-checked
  on a small cache before a full run.
- **Contract drift**: cache v2 layout is copied from DeepSpec upstream;
  pin the expected `git_sha` in the manifest and re-check after pulling
  DeepSpec updates.
- **Checkpoint repack**: DeepSpec's `save_pretrained` names must be
  mapped to the `mtp.*` safetensors layout that
  `deepseek4-quantize --dspark-support` expects; `--dspark-manifest`
  validates coverage before a full conversion.
- **Eval path**: DeepSpec's evaluator runs the target with HF
  `output_hidden_states`; for Flash the final acceptance eval runs in
  ds4 instead (load the converted support GGUF via `--mtp`, compare
  acceptance rate and per-token behavior against the released draft).

## 8. Online training: train while ds4 infers (no tens-of-TB disk)

Full-corpus caches are tens of TB (section 6). If that disk does not
exist, training can run against a **rolling cache window** fed live by
ds4. This works with the stock DeepSpec trainer — do **not** bolt an
IterableDataset onto it:

- The trainer is epoch/sampler based (`StatelessResumableDistributedSampler`,
  `base_trainer.py:296`; `num_train_epochs=10` by default), and per-epoch
  random anchor resampling (`modeling/dspark/common.py:109-120`) is real
  data augmentation — 10 epochs over the same cache = 10 different anchor
  views of each conversation. A single-pass stream throws that away and
  would require surgery on the sampler, epoch math, resume, and LR
  schedule.
- The producer is the bottleneck anyway. One training step consumes
  `global_batch_size x seq_len` = 512 x 4096 ≈ 2M tokens; a 5090 chews
  through tokens far faster than one Mac can prefill them (ds4 Metal
  prefill ≈ 1-3k tok/s — measure on the target machine). So the natural
  regime is: window fills slowly, trainer runs multiple epochs over the
  window. A rolling window is not a compromise here, it is the regime the
  hardware dictates. Multiple Macs can feed the same 5090 to shrink the
  gap.

Architecture (three pieces, two repos):

1. **Mac, ds4 producer** — new role combining Stage A + B per
   conversation, streaming instead of writing files:

   ```
   ./ds4 --role stream-gen -m ds4flash.gguf \
         --dataset regen.jsonl --tap-layers 1,11,21,31,41 \
         --export-to 5090host:7800 --max-length 4096
   ```

   Renders + tokenizes, prefills (teacher-forced, taps captured as a
   byproduct), pushes the sample over DS4X. Requires DS4X protocol
   additions (Phase 1 shipped hidden rows only):
   - `SAMPLE_BEGIN { sample_id, seq_len, n_layers, layer_ids[], format }`
      / `SAMPLE_END { sample_id, prefix_hash }` framing — training
      consumes whole conversations, not arbitrary chunks;
   - token ids + loss_mask sections in the sample stream (the subscriber
      cannot reconstruct them; today only the prefix hash crosses);
   - `bf16` as a wire format (matches the cache dtype, halves bandwidth:
      ~98 KB/token f32 -> ~49 KB/token for L=5; at 2k tok/s that is
      ~100 MB/s — needs 10GbE or Thunderbolt, marginal otherwise; f32
      stays available for validation runs).

2. **5090, cache writer** — `scripts/data/stream_cache_writer.py` in the
   DeepSpec repo: a DS4X subscriber (grow `tests/hidden_export_client.py`
   into a small client lib) that assembles complete samples and writes
   them with **DeepSpec's own** cache writer helpers
   (`deepspec/data/target_cache_dataset.py` `write_sample_bytes` /
   manifest builder) — the v2 format is produced by its owner's code, so
   contract drift fails loudly there, not in ds4. Writes into
   `cache_active/`, rotates into `cache_next/`, deletes the oldest shards
   beyond the configured window. Sample atomicity: only complete,
   hash-verified samples are written; a desync drops the in-flight
   sample, logs, and continues (dropped samples are tolerable, corrupt
   ones are not).

3. **5090, stock trainer, unmodified** — trained in *rounds*: point the
   config at `cache_active/`, train k epochs, checkpoint; when
   `cache_next/` has accumulated enough fresh data, swap the symlink and
   resume from the checkpoint on the new window. Window size is a user
   knob, not a constraint: e.g. 500 GB ≈ 10M tokens ≈ 2.5k conversations
   of 4096 (formula: window_bytes / (48 KB x 4096) samples).

Eval and correctness gates:

- Build one small **fixed offline eval cache** (a few GB) with the D3
  cache-gen path; every training round evals against it — comparable
  numbers across rounds, no streaming noise.
- **Streaming == offline gate**: run the same conversations through
  cache-gen (file) and stream-gen (wire) and require byte-identical
  sample payloads. This is the streaming path's correctness proof; run
  it on every protocol or tap change.

What remains genuinely online (future, optional): a true IterableDataset
single-pass mode for when producers get fast enough that replay is
unnecessary. Not worth the trainer surgery at current speed ratios.

## 9. Sequence packing in DeepSpec (port from speculators)

Observation: DeepSpec has no sequence packing; `CacheCollator`
(`target_cache_dataset.py:859-870`) right-pads to the batch max. Note the
nuance: all shipped configs run `local_batch_size=1`, so padding waste is
zero today — the win from packing is **compute density**: many target-cache
conversations are far shorter than 4096, and each one currently occupies a
whole micro-step. Packing several short samples into one token-budget row
recovers GPU utilization per step, which matters in the online regime
(section 8) where every trainer cycle counts. speculators
(`vllm-project/speculators`, local clone at
`/Users/sean/datadisk/projects/speculators`) has a proven implementation to
port: `MultipackDistributedBatchSamplerV2`
(`src/speculators/train/distributed_batch_sampler.py`, itself adapted from
`imoneoi/multipack_sampler` — check license before vendoring) plus a packed
collator and a flex_attention document mask.

How it works there: lengths for the whole dataset -> deterministic per-epoch
shuffle -> binary search the largest prefix that LPT-bin-packs into
`world_size` bins of `batch_max_length` tokens; every rank runs the same
algorithm and keeps its own bin (no communication); collate concatenates the
bin's samples into one `[1, budget]` zero-padded row with `document_ids`
(-1 = pad), per-doc reset `position_ids`, and per-position loss_mask;
attention isolation via a flex_attention `document_mask_mod` (causal ∧
same-doc ∧ non-pad). Resume replays the deterministic packing list and
slices off completed steps (`speculators/train/trainer.py:386-417`).

DeepSpec port plan (DSpark draft first):

1. **Quantify before building**: `samples.idx` records carry `seq_len` —
   mmap the index and compute the length histogram (no data reads). If
   most samples are near 4096, skip D8 entirely.
2. **Sampler**: port `MultipackDistributedBatchSamplerV2` as a
   `batch_sampler`; bins = world_size x grad_accum (or world_size with
   accum over consecutive bins); token budget configurable (start at 4096
   so memory upper bound == today's bs=1 worst case: KV_LEN = 4096 +
   512x7).
3. **Packed CacheCollator**: emits `[1, budget]` rows — concat samples,
   zero tail, `document_ids`, per-doc reset `position_ids`, loss_mask with
   each doc's first position forced to 0 (prevents "doc-A-tail anchor ->
   doc-B-first-token" false supervision; the cache format itself does not
   change).
4. **Two-coordinate modeling change** (the delicate part): anchor
   positions are used both as packed-row *global* coordinates (gather,
   mask prefix test) and as RoPE positions. Keep global coordinates for
   gather/mask, add doc-relative coordinates for RoPE; touch
   `create_dspark_attention_mask` (add same-doc condition — easy,
   flex_attention already the backend), `create_position_ids`,
   `build_anchor_candidate_mask` / `build_eval_mask` (doc-boundary aware),
   and the shared `arange` context positions in
   `modeling/dspark/qwen3/modeling.py:412`.
5. **Trainer plumbing**: replace `StatelessResumableDistributedSampler`
   usage in `base_trainer.py:295-314` (run_batch interface unchanged —
   the batch is still one row); `steps_per_epoch` becomes the packing
   output; resume switches from sample-offset math to (epoch, step) +
   deterministic packing replay; extend ckpt training state accordingly.

Free wins already in place: the DSpark loss all-reduces denominators
globally (uneven token counts per rank are fine), Markov/RNN heads keep
no cross-block state, and flex_attention is already the attention backend.

Correctness gates: (a) doc-isolation unit test — zero attention weight
across document_ids; (b) contamination test — no anchor/eval index ever
crosses a doc boundary; (c) training smoke — acceptance-rate parity on the
fixed offline eval cache vs the unpacked baseline. EAGLE3 can reuse the
sampler/collator later with its own model-side changes.

## 10. Official DSpark draft architecture and the 5090 memory reality

Facts established by inspecting the official HF checkpoint
(`~/models/deepseek-ai/DeepSeek-V4-Flash-DSpark`, tensor headers):

- The draft is **3 stages** (`mtp.0/1/2`), each a **full DS4-architecture
  block**: LoRA attention (`wq_a/wq_b` 1024->32768, `wkv` 512x4096,
  `wo_a/wo_b`), **MoE FFN with 256 routed experts** (fp4-packed, logical
  3x2048x4096 per expert) + shared expert, and **hyper-connections**
  (`hc_attn_*` / `hc_ffn_*` over the 4x4096 HC state), plus attention
  sinks. DeepSpec's stock qwen3/gemma4 draft modeling cannot represent
  this — a `deepspec/modeling/dspark/deepseekv4/` port is required.
  Reference implementation: `inference/model.py` in the same HF repo.
- Stage extras: `mtp.0` has `main_proj` [4096, 12288] + `main_norm`
  (3 target layers x 4096); `mtp.2` has `norm`, `hc_head`, Markov head
  (2x [129280, 256] bf16), `confidence_head.proj` [1, 4352].
- Official hyper-params (config.json): `dspark_block_size=5`,
  `dspark_target_layer_ids=[40,41,42]`, `dspark_noise_token_id=128799`,
  `dspark_markov_rank=256`. **Loading the official draft as init means the
  training config must match these**, and the ds4 cache must tap layers
  40,41,42 (MEAN_HC) — layer 42 is fine here: the HF-eval prohibition
  does not apply to ds4-produced caches.
- Parameter count: ~19.5B (fp4 MoE experts dominate, ~6.4G per stage).
  **Full bf16 training does not fit on the 5090 (32 GB)** — weights alone
  would be ~39 GB. The viable single-GPU regime: keep the 256-expert MoE
  FFNs **frozen in fp4** (dequant on the fly, ~10 GB resident) and train
  attention (~0.3 B), `main_proj`/norms, Markov + confidence heads
  (~0.1 B) — roughly 0.5 B trainable -> ~1 GB weights + 1 GB grads +
  ~4 GB Adam, plus frozen embed/lm_head (~4.2 GB bf16) and activations
  with gradient checkpointing. Full-fidelity retraining is a multi-GPU
  project, not a 5090 one.
- 5090 disk is 97% full (~32 GB free) — checkpoint/cache housekeeping is
  part of the run plan; keep smoke caches small and clean up old
  checkpoints aggressively.

## 11. Milestone plan

| Milestone | Status | Date | Notes |
|---|---|---|---|
| D1: design doc | done | 2026-07-26 | this file; updated with 5090 training host + existing conversion tooling; online mode added (section 8) |
| D2: engine tap sentinels (FINAL_NORM, EMBED) | done | 2026-07-26 | DS4_TAP_LAYER_EMBED(-1)/FINAL_NORM(-2) in ds4.h, session-local (export wire rejects negative ids; --hidden-export-layers fails at parse); sentinel rows always n_embd wide, per-slot row widths in the tap table. FINAL_NORM decode reuses the output head's norm buffer; prefill rows run the batched output-head kernel prefix factored out of the speculative-verify head (metal_graph_encode_output_norm_batch), so a row is by construction the lm-head input of its position. EMBED = mean-over-HC of the initial hidden state (== plain embedding for token input), captured on decode/prefill/slice/input-hc paths; CPU reference path mirrors both. tap-golden test: 5-slot set {EMBED, 0, n/2, n-1, FINAL_NORM}, fixture v2 (i32 ids + per-slot widths), RAW_HC row -> output head reproduces frontier logits bitwise, cross-batch check loose (argmax + max_abs<=4.0, bound from model-machine measurement — see the acceptance row). Warning-free make; --tap-golden-vectors first run on the model machine writes the fixture (run there 2026-07-26, green) |
| D3: cache-gen role + v2 writer | done | 2026-07-26 | `./ds4 --role cache-gen` (ds4_cachegen.c): regen.jsonl parsing (own minimal JSON reader in ds4_server.c style; v1 = exactly one assistant turn, last; multi-assistant/non-last/unknown-role/null-content skipped + counted), rendering mirrors render_deepseek_chat_prompt_text (tool-free subset, think enabled; loss_mask span from prefix-render token count vs full-render token count, EOS included), truncation at --max-length with --min-loss-tokens drop. Tap set = user layers (MEAN_HC, -1=embedding ok, -2 rejected, ascending enforced) + FINAL_NORM always; chunked prefill at the session's resolved prefill cap (ds4_session_prefill_cap; the raw ds4_engine_prefill_chunk option is 0 when unset) with per-chunk read_tap (hsexport discipline). ds4_cache_writer.c: v2 manifest/samples.idx/shard writer, 56B "<QIIQQQQQ" records (task draft said 48 — DeepSpec INDEX_RECORD_SIZE is the truth), shard rotation mirrors LocalTargetCacheWriter, manifest tmp+rename last, f32->bf16 RNE bit-ops (torch-verified table). Contract proof: tests/test_cache_writer.c (make test) + tests/validate_target_cache.py reads the synthetic cache with DeepSpec's own CacheDataset BIT-EXACT (run locally with the deepspec conda env, torch 2.9.1). git_sha via Makefile -DDS4_GIT_SHA. Warning-free make; model-independent tests green; real regen.jsonl end-to-end run on the model machine 2026-07-26 (see the acceptance row) |
| D2/D3 model-machine acceptance | done | 2026-07-26 | M4 Max 128 GB, IQ2XXS chat-v2 GGUF. `./ds4_test --tap-golden-vectors`: zero-drift (tapped vs untapped logits memcmp + 8 greedy steps), same-batch head consistency bitwise, fixture tests/test-vectors/tap-golden.bin written then rerun bitwise green; cross-batch (33-token vs 64-token prefill) max_abs=1.2739 with argmax match — the pre-set 1e-3 bound was never hardware-validated and is relaxed to 4.0 with a comment: the equivalent production scenario (incremental prefill 33->64 vs one-shot 64) shows max_abs=2.0, hidden row rel diff 4%/cos 0.9986 after 61 layers, i.e. the drift is engine-inherent batch-tiling noise, not tap-related (tap path itself is bit-exact). `--role cache-gen` on a 5-line regen.jsonl: 4 written (286 tokens, 220 loss tokens), 1 dropped+counted (multi-assistant), 13.41 MiB shards, 3.3 s (88 tok/s, tiny-sample). Two cache-gen bugs fixed en route: chunk source (raw option 0 -> session prefill cap) and tap-row base on sync rebuild (see Stage B acceptance). tests/validate_target_cache.py auto-branches synthetic/real; synthetic /tmp/cw-cache still bit-exact OK, real cache OK via DeepSpec CacheDataset (shapes/dtypes/layers/hidden/loss_mask>=14/all-finite). Sanity: loss_mask one contiguous tail run per sample; last-hidden L2 16.8..22.9 mean ~21; cosine L1vL11 mean 0.41, L31vL41 mean 0.55. Still open from the checklist: validate_train_cache vs the D5 draft config (runs with D5 on the 5090) |
| D4: HF assets on 5090 | done | 2026-07-26 | official repo at 5090:~/models/deepseek-ai/DeepSeek-V4-Flash-DSpark/; verify embed/lm_head + tokenizer at training time |
| D5: DeepSpec ds4_flash config + training smoke | pending | | small cache end-to-end on the 5090; acceptance-rate sanity |
| D6: draft checkpoint -> support GGUF + ds4 acceptance | pending | | repack script (DeepSpec names -> mtp.* layout), then deepseek4-quantize --dspark-support; compare vs released draft |
| D7a: DS4X sample framing + tokens/loss_mask + bf16 wire | pending | | SAMPLE_BEGIN/END, token/mask sections, bf16 encode |
| D7b: stream-gen producer role (ds4) | pending | | render+prefill+push per conversation |
| D7c: stream_cache_writer.py (DeepSpec repo) | pending | | DS4X client lib + v2 writer helpers + window rotation |
| D7d: online gates | pending | | streaming==offline byte-identity test; fixed offline eval cache; throughput measurement on target machines |
| D8: sequence packing (DeepSpec side) | pending | | length histogram from samples.idx first; port MultipackDistributedBatchSamplerV2 + packed collator + document mask; two-coordinate modeling change; sampler/schedule/resume rewrite; gates in section 9 |
