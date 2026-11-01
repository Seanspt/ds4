#ifndef DS4_CACHE_WRITER_H
#define DS4_CACHE_WRITER_H

/* DeepSpec target-cache v2 writer (DEEPSPEC_INTEGRATION.md Stage B).
 *
 * Self-contained module: no engine dependencies, plain stdio write I/O (the
 * project convention for disposable cache files — no mmap). Produces the
 * three-piece layout consumed by DeepSpec's
 * deepspec/data/target_cache_dataset.py:
 *
 *   manifest.json     written last, via tmp + rename (no half-built caches)
 *   samples.idx       N x 56B records, struct "<QIIQQQQQ":
 *                     (sample_id, shard_id, seq_len, off_input_ids,
 *                      off_attention_mask, off_loss_mask, off_target_hidden,
 *                      off_target_last_hidden) — offsets are byte offsets
 *                     inside the sample's shard file
 *   shard-%05d.bin    per sample 5 contiguous byte ranges in fixed order:
 *                     input_ids [T] int32 -> attention_mask [T] uint8 ->
 *                     loss_mask [T] uint8 -> target_hidden_states [T, L*H]
 *                     bf16 -> target_last_hidden_states [T, H] bf16
 *
 * Like DeepSpec's own writer (numpy .tobytes()) the byte streams are host
 * order; every supported ds4 target is little-endian. The index record is
 * packed with explicit LE byte stores. bf16 is converted from f32 with
 * round-to-nearest-even (pure bit ops, matching torch's f32 -> bfloat16).
 *
 * Shard rotation mirrors LocalTargetCacheWriter._ensure_shard: a sample is
 * never split; when the current shard is non-empty and the sample would
 * push it past max_shard_bytes, shard_id increments and a new file opens. */

#include <stddef.h>
#include <stdint.h>

#define DS4_CACHE_WRITER_VERSION 2
#define DS4_CACHE_WRITER_INDEX_RECORD_SIZE 56 /* struct "<QIIQQQQQ" */
#define DS4_CACHE_WRITER_DEFAULT_MAX_SHARD_BYTES (64ull * 1024 * 1024 * 1024)

typedef struct ds4_cache_writer ds4_cache_writer;

typedef struct {
    const char *out_dir;         /* created; must not exist or be empty */
    uint64_t max_shard_bytes;    /* 0 -> DS4_CACHE_WRITER_DEFAULT_* */
    int hidden_size;             /* H, e.g. 4096 */
    const int *layer_ids;        /* L ids, ascending; -1 = embedding allowed */
    int n_layers;                /* L, >= 1 */
    /* manifest extras (all copied; may be NULL except target_name) */
    const char *target_name;     /* target_model_name_or_path (required) */
    const char *source_jsonl;    /* source_jsonl_paths[0] */
    const char *git_sha;         /* ds4 repo sha */
    int max_length;
    int min_loss_tokens;
} ds4_cache_writer_options;

int ds4_cache_writer_open(ds4_cache_writer **out,
                          const ds4_cache_writer_options *opt,
                          char *err, size_t errlen);

/* Write one sample. sample_id must equal the count of samples written so
 * far (the reader requires dense, ordered ids). layer_hidden_f32[l] points
 * at seq_len * hidden_size f32 values of tap layer l (layers ascending by
 * id); last_hidden_f32 holds seq_len * hidden_size values of the final-norm
 * tap. input_ids and loss_mask are seq_len long; attention_mask is written
 * as all ones (the DeepSpec reader ignores it). The writer interleaves to
 * the [T, L*H] position-major layout and casts to bf16. */
int ds4_cache_writer_write_sample(ds4_cache_writer *w,
                                  uint64_t sample_id,
                                  const int32_t *input_ids,
                                  const uint8_t *loss_mask,
                                  uint32_t seq_len,
                                  const float *const *layer_hidden_f32,
                                  const float *last_hidden_f32,
                                  char *err, size_t errlen);

/* Flush + fsync shard/index, then write manifest.json (tmp + rename).
 * Frees the writer; *w is always consumed, even on error. */
int ds4_cache_writer_close(ds4_cache_writer *w, char *err, size_t errlen);

/* Stats for the caller's end-of-run report. */
uint64_t ds4_cache_writer_samples(const ds4_cache_writer *w);
uint64_t ds4_cache_writer_bytes(const ds4_cache_writer *w);

/* f32 -> bf16, round-to-nearest-even (exposed for the unit test). */
uint16_t ds4_f32_to_bf16(float f);

#endif
