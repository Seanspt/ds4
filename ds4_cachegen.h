#ifndef DS4_CACHEGEN_H
#define DS4_CACHEGEN_H

/* `./ds4 --role cache-gen`: offline DeepSpec target-cache producer
 * (DEEPSPEC_INTEGRATION.md Stage B). Reads a regen.jsonl-style dataset
 * (one {"messages":[...]} object per line, exactly one assistant turn as
 * the last message), renders each conversation with the ds4 chat template,
 * prefills it teacher-forced with hidden taps configured (chosen layers as
 * MEAN_HC + DS4_TAP_LAYER_FINAL_NORM), and writes DeepSpec target-cache v2
 * (ds4_cache_writer.c). No generation, no network. */

#include "ds4.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_CACHEGEN_DEFAULT_MAX_LENGTH 4096
#define DS4_CACHEGEN_DEFAULT_MIN_LOSS_TOKENS 14

typedef struct {
    const char *dataset_path;  /* required */
    const char *out_dir;       /* required */
    const char *tap_layers_csv;/* required; -1 (embedding) .. n_layer-1 */
    const char *target_name;   /* required; manifest target_model_name_or_path */
    int max_length;            /* 0 -> DS4_CACHEGEN_DEFAULT_MAX_LENGTH */
    int min_loss_tokens;       /* 0 -> DS4_CACHEGEN_DEFAULT_MIN_LOSS_TOKENS */
    uint64_t max_shard_bytes;  /* 0 -> writer default (64 GiB) */
    const char *git_sha;       /* manifest git_sha; may be NULL */
    bool (*cancel)(void *ud);  /* polled between samples/chunks; may be NULL */
    void *cancel_ud;
} ds4_cachegen_options;

/* Returns 0 on success (dropped samples are not failures; the run summary
 * on stderr reports them), 1 on error (err set), 2 when cancelled. */
int ds4_cachegen_run(ds4_engine *engine, const ds4_cachegen_options *opt,
                     char *err, size_t errlen);

#endif
