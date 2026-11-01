#include "ds4_cache_writer.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct ds4_cache_writer {
    char *out_dir;
    uint64_t max_shard_bytes;
    int hidden_size;
    int *layer_ids;
    int n_layers;
    char *target_name;
    char *source_jsonl;
    char *git_sha;
    int max_length;
    int min_loss_tokens;

    FILE *idx;               /* samples.idx, appended per sample */
    FILE *shard;             /* current shard-%05d.bin */
    uint32_t shard_id;       /* valid when shard != NULL */
    uint64_t shard_size;
    char (*shard_names)[32]; /* manifest shards[] entries, in order */
    int n_shards;
    int cap_shards;

    uint64_t num_samples;
    uint64_t total_bytes;    /* shard payload bytes written */

    uint16_t *bf16_scratch;  /* n_layers * hidden_size entries */
    uint8_t *ones;           /* attention_mask fill, grown as needed */
    size_t ones_cap;
};

uint16_t ds4_f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    /* Round-to-nearest-even on the dropped 16 bits: add half the ulp plus
     * the tie-break bit of the kept LSB, then truncate. NaNs keep their
     * payload bits (a NaN stays NaN for any payload used by inference). */
    const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
    u += bias;
    return (uint16_t)(u >> 16);
}

static char *cw_strdup(const char *s) {
    if (!s) return NULL;
    const size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static void cw_err(char *err, size_t errlen, const char *msg) {
    if (errlen) snprintf(err, errlen, "%s", msg);
}

/* struct "<QIIQQQQQ", packed with explicit little-endian stores. */
static void cw_pack_index_record(uint8_t out[DS4_CACHE_WRITER_INDEX_RECORD_SIZE],
                                 uint64_t sample_id,
                                 uint32_t shard_id,
                                 uint32_t seq_len,
                                 uint64_t off_input_ids,
                                 uint64_t off_attention_mask,
                                 uint64_t off_loss_mask,
                                 uint64_t off_target_hidden,
                                 uint64_t off_target_last_hidden) {
    const uint64_t fields[7] = {
        sample_id, shard_id, seq_len, off_input_ids, off_attention_mask,
        off_loss_mask, off_target_hidden,
    };
    size_t p = 0;
    for (int i = 0; i < 7; i++) {
        const int nbytes = (i == 0 || i >= 3) ? 8 : 4;
        for (int b = 0; b < nbytes; b++) {
            out[p++] = (uint8_t)(fields[i] >> (8 * b));
        }
    }
    for (int b = 0; b < 8; b++) {
        out[p++] = (uint8_t)(off_target_last_hidden >> (8 * b));
    }
}

static int cw_flush_fsync(FILE *fp) {
    if (fflush(fp) != 0) return -1;
    return fsync(fileno(fp));
}

static int cw_open_shard(ds4_cache_writer *w, char *err, size_t errlen) {
    if (w->n_shards == w->cap_shards) {
        const int ncap = w->cap_shards ? w->cap_shards * 2 : 8;
        char (*nn)[32] = realloc(w->shard_names,
                                 (size_t)ncap * sizeof(*nn));
        if (!nn) {
            cw_err(err, errlen, "cache writer: out of memory (shard list)");
            return -1;
        }
        w->shard_names = nn;
        w->cap_shards = ncap;
    }
    const uint32_t id = w->n_shards;
    snprintf(w->shard_names[w->n_shards], sizeof(w->shard_names[0]),
             "shard-%05u.bin", id);
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", w->out_dir,
             w->shard_names[w->n_shards]);
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        if (errlen) snprintf(err, errlen, "cache writer: cannot create %s", path);
        return -1;
    }
    w->shard = fp;
    w->shard_id = id;
    w->shard_size = 0;
    w->n_shards++;
    return 0;
}

static int cw_close_shard(ds4_cache_writer *w, char *err, size_t errlen) {
    if (!w->shard) return 0;
    if (cw_flush_fsync(w->shard) != 0) {
        cw_err(err, errlen, "cache writer: shard flush/fsync failed");
        fclose(w->shard);
        w->shard = NULL;
        return -1;
    }
    fclose(w->shard);
    w->shard = NULL;
    return 0;
}

int ds4_cache_writer_open(ds4_cache_writer **out,
                          const ds4_cache_writer_options *opt,
                          char *err, size_t errlen) {
    if (!out || !opt || !opt->out_dir || !opt->target_name ||
        opt->hidden_size <= 0 || !opt->layer_ids || opt->n_layers < 1) {
        cw_err(err, errlen, "cache writer: invalid options");
        return -1;
    }
    for (int i = 1; i < opt->n_layers; i++) {
        if (opt->layer_ids[i] <= opt->layer_ids[i - 1]) {
            cw_err(err, errlen,
                   "cache writer: layer_ids must be strictly ascending");
            return -1;
        }
    }
    /* Mirror prepare_target_cache_output_dir: the output directory must be
     * new or empty, so a stale cache is never mixed into a fresh run. */
    if (mkdir(opt->out_dir, 0755) != 0) {
        DIR *d = opendir(opt->out_dir);
        if (!d) {
            if (errlen) snprintf(err, errlen,
                                 "cache writer: cannot create %s", opt->out_dir);
            return -1;
        }
        struct dirent *de;
        int entries = 0;
        while ((de = readdir(d)) != NULL) {
            if (strcmp(de->d_name, ".") && strcmp(de->d_name, "..")) entries++;
        }
        closedir(d);
        if (entries != 0) {
            if (errlen) snprintf(err, errlen,
                                 "cache writer: output dir is not empty: %s",
                                 opt->out_dir);
            return -1;
        }
    }

    ds4_cache_writer *w = calloc(1, sizeof(*w));
    if (!w) {
        cw_err(err, errlen, "cache writer: out of memory");
        return -1;
    }
    w->out_dir = cw_strdup(opt->out_dir);
    w->target_name = cw_strdup(opt->target_name);
    w->source_jsonl = cw_strdup(opt->source_jsonl);
    w->git_sha = cw_strdup(opt->git_sha);
    w->layer_ids = malloc((size_t)opt->n_layers * sizeof(*w->layer_ids));
    w->bf16_scratch = malloc((size_t)opt->n_layers * (size_t)opt->hidden_size *
                             sizeof(*w->bf16_scratch));
    if (!w->out_dir || !w->target_name || !w->layer_ids || !w->bf16_scratch) {
        cw_err(err, errlen, "cache writer: out of memory");
        ds4_cache_writer_close(w, err, errlen);
        return -1;
    }
    memcpy(w->layer_ids, opt->layer_ids,
           (size_t)opt->n_layers * sizeof(*w->layer_ids));
    w->max_shard_bytes = opt->max_shard_bytes ?
        opt->max_shard_bytes : DS4_CACHE_WRITER_DEFAULT_MAX_SHARD_BYTES;
    w->hidden_size = opt->hidden_size;
    w->n_layers = opt->n_layers;
    w->max_length = opt->max_length;
    w->min_loss_tokens = opt->min_loss_tokens;

    char path[4096];
    snprintf(path, sizeof(path), "%s/samples.idx", w->out_dir);
    w->idx = fopen(path, "wb");
    if (!w->idx) {
        if (errlen) snprintf(err, errlen, "cache writer: cannot create %s", path);
        ds4_cache_writer_close(w, err, errlen);
        return -1;
    }
    *out = w;
    return 0;
}

static int cw_write_all(ds4_cache_writer *w, const void *ptr, size_t bytes,
                        char *err, size_t errlen) {
    if (bytes && fwrite(ptr, 1, bytes, w->shard) != bytes) {
        cw_err(err, errlen, "cache writer: shard write failed");
        return -1;
    }
    w->shard_size += bytes;
    w->total_bytes += bytes;
    return 0;
}

int ds4_cache_writer_write_sample(ds4_cache_writer *w,
                                  uint64_t sample_id,
                                  const int32_t *input_ids,
                                  const uint8_t *loss_mask,
                                  uint32_t seq_len,
                                  const float *const *layer_hidden_f32,
                                  const float *last_hidden_f32,
                                  char *err, size_t errlen) {
    if (!w || !w->idx || !input_ids || !loss_mask || !layer_hidden_f32 ||
        !last_hidden_f32 || seq_len == 0) {
        cw_err(err, errlen, "cache writer: invalid sample arguments");
        return -1;
    }
    if (sample_id != w->num_samples) {
        if (errlen) snprintf(err, errlen,
                             "cache writer: sample_id %llu out of order (expected %llu)",
                             (unsigned long long)sample_id,
                             (unsigned long long)w->num_samples);
        return -1;
    }
    const uint64_t h = (uint64_t)w->hidden_size;
    const uint64_t l = (uint64_t)w->n_layers;
    const uint64_t ids_bytes = (uint64_t)seq_len * 4;
    const uint64_t mask_bytes = seq_len;
    const uint64_t hidden_bytes = (uint64_t)seq_len * l * h * 2;
    const uint64_t last_bytes = (uint64_t)seq_len * h * 2;
    const uint64_t sample_bytes =
        ids_bytes + 2 * mask_bytes + hidden_bytes + last_bytes;

    /* Rotate like LocalTargetCacheWriter._ensure_shard: never split a
     * sample, never rotate an empty shard. */
    if (!w->shard) {
        if (cw_open_shard(w, err, errlen) != 0) return -1;
    } else if (w->shard_size > 0 &&
               w->shard_size + sample_bytes > w->max_shard_bytes) {
        if (cw_close_shard(w, err, errlen) != 0) return -1;
        if (cw_open_shard(w, err, errlen) != 0) return -1;
    }

    const uint64_t off_ids = w->shard_size;
    const uint64_t off_attn = off_ids + ids_bytes;
    const uint64_t off_loss = off_attn + mask_bytes;
    const uint64_t off_hidden = off_loss + mask_bytes;
    const uint64_t off_last = off_hidden + hidden_bytes;

    if (cw_write_all(w, input_ids, ids_bytes, err, errlen) != 0) return -1;

    if (w->ones_cap < mask_bytes) {
        uint8_t *nn = realloc(w->ones, mask_bytes);
        if (!nn) {
            cw_err(err, errlen, "cache writer: out of memory (mask)");
            return -1;
        }
        w->ones = nn;
        w->ones_cap = mask_bytes;
    }
    memset(w->ones, 1, mask_bytes);
    if (cw_write_all(w, w->ones, mask_bytes, err, errlen) != 0) return -1;
    if (cw_write_all(w, loss_mask, mask_bytes, err, errlen) != 0) return -1;

    /* [T, L*H] position-major, layers ascending within a position. */
    const size_t row_values = (size_t)(l * h);
    for (uint32_t t = 0; t < seq_len; t++) {
        uint16_t *dst = w->bf16_scratch;
        for (int li = 0; li < w->n_layers; li++) {
            const float *src = layer_hidden_f32[li] + (uint64_t)t * h;
            for (uint64_t j = 0; j < h; j++) *dst++ = ds4_f32_to_bf16(src[j]);
        }
        if (cw_write_all(w, w->bf16_scratch, row_values * 2, err, errlen) != 0) {
            return -1;
        }
    }
    for (uint32_t t = 0; t < seq_len; t++) {
        const float *src = last_hidden_f32 + (uint64_t)t * h;
        for (uint64_t j = 0; j < h; j++) {
            w->bf16_scratch[j] = ds4_f32_to_bf16(src[j]);
        }
        if (cw_write_all(w, w->bf16_scratch, (size_t)h * 2, err, errlen) != 0) {
            return -1;
        }
    }

    uint8_t rec[DS4_CACHE_WRITER_INDEX_RECORD_SIZE];
    cw_pack_index_record(rec, sample_id, w->shard_id, seq_len,
                         off_ids, off_attn, off_loss, off_hidden, off_last);
    if (fwrite(rec, 1, sizeof(rec), w->idx) != sizeof(rec)) {
        cw_err(err, errlen, "cache writer: index write failed");
        return -1;
    }
    w->num_samples++;
    return 0;
}

/* Minimal JSON string escaping (quotes, backslashes, control chars). */
static void cw_json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    for (; s && *s; s++) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc(c, fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", c);
        } else {
            fputc(c, fp);
        }
    }
    fputc('"', fp);
}

static int cw_write_manifest(ds4_cache_writer *w, char *err, size_t errlen) {
    char path[4096], tmp[4096];
    snprintf(path, sizeof(path), "%s/manifest.json", w->out_dir);
    snprintf(tmp, sizeof(tmp), "%s/manifest.json.tmp", w->out_dir);
    FILE *fp = fopen(tmp, "w");
    if (!fp) {
        if (errlen) snprintf(err, errlen, "cache writer: cannot create %s", tmp);
        return -1;
    }
    /* Keys sorted, matching DeepSpec's json.dump(sort_keys=True) layout. */
    fprintf(fp, "{\n");
    fprintf(fp, "  \"chat_template\": \"ds4\",\n");
    fprintf(fp, "  \"git_sha\": ");
    cw_json_string(fp, w->git_sha ? w->git_sha : "unknown");
    fprintf(fp, ",\n");
    fprintf(fp, "  \"hidden_dtype\": \"bfloat16\",\n");
    fprintf(fp, "  \"hidden_size\": %d,\n", w->hidden_size);
    fprintf(fp, "  \"index_record_size\": %d,\n",
            DS4_CACHE_WRITER_INDEX_RECORD_SIZE);
    fprintf(fp, "  \"mask_dtype\": \"uint8\",\n");
    fprintf(fp, "  \"max_length\": %d,\n", w->max_length);
    fprintf(fp, "  \"min_loss_tokens\": %d,\n", w->min_loss_tokens);
    fprintf(fp, "  \"num_samples\": %llu,\n",
            (unsigned long long)w->num_samples);
    fprintf(fp, "  \"num_shards\": %d,\n", w->n_shards);
    fprintf(fp, "  \"shards\": [");
    for (int i = 0; i < w->n_shards; i++) {
        fprintf(fp, "%s\n    {\n      \"file_name\": ", i ? "," : "");
        cw_json_string(fp, w->shard_names[i]);
        fprintf(fp, ",\n      \"shard_id\": %u\n    }", (unsigned)i);
    }
    fprintf(fp, "%s],\n", w->n_shards ? "\n  " : "");
    fprintf(fp, "  \"source_jsonl_paths\": [");
    if (w->source_jsonl) cw_json_string(fp, w->source_jsonl);
    fprintf(fp, "],\n");
    fprintf(fp, "  \"target_layer_ids\": [");
    for (int i = 0; i < w->n_layers; i++) {
        fprintf(fp, "%s%d", i ? ", " : "", w->layer_ids[i]);
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"target_model_name_or_path\": ");
    cw_json_string(fp, w->target_name);
    fprintf(fp, ",\n");
    fprintf(fp, "  \"token_dtype\": \"int32\",\n");
    fprintf(fp, "  \"version\": %d\n", DS4_CACHE_WRITER_VERSION);
    fprintf(fp, "}\n");
    if (cw_flush_fsync(fp) != 0) {
        fclose(fp);
        cw_err(err, errlen, "cache writer: manifest write failed");
        return -1;
    }
    fclose(fp);
    if (rename(tmp, path) != 0) {
        cw_err(err, errlen, "cache writer: manifest rename failed");
        return -1;
    }
    return 0;
}

int ds4_cache_writer_close(ds4_cache_writer *w, char *err, size_t errlen) {
    if (!w) return 0;
    int rc = 0;
    /* Finalize only a cleanly opened run; a half-open writer (open failure)
     * leaves no manifest behind, which marks the cache as invalid. */
    if (w->idx) {
        if (cw_close_shard(w, err, errlen) != 0) rc = -1;
        if (cw_flush_fsync(w->idx) != 0) {
            cw_err(err, errlen, "cache writer: index flush/fsync failed");
            rc = -1;
        }
        fclose(w->idx);
        w->idx = NULL;
        if (rc == 0 && cw_write_manifest(w, err, errlen) != 0) rc = -1;
    }
    free(w->out_dir);
    free(w->target_name);
    free(w->source_jsonl);
    free(w->git_sha);
    free(w->layer_ids);
    free(w->shard_names);
    free(w->bf16_scratch);
    free(w->ones);
    free(w);
    return rc;
}

uint64_t ds4_cache_writer_samples(const ds4_cache_writer *w) {
    return w ? w->num_samples : 0;
}

uint64_t ds4_cache_writer_bytes(const ds4_cache_writer *w) {
    return w ? w->total_bytes : 0;
}
