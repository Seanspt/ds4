/* Model-free tests for the DeepSpec target-cache v2 writer
 * (ds4_cache_writer.c, DEEPSPEC_INTEGRATION.md Stage B).
 *
 * Covers: f32 -> bf16 round-to-nearest-even bit patterns against known
 * cases (cross-checked with torch), the 56-byte "<QIIQQQQQ" index record
 * layout, shard section order/offsets, shard rotation at max_shard_bytes,
 * manifest contents, and the option/sample validation errors.
 *
 * The synthetic cache is deterministic and mirrored by
 * tests/validate_target_cache.py, which reads it back with DeepSpec's own
 * CacheDataset (the strongest contract proof). Set DS4_TEST_CACHE_WRITER_OUT
 * to keep the written cache for that script; otherwise a temp dir is used
 * and removed.
 *
 * Pure C99 + POSIX: builds and runs on any host. */

#include "../ds4_cache_writer.h"

#include <dirent.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_failed = 0;

#define CHECK(cond) do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            g_failed++; \
        } \
    } while (0)

/* ---------------------------------------------------------------------------
 * Synthetic cache contract (shared with tests/validate_target_cache.py):
 *   3 samples, seq_len T=5, hidden H=8, layer ids {-1, 3}, one shard each
 *   (max_shard_bytes = 300, one sample = 270 bytes).
 *   sample s:
 *     input_ids[t]        = s*100 + t
 *     loss_mask[t]        = t >= 2
 *     layer_hidden[l][i]  = ((s*7 + l*3 + i) % 17 - 8) * 0.5f   (i = t*8+h)
 *     last_hidden[i]      = 0.1f * ((s*31 + i) % 13) + 0.05f
 * ------------------------------------------------------------------------- */

#define TEST_T 5
#define TEST_H 8
#define TEST_LAYERS 2
#define TEST_SAMPLES 3
#define TEST_MAX_SHARD 300

static const int test_layer_ids[TEST_LAYERS] = { -1, 3 };

static float test_layer_value(int s, int l, int i) {
    return (float)((s * 7 + l * 3 + i) % 17 - 8) * 0.5f;
}

static float test_last_value(int s, int i) {
    return 0.1f * (float)((s * 31 + i) % 13) + 0.05f;
}

static float f32_from_bits(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void test_bf16_patterns(void) {
    static const struct {
        uint32_t f32;
        uint16_t bf16;
    } cases[] = {
        { 0x00000000u, 0x0000 }, /* 0 */
        { 0x80000000u, 0x8000 }, /* -0 */
        { 0x3F800000u, 0x3F80 }, /* 1.0 */
        { 0xC0200000u, 0xC020 }, /* -2.5 */
        { 0x477FE000u, 0x4780 }, /* 65504 rounds up (dropped bits > half) */
        { 0x7F7F0000u, 0x7F7F }, /* bf16 max */
        { 0x7F7FFFFFu, 0x7F80 }, /* f32 max rounds to bf16 inf */
        { 0x7F800000u, 0x7F80 }, /* inf */
        { 0xFF800000u, 0xFF80 }, /* -inf */
        { 0x7FC00000u, 0x7FC0 }, /* NaN preserved */
        { 0x3F808000u, 0x3F80 }, /* RNE tie -> even (down) */
        { 0x3F818000u, 0x3F82 }, /* RNE tie -> even (up) */
        { 0x3F808001u, 0x3F81 }, /* just above the tie rounds up */
        { 0x3F807FFFu, 0x3F80 }, /* just below the tie rounds down */
        { 0x00010000u, 0x0001 }, /* 2^-133: bf16 min denormal */
        { 0x00008000u, 0x0000 }, /* 2^-134: tie at denormal boundary -> even */
        { 0x00008001u, 0x0001 }, /* 2^-134 + 1 ulp rounds up */
        { 0x00007FFFu, 0x0000 }, /* below the tie rounds down */
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const uint16_t got = ds4_f32_to_bf16(f32_from_bits(cases[i].f32));
        if (got != cases[i].bf16) {
            fprintf(stderr, "bf16 case f32=0x%08X: expected 0x%04X, got 0x%04X\n",
                    cases[i].f32, cases[i].bf16, got);
            CHECK(false);
        }
    }
}

static uint64_t rd_u64le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static uint32_t rd_u32le(const uint8_t *p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static uint8_t *read_file(const char *path, size_t *size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    const long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *buf = malloc(n > 0 ? (size_t)n : 1);
    if (buf && fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(fp);
    if (size) *size = (size_t)n;
    return buf;
}

static void test_cache_roundtrip(const char *out_dir, bool keep) {
    char err[256] = {0};
    ds4_cache_writer *w = NULL;
    const ds4_cache_writer_options opt = {
        .out_dir = out_dir,
        .max_shard_bytes = TEST_MAX_SHARD,
        .hidden_size = TEST_H,
        .layer_ids = test_layer_ids,
        .n_layers = TEST_LAYERS,
        .target_name = "/fake/DeepSeek-V4-Flash-DSpark",
        .source_jsonl = "regen.jsonl",
        .git_sha = "deadbeef",
        .max_length = 4096,
        .min_loss_tokens = 14,
    };
    CHECK(ds4_cache_writer_open(&w, &opt, err, sizeof(err)) == 0);
    if (!w) return;

    int32_t ids[TEST_T];
    uint8_t loss[TEST_T];
    float hidden[TEST_LAYERS][TEST_T * TEST_H];
    float last[TEST_T * TEST_H];
    const float *layer_ptrs[TEST_LAYERS] = { hidden[0], hidden[1] };
    for (int s = 0; s < TEST_SAMPLES; s++) {
        for (int t = 0; t < TEST_T; t++) {
            ids[t] = s * 100 + t;
            loss[t] = t >= 2 ? 1 : 0;
        }
        for (int i = 0; i < TEST_T * TEST_H; i++) {
            hidden[0][i] = test_layer_value(s, 0, i);
            hidden[1][i] = test_layer_value(s, 1, i);
            last[i] = test_last_value(s, i);
        }
        CHECK(ds4_cache_writer_write_sample(w, (uint64_t)s, ids, loss, TEST_T,
                                            layer_ptrs, last,
                                            err, sizeof(err)) == 0);
    }
    /* sample_id must be dense and ordered. */
    CHECK(ds4_cache_writer_write_sample(w, 99, ids, loss, TEST_T,
                                        layer_ptrs, last,
                                        err, sizeof(err)) != 0);
    CHECK(ds4_cache_writer_samples(w) == TEST_SAMPLES);
    /* One sample: 5*4 + 2*5 + 5*2*8*2 + 5*8*2 = 270 bytes. */
    CHECK(ds4_cache_writer_bytes(w) == 3 * 270);
    CHECK(ds4_cache_writer_close(w, err, sizeof(err)) == 0);

    /* Index: 3 records of 56 bytes; sample s lives in shard s at offset 0. */
    char path[4096];
    snprintf(path, sizeof(path), "%s/samples.idx", out_dir);
    size_t idx_size = 0;
    uint8_t *idx = read_file(path, &idx_size);
    CHECK(idx != NULL);
    CHECK(idx_size == TEST_SAMPLES * DS4_CACHE_WRITER_INDEX_RECORD_SIZE);
    if (idx && idx_size == TEST_SAMPLES * DS4_CACHE_WRITER_INDEX_RECORD_SIZE) {
        for (int s = 0; s < TEST_SAMPLES; s++) {
            const uint8_t *r = idx + s * DS4_CACHE_WRITER_INDEX_RECORD_SIZE;
            CHECK(rd_u64le(r + 0) == (uint64_t)s);  /* sample_id */
            CHECK(rd_u32le(r + 8) == (uint32_t)s);  /* shard_id */
            CHECK(rd_u32le(r + 12) == TEST_T);      /* seq_len */
            CHECK(rd_u64le(r + 16) == 0);           /* off_input_ids */
            CHECK(rd_u64le(r + 24) == 20);          /* off_attention_mask */
            CHECK(rd_u64le(r + 32) == 25);          /* off_loss_mask */
            CHECK(rd_u64le(r + 40) == 30);          /* off_target_hidden */
            CHECK(rd_u64le(r + 48) == 190);         /* off_last_hidden */
        }
    }
    free(idx);

    /* Shards: 270 bytes each, 5 sections in fixed order. */
    for (int s = 0; s < TEST_SAMPLES; s++) {
        snprintf(path, sizeof(path), "%s/shard-%05d.bin", out_dir, s);
        size_t shard_size = 0;
        uint8_t *sh = read_file(path, &shard_size);
        CHECK(sh != NULL);
        CHECK(shard_size == 270);
        if (!sh || shard_size != 270) {
            free(sh);
            continue;
        }
        bool ok = true;
        for (int t = 0; t < TEST_T; t++) {
            if ((int32_t)rd_u32le(sh + t * 4) != s * 100 + t) ok = false;
            if (sh[20 + t] != 1) ok = false;                 /* attention */
            if (sh[25 + t] != (t >= 2 ? 1 : 0)) ok = false;  /* loss */
        }
        const uint8_t *hid = sh + 30;
        for (int t = 0; t < TEST_T && ok; t++) {
            for (int l = 0; l < TEST_LAYERS && ok; l++) {
                for (int hh = 0; hh < TEST_H && ok; hh++) {
                    const int i = t * TEST_H + hh;
                    const uint16_t want =
                        ds4_f32_to_bf16(test_layer_value(s, l, i));
                    const uint16_t got = (uint16_t)rd_u32le(
                        hid + ((size_t)t * TEST_LAYERS * TEST_H +
                               (size_t)l * TEST_H + hh) * 2);
                    if (got != want) ok = false;
                }
            }
        }
        const uint8_t *lst = sh + 190;
        for (int i = 0; i < TEST_T * TEST_H && ok; i++) {
            const uint16_t want = ds4_f32_to_bf16(test_last_value(s, i));
            const uint16_t got = (uint16_t)rd_u32le(lst + (size_t)i * 2);
            if (got != want) ok = false;
        }
        CHECK(ok);
        free(sh);
    }

    /* Manifest: required contract fields. */
    snprintf(path, sizeof(path), "%s/manifest.json", out_dir);
    size_t msize = 0;
    uint8_t *m = read_file(path, &msize);
    CHECK(m != NULL);
    if (m) {
        const char *j = (const char *)m;
        CHECK(strstr(j, "\"version\": 2") != NULL);
        CHECK(strstr(j, "\"num_samples\": 3") != NULL);
        CHECK(strstr(j, "\"num_shards\": 3") != NULL);
        CHECK(strstr(j, "\"hidden_dtype\": \"bfloat16\"") != NULL);
        CHECK(strstr(j, "\"token_dtype\": \"int32\"") != NULL);
        CHECK(strstr(j, "\"mask_dtype\": \"uint8\"") != NULL);
        CHECK(strstr(j, "\"index_record_size\": 56") != NULL);
        CHECK(strstr(j, "\"hidden_size\": 8") != NULL);
        CHECK(strstr(j, "\"target_layer_ids\": [-1, 3]") != NULL);
        CHECK(strstr(j, "\"target_model_name_or_path\": "
                       "\"/fake/DeepSeek-V4-Flash-DSpark\"") != NULL);
        CHECK(strstr(j, "\"shard-00000.bin\"") != NULL);
        CHECK(strstr(j, "\"shard-00002.bin\"") != NULL);
        CHECK(strstr(j, "\"chat_template\": \"ds4\"") != NULL);
        CHECK(strstr(j, "\"git_sha\": \"deadbeef\"") != NULL);
        free(m);
    }

    if (!keep) {
        for (int s = 0; s < TEST_SAMPLES; s++) {
            snprintf(path, sizeof(path), "%s/shard-%05d.bin", out_dir, s);
            unlink(path);
        }
        snprintf(path, sizeof(path), "%s/samples.idx", out_dir);
        unlink(path);
        snprintf(path, sizeof(path), "%s/manifest.json", out_dir);
        unlink(path);
        rmdir(out_dir);
    } else {
        fprintf(stderr, "test_cache_writer: cache kept at %s\n", out_dir);
    }
}

static void test_option_validation(const char *base_dir) {
    char err[256] = {0};
    char dir[4096];
    ds4_cache_writer *w = NULL;

    /* Non-ascending layer ids are rejected. */
    const int bad_layers[2] = { 3, -1 };
    snprintf(dir, sizeof(dir), "%s/cw-bad-%d", base_dir, (int)getpid());
    ds4_cache_writer_options opt = {
        .out_dir = dir,
        .hidden_size = TEST_H,
        .layer_ids = bad_layers,
        .n_layers = 2,
        .target_name = "x",
    };
    CHECK(ds4_cache_writer_open(&w, &opt, err, sizeof(err)) != 0);

    /* A non-empty output dir is rejected (stale cache protection). */
    snprintf(dir, sizeof(dir), "%s/cw-nonempty-%d", base_dir, (int)getpid());
    CHECK(mkdir(dir, 0755) == 0);
    char f[4096];
    snprintf(f, sizeof(f), "%s/stale", dir);
    FILE *fp = fopen(f, "w");
    if (fp) {
        fputc('x', fp);
        fclose(fp);
    }
    const int good_layers[1] = { 0 };
    opt.layer_ids = good_layers;
    opt.n_layers = 1;
    CHECK(ds4_cache_writer_open(&w, &opt, err, sizeof(err)) != 0);
    unlink(f);
    rmdir(dir);
}

int main(void) {
    test_bf16_patterns();

    const char *out = getenv("DS4_TEST_CACHE_WRITER_OUT");
    char tmp[4096] = {0};
    if (!out || !out[0]) {
        snprintf(tmp, sizeof(tmp), "/tmp/ds4-cw-test-XXXXXX");
        if (!mkdtemp(tmp)) {
            fprintf(stderr, "test_cache_writer: mkdtemp failed\n");
            return 1;
        }
        /* mkdtemp creates the (empty) dir, which the writer accepts. */
        out = tmp;
    }
    test_cache_roundtrip(out, out != tmp);
    test_option_validation("/tmp");

    if (g_failed) {
        fprintf(stderr, "test_cache_writer: %d checks failed\n", g_failed);
        return 1;
    }
    printf("test_cache_writer: all tests passed\n");
    return 0;
}
