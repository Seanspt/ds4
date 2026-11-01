#include "ds4_cachegen.h"

#include "ds4_cache_writer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Build-time ds4 repo sha for the manifest (Makefile -DDS4_GIT_SHA). */
#ifndef DS4_GIT_SHA
#define DS4_GIT_SHA "unknown"
#endif

/* Engine tap table capacity (DS4_MAX_TAP_LAYERS in ds4.c, kept internal
 * there): user layers + the appended FINAL_NORM must fit. */
#define CG_MAX_TAP_LAYERS 8

/* =========================================================================
 * Small growable byte buffer (same idiom as ds4_server.c's buf).
 * ========================================================================= */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} cg_buf;

static void cg_buf_reserve(cg_buf *b, size_t add) {
    if (b->len + add + 1 <= b->cap) return;
    size_t ncap = b->cap ? b->cap * 2 : 256;
    while (ncap < b->len + add + 1) ncap *= 2;
    b->ptr = realloc(b->ptr, ncap);
    if (!b->ptr) {
        fprintf(stderr, "ds4: cache-gen: out of memory\n");
        exit(1);
    }
    b->cap = ncap;
}

static void cg_buf_putn(cg_buf *b, const char *s, size_t n) {
    cg_buf_reserve(b, n);
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void cg_buf_puts(cg_buf *b, const char *s) {
    cg_buf_putn(b, s, strlen(s));
}

static char *cg_buf_take(cg_buf *b) {
    char *p = b->ptr ? b->ptr : strdup("");
    b->ptr = NULL;
    b->len = b->cap = 0;
    return p;
}

static void cg_buf_free(cg_buf *b) {
    free(b->ptr);
    b->ptr = NULL;
    b->len = b->cap = 0;
}

/* =========================================================================
 * Minimal JSON reader for the dataset lines (same style as the parser
 * primitives in ds4_server.c): only what {"messages":[...]} needs.
 * ========================================================================= */

static void cg_json_ws(const char **p) {
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') (*p)++;
}

static bool cg_json_lit(const char **p, const char *lit) {
    const size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int cg_json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool cg_json_u16(const char **p, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        const int d = cg_json_hex((*p)[i]);
        if (d < 0) return false;
        v = (v << 4) | (uint32_t)d;
    }
    *p += 4;
    *out = v;
    return true;
}

static void cg_utf8_put(cg_buf *b, uint32_t cp) {
    char tmp[4];
    size_t n;
    if (cp < 0x80) {
        tmp[0] = (char)cp;
        n = 1;
    } else if (cp < 0x800) {
        tmp[0] = (char)(0xC0 | (cp >> 6));
        tmp[1] = (char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        tmp[0] = (char)(0xE0 | (cp >> 12));
        tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[2] = (char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        tmp[0] = (char)(0xF0 | (cp >> 18));
        tmp[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        tmp[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[3] = (char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    cg_buf_putn(b, tmp, n);
}

/* Parse a JSON string into a fresh NUL-terminated buffer. */
static bool cg_json_string(const char **p, char **out) {
    if (**p != '"') return false;
    (*p)++;
    cg_buf b = {0};
    for (;;) {
        const char c = **p;
        if (c == '\0' || (unsigned char)c < 0x20) goto fail;
        (*p)++;
        if (c == '"') break;
        if (c != '\\') {
            cg_buf_putn(&b, &c, 1);
            continue;
        }
        const char esc = **p;
        (*p)++;
        switch (esc) {
        case '"': cg_buf_putn(&b, "\"", 1); break;
        case '\\': cg_buf_putn(&b, "\\", 1); break;
        case '/': cg_buf_putn(&b, "/", 1); break;
        case 'b': cg_buf_putn(&b, "\b", 1); break;
        case 'f': cg_buf_putn(&b, "\f", 1); break;
        case 'n': cg_buf_putn(&b, "\n", 1); break;
        case 'r': cg_buf_putn(&b, "\r", 1); break;
        case 't': cg_buf_putn(&b, "\t", 1); break;
        case 'u': {
            uint32_t cp;
            if (!cg_json_u16(p, &cp)) goto fail;
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                uint32_t lo;
                if ((*p)[0] != '\\' || (*p)[1] != 'u') goto fail;
                *p += 2;
                if (!cg_json_u16(p, &lo) || lo < 0xDC00 || lo > 0xDFFF) goto fail;
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
            cg_utf8_put(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    *out = cg_buf_take(&b);
    return true;
fail:
    cg_buf_free(&b);
    return false;
}

static bool cg_json_skip_value(const char **p);

static bool cg_json_skip_composite(const char **p, char open, char close) {
    if (**p != open) return false;
    (*p)++;
    cg_json_ws(p);
    if (**p == close) {
        (*p)++;
        return true;
    }
    for (;;) {
        if (open == '{') {
            char *key = NULL;
            cg_json_ws(p);
            if (!cg_json_string(p, &key)) return false;
            free(key);
            cg_json_ws(p);
            if (**p != ':') return false;
            (*p)++;
        }
        cg_json_ws(p);
        if (!cg_json_skip_value(p)) return false;
        cg_json_ws(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }
        if (**p == close) {
            (*p)++;
            return true;
        }
        return false;
    }
}

static bool cg_json_skip_value(const char **p) {
    cg_json_ws(p);
    const char c = **p;
    if (c == '{') return cg_json_skip_composite(p, '{', '}');
    if (c == '[') return cg_json_skip_composite(p, '[', ']');
    if (c == '"') {
        char *s = NULL;
        if (!cg_json_string(p, &s)) return false;
        free(s);
        return true;
    }
    if (c == 't') return cg_json_lit(p, "true");
    if (c == 'f') return cg_json_lit(p, "false");
    if (c == 'n') return cg_json_lit(p, "null");
    /* number */
    const char *q = *p;
    if (*q == '-') q++;
    if (*q < '0' || *q > '9') return false;
    while ((*q >= '0' && *q <= '9') || *q == '.' || *q == 'e' || *q == 'E' ||
           *q == '+' || *q == '-') {
        q++;
    }
    *p = q;
    return true;
}

/* =========================================================================
 * Dataset record: {"messages":[{"role","content","reasoning"?}, ...]}.
 * v1 subset: roles system/user/assistant, string content, exactly one
 * assistant turn as the last message.
 * ========================================================================= */

typedef struct {
    char *role;
    char *content;
    char *reasoning; /* may be NULL -> rendered as empty */
} cg_msg;

typedef enum {
    CG_DROP_OK = 0,
    CG_DROP_JSON,
    CG_DROP_NO_MESSAGES,
    CG_DROP_UNSUPPORTED_ROLE,
    CG_DROP_BAD_CONTENT,
    CG_DROP_MULTI_ASSISTANT,
    CG_DROP_ASSISTANT_NOT_LAST,
    CG_DROP_FEW_LOSS_TOKENS,
    CG_DROP_N
} cg_drop_reason;

static const char *const cg_drop_names[CG_DROP_N] = {
    [CG_DROP_JSON] = "malformed JSON",
    [CG_DROP_NO_MESSAGES] = "empty messages or no assistant turn",
    [CG_DROP_UNSUPPORTED_ROLE] = "unsupported role",
    [CG_DROP_BAD_CONTENT] = "non-string content",
    [CG_DROP_MULTI_ASSISTANT] = "more than one assistant turn",
    [CG_DROP_ASSISTANT_NOT_LAST] = "assistant turn is not last",
    [CG_DROP_FEW_LOSS_TOKENS] = "fewer loss tokens than --min-loss-tokens",
};

static void cg_msgs_free(cg_msg *msgs, int n) {
    for (int i = 0; i < n; i++) {
        free(msgs[i].role);
        free(msgs[i].content);
        free(msgs[i].reasoning);
    }
    free(msgs);
}

static bool cg_json_message(const char **p, cg_msg *m) {
    cg_json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    bool got_role = false;
    cg_json_ws(p);
    if (**p == '}') {
        (*p)++;
        return false; /* role is mandatory */
    }
    for (;;) {
        cg_json_ws(p);
        char *key = NULL;
        if (!cg_json_string(p, &key)) return false;
        cg_json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        cg_json_ws(p);
        if (!strcmp(key, "role")) {
            free(key);
            if (!cg_json_string(p, &m->role)) return false;
            got_role = true;
        } else if (!strcmp(key, "content")) {
            free(key);
            if (**p == 'n') {
                if (!cg_json_lit(p, "null")) return false;
                m->content = NULL; /* tool-call style message: unsupported */
            } else if (!cg_json_string(p, &m->content)) {
                return false;
            }
        } else if (!strcmp(key, "reasoning") || !strcmp(key, "reasoning_content")) {
            free(key);
            if (**p == 'n') {
                if (!cg_json_lit(p, "null")) return false;
            } else if (!cg_json_string(p, &m->reasoning)) {
                return false;
            }
        } else {
            free(key);
            if (!cg_json_skip_value(p)) return false;
        }
        cg_json_ws(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }
        if (**p == '}') {
            (*p)++;
            break;
        }
        return false;
    }
    return got_role;
}

/* Parse one dataset line. On success the message array outputs are set and
 * the caller validates the v1 message shape; CG_DROP_JSON means the line is
 * not a {"messages":[...]} object we can read. */
static cg_drop_reason cg_parse_dataset_line(const char *line,
                                            cg_msg **msgs_out, int *n_out) {
    const char *p = line;
    cg_json_ws(&p);
    if (*p != '{') return CG_DROP_JSON;
    p++;
    cg_msg *msgs = NULL;
    int n = 0, cap = 0;
    bool got_messages = false;
    cg_json_ws(&p);
    if (*p == '}') return CG_DROP_NO_MESSAGES;
    for (;;) {
        cg_json_ws(&p);
        char *key = NULL;
        if (!cg_json_string(&p, &key)) goto json_fail;
        cg_json_ws(&p);
        if (*p != ':') {
            free(key);
            goto json_fail;
        }
        p++;
        cg_json_ws(&p);
        if (strcmp(key, "messages") != 0) {
            free(key);
            if (!cg_json_skip_value(&p)) goto json_fail;
        } else {
            free(key);
            if (*p != '[') goto json_fail;
            p++;
            cg_json_ws(&p);
            if (*p != ']') {
                for (;;) {
                    if (n == cap) {
                        cap = cap ? cap * 2 : 8;
                        cg_msg *nm = realloc(msgs, (size_t)cap * sizeof(*nm));
                        if (!nm) goto json_fail;
                        msgs = nm;
                    }
                    memset(&msgs[n], 0, sizeof(msgs[n]));
                    if (!cg_json_message(&p, &msgs[n])) goto json_fail;
                    n++;
                    cg_json_ws(&p);
                    if (*p == ',') {
                        p++;
                        cg_json_ws(&p);
                        continue;
                    }
                    break;
                }
            }
            if (*p != ']') goto json_fail;
            p++;
            got_messages = true;
        }
        cg_json_ws(&p);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == '}') {
            p++;
            break;
        }
        goto json_fail;
    }
    cg_json_ws(&p);
    if (*p != '\0') goto json_fail;
    if (!got_messages || n == 0) {
        cg_msgs_free(msgs, n);
        return CG_DROP_NO_MESSAGES;
    }
    *msgs_out = msgs;
    *n_out = n;
    return CG_DROP_OK;

json_fail:
    cg_msgs_free(msgs, n);
    return CG_DROP_JSON;
}

static cg_drop_reason cg_validate_messages(const cg_msg *msgs, int n) {
    int n_assistant = 0;
    for (int i = 0; i < n; i++) {
        const char *r = msgs[i].role;
        if (strcmp(r, "system") && strcmp(r, "user") && strcmp(r, "assistant")) {
            return CG_DROP_UNSUPPORTED_ROLE;
        }
        if (!msgs[i].content) return CG_DROP_BAD_CONTENT;
        if (!strcmp(r, "assistant")) n_assistant++;
    }
    if (n_assistant == 0) return CG_DROP_NO_MESSAGES;
    if (n_assistant > 1) return CG_DROP_MULTI_ASSISTANT;
    if (strcmp(msgs[n - 1].role, "assistant")) return CG_DROP_ASSISTANT_NOT_LAST;
    return CG_DROP_OK;
}

/* =========================================================================
 * Chat rendering: mirror of render_deepseek_chat_prompt_text
 * (ds4_server.c) for the cache-gen v1 subset — roles system/user/assistant,
 * no tool calls, no tool schemas, thinking enabled. Must stay byte
 * identical with the server renderer for the same messages: the training
 * cache is built from the exact token stream ds4-server produced/consumed
 * (DEEPSPEC_INTEGRATION.md 3.3 and §7). The loss-mask prefix render is the
 * same function on the message list without the final assistant turn; it
 * ends with the pending "<｜Assistant｜><think>" trailer, so the full render
 * extends it byte for byte.
 * ========================================================================= */

static char *cg_render(const cg_msg *msgs, int n) {
    cg_buf system = {0};
    for (int i = 0; i < n; i++) {
        if (strcmp(msgs[i].role, "system")) continue;
        if (system.len) cg_buf_puts(&system, "\n\n");
        cg_buf_puts(&system, msgs[i].content);
    }
    int last_user_idx = -1;
    for (int i = 0; i < n; i++) {
        if (!strcmp(msgs[i].role, "user")) last_user_idx = i;
    }

    cg_buf out = {0};
    cg_buf_puts(&out, "<｜begin▁of▁sentence｜>");
    if (system.ptr) cg_buf_puts(&out, system.ptr);

    bool pending_assistant = false;
    for (int i = 0; i < n; i++) {
        const cg_msg *m = &msgs[i];
        if (!strcmp(m->role, "system")) {
            continue;
        } else if (!strcmp(m->role, "user")) {
            cg_buf_puts(&out, "<｜User｜>");
            cg_buf_puts(&out, m->content);
            pending_assistant = true;
        } else if (!strcmp(m->role, "assistant")) {
            if (pending_assistant) {
                cg_buf_puts(&out, "<｜Assistant｜>");
                if (i > last_user_idx) {
                    cg_buf_puts(&out, "<think>");
                    cg_buf_puts(&out, m->reasoning ? m->reasoning : "");
                    cg_buf_puts(&out, "</think>");
                } else {
                    cg_buf_puts(&out, "</think>");
                }
            }
            cg_buf_puts(&out, m->content);
            cg_buf_puts(&out, "<｜end▁of▁sentence｜>");
            pending_assistant = false;
        }
    }
    if (pending_assistant) cg_buf_puts(&out, "<｜Assistant｜><think>");

    cg_buf_free(&system);
    return cg_buf_take(&out);
}

/* =========================================================================
 * Run.
 * ========================================================================= */

typedef struct {
    uint64_t lines;
    uint64_t written;
    uint64_t dropped[CG_DROP_N];
    uint64_t tokens;
    uint64_t loss_tokens;
} cg_stats;

static double cg_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Parse and validate --tap-layers: entries -1..n_layer-1 (-2 is reserved;
 * FINAL_NORM is always added by cache-gen itself), strictly ascending
 * (which also dedups), at most CG_MAX_TAP_LAYERS-1 entries so the
 * appended FINAL_NORM fits the engine tap table. Returns the count. */
static int cg_parse_tap_layers(const char *csv, int n_layer, int *out,
                               int max_out, char *err, size_t errlen) {
    if (!csv || !csv[0]) {
        if (errlen) snprintf(err, errlen, "--tap-layers is empty");
        return -1;
    }
    int n = 0;
    const char *p = csv;
    while (*p) {
        char *end = NULL;
        const long v = strtol(p, &end, 10);
        if (end == p || v < -1 || v >= n_layer || n >= max_out) {
            if (errlen) snprintf(err, errlen,
                                 "invalid --tap-layers entry near \"%s\" "
                                 "(allowed: -1..%d, ascending; -2 is implicit)",
                                 p, n_layer - 1);
            return -1;
        }
        if (*end != '\0' && *end != ',') {
            if (errlen) snprintf(err, errlen,
                                 "invalid --tap-layers entry near \"%s\"", p);
            return -1;
        }
        if (n > 0 && (int)v <= out[n - 1]) {
            if (errlen) snprintf(err, errlen,
                                 "--tap-layers must be strictly ascending with no duplicates");
            return -1;
        }
        out[n++] = (int)v;
        p = end + (*end == ',');
    }
    if (n == 0) {
        if (errlen) snprintf(err, errlen, "--tap-layers is empty");
        return -1;
    }
    return n;
}

int ds4_cachegen_run(ds4_engine *engine, const ds4_cachegen_options *opt,
                     char *err, size_t errlen) {
    if (!engine || !opt || !opt->dataset_path || !opt->out_dir ||
        !opt->tap_layers_csv || !opt->target_name) {
        if (errlen) snprintf(err, errlen, "invalid cache-gen options");
        return 1;
    }
    if (ds4_engine_is_glm_dsa(engine)) {
        if (errlen) snprintf(err, errlen,
                             "cache-gen requires a DeepSeek model (GLM sessions have no tap support)");
        return 1;
    }
    const int max_length =
        opt->max_length > 0 ? opt->max_length : DS4_CACHEGEN_DEFAULT_MAX_LENGTH;
    const int min_loss = opt->min_loss_tokens > 0 ?
        opt->min_loss_tokens : DS4_CACHEGEN_DEFAULT_MIN_LOSS_TOKENS;
    const int n_layer = ds4_engine_layer_count(engine);
    const int h = ds4_engine_embd_dim(engine);
    if (n_layer < 1 || h <= 0) {
        if (errlen) snprintf(err, errlen, "cache-gen: engine shape unavailable");
        return 1;
    }

    int layer_ids[CG_MAX_TAP_LAYERS];
    const int n_tap_user = cg_parse_tap_layers(opt->tap_layers_csv, n_layer,
                                               layer_ids, CG_MAX_TAP_LAYERS - 1,
                                               err, errlen);
    if (n_tap_user < 0) return 1;

    /* Session tap set: user layers (-1 == DS4_TAP_LAYER_EMBED already) in
     * MEAN_HC (n_embd rows for every slot) + FINAL_NORM for the cache's
     * target_last_hidden_states. */
    int tap_set[CG_MAX_TAP_LAYERS];
    memcpy(tap_set, layer_ids, (size_t)n_tap_user * sizeof(int));
    tap_set[n_tap_user] = DS4_TAP_LAYER_FINAL_NORM;

    FILE *fp = fopen(opt->dataset_path, "r");
    if (!fp) {
        if (errlen) snprintf(err, errlen, "cache-gen: cannot open dataset %s",
                             opt->dataset_path);
        return 1;
    }

    ds4_cache_writer *writer = NULL;
    const ds4_cache_writer_options wopt = {
        .out_dir = opt->out_dir,
        .max_shard_bytes = opt->max_shard_bytes,
        .hidden_size = h,
        .layer_ids = layer_ids,
        .n_layers = n_tap_user,
        .target_name = opt->target_name,
        .source_jsonl = opt->dataset_path,
        .git_sha = opt->git_sha ? opt->git_sha : DS4_GIT_SHA,
        .max_length = max_length,
        .min_loss_tokens = min_loss,
    };
    if (ds4_cache_writer_open(&writer, &wopt, err, errlen) != 0) {
        fclose(fp);
        return 1;
    }

    ds4_session *sess = NULL;
    /* Per-layer f32 row buffers, sized for a full (truncated) sample. */
    float *layer_bufs[CG_MAX_TAP_LAYERS] = {0};
    float *last_buf = NULL;
    char *line = NULL;
    cg_stats st = {0};
    int rc = 1;
    const double t0 = cg_now();

    if (ds4_session_create(&sess, engine, max_length) != 0 || !sess) {
        if (errlen) snprintf(err, errlen, "cache-gen: session creation failed");
        goto done;
    }
    if (ds4_session_set_hidden_taps(sess, tap_set, n_tap_user + 1,
                                    DS4_TAP_FORMAT_MEAN_HC) != 0) {
        if (errlen) snprintf(err, errlen,
                             "cache-gen: hidden tap configuration failed");
        goto done;
    }
    for (int l = 0; l < n_tap_user; l++) {
        layer_bufs[l] = malloc((size_t)max_length * (size_t)h * sizeof(float));
    }
    last_buf = malloc((size_t)max_length * (size_t)h * sizeof(float));
    for (int l = 0; l < n_tap_user; l++) {
        if (!layer_bufs[l]) goto oom;
    }
    if (!last_buf) goto oom;

    /* Chunk by the SESSION prefill cap, not the raw engine option: the
     * engine resolves env/default (4096) internally, and the tap holds only
     * the most recent eval's rows, so cache-gen must read taps at exactly
     * the engine's own eval granularity. */
    const int chunk = ds4_session_prefill_cap(sess);
    if (chunk < 1) {
        if (errlen) snprintf(err, errlen, "cache-gen: session prefill cap unavailable");
        goto done;
    }
    size_t line_cap = 0;
    ssize_t line_len;
    rc = 0;
    while (rc == 0 && (line_len = getline(&line, &line_cap, fp)) >= 0) {
        while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
            line[--line_len] = '\0';
        }
        if (line_len == 0) continue;
        st.lines++;
        if (opt->cancel && opt->cancel(opt->cancel_ud)) {
            rc = 2;
            break;
        }

        cg_msg *msgs = NULL;
        int n_msgs = 0;
        cg_drop_reason drop = cg_parse_dataset_line(line, &msgs, &n_msgs);
        if (drop == CG_DROP_OK) drop = cg_validate_messages(msgs, n_msgs);
        if (drop != CG_DROP_OK) {
            st.dropped[drop]++;
            cg_msgs_free(msgs, n_msgs);
            continue;
        }

        /* loss_mask span: the prefix render (conversation up to the
         * assistant start) is a byte prefix of the full render, so the
         * token counts delimit the assistant span exactly. */
        char *prefix_text = cg_render(msgs, n_msgs - 1);
        char *full_text = cg_render(msgs, n_msgs);
        cg_msgs_free(msgs, n_msgs);
        ds4_tokens prefix_toks = {0}, full_toks = {0};
        ds4_tokenize_rendered_chat(engine, prefix_text, &prefix_toks);
        ds4_tokenize_rendered_chat(engine, full_text, &full_toks);
        free(prefix_text);
        free(full_text);
        int t_total = full_toks.len;
        const int t_loss0 = prefix_toks.len;
        ds4_tokens_free(&prefix_toks);
        if (t_total > max_length) t_total = max_length;
        const int loss_tokens = t_total - t_loss0;
        if (t_loss0 >= t_total || loss_tokens < min_loss) {
            st.dropped[CG_DROP_FEW_LOSS_TOKENS]++;
            ds4_tokens_free(&full_toks);
            continue;
        }
        full_toks.len = t_total; /* truncated view for the sync loop below */

        /* Teacher-forced prefill in engine prefill chunks, reading the tap
         * rows of every chunk exactly once (a tap holds only the most
         * recent eval's rows — same discipline as ds4_hsexport_emit). */
        int done = 0;
        bool ok = true;
        while (ok && done < t_total) {
            int target = done + chunk;
            if (target > t_total) target = t_total;
            /* ds4_session_sync extends a live tail by single-token decode
             * (a 1-row eval, useless for taps) when the suffix is shorter
             * than the engine resume-prefill minimum (4): never leave a
             * 1..3-token tail for the next chunk. */
            if (t_total - target > 0 && t_total - target < 4) {
                target = t_total - 4;
            }
            ds4_tokens sub = full_toks;
            sub.len = target;
            int start = ds4_session_common_prefix(sess, &sub);
            if (start < ds4_session_pos(sess)) {
                /* The new prompt is not a tail extension of the live
                 * checkpoint (a different conversation, or a divergent
                 * suffix): sync rebuilds from scratch and the tap rows
                 * cover 0..target-1, not start..target-1. */
                start = 0;
            }
            char sync_err[160] = {0};
            if (ds4_session_sync(sess, &sub, sync_err, sizeof(sync_err)) != 0) {
                if (errlen) snprintf(err, errlen, "cache-gen: prefill failed: %s",
                                     sync_err[0] ? sync_err : "unknown error");
                ok = false;
                rc = 1;
                break;
            }
            done = ds4_session_pos(sess);
            const int rows = done - start;
            if (rows <= 0) {
                if (errlen) snprintf(err, errlen,
                                     "cache-gen: prefill made no progress");
                ok = false;
                rc = 1;
                break;
            }
            for (int l = 0; l < n_tap_user && ok; l++) {
                const int got = ds4_session_read_tap(
                    sess, tap_set[l],
                    layer_bufs[l] + (size_t)start * (size_t)h, rows);
                if (got != rows) {
                    if (errlen) snprintf(err, errlen,
                                         "cache-gen: tap row mismatch on layer %d: expected %d, got %d",
                                         tap_set[l], rows, got);
                    ok = false;
                    rc = 1;
                }
            }
            if (ok) {
                const int got = ds4_session_read_tap(
                    sess, DS4_TAP_LAYER_FINAL_NORM,
                    last_buf + (size_t)start * (size_t)h, rows);
                if (got != rows) {
                    if (errlen) snprintf(err, errlen,
                                         "cache-gen: final-norm tap row mismatch: expected %d, got %d",
                                         rows, got);
                    ok = false;
                    rc = 1;
                }
            }
            if (opt->cancel && opt->cancel(opt->cancel_ud)) {
                rc = 2;
                break;
            }
        }
        if (rc != 0) {
            ds4_tokens_free(&full_toks);
            break;
        }

        /* input_ids / loss_mask sections. */
        int32_t *ids = malloc((size_t)t_total * sizeof(int32_t));
        uint8_t *mask = malloc((size_t)t_total);
        if (!ids || !mask) {
            free(ids);
            free(mask);
            ds4_tokens_free(&full_toks);
            goto oom;
        }
        for (int t = 0; t < t_total; t++) {
            ids[t] = (int32_t)full_toks.v[t];
            mask[t] = (t >= t_loss0) ? 1 : 0;
        }
        ds4_tokens_free(&full_toks);
        const int wrc = ds4_cache_writer_write_sample(
            writer, st.written, ids, mask, (uint32_t)t_total,
            (const float *const *)layer_bufs, last_buf, err, errlen);
        free(ids);
        free(mask);
        if (wrc != 0) {
            rc = 1;
            break;
        }
        st.written++;
        st.tokens += (uint64_t)t_total;
        st.loss_tokens += (uint64_t)loss_tokens;
        if (st.written % 100 == 0) {
            const double el = cg_now() - t0;
            fprintf(stderr,
                    "ds4: cache-gen: %llu samples, %llu tokens, %.0f tok/s\n",
                    (unsigned long long)st.written,
                    (unsigned long long)st.tokens,
                    el > 0 ? (double)st.tokens / el : 0.0);
        }
    }
    if (rc == 0 && ferror(fp)) {
        if (errlen) snprintf(err, errlen, "cache-gen: dataset read error");
        rc = 1;
    }
    goto done;

oom:
    if (errlen) snprintf(err, errlen, "cache-gen: out of memory");
    rc = 1;

done:
    free(line);
    for (int l = 0; l < CG_MAX_TAP_LAYERS; l++) free(layer_bufs[l]);
    free(last_buf);
    if (sess) ds4_session_free(sess);
    fclose(fp);
    const uint64_t shard_bytes = ds4_cache_writer_bytes(writer);
    char werr[256] = {0};
    if (ds4_cache_writer_close(writer, werr, sizeof(werr)) != 0 && rc == 0) {
        if (errlen) snprintf(err, errlen, "%s", werr);
        rc = 1;
    }
    const double el = cg_now() - t0;
    fprintf(stderr,
            "ds4: cache-gen %s: %llu samples written (%llu tokens, %llu loss tokens), "
            "%llu lines read, %.2f MiB in shards, %.1fs (%.0f tok/s)\n",
            rc == 0 ? "done" : rc == 2 ? "interrupted" : "failed",
            (unsigned long long)st.written,
            (unsigned long long)st.tokens,
            (unsigned long long)st.loss_tokens,
            (unsigned long long)st.lines,
            (double)shard_bytes / (1024.0 * 1024.0),
            el, el > 0 ? (double)st.tokens / el : 0.0);
    for (int d = 1; d < CG_DROP_N; d++) {
        if (st.dropped[d]) {
            fprintf(stderr, "ds4: cache-gen: dropped %llu samples: %s\n",
                    (unsigned long long)st.dropped[d], cg_drop_names[d]);
        }
    }
    return rc;
}
