/* Model-free tests for the hidden-state export channel (ds4_hsexport.c).
 *
 * Drives the real publisher over loopback TCP with the engine/session
 * surface stubbed by fixtures, so no GGUF is needed. Covers: HELLO
 * wildcard/exact/mismatch handshakes and the ack, HIDDEN_BATCH roundtrip
 * (f32 and f16 wire), the prefix-hash chain, position-gap desync poisoning,
 * bounded-queue backpressure (no loss, no reorder), the oversized-batch on
 * an empty queue, the no-subscriber drop/re-anchor path, and the train-sink
 * dump layout end to end.
 *
 * Pure C99 + POSIX sockets: builds and runs on any host. */

#define _POSIX_C_SOURCE 200809L

#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_hsexport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int g_failed = 0;

#define CHECK(cond) do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            g_failed++; \
        } \
    } while (0)

/* =========================================================================
 * Engine/session stubs (no model): the publisher sees a fake session whose
 * tap rows follow a deterministic pattern.
 * ========================================================================= */

#define FAKE_MODEL_ID 4242
#define FAKE_MODEL_NAME "ds4-hsx-fake"
#define FAKE_ROW_VALUES 8
#define FAKE_CTX 4096
#define FAKE_MAX_TOKENS 131072

typedef struct {
    int tokens[FAKE_MAX_TOKENS];
    ds4_tokens tl;
    int pos;
    int tap_rows; /* rows produced by the most recent eval */
    int tap_layers[DS4_HSEXPORT_MAX_LAYERS];
    int tap_n_layers;
    int tap_format;
} fake_session;

static float tap_pattern(int layer, int row, int col) {
    return (float)(layer * 100 + row * 10 + col);
}

int ds4_engine_model_id(ds4_engine *e) {
    (void)e;
    return FAKE_MODEL_ID;
}

const char *ds4_engine_model_name(ds4_engine *e) {
    (void)e;
    return FAKE_MODEL_NAME;
}

uint64_t ds4_engine_hidden_f32_values(ds4_engine *e) {
    (void)e;
    return FAKE_ROW_VALUES;
}

int ds4_session_ctx(ds4_session *s) {
    (void)s;
    return FAKE_CTX;
}

const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    fake_session *fs = (fake_session *)(void *)s;
    fs->tl.v = fs->tokens;
    fs->tl.len = fs->pos;
    fs->tl.cap = FAKE_MAX_TOKENS;
    return &fs->tl;
}

int ds4_session_set_hidden_taps(ds4_session *s, const int *layers,
                                int n_layers, int format) {
    fake_session *fs = (fake_session *)(void *)s;
    if (n_layers <= 0 || n_layers > DS4_HSEXPORT_MAX_LAYERS) return -1;
    for (int i = 0; i < n_layers; i++) fs->tap_layers[i] = layers[i];
    fs->tap_n_layers = n_layers;
    fs->tap_format = format;
    return 0;
}

int ds4_session_read_tap(const ds4_session *s, int layer,
                         float *out, int max_rows) {
    const fake_session *fs = (const fake_session *)(const void *)s;
    bool known = false;
    for (int i = 0; i < fs->tap_n_layers; i++) {
        if (fs->tap_layers[i] == layer) known = true;
    }
    if (!known) return -1;
    const int rows = max_rows < fs->tap_rows ? max_rows : fs->tap_rows;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < FAKE_ROW_VALUES; c++) {
            out[(size_t)r * FAKE_ROW_VALUES + c] = tap_pattern(layer, r, c);
        }
    }
    return rows;
}

/* Same FNV-1a as the real wrapper in ds4_distributed.c; the hardcoded
 * vectors in test_batch_roundtrip pin it independently of this copy. */
uint64_t ds4_dist_token_hash_update_span(uint64_t h, const int *tokens,
                                         uint32_t n_tokens) {
    for (uint32_t i = 0; i < n_tokens; i++) {
        const uint32_t t = (uint32_t)tokens[i];
        for (int b = 0; b < 4; b++) {
            h ^= (t >> (b * 8)) & 0xffu;
            h *= 1099511628211ull;
        }
    }
    return h;
}

/* =========================================================================
 * Small helpers.
 * ========================================================================= */

static void sleep_ms(long ms) {
    const struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static void put_u32(unsigned char *p, uint32_t v) {
    v = htonl(v);
    memcpy(p, &v, sizeof(v));
}

static uint32_t get_u32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return ntohl(v);
}

static int write_full(int fd, const void *buf, size_t len) {
    const unsigned char *p = buf;
    while (len > 0) {
        const ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int read_full(int fd, void *buf, size_t len) {
    unsigned char *p = buf;
    while (len > 0) {
        const ssize_t n = recv(fd, p, len, 0);
        if (n < 0) return -1;
        if (n == 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static int test_connect_rsv(int port, int rcvbuf) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    /* Set the receive buffer before connect so the negotiated window is
     * small from the SYN on: backpressure tests rely on the publisher
     * sender blocking on an unread socket instead of the loopback stack
     * absorbing whole batches. Setting it after connect is too late on
     * macOS (the window scale is already negotiated). */
    if (rcvbuf > 0)
        (void)setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    sa.sin_addr.s_addr = htonl(0x7f000001u);
    if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int test_connect(int port) {
    return test_connect_rsv(port, 0);
}

static int send_hello(int fd, uint32_t model_id, const char *name,
                      const int *layers, int n_layers, uint32_t ctx) {
    unsigned char payload[24 + DS4_HSEXPORT_MAX_LAYERS * 4 +
                          DS4_HSEXPORT_MAX_MODEL_NAME];
    const size_t name_len = name ? strlen(name) : 0;
    put_u32(payload + 0, model_id);
    put_u32(payload + 4, (uint32_t)n_layers);
    put_u32(payload + 8, DS4_TAP_FORMAT_RAW_HC);
    put_u32(payload + 12, DS4_HSEXPORT_MODE_STREAM);
    put_u32(payload + 16, ctx);
    put_u32(payload + 20, (uint32_t)name_len);
    size_t off = 24;
    for (int i = 0; i < n_layers; i++) {
        put_u32(payload + off, (uint32_t)layers[i]);
        off += 4;
    }
    if (name_len) memcpy(payload + off, name, name_len);
    off += name_len;

    unsigned char hdr[12];
    put_u32(hdr + 0, DS4_HSEXPORT_MAGIC);
    put_u32(hdr + 4, DS4_HSEXPORT_MSG_HELLO);
    put_u32(hdr + 8, (uint32_t)off);
    if (write_full(fd, hdr, sizeof(hdr)) != 0) return -1;
    return write_full(fd, payload, off);
}

/* Returns 1 on a frame, 0 on EOF, -1 on error. *out is malloc'd. */
static int recv_frame(int fd, uint32_t *type, unsigned char **out,
                      uint32_t *bytes) {
    unsigned char hdr[12];
    *out = NULL;
    *type = 0;
    *bytes = 0;
    const int rc = read_full(fd, hdr, sizeof(hdr));
    if (rc != 1) return rc;
    if (get_u32(hdr + 0) != DS4_HSEXPORT_MAGIC) return -1;
    *type = get_u32(hdr + 4);
    *bytes = get_u32(hdr + 8);
    *out = malloc(*bytes ? *bytes : 1);
    if (!*out) return -1;
    if (*bytes && read_full(fd, *out, *bytes) != 1) {
        free(*out);
        *out = NULL;
        return -1;
    }
    return 1;
}

static bool bytes_contain(const unsigned char *hay, size_t hlen,
                          const char *needle, size_t nlen) {
    if (nlen == 0 || hlen < nlen) return false;
    for (size_t i = 0; i + nlen <= hlen; i++) {
        if (memcmp(hay + i, needle, nlen) == 0) return true;
    }
    return false;
}

/* Skips the publisher's HELLO ack after a successful handshake. */
static int recv_ack(int fd) {
    uint32_t type, bytes;
    unsigned char *pl = NULL;
    const int rc = recv_frame(fd, &type, &pl, &bytes);
    if (rc != 1 || type != DS4_HSEXPORT_MSG_HELLO) {
        free(pl);
        return -1;
    }
    free(pl);
    return 0;
}

/* =========================================================================
 * Publisher fixture.
 * ========================================================================= */

typedef struct {
    ds4_hsexport *x;
    fake_session fs;
    int layers[2];
} pub_fixture;

static void fake_session_init(fake_session *fs) {
    memset(fs, 0, sizeof(*fs));
    for (int i = 0; i < FAKE_MAX_TOKENS; i++) fs->tokens[i] = i + 1;
}

static int pub_start(pub_fixture *pf, int port, int bits, uint64_t queue_bytes) {
    memset(pf, 0, sizeof(*pf));
    fake_session_init(&pf->fs);
    pf->layers[0] = 3;
    pf->layers[1] = 5;
    ds4_hsexport_options opt = {
        .host = "127.0.0.1",
        .port = port,
        .bits = bits,
        .queue_bytes = queue_bytes,
    };
    char err[256];
    pf->x = ds4_hsexport_start((ds4_engine *)(void *)pf, &opt, err, sizeof(err));
    if (!pf->x) {
        fprintf(stderr, "pub_start: %s\n", err);
        g_failed++;
        return -1;
    }
    if (ds4_hsexport_attach(pf->x, (ds4_session *)(void *)&pf->fs, 7,
                            pf->layers, 2, err, sizeof(err)) != 0) {
        fprintf(stderr, "pub_start attach: %s\n", err);
        g_failed++;
        ds4_hsexport_stop(pf->x);
        pf->x = NULL;
        return -1;
    }
    return 0;
}

static void pub_emit(pub_fixture *pf, fake_session *fs, int pos0, int n) {
    fs->tap_rows = n;
    fs->pos = pos0 + n;
    CHECK(ds4_hsexport_emit(pf->x, (ds4_session *)(void *)fs,
                            (uint32_t)pos0, (uint32_t)n) == 0);
}

/* =========================================================================
 * Tests.
 * ========================================================================= */

static void test_parse_malformed(void) {
    unsigned char buf[256];
    ds4_hsexport_batch_view v;
    CHECK(ds4_hsexport_parse_batch(NULL, 0, &v) == -1);
    CHECK(ds4_hsexport_parse_batch(buf, 10, &v) == -1);

    memset(buf, 0, sizeof(buf));
    put_u32(buf + 0, 0);          /* session hi */
    put_u32(buf + 4, 1);          /* session lo */
    put_u32(buf + 8, 0);          /* pos0 */
    put_u32(buf + 12, 2);         /* n_tokens */
    put_u32(buf + 16, 1);         /* n_layers */
    put_u32(buf + 20, DS4_TAP_FORMAT_RAW_HC);
    put_u32(buf + 24, 32);        /* bits */
    put_u32(buf + 28, 4);         /* row_values */
    /* hash + reserved left zero */
    const size_t fixed = sizeof(ds4_hsexport_batch_fixed);
    CHECK(fixed == 44);
    CHECK(ds4_hsexport_parse_batch(buf, fixed, &v) == -1);

    put_u32(buf + fixed + 0, 3);  /* layer id */
    put_u32(buf + fixed + 4, 7);  /* token ids */
    put_u32(buf + fixed + 8, 8);
    for (int i = 0; i < 8; i++) { /* one layer, two rows of four f32 */
        const float f = tap_pattern(3, i / 4, i % 4);
        memcpy(buf + fixed + 12 + (size_t)i * 4, &f, 4);
    }
    CHECK(ds4_hsexport_parse_batch(buf, fixed + 12 + 32, &v) == 0);
    CHECK(v.session_id == 1 && v.pos0 == 0 && v.n_tokens == 2);
    CHECK(v.n_layers == 1 && v.bits == 32 && v.row_values == 4);
    CHECK(v.layer_ids[0] == 3 && v.token_ids[0] == 7 && v.token_ids[1] == 8);
    CHECK(v.data_bytes == 32);
    float got = 0.0f;
    memcpy(&got, v.data + 4, 4);
    CHECK(got == tap_pattern(3, 0, 1));
    ds4_hsexport_batch_view_free(&v);

    put_u32(buf + 12, 0);         /* n_tokens 0 is invalid */
    CHECK(ds4_hsexport_parse_batch(buf, fixed, &v) == -1);
    put_u32(buf + 12, 2);
    put_u32(buf + 16, DS4_HSEXPORT_MAX_LAYERS + 1); /* too many layers */
    CHECK(ds4_hsexport_parse_batch(buf, fixed, &v) == -1);
}

/* One handshake per publisher instance: a disconnected subscriber only
 * frees its slot on the next send, so each sub-case restarts cleanly. */
static void test_handshake(void) {
    pub_fixture pf;
    uint32_t type, bytes;
    unsigned char *pl = NULL;

    /* Wildcard: the ack carries the active configuration. */
    CHECK(pub_start(&pf, 18331, 32, 0) == 0);
    if (pf.x) {
        const int fd = test_connect(18331);
        CHECK(fd >= 0);
        CHECK(send_hello(fd, 0, NULL, NULL, 0, 0) == 0);
        CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
        CHECK(type == DS4_HSEXPORT_MSG_HELLO);
        if (type == DS4_HSEXPORT_MSG_HELLO) {
            CHECK(get_u32(pl + 0) == FAKE_MODEL_ID);
            CHECK(get_u32(pl + 4) == 2); /* n_layers */
            CHECK(get_u32(pl + 8) == DS4_TAP_FORMAT_RAW_HC);
            CHECK(get_u32(pl + 12) == DS4_HSEXPORT_MODE_STREAM);
            CHECK(get_u32(pl + 16) == FAKE_CTX);
            const uint32_t nl = get_u32(pl + 20);
            CHECK(nl == strlen(FAKE_MODEL_NAME));
            CHECK(get_u32(pl + 24) == 3 && get_u32(pl + 28) == 5);
            CHECK(bytes == 24 + 8 + nl);
            CHECK(memcmp(pl + 32, FAKE_MODEL_NAME, nl) == 0);
        }
        free(pl);
        close(fd);
        ds4_hsexport_stop(pf.x);
    }

    /* Exact match. */
    CHECK(pub_start(&pf, 18331, 32, 0) == 0);
    if (pf.x) {
        const int good[2] = {3, 5};
        const int fd = test_connect(18331);
        CHECK(fd >= 0);
        CHECK(send_hello(fd, FAKE_MODEL_ID, FAKE_MODEL_NAME, good, 2, FAKE_CTX) == 0);
        CHECK(recv_ack(fd) == 0);
        close(fd);
        ds4_hsexport_stop(pf.x);
    }

    /* Layer set mismatch: rejected with an ERROR frame. */
    CHECK(pub_start(&pf, 18331, 32, 0) == 0);
    if (pf.x) {
        const int bad[2] = {3, 6};
        const int fd = test_connect(18331);
        CHECK(fd >= 0);
        CHECK(send_hello(fd, FAKE_MODEL_ID, FAKE_MODEL_NAME, bad, 2, 0) == 0);
        CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
        CHECK(type == DS4_HSEXPORT_MSG_ERROR);
        free(pl);
        close(fd);
        ds4_hsexport_stop(pf.x);
    }

    /* Wrong model id: rejected. */
    CHECK(pub_start(&pf, 18331, 32, 0) == 0);
    if (pf.x) {
        const int good[2] = {3, 5};
        const int fd = test_connect(18331);
        CHECK(fd >= 0);
        CHECK(send_hello(fd, 999, FAKE_MODEL_NAME, good, 2, 0) == 0);
        CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
        CHECK(type == DS4_HSEXPORT_MSG_ERROR);
        free(pl);
        close(fd);
        ds4_hsexport_stop(pf.x);
    }

    /* Partial wildcard (model id 0 with a layer set): rejected. */
    CHECK(pub_start(&pf, 18331, 32, 0) == 0);
    if (pf.x) {
        const int good[2] = {3, 5};
        const int fd = test_connect(18331);
        CHECK(fd >= 0);
        CHECK(send_hello(fd, 0, NULL, good, 2, 0) == 0);
        CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
        CHECK(type == DS4_HSEXPORT_MSG_ERROR);
        free(pl);
        close(fd);
        ds4_hsexport_stop(pf.x);
    }
}

static void test_batch_roundtrip(void) {
    pub_fixture pf;
    CHECK(pub_start(&pf, 18332, 32, 0) == 0);
    if (!pf.x) return;
    const int fd = test_connect(18332);
    CHECK(fd >= 0);
    CHECK(send_hello(fd, 0, NULL, NULL, 0, 0) == 0);
    CHECK(recv_ack(fd) == 0);

    pub_emit(&pf, &pf.fs, 0, 3);
    pub_emit(&pf, &pf.fs, 3, 2);

    uint32_t type, bytes;
    unsigned char *pl = NULL;
    ds4_hsexport_batch_view v;

    CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
    CHECK(type == DS4_HSEXPORT_MSG_HIDDEN_BATCH);
    CHECK(ds4_hsexport_parse_batch(pl, bytes, &v) == 0);
    CHECK(v.session_id == 7 && v.pos0 == 0 && v.n_tokens == 3);
    CHECK(v.n_layers == 2 && v.bits == 32 && v.row_values == FAKE_ROW_VALUES);
    CHECK(v.format == DS4_TAP_FORMAT_RAW_HC);
    CHECK(v.layer_ids[0] == 3 && v.layer_ids[1] == 5);
    CHECK(v.token_ids[0] == 1 && v.token_ids[1] == 2 && v.token_ids[2] == 3);
    /* Independent FNV-1a anchor, computed offline (see hidden_export_client.py). */
    CHECK(v.prefix_hash == 0x6fdedb25bbb27e53ull);
    CHECK(v.data_bytes == (size_t)2 * 3 * FAKE_ROW_VALUES * 4);
    for (int l = 0; l < 2; l++) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < FAKE_ROW_VALUES; c++) {
                float got = 0.0f;
                const size_t idx = ((size_t)l * 3 + (size_t)r) * FAKE_ROW_VALUES + (size_t)c;
                memcpy(&got, v.data + idx * 4, 4);
                if (got != tap_pattern(pf.layers[l], r, c)) {
                    CHECK(0);
                    l = 2;
                    r = 3;
                    break;
                }
            }
        }
    }
    ds4_hsexport_batch_view_free(&v);
    free(pl);

    CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
    CHECK(ds4_hsexport_parse_batch(pl, bytes, &v) == 0);
    CHECK(v.pos0 == 3 && v.n_tokens == 2);
    CHECK(v.token_ids[0] == 4 && v.token_ids[1] == 5);
    CHECK(v.prefix_hash == 0xb52f129774b3c062ull);
    ds4_hsexport_batch_view_free(&v);
    free(pl);

    close(fd);
    ds4_hsexport_stop(pf.x);
}

/* Independent f16 conversion for cross-checking the wire encoding; pinned
 * by the spot vectors below (computed with Python struct '>e'). */
static uint16_t test_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exp = (int)((bits >> 23) & 0xffu);
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0) return (uint16_t)sign;
    if (exp == 0xff) return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
    const int e = exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const int shift = 14 - e;
        uint32_t hm = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        if (rem > (1u << (shift - 1)) ||
            (rem == (1u << (shift - 1)) && (hm & 1u))) hm++;
        return (uint16_t)(sign | hm);
    }
    uint32_t hm = mant >> 13;
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) hm++;
    return (uint16_t)(sign | ((uint32_t)e << 10) | hm);
}

static void test_f16_wire(void) {
    /* Spot vectors first: they pin the rounding of the test converter and,
     * through it, the publisher's wire encoding. */
    CHECK(test_f32_to_f16(0.0f) == 0x0000);
    CHECK(test_f32_to_f16(1.0f) == 0x3c00);
    CHECK(test_f32_to_f16(-2.0f) == 0xc000);
    CHECK(test_f32_to_f16(0.5f) == 0x3800);
    CHECK(test_f32_to_f16(65504.0f) == 0x7bff);
    CHECK(test_f32_to_f16(0.1f) == 0x2e66);
    CHECK(test_f32_to_f16(300.0f) == 0x5cb0);
    CHECK(test_f32_to_f16(517.0f) == 0x600a);

    pub_fixture pf;
    CHECK(pub_start(&pf, 18333, 16, 0) == 0);
    if (!pf.x) return;
    const int fd = test_connect(18333);
    CHECK(fd >= 0);
    CHECK(send_hello(fd, 0, NULL, NULL, 0, 0) == 0);
    CHECK(recv_ack(fd) == 0);

    pub_emit(&pf, &pf.fs, 0, 2);

    uint32_t type, bytes;
    unsigned char *pl = NULL;
    ds4_hsexport_batch_view v;
    CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
    CHECK(ds4_hsexport_parse_batch(pl, bytes, &v) == 0);
    CHECK(v.bits == 16);
    CHECK(v.data_bytes == (size_t)2 * 2 * FAKE_ROW_VALUES * 2);
    bool ok = true;
    for (int l = 0; l < 2 && ok; l++) {
        for (int r = 0; r < 2 && ok; r++) {
            for (int c = 0; c < FAKE_ROW_VALUES && ok; c++) {
                const size_t idx = ((size_t)l * 2 + (size_t)r) * FAKE_ROW_VALUES + (size_t)c;
                uint16_t got;
                memcpy(&got, v.data + idx * 2, 2); /* payload is little-endian */
                const uint16_t want = test_f32_to_f16(tap_pattern(pf.layers[l], r, c));
                if (got != want) {
                    CHECK(0);
                    ok = false;
                }
            }
        }
    }
    ds4_hsexport_batch_view_free(&v);
    free(pl);
    close(fd);
    ds4_hsexport_stop(pf.x);
}

static void test_desync_gap(void) {
    pub_fixture pf;
    CHECK(pub_start(&pf, 18334, 32, 0) == 0);
    if (!pf.x) return;
    const int fd = test_connect(18334);
    CHECK(fd >= 0);
    CHECK(send_hello(fd, 0, NULL, NULL, 0, 0) == 0);
    CHECK(recv_ack(fd) == 0);

    pub_emit(&pf, &pf.fs, 0, 2);
    pub_emit(&pf, &pf.fs, 5, 1); /* position gap: expected pos 2 */

    uint32_t type, bytes;
    unsigned char *pl = NULL;
    ds4_hsexport_batch_view v;
    CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
    CHECK(type == DS4_HSEXPORT_MSG_HIDDEN_BATCH);
    CHECK(ds4_hsexport_parse_batch(pl, bytes, &v) == 0);
    CHECK(v.pos0 == 0 && v.n_tokens == 2);
    ds4_hsexport_batch_view_free(&v);
    free(pl);

    CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
    CHECK(type == DS4_HSEXPORT_MSG_ERROR);
    if (pl != NULL && type == DS4_HSEXPORT_MSG_ERROR) {
        CHECK(bytes_contain(pl, bytes, "position gap", 12));
    }
    free(pl);
    /* The poisoned connection is closed by the publisher. */
    CHECK(recv_frame(fd, &type, &pl, &bytes) == 0);

    close(fd);
    ds4_hsexport_stop(pf.x);
}

typedef struct {
    pub_fixture *pf;
    int pos0;
    int n;
    int count;
    volatile int done;
} emit_job;

static void *emit_thread(void *arg) {
    emit_job *j = arg;
    for (int i = 0; i < j->count; i++) {
        j->pf->fs.tap_rows = j->n;
        j->pf->fs.pos = j->pos0 + (i + 1) * j->n;
        if (ds4_hsexport_emit(j->pf->x, (ds4_session *)(void *)&j->pf->fs,
                              (uint32_t)(j->pos0 + i * j->n), (uint32_t)j->n) != 0) {
            fprintf(stderr, "emit_thread: emit failed at %d\n", i);
            g_failed++;
            break;
        }
    }
    j->done = 1;
    return NULL;
}

static void test_backpressure(void) {
    enum { N = 8192, BATCHES = 6 };
    const uint64_t per_batch = (uint64_t)2 * N * FAKE_ROW_VALUES * 4; /* 512 KiB */
    pub_fixture pf;
    CHECK(pub_start(&pf, 18335, 32, 2 * per_batch) == 0);
    if (!pf.x) return;
    /* Batches are sized far above what an unread loopback socket absorbs
     * (measured ~640 KiB on macOS; SO_RCVBUF is not honored there), so the
     * publisher sender stalls mid-batch and the queue is the only absorber.
     * A small receive buffer still helps on hosts that honor it. */
    const int fd = test_connect_rsv(18335, 4096);
    CHECK(fd >= 0);
    CHECK(send_hello(fd, 0, NULL, NULL, 0, 0) == 0);
    CHECK(recv_ack(fd) == 0);

    /* The subscriber stops reading; the queue (2 batches) plus the socket
     * buffers cannot absorb 6 batches, so the producer must block. */
    emit_job j = { &pf, 0, N, BATCHES, 0 };
    pthread_t th;
    CHECK(pthread_create(&th, NULL, emit_thread, &j) == 0);
    sleep_ms(500);
    CHECK(j.done == 0);

    /* Drain everything: no loss, no reorder, continuous hash chain. */
    uint64_t hash = DS4_DIST_TOKEN_HASH_INIT;
    for (int i = 0; i < BATCHES; i++) {
        uint32_t type, bytes;
        unsigned char *pl = NULL;
        ds4_hsexport_batch_view v;
        CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
        CHECK(type == DS4_HSEXPORT_MSG_HIDDEN_BATCH);
        CHECK(ds4_hsexport_parse_batch(pl, bytes, &v) == 0);
        if (v.pos0 != (uint32_t)(i * N) || v.n_tokens != (uint32_t)N) {
            CHECK(0);
        }
        hash = ds4_dist_token_hash_update_span(hash, v.token_ids, v.n_tokens);
        CHECK(hash == v.prefix_hash);
        ds4_hsexport_batch_view_free(&v);
        free(pl);
    }
    pthread_join(th, NULL);
    CHECK(j.done == 1);

    close(fd);
    ds4_hsexport_stop(pf.x);
}

static void test_oversized_empty_queue(void) {
    enum { N = 32768 };
    /* One batch is 2*32768*8*4 = 2 MiB, far over the 64 KiB queue budget and
     * far over what an unread loopback socket absorbs (~640 KiB on macOS),
     * so the publisher sender stalls mid-batch-1 and cannot pop batch 2. */
    pub_fixture pf;
    CHECK(pub_start(&pf, 18336, 32, 64 * 1024) == 0); /* cap << one batch */
    if (!pf.x) return;
    /* Small receive buffer as a second shrink where the host honors it. */
    const int fd = test_connect_rsv(18336, 4096);
    CHECK(fd >= 0);
    CHECK(send_hello(fd, 0, NULL, NULL, 0, 0) == 0);
    CHECK(recv_ack(fd) == 0);

    /* Batch 1: an empty queue always accepts, even oversized. */
    emit_job j1 = { &pf, 0, N, 1, 0 };
    pthread_t th1;
    CHECK(pthread_create(&th1, NULL, emit_thread, &j1) == 0);
    pthread_join(th1, NULL);
    CHECK(j1.done == 1);

    /* Wait until the sender popped batch 1 (it blocks on the unread socket),
     * so the queue is empty again. */
    bool drained = false;
    for (int i = 0; i < 100 && !drained; i++) {
        drained = ds4_hsexport_queued_bytes(pf.x) == 0;
        if (!drained) sleep_ms(20);
    }
    CHECK(drained);

    /* Batch 2: accepted immediately for the same reason. */
    emit_job j2 = { &pf, N, N, 1, 0 };
    pthread_t th2;
    CHECK(pthread_create(&th2, NULL, emit_thread, &j2) == 0);
    pthread_join(th2, NULL);
    CHECK(j2.done == 1);

    /* Batch 3: the queue is non-empty and over budget, so this blocks. */
    emit_job j3 = { &pf, 2 * N, N, 1, 0 };
    pthread_t th3;
    CHECK(pthread_create(&th3, NULL, emit_thread, &j3) == 0);
    sleep_ms(400);
    CHECK(j3.done == 0);

    /* Drain all three; order and positions must be intact. */
    for (int i = 0; i < 3; i++) {
        uint32_t type, bytes;
        unsigned char *pl = NULL;
        ds4_hsexport_batch_view v;
        CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
        CHECK(ds4_hsexport_parse_batch(pl, bytes, &v) == 0);
        CHECK(v.pos0 == (uint32_t)(i * N) && v.n_tokens == (uint32_t)N);
        ds4_hsexport_batch_view_free(&v);
        free(pl);
    }
    pthread_join(th3, NULL);
    CHECK(j3.done == 1);

    close(fd);
    ds4_hsexport_stop(pf.x);
}

static void test_no_subscriber_reanchor(void) {
    pub_fixture pf;
    CHECK(pub_start(&pf, 18337, 32, 0) == 0);
    if (!pf.x) return;

    /* No subscriber: emits advance the frontier without queueing. */
    for (int i = 0; i < 5; i++) pub_emit(&pf, &pf.fs, i * 2, 2);
    CHECK(ds4_hsexport_queued_bytes(pf.x) == 0);

    const int fd = test_connect(18337);
    CHECK(fd >= 0);
    CHECK(send_hello(fd, 0, NULL, NULL, 0, 0) == 0);
    CHECK(recv_ack(fd) == 0);

    /* The first batch after the connect re-anchors at the live frontier:
     * earlier positions are not replayed. */
    pub_emit(&pf, &pf.fs, 10, 2);
    uint32_t type, bytes;
    unsigned char *pl = NULL;
    ds4_hsexport_batch_view v;
    CHECK(recv_frame(fd, &type, &pl, &bytes) == 1);
    CHECK(type == DS4_HSEXPORT_MSG_HIDDEN_BATCH);
    CHECK(ds4_hsexport_parse_batch(pl, bytes, &v) == 0);
    CHECK(v.pos0 == 10 && v.n_tokens == 2);
    ds4_hsexport_batch_view_free(&v);
    free(pl);

    close(fd);
    ds4_hsexport_stop(pf.x);
}

typedef struct {
    ds4_hsexport_sink_options opt;
    volatile sig_atomic_t stop;
    char err[512];
    int rc;
    uint64_t batches;
    uint64_t tokens;
} sink_job;

static void *sink_thread(void *arg) {
    sink_job *j = arg;
    j->rc = ds4_hsexport_sink_run(&j->opt, &j->stop, j->err, sizeof(j->err));
    return NULL;
}

static uint32_t read_be32(FILE *fp, bool *ok) {
    unsigned char b[4];
    if (fread(b, 1, 4, fp) != 4) {
        *ok = false;
        return 0;
    }
    return get_u32(b);
}

static void test_sink_dump(void) {
    const char *path = "/tmp/ds4_hsx_test_dump.bin";
    pub_fixture pf;
    CHECK(pub_start(&pf, 18338, 32, 0) == 0);
    if (!pf.x) return;

    fake_session fs2;
    fake_session_init(&fs2);
    char err[256];
    CHECK(ds4_hsexport_attach(pf.x, (ds4_session *)(void *)&fs2, 9,
                              pf.layers, 2, err, sizeof(err)) == 0);

    sink_job j;
    memset(&j, 0, sizeof(j));
    j.opt.host = "127.0.0.1";
    j.opt.port = 18338;
    j.opt.dump_path = path;
    j.opt.batches_out = &j.batches;
    j.opt.tokens_out = &j.tokens;
    pthread_t th;
    CHECK(pthread_create(&th, NULL, sink_thread, &j) == 0);
    sleep_ms(300); /* let the sink connect before emitting */

    /* Interleaved emits on two sessions: the sink tracks both chains. */
    pub_emit(&pf, &pf.fs, 0, 2);
    pub_emit(&pf, &pf.fs, 2, 2);
    pub_emit(&pf, &fs2, 0, 2);
    pub_emit(&pf, &pf.fs, 4, 2);
    pub_emit(&pf, &fs2, 2, 2);
    sleep_ms(200);

    /* Stopping the publisher closes the subscriber socket: a clean EOF. */
    ds4_hsexport_stop(pf.x);
    pthread_join(th, NULL);
    if (j.rc != 0) fprintf(stderr, "sink: %s\n", j.err);
    CHECK(j.rc == 0);
    CHECK(j.batches == 5);
    CHECK(j.tokens == 10);

    /* Verify the dump: self-describing header, then verbatim batch records. */
    FILE *fp = fopen(path, "rb");
    CHECK(fp != NULL);
    if (!fp) return;
    bool ok = true;
    CHECK(read_be32(fp, &ok) == DS4_HSEXPORT_DUMP_MAGIC);
    CHECK(read_be32(fp, &ok) == DS4_HSEXPORT_DUMP_VERSION);
    CHECK(read_be32(fp, &ok) == FAKE_MODEL_ID);
    CHECK(read_be32(fp, &ok) == 2); /* n_layers */
    CHECK(read_be32(fp, &ok) == DS4_TAP_FORMAT_RAW_HC);
    CHECK(read_be32(fp, &ok) == 32); /* bits */
    CHECK(read_be32(fp, &ok) == FAKE_ROW_VALUES);
    CHECK(read_be32(fp, &ok) == FAKE_CTX);
    const uint32_t name_len = read_be32(fp, &ok);
    CHECK(name_len == strlen(FAKE_MODEL_NAME));
    CHECK(read_be32(fp, &ok) == 0);
    CHECK(read_be32(fp, &ok) == 0);
    CHECK(read_be32(fp, &ok) == 0);
    CHECK(read_be32(fp, &ok) == 3);
    CHECK(read_be32(fp, &ok) == 5);
    char name[sizeof(FAKE_MODEL_NAME)];
    CHECK(fread(name, 1, name_len, fp) == name_len);
    CHECK(memcmp(name, FAKE_MODEL_NAME, name_len) == 0);
    CHECK(ok);

    struct {
        uint64_t hash;
        uint32_t next_pos;
        bool anchored;
    } chain[2] = {0};
    const uint64_t ids[2] = {7, 9};
    int records = 0;
    for (;;) {
        unsigned char lenbuf[4];
        const size_t got = fread(lenbuf, 1, 4, fp);
        if (got == 0) break;
        CHECK(got == 4);
        const uint32_t len = get_u32(lenbuf);
        unsigned char *pl = malloc(len);
        CHECK(pl != NULL);
        CHECK(fread(pl, 1, len, fp) == len);
        ds4_hsexport_batch_view v;
        CHECK(ds4_hsexport_parse_batch(pl, len, &v) == 0);
        int slot = v.session_id == ids[0] ? 0 : 1;
        CHECK(v.session_id == ids[0] || v.session_id == ids[1]);
        if (!chain[slot].anchored) {
            chain[slot].hash = v.prefix_hash;
            chain[slot].next_pos = v.pos0 + v.n_tokens;
            chain[slot].anchored = true;
        } else {
            CHECK(v.pos0 == chain[slot].next_pos);
            const uint64_t h = ds4_dist_token_hash_update_span(
                chain[slot].hash, v.token_ids, v.n_tokens);
            CHECK(h == v.prefix_hash);
            chain[slot].hash = h;
            chain[slot].next_pos = v.pos0 + v.n_tokens;
        }
        float first = 0.0f;
        memcpy(&first, v.data, 4);
        CHECK(first == tap_pattern(3, 0, 0));
        ds4_hsexport_batch_view_free(&v);
        free(pl);
        records++;
    }
    CHECK(records == 5);
    CHECK(chain[0].next_pos == 6); /* session 7 covered positions 0..6 */
    CHECK(chain[1].next_pos == 4); /* session 9 covered positions 0..4 */
    fclose(fp);
    unlink(path);
}

int main(void) {
    test_parse_malformed();
    test_handshake();
    test_batch_roundtrip();
    test_f16_wire();
    test_desync_gap();
    test_backpressure();
    test_oversized_empty_queue();
    test_no_subscriber_reanchor();
    test_sink_dump();

    if (g_failed) {
        fprintf(stderr, "test_hsexport: %d failure(s)\n", g_failed);
        return 1;
    }
    puts("test_hsexport: all tests passed");
    return 0;
}
