/* =========================================================================
 * ds4_hsexport.c - Hidden-state export channel (HIDDEN_EXPORT.md).
 * =========================================================================
 *
 * One publisher per process, one subscriber at a time (Phase 1). The eval
 * paths enqueue HIDDEN_BATCH records built from the session tap buffers
 * (ds4_session_read_tap); a connection thread owns the socket: accept, HELLO
 * validation, then drain the queue. A full queue blocks the producer
 * (backpressure — training must not lose data); on disconnect the queue is
 * dropped and the next subscriber re-anchors its hash chain at the first
 * batch it receives.
 *
 * Same trust model as the distributed protocol: no encryption, no
 * authentication — trusted machines and trusted networks only.
 */

#include "ds4_hsexport.h"

#include "ds4_distributed.h" /* ds4_dist_token_hash_update_span */

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef MSG_NOSIGNAL
#define HSX_SEND_FLAGS MSG_NOSIGNAL
#else
#define HSX_SEND_FLAGS 0
#endif

#define HSX_DEFAULT_QUEUE_BYTES (256ull * 1024ull * 1024ull)
#define HSX_MAX_SESSIONS 16
#define HSX_DESYNC_MSG_LEN 160

typedef struct hsx_batch {
    struct hsx_batch *next;
    uint64_t session_id;
    uint32_t pos0;
    uint32_t n_tokens;
    uint64_t prefix_hash;
    size_t payload_f32_bytes;      /* queue accounting (wire may be f16) */
    int32_t tokens[];              /* n_tokens, then the f32 payload rows:
                                    * n_layers * n_tokens * row_values floats
                                    * follow the token ids in one block. */
} hsx_batch;

typedef struct {
    ds4_session *session;
    uint64_t id;
    uint32_t next_pos;      /* position the next exported batch must start at */
    uint64_t hash;          /* FNV-1a over tokens[0, next_pos) */
    uint64_t conn_gen;      /* connection generation this frontier belongs to */
    bool used;
} hsx_session;

struct ds4_hsexport {
    ds4_engine *engine;
    int listen_fd;
    struct sockaddr_storage bound; /* getsockname() of listen_fd */
    socklen_t bound_len;
    int bits;               /* wire payload width: 32 or 16 */
    uint64_t queue_cap;
    uint32_t row_values;    /* f32 values per RAW_HC row (hc_dim) */

    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv_items;
    pthread_cond_t cv_space;
    bool stopping;
    int sub_fd;             /* subscriber socket; -1 when none */
    uint64_t conn_gen;      /* bumped on every accepted subscriber */

    hsx_batch *head;
    hsx_batch *tail;
    uint64_t queued_bytes;
    char desync_msg[HSX_DESYNC_MSG_LEN]; /* set: send ERROR and close */

    /* Phase 1: one tap configuration shared by all attached sessions. */
    int layers[DS4_HSEXPORT_MAX_LAYERS];
    int n_layers;
    bool config_set;

    hsx_session sessions[HSX_MAX_SESSIONS];
};

/* =========================================================================
 * Wire helpers (same conventions as ds4_distributed.c).
 * ========================================================================= */

static int hsx_write_full(int fd, const void *buf, size_t len) {
    const unsigned char *p = buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, HSX_SEND_FLAGS);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int hsx_read_full(int fd, void *buf, size_t len) {
    unsigned char *p = buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static int hsx_write_frame_header(int fd, uint32_t type, uint32_t bytes) {
    uint32_t h[3] = { htonl(DS4_HSEXPORT_MAGIC), htonl(type), htonl(bytes) };
    return hsx_write_full(fd, h, sizeof(h));
}

static int hsx_send_error(int fd, const char *msg) {
    if (!msg) msg = "hidden export protocol error";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    if (hsx_write_frame_header(fd, DS4_HSEXPORT_MSG_ERROR, (uint32_t)len) != 0) {
        return -1;
    }
    return hsx_write_full(fd, msg, len);
}

/* f32 -> f16, identical conversion to dist_f32_to_f32's counterpart in
 * ds4_distributed.c so both channels share one wire encoding. */
static uint16_t hsx_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

/* Payload write mirroring dist_write_activation_payload: f32 goes out raw,
 * f16 is converted in bounded chunks. */
static int hsx_write_payload(int fd, const float *src, uint64_t values, uint32_t bits) {
    if (values == 0) return 0;
    if (!src) return -1;
    if (bits == 32u) {
        return hsx_write_full(fd, src, (size_t)values * sizeof(float));
    }
    if (bits != 16u) return -1;

    const uint64_t max_values = 1024u * 1024u;
    const uint64_t cap = values < max_values ? values : max_values;
    uint16_t *buf = malloc((size_t)cap * sizeof(buf[0]));
    if (!buf) return -1;
    uint64_t done = 0;
    int rc = 0;
    while (done < values) {
        uint64_t n = values - done;
        if (n > cap) n = cap;
        for (uint64_t i = 0; i < n; i++) buf[i] = hsx_f32_to_f16(src[done + i]);
        if (hsx_write_full(fd, buf, (size_t)n * sizeof(buf[0])) != 0) {
            rc = -1;
            break;
        }
        done += n;
    }
    free(buf);
    return rc;
}

static void hsx_u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo) {
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)(v & 0xffffffffu);
}

static int hsx_set_socket_options(int fd, bool handshake) {
    int one = 1;
    int rc = 0;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one)) != 0) rc = -1;
    /* The handshake must not hold the single subscriber slot forever, and a
     * stuck receiver must not wedge the sender thread behind a full queue. */
    struct timeval tv = {
        .tv_sec = handshake ? 30 : 60,
        .tv_usec = 0,
    };
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) rc = -1;
#ifdef SO_NOSIGPIPE
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) != 0) rc = -1;
#endif
    return rc;
}

/* =========================================================================
 * HIDDEN_BATCH send.
 * ========================================================================= */

static int hsx_send_batch(ds4_hsexport *x, int fd, const hsx_batch *b) {
    const float *payload =
        (const float *)(const void *)(b->tokens + b->n_tokens);
    const uint64_t values =
        (uint64_t)x->n_layers * b->n_tokens * x->row_values;
    const uint64_t wire_payload =
        values * (uint64_t)(x->bits == 16 ? 2 : 4);
    const uint64_t bytes64 =
        (uint64_t)sizeof(ds4_hsexport_batch_fixed) +
        (uint64_t)x->n_layers * sizeof(uint32_t) +
        (uint64_t)b->n_tokens * sizeof(int32_t) +
        wire_payload;
    if (wire_payload > UINT32_MAX || bytes64 > UINT32_MAX) return -1;

    ds4_hsexport_batch_fixed wire;
    hsx_u64_to_halves(b->session_id, &wire.session_hi, &wire.session_lo);
    wire.pos0 = b->pos0;
    wire.n_tokens = b->n_tokens;
    wire.n_layers = (uint32_t)x->n_layers;
    wire.format = DS4_TAP_FORMAT_RAW_HC;
    wire.bits = (uint32_t)x->bits;
    wire.row_values = x->row_values;
    hsx_u64_to_halves(b->prefix_hash, &wire.prefix_hash_hi, &wire.prefix_hash_lo);
    wire.reserved = 0;

    ds4_hsexport_batch_fixed be = {
        htonl(wire.session_hi), htonl(wire.session_lo),
        htonl(wire.pos0), htonl(wire.n_tokens),
        htonl(wire.n_layers), htonl(wire.format),
        htonl(wire.bits), htonl(wire.row_values),
        htonl(wire.prefix_hash_hi), htonl(wire.prefix_hash_lo),
        htonl(wire.reserved),
    };

    if (hsx_write_frame_header(fd, DS4_HSEXPORT_MSG_HIDDEN_BATCH,
                               (uint32_t)bytes64) != 0) return -1;
    if (hsx_write_full(fd, &be, sizeof(be)) != 0) return -1;
    for (int i = 0; i < x->n_layers; i++) {
        uint32_t id = htonl((uint32_t)x->layers[i]);
        if (hsx_write_full(fd, &id, sizeof(id)) != 0) return -1;
    }
    for (uint32_t i = 0; i < b->n_tokens; i++) {
        uint32_t t = htonl((uint32_t)b->tokens[i]);
        if (hsx_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    return hsx_write_payload(fd, payload, values, (uint32_t)x->bits);
}

/* =========================================================================
 * HELLO receive and validation.
 * ========================================================================= */

static int hsx_recv_hello(ds4_hsexport *x, int fd, char *err, size_t errlen) {
    uint32_t h[3];
    int rc = hsx_read_full(fd, h, sizeof(h));
    if (rc <= 0) {
        if (errlen && rc < 0) snprintf(err, errlen, "failed to read HELLO: %s", strerror(errno));
        return rc;
    }
    if (ntohl(h[0]) != DS4_HSEXPORT_MAGIC) {
        if (errlen) snprintf(err, errlen, "bad frame magic 0x%08x", ntohl(h[0]));
        return -1;
    }
    const uint32_t type = ntohl(h[1]);
    const uint32_t bytes = ntohl(h[2]);
    const uint32_t max_bytes = (uint32_t)sizeof(ds4_hsexport_hello_fixed) +
        DS4_HSEXPORT_MAX_LAYERS * (uint32_t)sizeof(uint32_t) +
        DS4_HSEXPORT_MAX_MODEL_NAME;
    if (type != DS4_HSEXPORT_MSG_HELLO ||
        bytes < sizeof(ds4_hsexport_hello_fixed) || bytes > max_bytes) {
        if (errlen) snprintf(err, errlen, "invalid HELLO frame (type %u, %u bytes)", type, bytes);
        return -1;
    }

    ds4_hsexport_hello_fixed wire;
    rc = hsx_read_full(fd, &wire, sizeof(wire));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    ds4_hsexport_hello_fixed hello = {
        ntohl(wire.model_id), ntohl(wire.n_layers), ntohl(wire.format),
        ntohl(wire.mode), ntohl(wire.ctx_size), ntohl(wire.model_name_len),
    };
    const uint32_t remaining = bytes - (uint32_t)sizeof(wire);
    if (hello.n_layers > DS4_HSEXPORT_MAX_LAYERS ||
        (uint64_t)hello.n_layers * sizeof(uint32_t) + hello.model_name_len !=
            remaining) {
        if (errlen) snprintf(err, errlen, "invalid HELLO layout (%u layers, name %u, %u remaining)",
                             hello.n_layers, hello.model_name_len, remaining);
        return -1;
    }

    int req_layers[DS4_HSEXPORT_MAX_LAYERS] = {0};
    for (uint32_t i = 0; i < hello.n_layers; i++) {
        uint32_t id = 0;
        rc = hsx_read_full(fd, &id, sizeof(id));
        if (rc <= 0) return rc == 0 ? 0 : -1;
        req_layers[i] = (int)ntohl(id);
    }
    char model_name[DS4_HSEXPORT_MAX_MODEL_NAME + 1u] = {0};
    if (hello.model_name_len) {
        rc = hsx_read_full(fd, model_name, hello.model_name_len);
        if (rc <= 0) return rc == 0 ? 0 : -1;
        if (memchr(model_name, '\0', hello.model_name_len) != NULL) {
            if (errlen) snprintf(err, errlen, "HELLO model name contains NUL bytes");
            return -1;
        }
        model_name[hello.model_name_len] = '\0';
    }

    if (!x->config_set) {
        if (errlen) snprintf(err, errlen, "publisher has no active tap configuration");
        return -1;
    }
    if (hello.mode != DS4_HSEXPORT_MODE_STREAM) {
        if (errlen) snprintf(err, errlen, "unsupported HELLO mode %u (Phase 1 is STREAM only)", hello.mode);
        return -1;
    }
    /* Wildcard handshake (train sinks): model_id 0 asks for the publisher's
     * active configuration, which is echoed back in the HELLO ack. */
    const bool wildcard = hello.model_id == 0;
    if (wildcard && (hello.n_layers != 0 || hello.model_name_len != 0)) {
        if (errlen) snprintf(err, errlen,
                             "partial HELLO wildcard: model_id 0 requires no layer set and no model name");
        return -1;
    }
    if (!wildcard) {
        if ((int)hello.model_id != ds4_engine_model_id(x->engine)) {
            if (errlen) snprintf(err, errlen, "model id mismatch (subscriber %u, publisher %d)",
                                 hello.model_id, ds4_engine_model_id(x->engine));
            return -1;
        }
        const char *want_name = ds4_engine_model_name(x->engine);
        if (!want_name) want_name = "unknown";
        if (strcmp(model_name, want_name) != 0) {
            if (errlen) snprintf(err, errlen, "model name mismatch (subscriber '%s', publisher '%s')",
                                 model_name, want_name);
            return -1;
        }
        if ((int)hello.n_layers != x->n_layers ||
            memcmp(req_layers, x->layers,
                   (size_t)x->n_layers * sizeof(req_layers[0])) != 0) {
            if (errlen) snprintf(err, errlen, "requested layer set does not match the active tap configuration");
            return -1;
        }
    }
    if (hello.format != DS4_TAP_FORMAT_RAW_HC) {
        if (errlen) snprintf(err, errlen, "tap format mismatch (subscriber %u, publisher %d)",
                             hello.format, DS4_TAP_FORMAT_RAW_HC);
        return -1;
    }
    if (hello.ctx_size != 0) {
        for (int i = 0; i < HSX_MAX_SESSIONS; i++) {
            if (!x->sessions[i].used) continue;
            if ((int)hello.ctx_size != ds4_session_ctx(x->sessions[i].session)) {
                if (errlen) snprintf(err, errlen, "ctx size mismatch (subscriber %u)", hello.ctx_size);
                return -1;
            }
            break;
        }
    }
    return 1;
}

/* Answers a successful HELLO with the active configuration so wildcard
 * subscribers learn the model identity, layer set, and ctx size. */
static int hsx_send_hello_ack(ds4_hsexport *x, int fd) {
    const char *name = ds4_engine_model_name(x->engine);
    if (!name) name = "unknown";
    const size_t name_len = strlen(name);
    if (name_len > DS4_HSEXPORT_MAX_MODEL_NAME) return -1;

    pthread_mutex_lock(&x->mu);
    const int n_layers = x->n_layers;
    int layers[DS4_HSEXPORT_MAX_LAYERS];
    memcpy(layers, x->layers, (size_t)n_layers * sizeof(layers[0]));
    uint32_t ctx = 0;
    for (int i = 0; i < HSX_MAX_SESSIONS; i++) {
        if (x->sessions[i].used) {
            ctx = (uint32_t)ds4_session_ctx(x->sessions[i].session);
            break;
        }
    }
    pthread_mutex_unlock(&x->mu);

    ds4_hsexport_hello_fixed wire = {
        htonl((uint32_t)ds4_engine_model_id(x->engine)),
        htonl((uint32_t)n_layers),
        htonl(DS4_TAP_FORMAT_RAW_HC),
        htonl(DS4_HSEXPORT_MODE_STREAM),
        htonl(ctx),
        htonl((uint32_t)name_len),
    };
    const uint32_t bytes = (uint32_t)sizeof(wire) +
        (uint32_t)n_layers * (uint32_t)sizeof(uint32_t) + (uint32_t)name_len;
    if (hsx_write_frame_header(fd, DS4_HSEXPORT_MSG_HELLO, bytes) != 0) return -1;
    if (hsx_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (int i = 0; i < n_layers; i++) {
        uint32_t id = htonl((uint32_t)layers[i]);
        if (hsx_write_full(fd, &id, sizeof(id)) != 0) return -1;
    }
    return hsx_write_full(fd, name, name_len);
}

/* =========================================================================
 * Connection / sender thread.
 * ========================================================================= */

static void hsx_queue_clear(ds4_hsexport *x) {
    hsx_batch *b = x->head;
    while (b) {
        hsx_batch *next = b->next;
        free(b);
        b = next;
    }
    x->head = NULL;
    x->tail = NULL;
    x->queued_bytes = 0;
}

static void hsx_serve_subscriber(ds4_hsexport *x, int fd) {
    char hello_err[256] = {0};
    int rc = hsx_recv_hello(x, fd, hello_err, sizeof(hello_err));
    if (rc <= 0) {
        if (rc < 0) {
            fprintf(stderr, "ds4: hidden export subscriber rejected: %s\n",
                    hello_err[0] ? hello_err : "handshake failed");
            hsx_send_error(fd, hello_err[0] ? hello_err : "handshake failed");
        }
        return;
    }
    pthread_mutex_lock(&x->mu);
    x->sub_fd = fd;
    x->conn_gen++; /* attached sessions re-anchor on their next emit */
    pthread_mutex_unlock(&x->mu);
    /* Set sub_fd before the ack so a subscriber that has received the ack
     * never races an emit into the no-subscriber path. */
    if (hsx_send_hello_ack(x, fd) != 0) {
        fprintf(stderr, "ds4: hidden export subscriber dropped: handshake ack failed\n");
        pthread_mutex_lock(&x->mu);
        x->sub_fd = -1;
        pthread_mutex_unlock(&x->mu);
        return;
    }
    fprintf(stderr, "ds4: hidden export subscriber connected\n");

    bool ok = true;
    for (;;) {
        pthread_mutex_lock(&x->mu);
        while (ok && !x->stopping && !x->head && !x->desync_msg[0]) {
            pthread_cond_wait(&x->cv_items, &x->mu);
        }
        if (x->stopping) {
            pthread_mutex_unlock(&x->mu);
            break;
        }
        /* Drain every queued batch before the poison frame: the batches were
         * accepted while the stream was still consistent, so the subscriber
         * should see them, then the ERROR, then EOF. */
        if (x->head) {
            hsx_batch *b = x->head;
            x->head = b->next;
            if (!x->head) x->tail = NULL;
            x->queued_bytes -= b->payload_f32_bytes;
            pthread_cond_broadcast(&x->cv_space);
            pthread_mutex_unlock(&x->mu);

            ok = hsx_send_batch(x, fd, b) == 0;
            free(b);
            if (!ok) {
                fprintf(stderr, "ds4: hidden export subscriber send failed: %s\n",
                        strerror(errno));
            }
            continue;
        }
        if (x->desync_msg[0]) {
            char msg[HSX_DESYNC_MSG_LEN];
            snprintf(msg, sizeof(msg), "%s", x->desync_msg);
            pthread_mutex_unlock(&x->mu);
            fprintf(stderr, "ds4: hidden export desync: %s\n", msg);
            hsx_send_error(fd, msg);
            break;
        }
        pthread_mutex_unlock(&x->mu);
    }

    pthread_mutex_lock(&x->mu);
    x->sub_fd = -1;
    x->desync_msg[0] = '\0';
    hsx_queue_clear(x);
    pthread_cond_broadcast(&x->cv_space);
    pthread_mutex_unlock(&x->mu);
    fprintf(stderr, "ds4: hidden export subscriber disconnected\n");
}

static void *hsx_conn_thread(void *arg) {
    ds4_hsexport *x = arg;
    for (;;) {
        pthread_mutex_lock(&x->mu);
        const bool stopping = x->stopping;
        pthread_mutex_unlock(&x->mu);
        if (stopping) break;

        int fd = accept(x->listen_fd, NULL, NULL);
        if (fd < 0) {
            pthread_mutex_lock(&x->mu);
            const bool stop = x->stopping;
            pthread_mutex_unlock(&x->mu);
            if (stop) break;
            if (errno == EINTR) continue;
            fprintf(stderr, "ds4: hidden export accept failed: %s\n", strerror(errno));
            break;
        }
        pthread_mutex_lock(&x->mu);
        const bool stop = x->stopping;
        pthread_mutex_unlock(&x->mu);
        if (stop) {
            /* Woken by the stop-time dummy connection, not a subscriber. */
            close(fd);
            break;
        }
        (void)hsx_set_socket_options(fd, true);
        /* Phase 1 serves one subscriber at a time; since the thread is busy
         * with the current connection, any extra connector simply waits in
         * the listen backlog until the slot frees. */
        hsx_serve_subscriber(x, fd);
        close(fd);
    }
    return NULL;
}

/* =========================================================================
 * Public API.
 * ========================================================================= */

ds4_hsexport *ds4_hsexport_start(ds4_engine *engine,
                                 const ds4_hsexport_options *opt,
                                 char *err,
                                 size_t errlen) {
    if (!engine || !opt || opt->port <= 0 || opt->port > 65535) {
        if (errlen) snprintf(err, errlen, "hidden export needs a valid listen port");
        return NULL;
    }
    if (opt->bits != 0 && opt->bits != 32 && opt->bits != 16) {
        if (errlen) snprintf(err, errlen, "hidden export bits must be 32 or 16");
        return NULL;
    }

    ds4_hsexport *x = calloc(1, sizeof(*x));
    if (!x) {
        if (errlen) snprintf(err, errlen, "out of memory");
        return NULL;
    }
    x->engine = engine;
    x->listen_fd = -1;
    x->sub_fd = -1;
    x->bits = opt->bits ? opt->bits : 32;
    x->queue_cap = opt->queue_bytes ? opt->queue_bytes : HSX_DEFAULT_QUEUE_BYTES;
    x->row_values = (uint32_t)ds4_engine_hidden_f32_values(engine);
    pthread_mutex_init(&x->mu, NULL);
    pthread_cond_init(&x->cv_items, NULL);
    pthread_cond_init(&x->cv_space, NULL);

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", opt->port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const char *host = (opt->host && opt->host[0]) ? opt->host : "127.0.0.1";
    struct addrinfo *res = NULL;
    const int gai = getaddrinfo(host, port_str, &hints, &res);
    if (gai != 0) {
        if (errlen) snprintf(err, errlen, "hidden export address %s:%s: %s",
                             host, port_str, gai_strerror(gai));
        goto fail;
    }
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        x->listen_fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (x->listen_fd < 0) continue;
        int one = 1;
        (void)setsockopt(x->listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(x->listen_fd, ai->ai_addr, ai->ai_addrlen) == 0 &&
            listen(x->listen_fd, 4) == 0) {
            break;
        }
        close(x->listen_fd);
        x->listen_fd = -1;
    }
    freeaddrinfo(res);
    if (x->listen_fd < 0) {
        if (errlen) snprintf(err, errlen, "hidden export cannot bind %s:%s: %s",
                             host, port_str, strerror(errno));
        goto fail;
    }
    x->bound_len = (socklen_t)sizeof(x->bound);
    if (getsockname(x->listen_fd, (struct sockaddr *)&x->bound, &x->bound_len) != 0) {
        x->bound_len = 0;
    }
    if (pthread_create(&x->thread, NULL, hsx_conn_thread, x) != 0) {
        if (errlen) snprintf(err, errlen, "hidden export thread creation failed");
        close(x->listen_fd);
        x->listen_fd = -1;
        goto fail;
    }
    fprintf(stderr, "ds4: hidden export listening on %s:%d (f%d payload)\n",
            host, opt->port, x->bits);
    return x;

fail:
    pthread_mutex_destroy(&x->mu);
    pthread_cond_destroy(&x->cv_items);
    pthread_cond_destroy(&x->cv_space);
    free(x);
    return NULL;
}

void ds4_hsexport_stop(ds4_hsexport *x) {
    if (!x) return;
    pthread_mutex_lock(&x->mu);
    x->stopping = true;
    pthread_cond_broadcast(&x->cv_items);
    pthread_cond_broadcast(&x->cv_space);
    const int sub_fd = x->sub_fd;
    pthread_mutex_unlock(&x->mu);
    if (sub_fd >= 0) (void)shutdown(sub_fd, SHUT_RDWR);
    if (x->listen_fd >= 0) (void)shutdown(x->listen_fd, SHUT_RDWR);
    /* shutdown() does not wake a blocked accept() on a listening socket on
     * every platform (notably macOS), so poke the listener with a dummy
     * connection; the thread re-checks `stopping` right after accept. */
    if (x->bound_len > 0) {
        struct sockaddr_storage sa = x->bound;
        if (sa.ss_family == AF_INET) {
            struct sockaddr_in *a = (struct sockaddr_in *)(void *)&sa;
            if (a->sin_addr.s_addr == htonl(INADDR_ANY)) {
                a->sin_addr.s_addr = htonl(0x7f000001u);
            }
        } else if (sa.ss_family == AF_INET6) {
            struct sockaddr_in6 *a = (struct sockaddr_in6 *)(void *)&sa;
            if (IN6_IS_ADDR_UNSPECIFIED(&a->sin6_addr)) {
                a->sin6_addr = in6addr_loopback;
            }
        }
        const int wake = socket(sa.ss_family, SOCK_STREAM, 0);
        if (wake >= 0) {
            (void)connect(wake, (struct sockaddr *)(void *)&sa, x->bound_len);
            close(wake);
        }
    }
    pthread_join(x->thread, NULL);
    if (x->listen_fd >= 0) close(x->listen_fd);
    hsx_queue_clear(x);
    pthread_mutex_destroy(&x->mu);
    pthread_cond_destroy(&x->cv_items);
    pthread_cond_destroy(&x->cv_space);
    free(x);
}

static hsx_session *hsx_session_find(ds4_hsexport *x, ds4_session *session) {
    for (int i = 0; i < HSX_MAX_SESSIONS; i++) {
        if (x->sessions[i].used && x->sessions[i].session == session) {
            return &x->sessions[i];
        }
    }
    return NULL;
}

int ds4_hsexport_attach(ds4_hsexport *x,
                        ds4_session *session,
                        uint64_t session_id,
                        const int *layers,
                        int n_layers,
                        char *err,
                        size_t errlen) {
    if (!x || !session || !layers ||
        n_layers <= 0 || n_layers > DS4_HSEXPORT_MAX_LAYERS) {
        if (errlen) snprintf(err, errlen, "invalid hidden export attach arguments");
        return 1;
    }
    /* The engine validates layer ids, duplicates, and rejects GLM/TP/
     * distributed-coordinator sessions. */
    if (ds4_session_set_hidden_taps(session, layers, n_layers,
                                    DS4_TAP_FORMAT_RAW_HC) != 0) {
        if (errlen) snprintf(err, errlen, "hidden export tap configuration failed");
        return 1;
    }

    pthread_mutex_lock(&x->mu);
    if (x->config_set) {
        if (x->n_layers != n_layers ||
            memcmp(x->layers, layers,
                   (size_t)n_layers * sizeof(layers[0])) != 0) {
            pthread_mutex_unlock(&x->mu);
            if (errlen) snprintf(err, errlen,
                                 "hidden export supports one tap layer set per publisher");
            return 1;
        }
    } else {
        memcpy(x->layers, layers, (size_t)n_layers * sizeof(layers[0]));
        x->n_layers = n_layers;
        x->config_set = true;
    }
    hsx_session *slot = hsx_session_find(x, session);
    if (!slot) {
        /* Re-attaching a session id (e.g. after the owning session was
         * recreated) replaces the stale registration. */
        for (int i = 0; i < HSX_MAX_SESSIONS; i++) {
            if (x->sessions[i].used && x->sessions[i].id == session_id) {
                x->sessions[i].used = false;
            }
        }
        for (int i = 0; i < HSX_MAX_SESSIONS; i++) {
            if (!x->sessions[i].used) {
                slot = &x->sessions[i];
                break;
            }
        }
    }
    if (!slot) {
        pthread_mutex_unlock(&x->mu);
        if (errlen) snprintf(err, errlen, "hidden export session table is full");
        return 1;
    }
    slot->session = session;
    slot->id = session_id;
    slot->used = true;
    slot->conn_gen = 0; /* forces a frontier re-anchor on the next emit */
    slot->next_pos = 0;
    slot->hash = DS4_DIST_TOKEN_HASH_INIT;
    pthread_mutex_unlock(&x->mu);
    return 0;
}

void ds4_hsexport_detach(ds4_hsexport *x, ds4_session *session) {
    if (!x || !session) return;
    pthread_mutex_lock(&x->mu);
    hsx_session *slot = hsx_session_find(x, session);
    if (slot) slot->used = false;
    pthread_mutex_unlock(&x->mu);
}

uint64_t ds4_hsexport_queued_bytes(ds4_hsexport *x) {
    if (!x) return 0;
    pthread_mutex_lock(&x->mu);
    const uint64_t bytes = x->queued_bytes;
    pthread_mutex_unlock(&x->mu);
    return bytes;
}

int ds4_hsexport_emit(ds4_hsexport *x,
                      ds4_session *session,
                      uint32_t pos0,
                      uint32_t n_tokens) {
    if (!x || !session || n_tokens == 0) return 1;

    pthread_mutex_lock(&x->mu);
    hsx_session *sess = hsx_session_find(x, session);
    if (!sess) {
        pthread_mutex_unlock(&x->mu);
        return 1;
    }
    if (x->sub_fd < 0) {
        /* No subscriber: advance the frontier without queueing. The next
         * connection re-anchors anyway, so the hash stays stale here. */
        sess->next_pos = pos0 + n_tokens;
        pthread_mutex_unlock(&x->mu);
        return 0;
    }
    if (sess->conn_gen != x->conn_gen) {
        /* First emit after a (re)connect: adopt the live frontier as the new
         * stream anchor. The subscriber adopts this batch's prefix_hash. */
        sess->next_pos = pos0;
        sess->hash = DS4_DIST_TOKEN_HASH_INIT;
        sess->conn_gen = x->conn_gen;
        const ds4_tokens *tl = ds4_session_tokens(session);
        if (tl && tl->v && tl->len >= (int)pos0) {
            sess->hash = ds4_dist_token_hash_update_span(sess->hash, tl->v, pos0);
        }
    }
    const uint32_t expect = sess->next_pos;
    const uint64_t base_hash = sess->hash;
    pthread_mutex_unlock(&x->mu);

    if (pos0 != expect) {
        /* Position gap: silent recovery would corrupt a training stream, so
         * poison the connection (ERROR + close) per the protocol. */
        pthread_mutex_lock(&x->mu);
        if (!x->desync_msg[0]) {
            snprintf(x->desync_msg, sizeof(x->desync_msg),
                     "hidden export position gap on session %llu: expected %u, got %u",
                     (unsigned long long)sess->id, expect, pos0);
            pthread_cond_broadcast(&x->cv_items);
        }
        sess->next_pos = pos0 + n_tokens;
        pthread_mutex_unlock(&x->mu);
        return 0;
    }

    /* Collect token ids and extend the hash over this batch. */
    const ds4_tokens *tl = ds4_session_tokens(session);
    if (!tl || !tl->v || tl->len < (int)(pos0 + n_tokens)) return 1;
    const uint64_t hash =
        ds4_dist_token_hash_update_span(base_hash, tl->v + pos0, n_tokens);

    const uint64_t payload_values =
        (uint64_t)x->n_layers * n_tokens * x->row_values;
    const uint64_t payload_bytes = payload_values * sizeof(float);
    if (payload_values > UINT32_MAX ||
        payload_bytes > (uint64_t)SIZE_MAX - sizeof(hsx_batch) -
            (uint64_t)n_tokens * sizeof(int32_t)) {
        return 1;
    }
    hsx_batch *b = malloc(sizeof(*b) +
                          (size_t)n_tokens * sizeof(b->tokens[0]) +
                          (size_t)payload_bytes);
    if (!b) return 1;
    b->next = NULL;
    b->session_id = sess->id;
    b->pos0 = pos0;
    b->n_tokens = n_tokens;
    b->prefix_hash = hash;
    b->payload_f32_bytes = (size_t)payload_bytes;
    memcpy(b->tokens, tl->v + pos0, (size_t)n_tokens * sizeof(b->tokens[0]));

    float *dst = (float *)(void *)(b->tokens + n_tokens);
    for (int l = 0; l < x->n_layers; l++) {
        const int rows = ds4_session_read_tap(session,
                                              x->layers[l],
                                              dst + (uint64_t)l * n_tokens *
                                                    x->row_values,
                                              (int)n_tokens);
        if (rows != (int)n_tokens) {
            /* The taps do not cover this span (e.g. an eval path without tap
             * readback): treat it like a position gap. */
            free(b);
            pthread_mutex_lock(&x->mu);
            if (!x->desync_msg[0]) {
                snprintf(x->desync_msg, sizeof(x->desync_msg),
                         "hidden export tap row mismatch on session %llu layer %d: expected %u, got %d",
                         (unsigned long long)sess->id, x->layers[l],
                         (int)n_tokens, rows);
                pthread_cond_broadcast(&x->cv_items);
            }
            sess->next_pos = pos0 + n_tokens;
            pthread_mutex_unlock(&x->mu);
            return 0;
        }
    }

    /* Bounded queue: block while a subscriber is connected and the budget is
     * exhausted. An empty queue always accepts one batch, even oversized, so
     * a single large prefill chunk can never deadlock. */
    pthread_mutex_lock(&x->mu);
    while (x->sub_fd >= 0 && !x->stopping && x->head &&
           x->queued_bytes + b->payload_f32_bytes > x->queue_cap) {
        pthread_cond_wait(&x->cv_space, &x->mu);
    }
    if (x->sub_fd < 0 || x->stopping) {
        sess->next_pos = pos0 + n_tokens;
        pthread_mutex_unlock(&x->mu);
        free(b);
        return 0;
    }
    if (x->tail) x->tail->next = b;
    else x->head = b;
    x->tail = b;
    x->queued_bytes += b->payload_f32_bytes;
    sess->next_pos = pos0 + n_tokens;
    sess->hash = hash;
    pthread_cond_signal(&x->cv_items);
    pthread_mutex_unlock(&x->mu);
    return 0;
}

/* Parses a comma-separated layer list ("12,24,36") into `layers`. Returns
 * the number of layers (>= 1) on success, -1 on malformed input or overflow. */
int ds4_hsexport_parse_layers(const char *csv, int *layers, int max_layers) {
    if (!csv || !layers || max_layers <= 0) return -1;
    int n = 0;
    const char *p = csv;
    while (*p) {
        char *end = NULL;
        const long v = strtol(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX || n >= max_layers) return -1;
        if (*end != '\0' && *end != ',') return -1;
        layers[n++] = (int)v;
        p = end + (*end == ',');
    }
    return n > 0 ? n : -1;
}

/* =========================================================================
 * HIDDEN_BATCH parse (consumer side).
 * ========================================================================= */

void ds4_hsexport_batch_view_free(ds4_hsexport_batch_view *view) {
    if (!view) return;
    free(view->layer_ids);
    free(view->token_ids);
    view->layer_ids = NULL;
    view->token_ids = NULL;
}

int ds4_hsexport_parse_batch(const void *payload,
                             size_t bytes,
                             ds4_hsexport_batch_view *view) {
    if (!payload || !view) return -1;
    memset(view, 0, sizeof(*view));
    if (bytes < sizeof(ds4_hsexport_batch_fixed)) return -1;
    const unsigned char *p = payload;

    uint32_t f[11];
    memcpy(f, p, sizeof(f));
    for (size_t i = 0; i < 11; i++) f[i] = ntohl(f[i]);
    view->session_id = ((uint64_t)f[0] << 32) | f[1];
    view->pos0 = f[2];
    view->n_tokens = f[3];
    view->n_layers = f[4];
    view->format = f[5];
    view->bits = f[6];
    view->row_values = f[7];
    view->prefix_hash = ((uint64_t)f[8] << 32) | f[9];
    /* f[10] is reserved. */

    if (view->n_layers < 1 || view->n_layers > DS4_HSEXPORT_MAX_LAYERS) return -1;
    if (view->bits != 16 && view->bits != 32) return -1;
    if (view->n_tokens < 1 || view->row_values == 0) return -1;

    uint64_t expect = view->n_layers;
    if (expect > UINT64_MAX / view->n_tokens) return -1;
    expect *= view->n_tokens;
    if (expect > UINT64_MAX / view->row_values) return -1;
    expect *= view->row_values;
    if (expect > UINT64_MAX / 4u) return -1;
    expect *= view->bits / 8u;

    size_t off = sizeof(ds4_hsexport_batch_fixed);
    const uint64_t ids_bytes =
        (uint64_t)view->n_layers * 4u + (uint64_t)view->n_tokens * 4u;
    if ((uint64_t)bytes < (uint64_t)off + ids_bytes) return -1;
    if ((uint64_t)bytes - (uint64_t)off - ids_bytes != expect) return -1;

    view->layer_ids = malloc((size_t)view->n_layers * sizeof(view->layer_ids[0]));
    view->token_ids = malloc((size_t)view->n_tokens * sizeof(view->token_ids[0]));
    if (!view->layer_ids || !view->token_ids) {
        ds4_hsexport_batch_view_free(view);
        return -1;
    }
    for (uint32_t i = 0; i < view->n_layers; i++) {
        uint32_t v;
        memcpy(&v, p + off + (size_t)i * 4u, sizeof(v));
        view->layer_ids[i] = ntohl(v);
    }
    off += (size_t)view->n_layers * 4u;
    for (uint32_t i = 0; i < view->n_tokens; i++) {
        uint32_t v;
        memcpy(&v, p + off + (size_t)i * 4u, sizeof(v));
        view->token_ids[i] = (int32_t)ntohl(v);
    }
    off += (size_t)view->n_tokens * 4u;
    view->data = p + off;
    view->data_bytes = bytes - off;
    return 0;
}

/* =========================================================================
 * Train sink (HIDDEN_EXPORT.md 3.4).
 * ========================================================================= */

#define HSX_SINK_MAX_FRAME (1u << 31)
#define HSX_SINK_MAX_SESSIONS 16

typedef struct {
    uint64_t session_id;
    uint32_t next_pos;
    uint64_t hash;
    bool anchored;
} hsx_sink_session;

static void hsx_sink_fail(char *err, size_t errlen, const char *fmt, ...) {
    if (!errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

/* Like hsx_read_full but polls the stop flag: the socket has a 1 s receive
 * timeout, so a timeout simply retries. Partial reads are kept, so frames
 * never desync. Returns 1 on success, 0 on orderly EOF, -2 on error,
 * -3 when stop was requested. */
static int hsx_sink_read_full(int fd, void *buf, size_t len,
                              const volatile sig_atomic_t *stop) {
    unsigned char *p = buf;
    while (len > 0) {
        if (stop && *stop) return -3;
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return -2;
        }
        if (n == 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

/* Reads one frame; return codes mirror hsx_sink_read_full. On success
 * *payload is malloc'd (NULL when bytes == 0). */
static int hsx_sink_recv_frame(int fd, uint32_t *type, unsigned char **payload,
                               uint32_t *bytes,
                               const volatile sig_atomic_t *stop) {
    uint32_t h[3];
    int rc = hsx_sink_read_full(fd, h, sizeof(h), stop);
    if (rc != 1) return rc;
    if (ntohl(h[0]) != DS4_HSEXPORT_MAGIC) return -2;
    *type = ntohl(h[1]);
    *bytes = ntohl(h[2]);
    if (*bytes > HSX_SINK_MAX_FRAME) return -2;
    *payload = NULL;
    if (*bytes > 0) {
        *payload = malloc(*bytes);
        if (!*payload) return -2;
        rc = hsx_sink_read_full(fd, *payload, *bytes, stop);
        if (rc != 1) {
            free(*payload);
            *payload = NULL;
            return rc;
        }
    }
    return 1;
}

static int hsx_sink_connect(const char *host, int port, char *err, size_t errlen) {
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = NULL;
    const int gai = getaddrinfo(host, port_str, &hints, &res);
    if (gai != 0) {
        hsx_sink_fail(err, errlen, "cannot resolve %s:%s: %s",
                      host, port_str, gai_strerror(gai));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        hsx_sink_fail(err, errlen, "cannot connect to %s:%s: %s",
                      host, port_str, strerror(errno));
    }
    return fd;
}

static int hsx_sink_send_hello(int fd, const ds4_hsexport_sink_options *opt) {
    const bool wildcard = opt->model_id == 0;
    const char *name = wildcard ? "" : opt->model_name;
    const size_t name_len = strlen(name);
    const int n_layers = wildcard ? 0 : opt->n_layers;
    ds4_hsexport_hello_fixed wire = {
        htonl(opt->model_id),
        htonl((uint32_t)n_layers),
        htonl(DS4_TAP_FORMAT_RAW_HC),
        htonl(DS4_HSEXPORT_MODE_STREAM),
        htonl(opt->ctx_size),
        htonl((uint32_t)name_len),
    };
    const uint32_t bytes = (uint32_t)sizeof(wire) +
        (uint32_t)n_layers * (uint32_t)sizeof(uint32_t) + (uint32_t)name_len;
    if (hsx_write_frame_header(fd, DS4_HSEXPORT_MSG_HELLO, bytes) != 0) return -1;
    if (hsx_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (int i = 0; i < n_layers; i++) {
        uint32_t id = htonl((uint32_t)opt->layers[i]);
        if (hsx_write_full(fd, &id, sizeof(id)) != 0) return -1;
    }
    return hsx_write_full(fd, name, name_len);
}

/* Parses the publisher's HELLO ack payload into host-order fields. */
static int hsx_sink_parse_hello(const unsigned char *p, size_t bytes,
                                ds4_hsexport_hello_fixed *hello,
                                int *layers, int max_layers,
                                char *name, size_t name_cap) {
    if (!p || bytes < sizeof(ds4_hsexport_hello_fixed)) return -1;
    uint32_t f[6];
    memcpy(f, p, sizeof(f));
    for (size_t i = 0; i < 6; i++) f[i] = ntohl(f[i]);
    hello->model_id = f[0];
    hello->n_layers = f[1];
    hello->format = f[2];
    hello->mode = f[3];
    hello->ctx_size = f[4];
    hello->model_name_len = f[5];
    size_t off = sizeof(ds4_hsexport_hello_fixed);
    if (hello->n_layers < 1 || hello->n_layers > DS4_HSEXPORT_MAX_LAYERS ||
        (int)hello->n_layers > max_layers) return -1;
    if (hello->model_name_len > DS4_HSEXPORT_MAX_MODEL_NAME ||
        (size_t)hello->model_name_len >= name_cap) return -1;
    if ((uint64_t)hello->n_layers * 4u + hello->model_name_len !=
        (uint64_t)(bytes - off)) return -1;
    for (uint32_t i = 0; i < hello->n_layers; i++) {
        uint32_t v;
        memcpy(&v, p + off + (size_t)i * 4u, sizeof(v));
        layers[i] = (int)ntohl(v);
    }
    off += (size_t)hello->n_layers * 4u;
    memcpy(name, p + off, hello->model_name_len);
    name[hello->model_name_len] = '\0';
    return 0;
}

static int hsx_dump_write_u32(FILE *fp, uint32_t v) {
    v = htonl(v);
    return fwrite(&v, sizeof(v), 1, fp) == 1 ? 0 : -1;
}

static int hsx_dump_write_header(FILE *fp, const ds4_hsexport_hello_fixed *hello,
                                 const int *layers, const char *name,
                                 const ds4_hsexport_batch_view *first) {
    const uint32_t hdr[DS4_HSEXPORT_DUMP_HEADER_U32] = {
        DS4_HSEXPORT_DUMP_MAGIC,
        DS4_HSEXPORT_DUMP_VERSION,
        hello->model_id,
        hello->n_layers,
        hello->format,
        first->bits,
        first->row_values,
        hello->ctx_size,
        hello->model_name_len,
        0, 0, 0, /* reserved */
    };
    for (size_t i = 0; i < DS4_HSEXPORT_DUMP_HEADER_U32; i++) {
        if (hsx_dump_write_u32(fp, hdr[i]) != 0) return -1;
    }
    for (uint32_t i = 0; i < hello->n_layers; i++) {
        if (hsx_dump_write_u32(fp, (uint32_t)layers[i]) != 0) return -1;
    }
    return fwrite(name, 1, hello->model_name_len, fp) == hello->model_name_len ?
        0 : -1;
}

int ds4_hsexport_sink_run(const ds4_hsexport_sink_options *opt,
                          const volatile sig_atomic_t *stop,
                          char *err, size_t errlen) {
    if (!opt || !opt->host || !opt->host[0] ||
        opt->port <= 0 || opt->port > 65535 ||
        !opt->dump_path || !opt->dump_path[0]) {
        hsx_sink_fail(err, errlen, "train sink needs --hidden-source HOST PORT and --dump FILE");
        return 1;
    }
    const bool wildcard = opt->model_id == 0;
    if (wildcard && (opt->model_name || opt->layers || opt->n_layers > 0)) {
        hsx_sink_fail(err, errlen,
                      "partial sink handshake: model id 0 takes no model name or layer set");
        return 1;
    }
    if (!wildcard &&
        (!opt->model_name || !opt->layers ||
         opt->n_layers <= 0 || opt->n_layers > DS4_HSEXPORT_MAX_LAYERS)) {
        hsx_sink_fail(err, errlen,
                      "explicit sink handshake needs a model id, model name, and a layer set");
        return 1;
    }

    int fd = hsx_sink_connect(opt->host, opt->port, err, errlen);
    if (fd < 0) return 1;
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    /* 1 s receive timeout so the stop flag is polled during idle streams. */
    struct timeval rcv = { .tv_sec = 1, .tv_usec = 0 };
    struct timeval snd = { .tv_sec = 30, .tv_usec = 0 };
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rcv, sizeof(rcv));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &snd, sizeof(snd));

    int ret = 1;
    FILE *fp = NULL;
    unsigned char *payload = NULL;
    ds4_hsexport_hello_fixed hello;
    memset(&hello, 0, sizeof(hello));
    int layers[DS4_HSEXPORT_MAX_LAYERS] = {0};
    char model_name[DS4_HSEXPORT_MAX_MODEL_NAME + 1u] = {0};
    uint64_t batches = 0, tokens = 0, bytes_written = 0;
    hsx_sink_session sessions[HSX_SINK_MAX_SESSIONS];
    memset(sessions, 0, sizeof(sessions));
    uint32_t first_bits = 0, first_rows = 0;
    bool header_written = false;

    if (hsx_sink_send_hello(fd, opt) != 0) {
        hsx_sink_fail(err, errlen, "failed to send HELLO: %s", strerror(errno));
        goto out;
    }

    /* The first frame must be the publisher's HELLO ack (or an ERROR). */
    for (;;) {
        uint32_t type = 0, bytes = 0;
        const int rc = hsx_sink_recv_frame(fd, &type, &payload, &bytes, stop);
        if (rc == -3) {
            hsx_sink_fail(err, errlen, "interrupted during handshake");
            goto out;
        }
        if (rc == 0) {
            hsx_sink_fail(err, errlen, "publisher closed the connection during handshake");
            goto out;
        }
        if (rc != 1) {
            hsx_sink_fail(err, errlen, "handshake read failed: %s", strerror(errno));
            goto out;
        }
        if (type == DS4_HSEXPORT_MSG_ERROR) {
            hsx_sink_fail(err, errlen, "publisher rejected HELLO: %.*s",
                          (int)bytes, (char *)payload);
            goto out;
        }
        if (type != DS4_HSEXPORT_MSG_HELLO) {
            hsx_sink_fail(err, errlen,
                          "protocol error: expected HELLO ack, got frame type %u", type);
            goto out;
        }
        if (hsx_sink_parse_hello(payload, bytes, &hello, layers,
                                 DS4_HSEXPORT_MAX_LAYERS,
                                 model_name, sizeof(model_name)) != 0) {
            hsx_sink_fail(err, errlen, "malformed HELLO ack");
            goto out;
        }
        free(payload);
        payload = NULL;
        break;
    }

    fprintf(stderr,
            "ds4: train sink connected to %s:%d: model='%s' id=%u layers=%u format=%u ctx=%u\n",
            opt->host, opt->port, model_name, hello.model_id,
            hello.n_layers, hello.format, hello.ctx_size);

    fp = fopen(opt->dump_path, "wb");
    if (!fp) {
        hsx_sink_fail(err, errlen, "cannot open dump file %s: %s",
                      opt->dump_path, strerror(errno));
        goto out;
    }

    for (;;) {
        uint32_t type = 0, bytes = 0;
        const int rc = hsx_sink_recv_frame(fd, &type, &payload, &bytes, stop);
        if (rc == -3 || rc == 0) break; /* stop requested or publisher closed */
        if (rc != 1) {
            hsx_sink_fail(err, errlen, "stream read failed: %s", strerror(errno));
            goto out;
        }
        if (type == DS4_HSEXPORT_MSG_ERROR) {
            hsx_sink_fail(err, errlen, "publisher reported desync: %.*s",
                          (int)bytes, (char *)payload);
            goto out;
        }
        if (type != DS4_HSEXPORT_MSG_HIDDEN_BATCH) {
            hsx_sink_fail(err, errlen, "protocol error: unexpected frame type %u", type);
            goto out;
        }
        ds4_hsexport_batch_view view;
        if (ds4_hsexport_parse_batch(payload, bytes, &view) != 0) {
            hsx_sink_fail(err, errlen, "malformed HIDDEN_BATCH frame");
            goto out;
        }
        /* The stream shape must stay constant across the whole connection. */
        bool shape_ok = view.format == DS4_TAP_FORMAT_RAW_HC &&
            view.n_layers == hello.n_layers;
        for (uint32_t i = 0; shape_ok && i < view.n_layers; i++) {
            shape_ok = (int)view.layer_ids[i] == layers[i];
        }
        if (shape_ok && header_written) {
            shape_ok = view.bits == first_bits && view.row_values == first_rows;
        }
        if (!shape_ok) {
            hsx_sink_fail(err, errlen, "HIDDEN_BATCH shape changed mid-stream");
            ds4_hsexport_batch_view_free(&view);
            goto out;
        }
        /* Per-session position continuity and prefix-hash chain. */
        hsx_sink_session *sess = NULL;
        for (int i = 0; i < HSX_SINK_MAX_SESSIONS; i++) {
            if (sessions[i].anchored && sessions[i].session_id == view.session_id) {
                sess = &sessions[i];
                break;
            }
        }
        if (!sess) {
            for (int i = 0; i < HSX_SINK_MAX_SESSIONS; i++) {
                if (!sessions[i].anchored) {
                    sess = &sessions[i];
                    break;
                }
            }
        }
        if (!sess) {
            hsx_sink_fail(err, errlen, "too many concurrent export sessions");
            ds4_hsexport_batch_view_free(&view);
            goto out;
        }
        if (!sess->anchored) {
            /* First batch of this session on this connection: adopt the
             * anchor without verification (the prefix tokens are unknown). */
            sess->session_id = view.session_id;
            sess->hash = view.prefix_hash;
            sess->next_pos = view.pos0 + view.n_tokens;
            sess->anchored = true;
        } else {
            if (view.pos0 != sess->next_pos) {
                hsx_sink_fail(err, errlen,
                              "position gap on session %llu: expected %u, got %u",
                              (unsigned long long)view.session_id,
                              sess->next_pos, view.pos0);
                ds4_hsexport_batch_view_free(&view);
                goto out;
            }
            const uint64_t h = ds4_dist_token_hash_update_span(
                sess->hash, view.token_ids, view.n_tokens);
            if (h != view.prefix_hash) {
                hsx_sink_fail(err, errlen,
                              "prefix hash mismatch on session %llu at pos %u",
                              (unsigned long long)view.session_id, view.pos0);
                ds4_hsexport_batch_view_free(&view);
                goto out;
            }
            sess->hash = h;
            sess->next_pos = view.pos0 + view.n_tokens;
        }
        if (!header_written) {
            first_bits = view.bits;
            first_rows = view.row_values;
            if (hsx_dump_write_header(fp, &hello, layers, model_name, &view) != 0) {
                hsx_sink_fail(err, errlen, "dump write failed: %s", strerror(errno));
                ds4_hsexport_batch_view_free(&view);
                goto out;
            }
            header_written = true;
        }
        if (hsx_dump_write_u32(fp, bytes) != 0 ||
            fwrite(payload, 1, bytes, fp) != bytes) {
            hsx_sink_fail(err, errlen, "dump write failed: %s", strerror(errno));
            ds4_hsexport_batch_view_free(&view);
            goto out;
        }
        bytes_written += 4u + bytes;
        batches++;
        tokens += view.n_tokens;
        ds4_hsexport_batch_view_free(&view);
        free(payload);
        payload = NULL;
    }

    ret = 0;
out:
    free(payload);
    if (fp && fclose(fp) != 0 && ret == 0) {
        hsx_sink_fail(err, errlen, "dump close failed: %s", strerror(errno));
        ret = 1;
    }
    close(fd);
    if (opt->batches_out) *opt->batches_out = batches;
    if (opt->tokens_out) *opt->tokens_out = tokens;
    fprintf(stderr, "ds4: train sink done: %llu batches, %llu tokens, %.2f MiB written to %s\n",
            (unsigned long long)batches, (unsigned long long)tokens,
            (double)bytes_written / (1024.0 * 1024.0), opt->dump_path);
    if (ret != 0 && errlen) fprintf(stderr, "ds4: train sink error: %s\n", err);
    return ret;
}
