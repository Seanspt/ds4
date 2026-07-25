#ifndef DS4_HSEXPORT_H
#define DS4_HSEXPORT_H

#include <stdbool.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4.h"

/* Hidden-state export channel (HIDDEN_EXPORT.md). A publisher streams the
 * hidden rows captured by the session tap table (ds4_session_set_hidden_taps)
 * to ONE subscriber over TCP: a training sink or, later, a remote draft.
 *
 * Same trust model as the distributed protocol: no encryption, no
 * authentication. Run on trusted machines and trusted networks only.
 *
 * Wire format: 12-byte big-endian frame header { u32 magic "DS4X", u32 type,
 * u32 bytes }, then the payload. Message types and the fixed records below
 * are protocol constants; payload floats are f32 or f16 per batch (bits
 * field), layer-major (per layer, n_tokens consecutive rows). */

#define DS4_HSEXPORT_MAGIC 0x44533458u /* "DS4X" */
#define DS4_HSEXPORT_MSG_HELLO 1u        /* subscriber -> publisher */
#define DS4_HSEXPORT_MSG_ERROR 2u        /* either direction; fatal */
#define DS4_HSEXPORT_MSG_HIDDEN_BATCH 3u /* publisher -> subscriber */
#define DS4_HSEXPORT_MODE_STREAM 1u
#define DS4_HSEXPORT_MAX_LAYERS 8
#define DS4_HSEXPORT_MAX_MODEL_NAME 127u

/* HELLO payload: this fixed record (all fields big-endian on the wire),
 * then u32 layer_ids[n_layers], then model_name[model_name_len]. The
 * requested layer set and tap format must exactly match the publisher's
 * active tap configuration or the publisher replies ERROR and closes.
 *
 * Wildcard handshake (train sinks that do not know the model): model_id == 0
 * with n_layers == 0 and model_name_len == 0 asks the publisher for its
 * active configuration; partial wildcards are rejected. On success the
 * publisher answers with a HELLO frame of its own carrying the active
 * model_id, layer set, format, ctx_size, and model name before the first
 * HIDDEN_BATCH. */
typedef struct {
    uint32_t model_id;
    uint32_t n_layers;
    uint32_t format;        /* DS4_TAP_FORMAT_* */
    uint32_t mode;          /* DS4_HSEXPORT_MODE_STREAM */
    uint32_t ctx_size;      /* 0 = don't care */
    uint32_t model_name_len;
} ds4_hsexport_hello_fixed;

/* HIDDEN_BATCH payload: this fixed record (big-endian), then
 * u32 layer_ids[n_layers], i32 token_ids[n_tokens], then
 * n_layers * n_tokens * row_values * (bits/8) payload bytes. prefix_hash is
 * the FNV-1a token hash (ds4_distributed.h) of the whole session timeline
 * including this batch: tokens[0, pos0 + n_tokens). After a (re)connect the
 * subscriber must adopt the first batch's prefix_hash as its chain anchor.
 * Payload elements (f32/f16) are little-endian: every supported host is LE
 * and the fixed records are the only byte-swapped part of the protocol. */
typedef struct {
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t n_layers;
    uint32_t format;        /* DS4_TAP_FORMAT_*: row layout of the taps */
    uint32_t bits;          /* payload element width: 32 or 16 */
    uint32_t row_values;    /* f32 values per row (hc_dim or n_embd) */
    uint32_t prefix_hash_hi;
    uint32_t prefix_hash_lo;
    uint32_t reserved;      /* 0 */
} ds4_hsexport_batch_fixed;

typedef struct ds4_hsexport ds4_hsexport;

typedef struct {
    const char *host;     /* bind address; NULL or "" = 127.0.0.1 */
    int port;             /* required */
    int bits;             /* wire payload width: 32 (default) or 16 */
    uint64_t queue_bytes; /* backpressure budget; 0 = default (256 MiB) */
} ds4_hsexport_options;

/* Starts the listener and the connection/sender thread. */
ds4_hsexport *ds4_hsexport_start(ds4_engine *engine,
                                 const ds4_hsexport_options *opt,
                                 char *err,
                                 size_t errlen);
void ds4_hsexport_stop(ds4_hsexport *x);

/* Registers a session for export: configures its tap table (RAW_HC capture;
 * the wire width is a separate publisher option) and starts tracking its
 * position/hash frontier. All attached sessions must use the same layer set
 * (Phase 1: one tap configuration per publisher). Re-attaching the same
 * session_id replaces the previous registration. */
int ds4_hsexport_attach(ds4_hsexport *x,
                        ds4_session *session,
                        uint64_t session_id,
                        const int *layers,
                        int n_layers,
                        char *err,
                        size_t errlen);
void ds4_hsexport_detach(ds4_hsexport *x, ds4_session *session);

/* Current queue occupancy in bytes (monitoring and protocol tests). */
uint64_t ds4_hsexport_queued_bytes(ds4_hsexport *x);

/* Enqueues one HIDDEN_BATCH covering the eval of tokens [pos0, pos0+n_tokens)
 * as captured in the session's tap buffers (decode: n_tokens == 1; prefill
 * chunk: one row per token). With no subscriber connected the frontier
 * advances without queueing. With a subscriber, a full queue blocks the
 * caller (backpressure; training must not lose data), and a position gap
 * poisons the connection (ERROR frame, close) instead of silently skipping.
 * Returns 0 on success or a handled desync, 1 on misuse. */
int ds4_hsexport_emit(ds4_hsexport *x,
                      ds4_session *session,
                      uint32_t pos0,
                      uint32_t n_tokens);

/* Parses the --hidden-export-layers comma list ("40,41,42") into layers[].
 * Returns the layer count, or <0 on malformed input. */
int ds4_hsexport_parse_layers(const char *csv, int *layers, int max_layers);

/* =========================================================================
 * Consumer side: batch parsing and the train sink (HIDDEN_EXPORT.md 3.4).
 * ========================================================================= */

/* Parsed view of one HIDDEN_BATCH payload. layer_ids and token_ids are
 * malloc'd host-order arrays owned by the view (free with
 * ds4_hsexport_batch_view_free); data references the payload buffer. */
typedef struct {
    uint64_t session_id;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t n_layers;
    uint32_t format;            /* DS4_TAP_FORMAT_* */
    uint32_t bits;              /* payload element width: 32 or 16 */
    uint32_t row_values;
    uint64_t prefix_hash;
    uint32_t *layer_ids;        /* n_layers entries */
    int32_t *token_ids;         /* n_tokens entries */
    const unsigned char *data;  /* n_layers*n_tokens*row_values*(bits/8) bytes */
    size_t data_bytes;
} ds4_hsexport_batch_view;

/* Validates and parses one HIDDEN_BATCH payload. Returns 0 on success,
 * -1 on malformed input. */
int ds4_hsexport_parse_batch(const void *payload,
                             size_t bytes,
                             ds4_hsexport_batch_view *view);
void ds4_hsexport_batch_view_free(ds4_hsexport_batch_view *view);

/* Dump file layout (self-describing, big-endian, append-only):
 *
 *   header: u32 magic DS4_HSEXPORT_DUMP_MAGIC, u32 version,
 *           u32 model_id, u32 n_layers, u32 format, u32 bits,
 *           u32 row_values, u32 ctx_size, u32 model_name_len,
 *           u32 reserved x3, then u32 layer_ids[n_layers],
 *           then model_name[model_name_len].
 *   record: u32 record_bytes, then one verbatim HIDDEN_BATCH payload
 *           (batch_fixed + layer_ids + token_ids + data).
 *
 * The header is written when the first batch arrives (row_values/bits come
 * from the wire, not from the handshake). */
#define DS4_HSEXPORT_DUMP_MAGIC 0x44533448u /* "DS4H" */
#define DS4_HSEXPORT_DUMP_VERSION 1u
#define DS4_HSEXPORT_DUMP_HEADER_U32 12u

typedef struct {
    const char *host;      /* publisher address (required) */
    int port;              /* publisher port (required) */
    const char *dump_path; /* output file (required) */
    /* Handshake: all zero/NULL = wildcard (adopt the publisher's active
     * configuration). Otherwise an exact-match request. */
    uint32_t model_id;
    const char *model_name;
    const int *layers;
    int n_layers;
    uint32_t ctx_size;     /* 0 = don't care */
    /* Progress counters (may be NULL), filled on return. */
    uint64_t *batches_out;
    uint64_t *tokens_out;
} ds4_hsexport_sink_options;

/* Runs the train sink: connect, handshake, then verify (prefix-hash chain,
 * per-session position continuity, consistent shape) and dump every
 * HIDDEN_BATCH until the publisher closes the connection or `stop` (may be
 * NULL) is set. Desync is fatal, never silent. Returns 0 on a clean end,
 * 1 on error (err is filled). */
int ds4_hsexport_sink_run(const ds4_hsexport_sink_options *opt,
                          const volatile sig_atomic_t *stop,
                          char *err,
                          size_t errlen);

#endif /* DS4_HSEXPORT_H */
