#!/usr/bin/env python3
"""Minimal hidden-state export subscriber for ds4 / ds4-server (HIDDEN_EXPORT.md).

Connects to a publisher started with --hidden-export-listen HOST PORT, performs
the HELLO handshake, then prints one line per HIDDEN_BATCH frame. Optionally
verifies the FNV-1a prefix-hash chain incrementally (--verify-hash).

Protocol constants (keep in sync with ds4_hsexport.h):

    MAGIC        = 0x44533458 ("DS4X")
    MSG_HELLO    = 1   subscriber -> publisher
    MSG_ERROR    = 2   either direction; fatal, payload is a UTF-8 message
    MSG_HIDDEN_BATCH = 3   publisher -> subscriber
    MODE_STREAM  = 1

Frame header: 12 bytes big-endian { u32 magic, u32 type, u32 payload_bytes }.

HELLO payload: ds4_hsexport_hello_fixed (6 x u32 BE: model_id, n_layers,
format, mode, ctx_size, model_name_len), then u32 layer_ids[n_layers], then
model_name[model_name_len] (no NUL). The layer set must exactly match the
publisher's --hidden-export-layers; format must be 0 (RAW_HC); ctx_size 0
means "don't care". A model_id of 0 with no layer set and no model name is
the wildcard handshake: the publisher's active configuration is adopted. On
success the publisher answers with its own HELLO frame (the ack) carrying
the active model id, layer set, format, ctx_size, and model name.

HIDDEN_BATCH payload: ds4_hsexport_batch_fixed (11 x u32 BE: session_hi,
session_lo, pos0, n_tokens, n_layers, format, bits, row_values,
prefix_hash_hi, prefix_hash_lo, reserved), then u32 layer_ids[n_layers],
then i32 token_ids[n_tokens], then n_layers * n_tokens * row_values payload
elements of bits/8 bytes each, layer-major (per layer, n_tokens consecutive
rows). prefix_hash is FNV-1a over little-endian token ids of the whole
session timeline including this batch (INIT 1469598103934665603, PRIME
1099511628211; same as ds4_distributed.c). After a (re)connect the subscriber
adopts the first batch's prefix_hash as its chain anchor.

Usage:
    hidden_export_client.py HOST PORT [--model-id N --model-name NAME \
        --layers 40,41,42] [--ctx-size N] [--verify-hash] [--max-batches N]
    hidden_export_client.py --self-test

Without --model-id/--layers the client performs the wildcard handshake and
adopts the publisher's active configuration from the HELLO ack.
"""

import argparse
import socket
import struct
import sys

MAGIC = 0x44533458
MSG_HELLO = 1
MSG_ERROR = 2
MSG_HIDDEN_BATCH = 3
MODE_STREAM = 1
MAX_LAYERS = 8
TAP_FORMAT_RAW_HC = 0

HASH_INIT = 1469598103934665603
HASH_PRIME = 1099511628211
HASH_MASK = (1 << 64) - 1

HEADER = struct.Struct(">III")
HELLO_FIXED = struct.Struct(">6I")
BATCH_FIXED = struct.Struct(">11I")


class ProtocolError(Exception):
    pass


def hash_update(h, token):
    """One FNV-1a step over a little-endian token id (matches the C side)."""
    t = token & 0xFFFFFFFF
    for i in range(4):
        h ^= (t >> (i * 8)) & 0xFF
        h = (h * HASH_PRIME) & HASH_MASK
    return h


def hash_span(h, tokens):
    for tok in tokens:
        h = hash_update(h, tok)
    return h


def build_hello(model_id, layers, model_name, ctx_size=0, fmt=TAP_FORMAT_RAW_HC):
    name = model_name.encode("utf-8")
    if "\0" in model_name or len(name) > 127:
        raise ValueError("invalid model name")
    if model_id == 0:
        # Wildcard handshake: adopt the publisher's active configuration
        # (returned in the HELLO ack). Partial wildcards are rejected.
        if layers or model_name:
            raise ValueError("model_id 0 takes no layer set or model name")
    elif not 1 <= len(layers) <= MAX_LAYERS:
        raise ValueError("need 1..%d layers" % MAX_LAYERS)
    fixed = HELLO_FIXED.pack(model_id, len(layers), fmt, MODE_STREAM,
                             ctx_size, len(name))
    ids = b"".join(struct.pack(">I", layer) for layer in layers)
    payload = fixed + ids + name
    return HEADER.pack(MAGIC, MSG_HELLO, len(payload)) + payload


def parse_hello(payload):
    """Parses a HELLO payload (the publisher's ack). Returns
    (model_id, layer_ids, format, mode, ctx_size, model_name)."""
    if len(payload) < HELLO_FIXED.size:
        raise ProtocolError("HELLO too short: %d bytes" % len(payload))
    model_id, n_layers, fmt, mode, ctx_size, name_len = \
        HELLO_FIXED.unpack_from(payload, 0)
    off = HELLO_FIXED.size
    if n_layers < 1 or n_layers > MAX_LAYERS:
        raise ProtocolError("bad HELLO n_layers %d" % n_layers)
    if len(payload) != off + 4 * n_layers + name_len:
        raise ProtocolError("bad HELLO layout")
    layer_ids = list(struct.unpack_from(">%dI" % n_layers, payload, off))
    off += 4 * n_layers
    model_name = payload[off:off + name_len].decode("utf-8", "replace")
    return model_id, layer_ids, fmt, mode, ctx_size, model_name


def parse_batch(payload):
    """Parses one HIDDEN_BATCH payload. Returns (meta, layer_ids, token_ids,
    payload_bytes_after_headers). Raises ProtocolError on malformed input."""
    if len(payload) < BATCH_FIXED.size:
        raise ProtocolError("HIDDEN_BATCH too short: %d bytes" % len(payload))
    fields = BATCH_FIXED.unpack_from(payload, 0)
    meta = {
        "session_id": (fields[0] << 32) | fields[1],
        "pos0": fields[2],
        "n_tokens": fields[3],
        "n_layers": fields[4],
        "format": fields[5],
        "bits": fields[6],
        "row_values": fields[7],
        "prefix_hash": (fields[8] << 32) | fields[9],
        "reserved": fields[10],
    }
    if meta["n_layers"] < 1 or meta["n_layers"] > MAX_LAYERS:
        raise ProtocolError("bad n_layers %d" % meta["n_layers"])
    if meta["bits"] not in (16, 32):
        raise ProtocolError("bad bits %d" % meta["bits"])
    off = BATCH_FIXED.size
    need = 4 * meta["n_layers"] + 4 * meta["n_tokens"]
    if len(payload) < off + need:
        raise ProtocolError("HIDDEN_BATCH truncated id sections")
    layer_ids = list(struct.unpack_from(">%dI" % meta["n_layers"], payload, off))
    off += 4 * meta["n_layers"]
    token_ids = list(struct.unpack_from(">%di" % meta["n_tokens"], payload, off))
    off += 4 * meta["n_tokens"]
    data = payload[off:]
    expect = (meta["n_layers"] * meta["n_tokens"] * meta["row_values"]
              * (meta["bits"] // 8))
    if len(data) != expect:
        raise ProtocolError("payload size mismatch: got %d, want %d"
                            % (len(data), expect))
    return meta, layer_ids, token_ids, data


def recv_full(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed after %d/%d bytes" % (len(buf), n))
        buf += chunk
    return bytes(buf)


def recv_frame(sock):
    header = recv_full(sock, HEADER.size)
    magic, ftype, nbytes = HEADER.unpack(header)
    if magic != MAGIC:
        raise ProtocolError("bad frame magic 0x%08x" % magic)
    return ftype, recv_full(sock, nbytes)


def run_subscriber(args):
    layers = [int(x) for x in args.layers.split(",")] if args.layers else []
    hello = build_hello(args.model_id, layers, args.model_name,
                        ctx_size=args.ctx_size)
    sock = socket.create_connection((args.host, args.port))
    sock.sendall(hello)

    # The first frame must be the publisher's HELLO ack (or an ERROR).
    ftype, payload = recv_frame(sock)
    if ftype == MSG_ERROR:
        print("publisher rejected HELLO: %s" % payload.decode("utf-8", "replace"),
              file=sys.stderr)
        sock.close()
        return 1
    if ftype != MSG_HIDDEN_BATCH and ftype != MSG_HELLO:
        sock.close()
        raise ProtocolError("unexpected first frame type %d" % ftype)
    if ftype == MSG_HELLO:
        model_id, ack_layers, fmt, mode, ctx_size, model_name = \
            parse_hello(payload)
        print("handshake: model='%s' id=%d layers=%s format=%d mode=%d ctx=%d"
              % (model_name, model_id, ack_layers, fmt, mode, ctx_size))
        if layers and ack_layers != layers:
            sock.close()
            raise ProtocolError("ack layer set %s != requested %s"
                                % (ack_layers, layers))
        layers = ack_layers
        ftype = None  # read the first batch frame fresh

    hash_state = None  # None until the first batch anchors the chain
    batches = 0
    tokens = 0
    payload_bytes = 0
    try:
        while args.max_batches <= 0 or batches < args.max_batches:
            if ftype is None:
                ftype, payload = recv_frame(sock)
            if ftype == MSG_ERROR:
                print("ERROR frame from publisher: %s"
                      % payload.decode("utf-8", "replace"), file=sys.stderr)
                return 1
            if ftype != MSG_HIDDEN_BATCH:
                raise ProtocolError("unexpected frame type %d" % ftype)
            ftype = None
            meta, layer_ids, token_ids, data = parse_batch(payload)
            if layer_ids != layers:
                raise ProtocolError("batch layer set %s != negotiated %s"
                                    % (layer_ids, layers))
            if hash_state is None:
                # First batch after connect anchors the chain: adopt its
                # prefix_hash unverified (the prefix tokens are unknown).
                hash_state = meta["prefix_hash"]
                anchor = " (anchor)"
            else:
                hash_state = hash_span(hash_state, token_ids)
                anchor = ""
                if args.verify_hash and hash_state != meta["prefix_hash"]:
                    raise ProtocolError(
                        "prefix hash mismatch at pos %d: computed %016x, batch %016x"
                        % (meta["pos0"], hash_state, meta["prefix_hash"]))
            batches += 1
            tokens += meta["n_tokens"]
            payload_bytes += len(data)
            print("batch %d: session=%d pos=%d n_tokens=%d layers=%s "
                  "bits=%d rows=%d hash=%016x%s"
                  % (batches, meta["session_id"], meta["pos0"],
                     meta["n_tokens"], layer_ids, meta["bits"],
                     meta["row_values"], meta["prefix_hash"], anchor))
    except (EOFError, ConnectionError) as exc:
        print("connection ended: %s" % exc, file=sys.stderr)
    finally:
        sock.close()
    print("done: %d batches, %d tokens, %.2f MiB payload"
          % (batches, tokens, payload_bytes / (1024.0 * 1024.0)))
    return 0


def self_test():
    # A well-formed two-layer, three-token f32 batch round-trips the parser.
    layers = [40, 41]
    tokens = [101, 102, 103]
    row_values = 4
    h = hash_span(HASH_INIT, tokens)
    data = struct.pack(">%df" % (len(layers) * len(tokens) * row_values),
                       *([0.5] * len(layers) * len(tokens) * row_values))
    fixed = BATCH_FIXED.pack(0, 7, 1000, len(tokens), len(layers),
                             TAP_FORMAT_RAW_HC, 32, row_values,
                             (h >> 32) & 0xFFFFFFFF, h & 0xFFFFFFFF, 0)
    payload = (fixed
               + b"".join(struct.pack(">I", x) for x in layers)
               + struct.pack(">3i", *tokens)
               + data)
    meta, got_layers, got_tokens, got_data = parse_batch(payload)
    assert meta["session_id"] == 7 and meta["pos0"] == 1000
    assert meta["n_tokens"] == 3 and meta["prefix_hash"] == h
    assert got_layers == layers and got_tokens == tokens and got_data == data

    # The incremental chain over a follow-up batch matches a from-scratch hash.
    more = [104]
    h2 = hash_span(h, more)
    assert h2 == hash_span(HASH_INIT, tokens + more)

    # f16 payload sizing.
    fixed16 = BATCH_FIXED.pack(0, 1, 0, 2, 1, 0, 16, 8, 0, 0, 0)
    payload16 = (fixed16 + struct.pack(">I", 5) + struct.pack(">2i", 1, 2)
                 + b"\x00" * (1 * 2 * 8 * 2))
    meta16, _, _, data16 = parse_batch(payload16)
    assert meta16["bits"] == 16 and len(data16) == 32

    # Malformed inputs are rejected.
    for bad in (payload[:BATCH_FIXED.size - 1],           # short fixed record
                payload[:-1],                              # truncated payload
                payload[:BATCH_FIXED.size] + b"\x00" * 3): # missing sections
        try:
            parse_batch(bad)
        except ProtocolError:
            pass
        else:
            raise AssertionError("accepted malformed batch (%d bytes)" % len(bad))

    # HELLO round-trip against the documented layout.
    frame = build_hello(42, [40, 41], "deepseek-v4-flash", ctx_size=32768)
    magic, ftype, nbytes = HEADER.unpack_from(frame, 0)
    assert magic == MAGIC and ftype == MSG_HELLO
    assert nbytes == len(frame) - HEADER.size
    fields = HELLO_FIXED.unpack_from(frame, HEADER.size)
    assert fields == (42, 2, TAP_FORMAT_RAW_HC, MODE_STREAM, 32768,
                      len("deepseek-v4-flash"))

    # Wildcard handshake builds a bare request, and a publisher ack parses.
    wild = build_hello(0, [], "", ctx_size=0)
    magic, ftype, nbytes = HEADER.unpack_from(wild, 0)
    assert magic == MAGIC and ftype == MSG_HELLO
    assert nbytes == HELLO_FIXED.size
    try:
        build_hello(0, [40], "")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a partial wildcard HELLO")
    ack_payload = (HELLO_FIXED.pack(42, 2, TAP_FORMAT_RAW_HC, MODE_STREAM,
                                    32768, len("deepseek-v4-flash"))
                   + struct.pack(">2I", 40, 41) + b"deepseek-v4-flash")
    model_id, ack_layers, fmt, mode, ctx_size, model_name = parse_hello(ack_payload)
    assert (model_id, ack_layers, fmt, mode, ctx_size, model_name) == \
        (42, [40, 41], TAP_FORMAT_RAW_HC, MODE_STREAM, 32768, "deepseek-v4-flash")

    print("self-test OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("host", nargs="?", default="127.0.0.1")
    ap.add_argument("port", nargs="?", type=int, default=0)
    ap.add_argument("--model-id", type=int, default=0)
    ap.add_argument("--model-name", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--ctx-size", type=int, default=0)
    ap.add_argument("--verify-hash", action="store_true")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="stop after N batches (0 = unlimited)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.port:
        ap.error("subscriber mode needs PORT")
    if args.model_id == 0 and (args.layers or args.model_name):
        ap.error("--model-id 0 is the wildcard handshake: no --layers/--model-name")
    if args.model_id != 0 and not args.layers:
        ap.error("an explicit handshake needs --layers")
    try:
        return run_subscriber(args)
    except ProtocolError as exc:
        print("protocol error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
