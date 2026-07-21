#!/bin/sh
# Distributed DSpark acceptance fixture: loopback coordinator+worker pair.
#
# Compares distributed generation with and without DSpark speculation; the
# outputs must match token for token, and the worker must actually emit draft
# proposals. Skips (exit 0) when the binary, model, or support GGUF is absent.
#
# Env overrides:
#   DS4_BIN                 ds4 executable (default ./ds4)
#   DS4_TEST_MODEL          model GGUF with token_embd (default ./ds4flash.gguf)
#   DS4_DSPARK_SUPPORT      DSpark support GGUF
#   DS4_DIST_COORD_LAYERS   coordinator slice (default 0:37)
#   DS4_DIST_WORK_LAYERS    worker slice, must end with :output (default 38:output)
#   DS4_DIST_FIXTURE_TOKENS tokens to generate per case (default 32)
#   DS4_DIST_FIXTURE_PORT   base TCP port (default 17871)
set -eu

DS4_BIN=${DS4_BIN:-./ds4}
MODEL=${DS4_TEST_MODEL:-./ds4flash.gguf}
SUPPORT=${DS4_DSPARK_SUPPORT:-gguf/DeepSeek-V4-Flash-DSpark-support.gguf}
COORD_LAYERS=${DS4_DIST_COORD_LAYERS:-0:37}
WORK_LAYERS=${DS4_DIST_WORK_LAYERS:-38:output}
TOKENS=${DS4_DIST_FIXTURE_TOKENS:-32}
BASE_PORT=${DS4_DIST_FIXTURE_PORT:-17871}

if [ ! -x "$DS4_BIN" ]; then
    echo "dist-dspark-fixture: skipped, missing executable $DS4_BIN" >&2
    exit 0
fi
if [ ! -f "$MODEL" ]; then
    echo "dist-dspark-fixture: skipped, missing model $MODEL" >&2
    exit 0
fi
if [ ! -f "$SUPPORT" ]; then
    echo "dist-dspark-fixture: skipped, missing DSpark support model $SUPPORT" >&2
    exit 0
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-dist-dspark.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM

# run_dist <name> <port> <prompt> <coord extra args> <worker extra args>
# Generates $tmpdir/<name>.out (coordinator stdout) and .cerr/.werr logs.
run_dist() {
    name=$1
    port=$2
    prompt=$3
    coord_extra=$4
    worker_extra=$5

    # shellcheck disable=SC2086
    "$DS4_BIN" -m "$MODEL" \
        --role worker --layers "$WORK_LAYERS" \
        --coordinator 127.0.0.1 "$port" \
        --debug $worker_extra \
        >"$tmpdir/$name.wout" 2>"$tmpdir/$name.werr" &
    worker_pid=$!

    # Give the worker a moment to open its data listener; the coordinator
    # itself waits for full route coverage before generating.
    sleep 2
    rc=0
    # shellcheck disable=SC2086
    "$DS4_BIN" -m "$MODEL" \
        --role coordinator --layers "$COORD_LAYERS" \
        --listen 127.0.0.1 "$port" \
        --tokens "$TOKENS" --temp 0 --nothink -p "$prompt" \
        $coord_extra \
        >"$tmpdir/$name.out" 2>"$tmpdir/$name.cerr" || rc=$?

    kill "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
    return $rc
}

run_case() {
    id=$1
    prompt=$2
    port=$((BASE_PORT + ${3:-0}))

    run_dist "$id.base" "$port" "$prompt" "" "" || {
        echo "dist-dspark-fixture: distributed baseline failed for $id" >&2
        tail -n 5 "$tmpdir/$id.base.cerr" >&2 || true
        return 1
    }
    run_dist "$id.spec" "$port" "$prompt" "--dspark" "--dspark --mtp $SUPPORT" || {
        echo "dist-dspark-fixture: distributed dspark run failed for $id" >&2
        tail -n 5 "$tmpdir/$id.spec.cerr" >&2 || true
        return 1
    }

    if ! cmp -s "$tmpdir/$id.base.out" "$tmpdir/$id.spec.out"; then
        echo "dist-dspark-fixture: output mismatch for $id" >&2
        echo "baseline:" >&2
        sed 's/^/  /' "$tmpdir/$id.base.out" >&2
        echo "dspark:" >&2
        sed 's/^/  /' "$tmpdir/$id.spec.out" >&2
        return 1
    fi
    if ! grep -q 'dist worker dspark draft len=' "$tmpdir/$id.spec.werr"; then
        echo "dist-dspark-fixture: worker produced no draft proposals for $id" >&2
        return 1
    fi

    base_tps=$(sed -n 's/.*generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$tmpdir/$id.base.cerr" | tail -n 1)
    spec_tps=$(sed -n 's/.*generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$tmpdir/$id.spec.cerr" | tail -n 1)
    drafts=$(grep -c 'dist worker dspark draft len=' "$tmpdir/$id.spec.werr" || true)
    printf '%s\tbaseline_tps=%s\tspec_tps=%s\tdraft_events=%s\n' \
        "$id" "${base_tps:-n/a}" "${spec_tps:-n/a}" "$drafts"
}

echo "id	baseline_tps	spec_tps	draft_events"
run_case hello 'Hello' 0
run_case redis 'Explain Redis in one sentence.' 2
run_case c_add 'Complete this C function: int add(int a, int b) {' 4
