# AGENTS.md

Guidance for AI coding agents working in this repository. This file describes
the project as it actually is; when in doubt, trust the source and the
sub-READMEs over this summary.

## Project Overview

**DwarfStar** (`ds4.c`) is a small, self-contained native inference engine
optimized first for **DeepSeek V4 Flash**. It also supports **GLM 5.2** and, on
very high-memory machines, **DeepSeek V4 PRO**. It is **not a general GGUF
runner**: it only works with the specific GGUF files listed in `README.md`
(downloaded via `./download_model.sh`, stored under `./gguf/`, with
`./ds4flash.gguf` as the default model path). Arbitrary GGUF files will not
have the expected tensor layout, quantization mix, or metadata.

The whole product is built and tested together: model loading, tokenizer,
prompt rendering, DSML tool calls, KV state, the HTTP server, and the coding
agent. The repository also contains offline tooling for GGUF quantization,
imatrix collection, quality scoring, and speed benchmarking.

Supported backends:

- **Metal** (primary target) on Macs with 96 GB or more; smaller machines use
  SSD streaming.
- **NVIDIA CUDA**, including multi-GPU servers (tested with 8xL40S) and DGX
  Spark / GB10.
- **ROCm** on AMD Strix Halo (gfx1151), e.g. the Framework Desktop. See
  `STRIXHALO.md` for machine setup.
- **CPU**, reference/debug only — never the production target.

The project exists thanks to llama.cpp/GGML: GGUF quant layouts, CPU quant/dot
logic, and certain kernels are retained or adapted under the MIT license (see
`LICENSE`). The codebase is developed with strong AI assistance under human
direction (see "AI full disclosure" in `README.md`). Status: beta quality,
fast-changing; a full QA run (`QA_BEFORE_RELEASES.md`) gates each release.

Note: a legacy `AGENT.md` (singular) with the original agent notes also exists
in the root; its content is folded into this file.

## Technology Stack and Architecture

- **Language**: C99 (`-std=c99` for engine code, `-std=c11` for quality tools),
  compiled with `cc -O3 -ffast-math -Wall -Wextra`. Objective-C (with ARC)
  only where Metal requires it (`ds4_metal.m`). **Do not introduce C++.**
- **GPU kernels**: Metal Shading Language under `metal/*.metal`; CUDA in
  `ds4_cuda.cu`; ROCm/HIP in `ds4_rocm*.cu` plus headers under `rocm/*.cuh`.
- **Build system**: a single hand-written `Makefile` (no CMake, no package
  manager manifests — there is no `pyproject.toml`/`package.json`/`Cargo.toml`;
  Python appears only as standalone helper scripts).
- **No external library dependencies** for the engine itself; `rax.c`/`rax.h`
  (radix tree, from Redis) and `linenoise.c` (line editing) are vendored.

### Public API boundary

`ds4.h` is the narrow public engine boundary: `ds4_engine` is the loaded model,
`ds4_session` is one mutable inference timeline owning the live KV cache and
logits. Callers provide full token prefixes and `ds4_session_sync()` reuses,
extends, or rebuilds graph state. **Keep this header narrow** — CLI/server code
must not depend on tensor internals.

### Runtime architecture highlights

- **Whole-model Metal graph inference** is the production path on macOS. Model
  loading is mmap-backed for the resident case; do not eagerly copy the GGUF.
- **SSD streaming** (`--ssd-streaming`): non-routed weights stay resident;
  routed MoE experts live in an in-memory cache fed by explicit fast disk
  reads. Loading of missing experts must be hidden behind shared-expert /
  cached-expert inference, and layer loads hidden behind current-layer prefill
  time.
- **Sessions and KV reuse**: long local agent sessions work through live KV
  prefix reuse plus disk KV checkpoints (`.kv` files: `KVC` header + rendered
  text + `DSV4` session payload + optional tool-id map; format documented in
  `README.md`). Payload serialization lives in the engine; persistence policy
  lives in server/agent code.
- **Server** (`ds4-server`): OpenAI/Anthropic-compatible HTTP API
  (`/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/messages`,
  `/v1/models`), SSE streaming, DSML tool-call mapping with exact-replay of
  sampled DSML blocks, optional `--batched-session N` multi-KV serving, and
  the disk KV cache (`--kv-disk-dir`).
- **Agent** (`ds4-agent`): native coding agent that drives inference in-process
  (no socket boundary); sessions are the on-disk KV cache under
  `~/.ds4/kvcache`.
- **Distributed pipeline parallelism** (`ds4_distributed.c`): layer ranges
  split across machines (`--role coordinator/worker --layers A:B`), activations
  over plain TCP; accelerates prefill and fits larger models, never generation.
- **Tensor parallelism** (`ds4_tp.c`): two identical Macs over Thunderbolt 5
  (RDMA or TCP fallback) split per-layer matvecs 50/50; or CUDA multi-GPU
  tensor/expert parallelism via `--cuda-tensor-parallel` with explicit
  `--gpu-devices` ordering.
- **Speculative decoding**: DSpark draft model (`--mtp SUPPORT.gguf --dspark`,
  experimental, greedy-only) replacing the legacy one-stage MTP; GLM has its
  own in-GGUF MTP block (`--glm-mtp`).

## Repository Layout

- `ds4.c` (~2.9 MB): model loading, tokenizer, CPU reference code, Metal graph
  scheduling, sessions, disk-cache payload serialization. The core of the
  project.
- `ds4.h`: public engine/session API (see above).
- `ds4_cli.c`: CLI, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: HTTP API, worker queue, streaming, tool-call mapping, disk KV
  cache policy.
- `ds4_agent.c`: native coding agent; `ds4_web.c/.h`: its web/tool helpers.
- `ds4_bench.c`: speed benchmark (instantaneous prefill/generation t/s at
  context frontiers, CSV output).
- `ds4_eval.c`: capability regression benchmark (embedded 92-item GPQA /
  SuperGPQA / AIME / COMPSEC subset with TUI and `--self-test-extractors`).
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: Metal compute kernels (moe, dense, flash_attn, dsv4_kv, ...).
- `ds4_cuda.cu`: CUDA backend; `ds4_iq2_tables_cuda.inc`: shared tables.
- `ds4_rocm.cu`, `ds4_rocm_compat.cu`, `ds4_rocm_unavailable.cu`, `rocm/*.cuh`:
  ROCm/Strix Halo backend.
- `ds4_distributed.c/.h`: pipeline-parallel distributed inference.
- `ds4_tp.c/.h`: two-machine tensor parallelism (RDMA/TCP).
- `ds4_ssd.c/.h`: SSD streaming helpers; `ds4_layer_pack.c/.h`: layer packing.
- `ds4_kvstore.c/.h`: KV cache storage; `ds4_gpu_args.c/.h`, `ds4_gpu_mgpu.h`,
  `ds4_gpu.h`: shared GPU option parsing and backend interface.
- `ds4_hsexport.c/.h`: hidden-state export channel (HIDDEN_EXPORT.md): DS4X
  TCP framing, one subscriber, tap-driven HIDDEN_BATCH streaming with
  backpressure, plus the model-free train-sink consumer
  (`./ds4 --role train-sink --hidden-source HOST PORT --dump FILE`).
  Publisher wired into `ds4` (REPL) and `ds4-server` via
  `--hidden-export-*`.
- `ds4_help.c/.h`: `--help` rendering shared by all binaries.
- `rax.c/.h`, `rax_malloc.h`: vendored radix tree; `linenoise.c/.h`: REPL line
  editing.
- `gguf-tools/`: offline GGUF quantizer (`deepseek4-quantize.c`, `quants.[ch]`
  implementing `q8_0`, `q8_K`, `q4_K`, `q2_K`, `iq2_xxs`), `imatrix/`
  calibration tooling, `quality-testing/` official-continuation scorer.
- `tests/`: C test runners, shell fixtures, prompt files, `test-vectors/`
  (official DeepSeek API continuation vectors + local golden vectors).
- `speed-bench/`: benchmark prompt (`promessi_sposi.txt`), CSV results,
  `plot_speed.py` chart generator.
- `dir-steering/`: directional (activation) steering data, tools, examples.
- `misc/`: ignored notes, experiments, old planning material (gitignored).
- `download_model.sh`: downloads supported GGUFs from
  `huggingface.co/antirez/deepseek-v4-gguf` into `./gguf/`.
- `README.md`: full user documentation. `CONTRIBUTING.md`: regression-testing
  guide — **read it before sending a PR**. `QA_BEFORE_RELEASES.md`: release
  gate matrix. `MODEL_CARD.md`, `STRIXHALO.md`: model and ROCm-host notes.

## Build Commands

macOS (Metal is the default target):

```sh
make                  # builds ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, ./ds4-agent
make cpu              # CPU-only diagnostics build of the same five binaries
make clean
```

Linux: plain `make` only prints help; pick an explicit target:

```sh
make cuda-spark                    # DGX Spark / GB10 (no explicit -arch: fastest there)
make cuda-generic                  # generic local CUDA GPU (CUDA_ARCH=native)
make cuda CUDA_ARCH=sm_120         # explicit nvcc -arch
make strix-halo                    # ROCm for Strix Halo / gfx1151 (make rocm is an alias)
make cpu                           # CPU-only build (also on Linux)
```

Offline GGUF tooling:

```sh
make -C gguf-tools                 # quantizer etc.
make -C gguf-tools quality-score   # official-continuation scorer
```

Useful overridable variables: `CC`, `CUDA_HOME`, `NVCC`, `HIPCC`, `ROCM_ARCH`
(default `gfx1151`), `DEBUG_FLAGS`.

## Testing Instructions

Build validation is `make`; warning-free output is a release requirement.

```sh
make test        # unit/regression suite: ds4-eval extractor self-test,
                 # ds4_agent_test, ds4_test, test_layer_pack, test_hsexport,
                 # test_engine_mgpu_placement, test_gpu_args (+CLI script),
                 # test_sampling on Linux, q4k-dot-test
```

`make test` needs a model (default `ds4flash.gguf`, override with
`DS4_TEST_MODEL=/path/to/model.gguf`) and Metal. Use live server tests only
when intentionally testing the API surface. The main C runner is
`./ds4_test` (no args == `--all`); useful narrower checks:

```sh
./ds4_test --server            # API parsing, chat rendering, streaming, tool calls, KV bookkeeping
./ds4_test --logprob-vectors   # vs official DeepSeek API continuation vectors
./ds4_test --local-golden-vectors  # vs known-sane local logits fixture
./ds4_test --tap-golden-vectors    # hidden-tap zero-drift + golden tap rows
./ds4_test --long-context      # long-context fact recall regression
./ds4_test --tool-call-quality # DSML tool-call emission quality
./ds4_test --metal-kernels     # isolated Metal kernel numeric checks
```

Model/backend-specific tests:

```sh
DS4_TEST_MODEL=... DS4_TEST_SESSION_COUNT=4 make test-metal-session-batch   # Metal batching
DS4_TEST_MODEL=... make test-cuda-session-batch                             # CUDA multi-GPU
DS4_TEST_MODEL=... make test-cuda-mixed-batch
make cuda-regression           # CUDA long-context smoke (CUDA build only)
make dspark-verify-depth       # DSpark speculative smoke (needs support GGUF)
make mtp-verify-depth          # legacy MTP smoke (needs MTP GGUF)
tests/glm_long_context_smoke.sh /path/to/glm.gguf
```

Deterministic generation-drift gate (token counts must match the baseline
table in `README.md`):

```sh
./ds4-eval -m ds4flash.gguf --plain --questions 4 --tokens 2048 --temp 0 --seed 1
```

Speed regressions: use `./ds4-bench` with `speed-bench/promessi_sposi.txt`
(see `CONTRIBUTING.md` for the canonical sweep) and compare `prefill_tps` /
`gen_tps` CSVs on the same machine, quant, and thermal state.

Quantization quality: score GGUFs against official continuations with
`gguf-tools/quality-testing/score_official` and `compare_scores.py`
(lower `avg_nll` is better; see `gguf-tools/quality-testing/README.md`).

Release process: `QA_BEFORE_RELEASES.md` is the full gate — clean tree,
warning-free builds on every backend host, `make test`, vector checks,
official continuation quality gates, remote Metal/CUDA/ROCm machines,
distributed and agent manual checks.

## Development Conventions and Code Style

- **Correctness before speed.** Never keep a faster path with unexplained
  attention, KV cache, or logits drift. The only acceptable speed regression
  is one required by an important correctness fix.
- **Keep it small, sharp, and elegant.** Don't settle for the first thing that
  comes to mind; find the minimal working design. No slop: no fragile
  case-patching, no dead code, no code more complicated than it should be.
- **Comment important inference code** where model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
  Comments live beside the implementation (not in separate design documents),
  are instructive and compact, and explain *why* a shape, ordering, cache
  boundary, or memory choice exists. Match the surrounding comment density and
  style; all code and comments are in English.
- **Keep public APIs narrow.** CLI/server code should not know tensor
  internals.
- **No permanent semantic variants behind flags.** Diagnostic switches are fine
  when they validate the one release path.
- **No C++.** C99 engine code; Objective-C only for the Metal runtime.
- **Do not break sibling backends.** A fix to one path must not affect SSD
  streaming, CUDA, distributed inference, or the Metal default path. After
  major changes: verify the normal Metal path and its speed, test SSD
  streaming, test distributed inference if it could be affected (ask the user
  first), and check whether CUDA could be broken (ask the user for access to
  the CUDA machine to verify).
- Compiler warnings are treated as build failures in the release gate; run
  `git diff --check` for whitespace before committing.
- PRs/commits should record the commands run, machine/backend, model quant,
  and notable failures (see `CONTRIBUTING.md`).

## Safety and Security Considerations

- **Avoid large CPU inference runs on macOS**: the CPU path has previously
  exposed kernel VM failures with very large mappings and can crash the
  system. The CPU backend exists for reference/debug only.
- **Do not run multiple huge model processes concurrently.** The instance lock
  is intentional.
- The **distributed protocol has no encryption or authentication** and is not
  release-stable: run coordinator and workers built from the same commit, on
  trusted machines and trusted networks only.
- `ds4-server` binds to localhost by default; use `--host 0.0.0.0` only
  deliberately, and `--cors` only for browser clients (headers only, it does
  not expose the server by itself).
- **Disk KV cache files contain the verbatim cached prompt** (the rendered text
  is the lookup identity). Treat the cache directory as sensitive; it is
  disposable and can simply be deleted.
- The cache-directory and session files use ordinary `read`/`write` I/O, not
  `mmap`, to avoid adding VM mappings to a process that already maps the model.
