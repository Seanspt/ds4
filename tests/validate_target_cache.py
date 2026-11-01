#!/usr/bin/env python3
"""Contract validation for a ds4 cache-gen / test_cache_writer target cache.

Reads a target-cache v2 directory with DeepSpec's OWN CacheDataset
(deepspec/data/target_cache_dataset.py) and verifies it. Two modes, selected
by the manifest's target_model_name_or_path:

- "/fake/DeepSeek-V4-Flash-DSpark": the deterministic synthetic contract
  produced by tests/test_cache_writer.c (3 samples, T=5, H=8, layers
  [-1, 3]), compared bit-exact:

    input_ids[t]        = s*100 + t
    loss_mask[t]        = t >= 2
    layer_hidden[l][i]  = ((s*7 + l*3 + i) % 17 - 8) * 0.5
    last_hidden[i]      = 0.1 * ((s*31 + i) % 13) + 0.05   (f32 arithmetic)

- anything else: a real cache-gen output. Generic contract checks per
  sample: dtypes and shapes (input_ids int32 [T], loss_mask uint8 [T],
  target_hidden_states bf16 [T, L*H], target_last_hidden_states bf16
  [T, H]), loss_mask values in {0,1} with sum >= MIN_LOSS_TOKENS, and all
  hidden values finite after bf16 -> f32 conversion.

Usage:
    DS4_TEST_CACHE_WRITER_OUT=/tmp/cw-cache ./tests/test_cache_writer
    python3 tests/validate_target_cache.py /tmp/cw-cache
    python3 tests/validate_target_cache.py /tmp/cg_test/cache 14 1,11,21,31,41

Needs python3 with torch + numpy and a checkout of the DeepSpec repo
(DEEPSPEC_REPO env or /Volumes/datadisk/projects/DeepSpec). On the 5090
training host, point DEEPSPEC_REPO at the local DeepSpec clone.
"""

import os
import sys

import numpy as np
import torch

DEEPSPEC_REPO = os.environ.get("DEEPSPEC_REPO", "/Volumes/datadisk/projects/DeepSpec")
sys.path.insert(0, DEEPSPEC_REPO)

from deepspec.data.target_cache_dataset import (  # noqa: E402
    CacheDataset,
    load_target_cache_manifest,
)

SYNTHETIC_TARGET_NAME = "/fake/DeepSeek-V4-Flash-DSpark"

T = 5
H = 8
LAYERS = [-1, 3]
SAMPLES = 3


def expected_sample(s):
    ids = np.arange(s * 100, s * 100 + T, dtype=np.int32)
    loss = np.array([1 if t >= 2 else 0 for t in range(T)], dtype=np.uint8)
    n = T * H
    idx = np.arange(n, dtype=np.int64)
    layer_vals = [((s * 7 + l * 3 + idx) % 17 - 8).astype(np.float32) * np.float32(0.5)
                  for l in range(len(LAYERS))]
    last = (np.float32(0.1) * ((s * 31 + idx) % 13).astype(np.float32)
            + np.float32(0.05)).astype(np.float32)
    # [T, L*H] position-major, layers ascending within a position.
    hidden = np.stack(layer_vals, axis=0).reshape(len(LAYERS), T, H)
    hidden = np.transpose(hidden, (1, 0, 2)).reshape(T, len(LAYERS) * H)
    return ids, loss, hidden, last.reshape(T, H)


def bf16_bits(f32_array):
    t = torch.from_numpy(np.ascontiguousarray(f32_array.astype(np.float32)))
    return t.to(torch.bfloat16).view(torch.uint16).numpy()


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print(__doc__)
        return 2
    cache_dir = sys.argv[1]
    min_loss_tokens = int(sys.argv[2]) if len(sys.argv) >= 3 else 14
    expected_layers = (
        [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) >= 4 else None
    )

    # DeepSpec's own manifest validation runs inside load_target_cache_manifest.
    manifest = load_target_cache_manifest(cache_dir)

    if manifest.get("target_model_name_or_path") == SYNTHETIC_TARGET_NAME:
        return validate_synthetic(cache_dir, manifest)
    return validate_real(cache_dir, manifest, min_loss_tokens, expected_layers)


def validate_synthetic(cache_dir, manifest):
    assert manifest["target_layer_ids"] == LAYERS, manifest["target_layer_ids"]
    assert manifest["hidden_size"] == H
    assert manifest["num_samples"] == SAMPLES
    assert manifest["num_shards"] == SAMPLES

    ds = CacheDataset(cache_dir)
    assert len(ds) == SAMPLES
    for s in range(SAMPLES):
        item = ds[s]
        ids, loss, hidden, last = expected_sample(s)
        assert item["input_ids"].dtype == torch.int32
        assert item["loss_mask"].dtype == torch.uint8
        assert item["target_hidden_states"].dtype == torch.bfloat16
        assert item["target_last_hidden_states"].dtype == torch.bfloat16
        assert item["input_ids"].shape == (T,)
        assert item["target_hidden_states"].shape == (T, len(LAYERS) * H)
        assert item["target_last_hidden_states"].shape == (T, H)
        np.testing.assert_array_equal(item["input_ids"].numpy(), ids)
        np.testing.assert_array_equal(item["loss_mask"].numpy(), loss)
        # Bit-exact bf16 comparison (the writer's RNE cast must match torch).
        got_h = item["target_hidden_states"].view(torch.uint16).numpy()
        got_l = item["target_last_hidden_states"].view(torch.uint16).numpy()
        np.testing.assert_array_equal(got_h, bf16_bits(hidden))
        np.testing.assert_array_equal(got_l, bf16_bits(last))
    ds.close()
    print(f"validate_target_cache: OK ({SAMPLES} samples, DeepSpec CacheDataset "
          f"bit-exact) — {cache_dir}")
    return 0


def validate_real(cache_dir, manifest, min_loss_tokens, expected_layers):
    layers = [int(x) for x in manifest["target_layer_ids"]]
    h = int(manifest["hidden_size"])
    n = int(manifest["num_samples"])
    if expected_layers is not None:
        assert layers == expected_layers, (layers, expected_layers)

    ds = CacheDataset(cache_dir)
    assert len(ds) == n
    total_tokens = 0
    total_loss = 0
    for s in range(n):
        item = ds[s]
        t = item["input_ids"].shape[0]
        assert item["input_ids"].dtype == torch.int32
        assert item["loss_mask"].dtype == torch.uint8
        assert item["target_hidden_states"].dtype == torch.bfloat16
        assert item["target_last_hidden_states"].dtype == torch.bfloat16
        assert item["loss_mask"].shape == (t,)
        assert item["target_hidden_states"].shape == (t, len(layers) * h)
        assert item["target_last_hidden_states"].shape == (t, h)
        mask = item["loss_mask"].numpy()
        assert set(np.unique(mask)) <= {0, 1}, np.unique(mask)
        loss_n = int(mask.sum())
        assert loss_n >= min_loss_tokens, (s, loss_n, min_loss_tokens)
        for key in ("target_hidden_states", "target_last_hidden_states"):
            vals = item[key].to(torch.float32)
            assert torch.isfinite(vals).all(), (s, key)
        total_tokens += t
        total_loss += loss_n
    ds.close()
    print(f"validate_target_cache: OK ({n} samples, {total_tokens} tokens, "
          f"{total_loss} loss tokens, layers={layers}, hidden={h}, "
          f"min_loss_tokens>={min_loss_tokens}, all finite) — {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
