"""Exercise 2 -- softmax and layer_norm: memory-bound ops, and fusing them.

    python 02_normalization_ops/ops_bench.py

Unlike GEMM, softmax and layer_norm do O(n) arithmetic on O(n) data -- no
reuse is possible no matter how you tile it. They are memory-bound by
construction: the achieved bandwidth (bytes moved / time) is the number that
matters, not GFLOP/s.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from bench import benchmark  # noqa: E402

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# (batch, seq_len, hidden) -- shapes like a transformer activation tensor.
SHAPES = [
    (1, 512, 4096),
    (8, 512, 4096),
    (32, 512, 4096),
    (32, 4096, 4096),
]


# ============================================================================
# TODO 1 -- achieved memory bandwidth for one run
#
# Both softmax and layer_norm read a [batch, seq, hidden] tensor once and
# write a same-shaped tensor once. Ignore layer_norm's small reduction
# intermediates -- they're negligible next to the full tensor traffic.
#
# bytes = 2 * numel * dtype.itemsize    (one read, one write)
# GB/s  = bytes / (ms * 1e-3) / 1e9
# ============================================================================


def achieved_gbs(numel: int, dtype: torch.dtype, ms: float) -> float:
    raise NotImplementedError("TODO 1")


# ============================================================================
# TODO 2 -- benchmark softmax and layer_norm at one (shape, dtype)
#
# Allocate x: [batch, seq, hidden] on the GPU in the given dtype. Time
#   torch.softmax(x, dim=-1)
#   F.layer_norm(x, (hidden,))
# with the shared benchmark() util. Return both elapsed-ms values.
# ============================================================================


def bench_ops(batch: int, seq: int, hidden: int, dtype: torch.dtype):
    raise NotImplementedError("TODO 2")


# ============================================================================
# TODO 3 (stretch) -- kernel fusion
#
# Build a function that does F.layer_norm(torch.softmax(x, dim=-1), (hidden,))
# as two separate calls, and a torch.compile'd version of the same thing.
# Benchmark both at the smallest shape in SHAPES and the largest.
#
# Predict before running: at which shape (smallest or largest) do you expect
# fusion to help more, and why? Tie this back to Phase 1 Exercise 5 -- what
# does fusion actually save you, mechanically?
# ============================================================================


def bench_fusion(batch: int, seq: int, hidden: int, dtype: torch.dtype):
    raise NotImplementedError("TODO 3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "fp16", "bf16"],
                        choices=DTYPES.keys())
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible to PyTorch.")

    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}\n")
    print("Compare the GB/s columns below against your GPU's peak DRAM "
          "bandwidth (printed by phase1's printDeviceInfo()).\n")

    header = (f"{'shape (b,s,h)':>20} {'dtype':>6} {'softmax ms':>11} "
              f"{'softmax GB/s':>13} {'layernorm ms':>13} {'layernorm GB/s':>15}")
    print(header)
    print("-" * len(header))

    for batch, seq, hidden in SHAPES:
        for name in args.dtypes:
            dtype = DTYPES[name]
            sm_ms, ln_ms = bench_ops(batch, seq, hidden, dtype)
            numel = batch * seq * hidden
            sm_gbs = achieved_gbs(numel, dtype, sm_ms)
            ln_gbs = achieved_gbs(numel, dtype, ln_ms)
            label = f"({batch},{seq},{hidden})"
            print(f"{label:>20} {name:>6} {sm_ms:>11.4f} {sm_gbs:>13.1f} "
                  f"{ln_ms:>13.4f} {ln_gbs:>15.1f}")
        print()


if __name__ == "__main__":
    main()
