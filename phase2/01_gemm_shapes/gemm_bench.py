"""Exercise 1 -- GEMM across shapes, dtypes, and batch sizes.

    python 01_gemm_shapes/gemm_bench.py
    python 01_gemm_shapes/gemm_bench.py --dtypes fp32 fp16 bf16

You already benchmarked square NxNxN GEMM against cuBLAS in
`phase1/04_matmul`. This exercise asks a different question: the *shape* of
the GEMM, not just its size, determines whether it is compute-bound or
memory-bound. `SHAPES` below sweeps from a prefill-like shape (many rows) down
to a decode-like shape (one row) -- the exact comparison the matmul.cu README
asked you to reason about by hand. Now you measure it.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from bench import benchmark  # noqa: E402

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# (label, M, K, N). K=N=4096 throughout -- a plausible hidden size -- only M
# (rows of activations) changes. M = batch * seq_len in real usage.
SHAPES = [
    ("prefill, seq=4096", 4096, 4096, 4096),
    ("prefill, seq=512", 512, 4096, 4096),
    ("decode, batch=32", 32, 4096, 4096),
    ("decode, batch=1", 1, 4096, 4096),
]


# ============================================================================
# TODO 1 -- arithmetic intensity for one (M, K, N) shape
#
# Same definition as matmul.cu: FLOP/byte, assuming each matrix is read/written
# exactly once. 2*M*N*K FLOPs (one multiply, one add per output element per k).
# Bytes = (M*K + K*N + M*N) * dtype size, all three matrices.
#
# This is the number the matmul.cu README asked you to compute by hand for the
# M=1 case. Write the formula instead of doing it by hand this time.
# ============================================================================


def arithmetic_intensity(m: int, k: int, n: int, dtype: torch.dtype) -> float:
    raise NotImplementedError("TODO 1")


# ============================================================================
# TODO 2 -- benchmark one (M, K, N, dtype) config
#
# Allocate x: [M, K] and w: [K, N] on the GPU in the given dtype, then time
# `x @ w` with the shared `benchmark()` util (imported above -- do not
# reimplement CUDA-event timing here, you already know why wall clock lies).
#
# Return the elapsed ms.
# ============================================================================


def bench_gemm(m: int, k: int, n: int, dtype: torch.dtype) -> float:
    raise NotImplementedError("TODO 2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "fp16", "bf16"],
                        choices=DTYPES.keys())
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible to PyTorch.")

    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}\n")

    header = (f"{'shape':>20} {'dtype':>6} {'ms':>10} {'GFLOP/s':>12} "
              f"{'FLOP/byte':>10}")
    print(header)
    print("-" * len(header))

    for label, m, k, n in SHAPES:
        for name in args.dtypes:
            dtype = DTYPES[name]
            ms = bench_gemm(m, k, n, dtype)
            gflop = 2.0 * m * n * k / 1e9
            ai = arithmetic_intensity(m, k, n, dtype)
            print(f"{label:>20} {name:>6} {ms:>10.4f} {gflop / (ms * 1e-3):>12.1f} "
                  f"{ai:>10.1f}")
        print()


if __name__ == "__main__":
    main()
