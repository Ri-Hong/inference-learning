"""Compare torch.matmul against the hand-written kernels in matmul.cu.

Run this right after ./bin/matmul so the numbers are directly comparable --
same GPU, same sizes, same units.

    python 04_matmul/torch_matmul_bench.py
    python 04_matmul/torch_matmul_bench.py --sizes 512 1024 2048 4096

Two things to watch for:

1. torch.matmul dispatches to cuBLAS, so at fp32 it should land on essentially
   the same GFLOP/s as the cuBLAS row in matmul.cu. If it does not, the
   difference is measurement, not math -- most likely a missing synchronize.

2. The tf32 and fp16 rows use Tensor Cores. They do the same logical work at a
   fraction of the time, which is the entire reason inference runs in reduced
   precision. That is Phase 2's subject; this is a preview.
"""

import argparse
import time

import torch


def benchmark(fn, warmup=5, iters=20):
    """Time a GPU op correctly.

    CUDA launches are asynchronous: the Python call returns before the GPU has
    done anything. Without the synchronize you would be timing the launch, and
    you would report a matmul running in microseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def naive_timing_trap(a, b, iters=3):
    """The wrong way, kept here so you can see how wrong it is.

    Few iterations on purpose: enqueue enough work and the queue backs up and
    the host blocks anyway, which accidentally makes the wrong method look
    right.
    """
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    return (time.perf_counter() - t0) * 1000 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible to PyTorch.")

    dev = torch.device("cuda")
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    print(f"CUDA {torch.version.cuda}\n")

    configs = [
        ("fp32", torch.float32, False),
        ("tf32", torch.float32, True),
        ("fp16", torch.float16, False),
        ("bf16", torch.bfloat16, False),
    ]

    header = f"{'N':>6} {'dtype':>8} {'ms':>10} {'GFLOP/s':>12} {'vs fp32':>10}"
    print(header)
    print("-" * len(header))

    for n in args.sizes:
        gflop = 2.0 * n * n * n / 1e9
        fp32_ms = None

        for name, dtype, allow_tf32 in configs:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32

            a = torch.randn(n, n, device=dev, dtype=dtype)
            b = torch.randn(n, n, device=dev, dtype=dtype)

            ms = benchmark(lambda: a @ b)
            if name == "fp32":
                fp32_ms = ms

            speedup = f"{fp32_ms / ms:.2f}x" if fp32_ms else "-"
            print(f"{n:>6} {name:>8} {ms:>10.3f} {gflop / (ms * 1e-3):>12.1f} "
                  f"{speedup:>10}")

            del a, b
            torch.cuda.empty_cache()
        print()

    torch.backends.cuda.matmul.allow_tf32 = False

    # The classic mistake, shown side by side with the truth.
    n = 4096
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    wrong = naive_timing_trap(a, b)
    right = benchmark(lambda: a @ b)
    print(f"Timing a {n}x{n} fp32 matmul:")
    print(f"  wall clock, no synchronize : {wrong:8.3f} ms")
    print(f"  CUDA events, synchronized  : {right:8.3f} ms")
    print(f"  ratio                      : {right / wrong:8.2f}x")
    print("\nIf the first number is smaller, you measured how long it takes to")
    print("put work in a queue, not how long the GPU took to do it.")


if __name__ == "__main__":
    main()
