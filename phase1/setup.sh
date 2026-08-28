#!/usr/bin/env bash
# Check the toolchain and build Phase 1. Run this first on a fresh pod.
set -u

fail=0

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "  nvidia-smi not found — no NVIDIA driver visible."
    fail=1
fi

echo
echo "=== Compiler ==="
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -2
else
    echo "  nvcc not found. This image has the CUDA runtime but not the toolkit."
    echo "  Fix:  apt-get update && apt-get install -y cuda-toolkit-12-4"
    echo "        export PATH=/usr/local/cuda/bin:\$PATH"
    fail=1
fi

echo
echo "=== PyTorch (exercise 4 comparison only) ==="
python3 -c 'import torch; print(f"  torch {torch.__version__}, cuda={torch.cuda.is_available()}")' \
    2>/dev/null || echo "  not installed — 04_matmul/torch_matmul_bench.py will not run"

if [ "$fail" -ne 0 ]; then
    echo
    echo "Toolchain incomplete, not building."
    exit 1
fi

echo
echo "=== Build ==="
make clean && make || exit 1

echo
echo "Built. Now go implement the TODOs — every kernel is a stub, so the"
echo "correctness columns will all read NO until you do."
echo
echo "  ./bin/vector_add      start here"
echo "  make run              all five in order"
