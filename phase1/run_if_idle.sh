#!/bin/bash
# Usage: ./run_if_idle.sh <make-target> [threshold%] [gpu-index] [extra binary args...]
#
# Builds <make-target>, checks GPU utilization via nvidia-smi, and only runs
# ./bin/<make-target> if utilization is below threshold% (default 50, GPU 0).
set -e

TARGET="$1"
THRESHOLD="${2:-50}"
GPU="${3:-0}"

if [ -z "$TARGET" ]; then
  echo "Usage: $0 <make-target> [threshold%] [gpu-index] [extra binary args...]"
  exit 1
fi

make "$TARGET"

UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU")

echo "GPU $GPU utilization: ${UTIL}%"

if [ "$UTIL" -lt "$THRESHOLD" ]; then
  echo "Below ${THRESHOLD}% threshold — running ./bin/$TARGET"
  ./bin/"$TARGET" "${@:4}"
else
  echo "GPU $GPU is at ${UTIL}%, at or above ${THRESHOLD}% threshold — not running."
  exit 1
fi
