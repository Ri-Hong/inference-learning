#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err_));                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Kernel launches report errors asynchronously, so check both the launch
// itself and the work that followed it.
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

struct GpuTimer {
  cudaEvent_t beg, end;

  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
  }
  ~GpuTimer() {
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }

  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(beg, s)); }
  void stop(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(end, s));
    CUDA_CHECK(cudaEventSynchronize(end));
  }
  float elapsedMs() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    return ms;
  }
};

struct CpuTimer {
  std::chrono::high_resolution_clock::time_point beg, end;

  void start() { beg = std::chrono::high_resolution_clock::now(); }
  void stop() { end = std::chrono::high_resolution_clock::now(); }
  double elapsedMs() const {
    return std::chrono::duration<double, std::milli>(end - beg).count();
  }
};

// Run `f` a few times to warm up (context init, caches, clocks), then time
// `iters` runs and return the average milliseconds per run.
template <typename F>
float timeGpuMs(F &&f, int warmup = 5, int iters = 50) {
  for (int i = 0; i < warmup; ++i) f();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t;
  t.start();
  for (int i = 0; i < iters; ++i) f();
  t.stop();
  return t.elapsedMs() / static_cast<float>(iters);
}

// Silent version, for tables where a "NO" in the correctness column is the
// expected state until a kernel is implemented.
inline bool allCloseQuiet(const float *a, const float *b, size_t n,
                          float rtol = 1e-4f, float atol = 1e-4f) {
  for (size_t i = 0; i < n; ++i) {
    float diff = std::fabs(a[i] - b[i]);
    if (diff > atol + rtol * std::fabs(b[i])) return false;
  }
  return true;
}

// Same check, but prints the first mismatch. Use this while debugging a kernel.
inline bool allClose(const float *a, const float *b, size_t n,
                     float rtol = 1e-4f, float atol = 1e-4f) {
  for (size_t i = 0; i < n; ++i) {
    float diff = std::fabs(a[i] - b[i]);
    if (diff > atol + rtol * std::fabs(b[i])) {
      std::printf("  mismatch at %zu: %.6f vs %.6f (diff %.3e)\n", i, a[i],
                  b[i], diff);
      return false;
    }
  }
  return true;
}

inline void fillRandom(float *p, size_t n, unsigned seed = 1234) {
  std::srand(seed);
  for (size_t i = 0; i < n; ++i) {
    p[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
  }
}

// Theoretical peak DRAM bandwidth in GB/s. Every kernel you write should be
// compared against this number, not against another kernel in isolation.
//
// cudaDeviceProp::memoryClockRate is deprecated in CUDA 12 and reports 0 on
// some recent devices. When that happens, look the real figure up in the
// vendor spec sheet and pass it in:
//
//     PEAK_GBS=3350 ./bin/transpose
inline double peakBandwidthGBs(const cudaDeviceProp &p) {
  if (const char *env = std::getenv("PEAK_GBS")) {
    double v = std::atof(env);
    if (v > 0.0) return v;
  }
  return 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
}

inline cudaDeviceProp printDeviceInfo() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

  std::printf("=== Device %d: %s ===\n", dev, p.name);
  std::printf("  compute capability : %d.%d\n", p.major, p.minor);
  std::printf("  SMs                : %d\n", p.multiProcessorCount);
  std::printf("  warp size          : %d\n", p.warpSize);
  std::printf("  max threads/block  : %d\n", p.maxThreadsPerBlock);
  std::printf("  max threads/SM     : %d\n", p.maxThreadsPerMultiProcessor);
  std::printf("  shared mem/block   : %zu KB\n", p.sharedMemPerBlock / 1024);
  std::printf("  shared mem/SM      : %zu KB\n",
              p.sharedMemPerMultiprocessor / 1024);
  std::printf("  regs/block         : %d\n", p.regsPerBlock);
  std::printf("  global memory      : %.2f GB\n", p.totalGlobalMem / 1e9);
  std::printf("  memory bus width   : %d bits\n", p.memoryBusWidth);
  double peak = peakBandwidthGBs(p);
  std::printf("  peak DRAM bandwidth: %.1f GB/s\n", peak);
  if (peak <= 0.0) {
    std::printf("  !! memoryClockRate reported 0 (deprecated on this CUDA "
                "version).\n"
                "     The \"%% of peak\" columns will be meaningless. Look up "
                "this GPU's\n"
                "     bandwidth and rerun with PEAK_GBS=<number>.\n");
  }
  std::printf("\n");
  return p;
}
