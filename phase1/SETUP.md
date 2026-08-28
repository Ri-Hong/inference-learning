# Getting a GPU (RunPod)

There is no NVIDIA GPU on this machine, so the exercises need a remote one.
These notes assume RunPod, but nothing here is RunPod-specific beyond the first
section — any box with an NVIDIA GPU and the CUDA toolkit works.

## 1. Pick a pod

Any modern NVIDIA GPU teaches Phase 1 fine. What matters is that you use the
**same GPU for every run**, because none of the numbers are comparable across
devices.

Reasonable choices, cheapest first:

| GPU | `ARCH` | Notes |
|---|---|---|
| RTX A4000 / A5000 | `sm_86` | cheapest thing that works |
| RTX 3090 | `sm_86` | |
| RTX 4090 | `sm_89` | good price/perf, fast fp16 |
| L4 / L40S | `sm_89` | datacenter equivalent |
| A100 40/80GB | `sm_80` | HBM, much higher bandwidth ceiling |
| H100 | `sm_90` | overkill for Phase 1 |

An A100 or H100 is worth one session later on, because HBM changes the
bandwidth numbers by an order of magnitude and makes the roofline argument much
more vivid. It is not worth paying for while you are still writing kernels.

Community Cloud is cheaper than Secure Cloud and fine for this. Spot/interruptible
is also fine — nothing here runs long enough to care.

## 2. Pick a template with `nvcc` in it

This is the one thing that actually goes wrong. Many images ship the CUDA
*runtime* but not the *toolkit*, so `torch` works and `nvcc` does not.

Use a **PyTorch** template (they generally include the toolkit), or an explicit
`nvidia/cuda:12.x.x-devel-ubuntu22.04` image. Either way, verify first:

```bash
nvidia-smi          # driver + GPU present
nvcc --version      # compiler present  <- this is the one that fails
```

If `nvcc` is missing:

```bash
apt-get update && apt-get install -y cuda-toolkit-12-4
export PATH=/usr/local/cuda/bin:$PATH
```

For exercise 4 you also need `libcublas`, which comes with the toolkit, and
PyTorch for the comparison script.

## 3. Connect

The web terminal in the RunPod console is enough to get started. For real work,
add your public key to the pod's SSH settings and connect over SSH — then you
can use VS Code Remote or JetBrains Gateway and edit the `.cu` files in a normal
editor instead of `vim` in a browser tab.

```bash
ssh <pod-user>@<host> -p <port> -i ~/.ssh/id_ed25519
```

The exact host and port are shown in the pod's Connect panel.

## 4. Get the code and build

```bash
cd /workspace                                    # persists across pod restarts
git clone https://github.com/Ri-Hong/inference-learning.git
cd inference-learning/phase1
./setup.sh                                       # checks the toolchain, builds
```

Or by hand:

```bash
make                    # -arch=native, picks up whatever GPU is present
make ARCH=sm_89         # or name it explicitly
make run                # build and run all five
```

`/workspace` is on the persistent volume. Anything you write elsewhere in the
container is lost when the pod is stopped, so keep the repo and your results
there.

## 5. Save your results before you stop the pod

Results only mean something with the GPU recorded next to them. Capture the
environment along with the output:

```bash
nvidia-smi --query-gpu=name,memory.total,clocks.max.memory --format=csv
nvcc --version
make run 2>&1 | tee results/$(date +%Y-%m-%d)-$(nvidia-smi --query-gpu=name \
    --format=csv,noheader | tr ' ' '-').txt
```

Then commit and push from the pod, or `scp` the file back.

## 6. Stop the pod

Billing is per second while a pod is running, and **stopped pods still bill for
storage**. Terminate the pod when you are done for real; stop it if you want the
volume to survive until tomorrow.

## Gotchas

- **`% of peak` shows 0 or inf.** `cudaDeviceProp::memoryClockRate` is
  deprecated in CUDA 12 and reports 0 on some devices. Look up the GPU's real
  bandwidth and rerun with `PEAK_GBS=<number> ./bin/transpose`.
- **Numbers wobble between runs.** Datacenter GPUs clock down under sustained
  load and shared hosts have noisy neighbours. Re-run before believing a small
  difference; `timeGpuMs` already averages and warms up.
- **A consumer card (3090/4090) is not a small A100.** Very different memory
  bandwidth and fp32-vs-fp16 ratios. Fine for learning the mechanisms, wrong for
  extrapolating to serving hardware.
