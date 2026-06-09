# GPU Checkpoint/Restore for gVisor

Checkpoint and restore GPU containers in gVisor without losing GPU state. Works for single-GPU and multi-GPU. Supports cold restore (sentry dies, new sentry starts). Verified with PyTorch.

## How It Works

`cuCheckpointProcessCheckpoint` copies GPU VRAM to host memory and terminates the CUDA session. gVisor serializes the host memory (which now contains the GPU state) to disk. On restore, gVisor loads the memory back, and `cuCheckpointProcessRestore` reads from host memory and creates a new CUDA session on the GPU.

```
SAVE:
  1. cuCheckpointProcessLock     → atomically freeze all GPUs
  2. cuCheckpointProcessCheckpoint → copy GPU VRAM to host memory (+519MB RSS for 64MB GPU data)
  3. runsc checkpoint             → serialize everything to disk (host memory includes GPU state)
  4. Sentry exits                 → checkpoint file is portable

RESTORE:
  5. runsc restore                → new sentry, loads checkpoint, process memory restored
  6. cuCheckpointProcessRestore   → reads GPU state from host memory, creates new CUDA session
  7. cuCheckpointProcessUnlock    → resume GPU work
  8. App continues                → GPU memory intact, patterns verified
```

The GPU state is in **process memory**, not the driver session. The checkpoint file is portable across machines with the same GPU type.

## Key Discovery

`cuCheckpointProcessCheckpoint` doesn't just freeze GPU state — it **moves** it to host memory. RSS grows by ~8x the GPU allocation size. This host memory is regular anonymous pages that gVisor serializes naturally. On restore, `cuCheckpointProcessRestore` reads from that memory and creates a fresh CUDA session — it doesn't need the old driver session.

This is how Modal does it. Confirmed via their [GPU Memory Snapshots blog](https://modal.com/blog/gpu-mem-snapshots).

## Test Results

### Verified on H100, driver 580.105.08

| Test | Type | Sentry | Result |
|------|------|--------|--------|
| Single-GPU C test | `--leave-running` | Same | restore=0, 0xBEEF0001 ✓ |
| Multi-GPU C test (2x H100) | `--leave-running` | Same | restore=0, 0xCAFE0000+0xCAFE0001 ✓ |
| PyTorch nn.Linear (2x H100) | `--leave-running` | Same | restore=0, tensors+model ✓ |
| **Single-GPU cold restore** | **Raw runsc** | **NEW** | **restore=0, 0xBEEF1234, 15 ticks ✓** |
| **Multi-GPU cold restore (2x H100)** | **Raw runsc** | **NEW** | **restore=0, 0xCAFE0000+0xCAFE0001, 15 ticks ✓** |
| **Multi-GPU + NCCL cold restore** | **Raw runsc** | **NEW** | **restore=0, patterns+NCCL allreduce, 14 ticks ✓** |

**Cold restore output (sentry died, new sentry started):**
```
tick=10 ok val=0xBEEF1234    ← before checkpoint
SAVE:lock=0                   ← GPU frozen
SAVE:ckpt=0                   ← GPU state → host memory
tick=12 gpu_locked            ← app waiting
[checkpoint saved to disk: 22MB]
[sentry exits]
[NEW sentry starts, loads checkpoint]
RESTORE:restore=0             ← GPU state restored from host memory
RESTORE:unlock=0              ← GPU unlocked
tick=14 ok val=0xBEEF1234     ← GPU pattern verified!
tick=16 ok val=0xBEEF1234
... (15 consecutive ticks verified)
```

**PyTorch output (2x H100, --leave-running):**
```
GPU 0: tensor 0xCAFE, GPU 1: tensor 0xCAFF
Model: nn.Linear forward pass ✓
SAVE:lock=0, SAVE:ckpt=0
RESTORE:restore=0, RESTORE:unlock=0
tick=16 tensors_ok=True model_ok=True ✓
```

**Multi-GPU + NCCL cold restore output (2x H100, sentry died):**
```
NCCL initialized, allreduce across 2 GPUs
GPU 0: 0xCAFE0000, GPU 1: 0xCAFE0001
tick=12 patterns_ok=1 nccl_ok=1    ← before checkpoint
SAVE:lock=0                         ← atomic multi-GPU freeze
SAVE:ckpt=0                         ← GPU+NCCL state → host memory (666MB)
tick=14 gpu_locked
[sentry exits, checkpoint: 666MB]
[NEW sentry starts, loads checkpoint]
RESTORE:restore=0                   ← restored from host memory
RESTORE:unlock=0
tick=16 patterns_ok=1 nccl_ok=1     ← BOTH patterns + NCCL verified!
... (14 consecutive ticks, all pass)
```

Key fix for NCCL: use `--network=none` which creates a loopback interface inside gVisor via `createDefaultLoopbackInterface()`. Set `NCCL_SOCKET_IFNAME=lo` and `NCCL_SHM_DISABLE=1`. Default sandbox networking exposes 0 interfaces without Docker, and `--network=host` blocks checkpoint.

### Pending
- Cross-machine (different physical machine — same mechanism, untested)
- Large models (tested nn.Linear, not LLMs)

## The Multi-GPU Problem

Multiple GPUs communicate through NCCL. If you lock GPUs one at a time, you deadlock — GPU 0 freezes while GPU 1 waits for GPU 0's transfer to complete. `cuCheckpointProcessLock` freezes ALL GPUs owned by a PID atomically. Through nvproxy, all container processes share the sentry PID, so one lock call captures everything.

## Architecture

### Save Path

1. **Save-restore-exec helper** runs inside the sandbox, signals SIGUSR1 to PID 1
2. **App's signal handler** calls `cuCheckpointProcessLock(getpid())` + `cuCheckpointProcessCheckpoint(getpid())`
3. GPU VRAM is copied to host memory. CUDA session terminated. App sets `gpu_locked = 1`.
4. **gVisor** serializes everything to disk (process memory now includes GPU state)
5. Sentry exits (or stays alive with `--leave-running`)

### Restore Path

6. New sentry starts, opens fresh `/dev/nvidia*` FDs
7. gVisor loads checkpoint — process memory restored (including GPU state in host RAM)
8. Save-restore-exec helper signals SIGUSR2 to PID 1
9. App calls `cuCheckpointProcessRestore(getpid())` — reads from host memory, creates new CUDA session
10. App calls `cuCheckpointProcessUnlock(getpid())` — resumes GPU work
11. App continues — GPU memory intact

### Why Cold Restore Works

The GPU state is in the **process's host memory** after checkpoint, not in the NVIDIA driver session. When the sentry dies and a new one starts, the driver session is gone — but it doesn't matter. `cuCheckpointProcessRestore` creates a new session from the host memory data. Verified: RSS grows by 519MB after checkpointing 64MB of GPU data.

## Research Findings

Exhaustive testing across 25 Lambda Labs instances. See [RESEARCH.md](RESEARCH.md) for the full log.

### What We Proved

| Finding | Evidence |
|---------|----------|
| GPU state goes to host memory after checkpoint | RSS +519MB for 64MB GPU data |
| Cold restore works (new sentry) | restore=0, pattern verified, 15 ticks |
| `--leave-running` works for same-machine | restore=0, single + multi-GPU + PyTorch |
| V100 (Volta) not supported by nvproxy | cuInit returns CUDA_ERROR_NO_DEVICE |
| CRIU can restore CUDA processes | restore=0, 0xDEADBEEF verified |
| CRIU cannot dump the gVisor sentry | CLONE_VM children block CRIU |
| Cross-process restore fails (401) | Tested: SCM_RIGHTS, TID match, persistence mode, migration API |
| Raw runsc needs full nvidia lib stack | dlopen chain: ptxjitcompiler, gpucomp, etc. |
| Docker rootfs cleanup blocks cold restore via Docker | Use raw runsc or Modal-style rootfs management |

### Driver Session Binding (Academic Interest)

The NVIDIA driver binds checkpoint data to process identity. Cross-process restore always returns 401:

| Test | Result |
|------|--------|
| Same `struct file` via SCM_RIGHTS | restore=401 |
| Same TID via ns_last_pid | restore=401 |
| Persistence mode ON | restore=401 |
| Driver 580 migration API | restore=401 |
| CRIU (same PID + task_struct) | restore=0 ✓ |

This is irrelevant for the working solution since `cuCheckpointProcessRestore` reads from host memory in the same process, not cross-process.

## Files

| File | Purpose |
|------|---------|
| `gvisor-nvproxy-checkpoint.patch` | Base gVisor patch: FD reopen, stateify annotations, PMA dropping |
| `cmd/gvisor-gpu-ckpt/main.go` | Helper binary: 3 checkpoint modes (direct, signal, hybrid) |
| `cmd/gvisor-gpu-ckpt/cuda.go` | Runtime bindings to NVIDIA's cuda-checkpoint API via `dlopen` |
| `test/gpu_checkpoint_test.c` | Working checkpoint test (signal-based, used for all verified tests) |
| `test/gpu_checkpoint_test_single.c` | Single-GPU variant |
| `test/cold_restore_test.sh` | Cold restore test script (raw runsc, no Docker) |
| `test/torch_checkpoint_test.py` | PyTorch model checkpoint test |
| `test/host_mem_test.c` | Proves GPU state goes to host memory (+519MB RSS) |
| `test/criu_positive_test.c` | CRIU restore proof (restore=0, 0xDEADBEEF) |
| `test/tid_experiment.c` | TID matching experiment (restore=401) |
| `test/fd_pass_test.c` | SCM_RIGHTS experiment (restore=401) |
| `test/cross_proc_migration.c` | Driver 580 migration experiment (restore=401) |
| `test/apply_dma_save.py` | BAR DMA save patch for gVisor |
| `test/apply_rm_replay.py` | RM object replay patch for gVisor |
| `pkg/sentry/devices/nvproxy/` | nvproxy implementation guides for RM replay, mmap fields |
| `pkg/sentry/mm/` | BAR DMA save implementation guide |
| `test/gpu/multi_gpu_ckpt_test.cu` | Multi-GPU VRAM pattern verification test (no NCCL) |
| `test/gpu/nccl_ckpt_test.cu` | Multi-GPU NCCL AllReduce + pattern verification test |
| `RESEARCH.md` | Complete research log: all experiments, findings, architecture |

## Checkpoint Modes

The helper binary supports three modes via `GVISOR_GPU_CHECKPOINT_MODE` env var:

| Mode | How it works | When to use |
|------|-------------|-------------|
| `direct` (default) | Helper calls `cuCheckpointProcess*` directly on PID 1 | Single/multi-GPU, with or without NCCL (no CUDA graphs) |
| `signal` | Helper sends SIGUSR1/SIGUSR2 to PID 1, app handles everything | App needs full control (e.g., custom NCCL lifecycle) |
| `hybrid` | Helper locks GPUs atomically, then signals app | Multi-GPU + NCCL + CUDA graphs (app handles NCCL suspend/resume) |

Direct mode handles all GPU contexts for a PID in one atomic call. No per-GPU iteration needed. Works transparently even with active NCCL communicators.

### Known Limitation

CUDA graphs + NCCL + NVLink causes a segfault in `cuCheckpointProcessCheckpoint` (NVIDIA driver bug). Without CUDA graphs, direct mode works for everything.

## Network Mode for NCCL

NCCL needs a socket interface for bootstrap (`getifaddrs()` must find at least one). gVisor's network modes interact with checkpoint:

| Network mode | NCCL works? | Checkpoint works? | Why |
|-------------|-------------|-------------------|-----|
| `--network=none` | Yes | Yes | Creates loopback explicitly via `createDefaultLoopbackInterface()` |
| `--network=sandbox` (default) | No | Yes | Reads host netns — empty without Docker, so 0 interfaces |
| `--network=host` | Yes | No | Uses kernel networking (hostinet), not serializable |

For NCCL workloads, set these env vars in your OCI config:
```json
{
  "env": [
    "NCCL_SOCKET_IFNAME=lo",
    "NCCL_SHM_DISABLE=1"
  ]
}
```

## Requirements

- NVIDIA driver 570+ (cuda-checkpoint API support)
- gVisor with nvproxy enabled
- Turing+ GPU (T4, A100, H100 — NOT V100/Volta)
- NVIDIA persistence mode (`nvidia-smi -pm 1`)
- Linux
- For raw runsc: full NVIDIA library stack in rootfs
- For NCCL: `--network=none`, `NCCL_SOCKET_IFNAME=lo`, `NCCL_SHM_DISABLE=1`

## Usage

```bash
# Build helper binary (must be built on Linux with CGO)
cd cmd/gvisor-gpu-ckpt && go build -o gvisor-gpu-ckpt .

# Apply gVisor patch
cd /path/to/gvisor && git apply /path/to/gvisor-nvproxy-checkpoint.patch

# Start container with GPU
# Use --network=none for NCCL workloads
runsc --nvproxy --nvproxy-driver-version=580.105.08 \
  --network=none \
  run --bundle /path/to/bundle $CONTAINER_ID

# Checkpoint (lock GPUs, save state, sentry can exit)
runsc checkpoint \
  --save-restore-exec-argv=/bin/gvisor-gpu-ckpt \
  --image-path=/tmp/checkpoint \
  $CONTAINER_ID

# Cold restore (new sentry, GPU state from host memory)
runsc --nvproxy --nvproxy-driver-version=580.105.08 \
  --network=none \
  restore \
  --image-path=/tmp/checkpoint \
  --bundle=/path/to/bundle \
  $RESTORED_CONTAINER_ID
```

PR: [google/gvisor#13230](https://github.com/google/gvisor/pull/13230)
