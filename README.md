# Multi-GPU Checkpoint/Restore for gVisor

gVisor cannot checkpoint GPU containers. Every save/restore method in nvproxy calls `panic("not implemented")` — try to checkpoint a container with even one GPU and the sentry crashes. Multi-GPU is worse: NCCL peer-to-peer transfers between GPUs deadlock if you try to lock them one at a time.

This project fixes both. The key contribution is **atomic multi-GPU checkpoint**: all GPUs in a container are locked, checkpointed, and restored in a single operation with no deadlock window.

## Why Multi-GPU Checkpoint is Hard

A container with 8 GPUs has active NCCL channels between them — GPU 0 is mid-transfer to GPU 1, GPU 1 is sending to GPU 2, and so on. If you lock GPU 0 first, GPU 1 is waiting for GPU 0 to finish its transfer before it can be locked. GPU 1 can't complete because GPU 0 is frozen. Deadlock.

You need all GPUs frozen at the same instant before you checkpoint any of them. But how do you atomically lock 8 GPUs without hitting exactly this problem?

## The Solution

In gVisor, **one sentry process owns all GPUs in a container**. A container with 8 H100s has one sentry PID holding all 8 GPU contexts — not 8 processes with 1 GPU each.

NVIDIA's `cuCheckpointProcessLock` takes a PID and locks **every GPU context that PID owns** in a single atomic call:

```
cuCheckpointProcessLock(sentryPID)
  → locks GPU 0, GPU 1, GPU 2, ... GPU 7 — all at once
  → no window where some GPUs are locked and others aren't
  → NCCL can't deadlock because there's no partial state
```

One call. All GPUs. Atomic. Single-GPU is just the trivial case of the same mechanism.

## What I Built

**nvproxy FD lifecycle** (`save_restore_impl.go`) — Replaced 6 `panic("not implemented")` methods with real checkpoint/restore logic. On restore, device files (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) are reopened on the new host, the event notification system re-registers against new FDs, and memory-mapped file handles update to point at the new devices. Found and fixed a bug where the waiter queue entry was double-registered after restore, creating an infinite loop in event notification.

**GPU memory invalidation** (`save_restore.go`) — GPU device memory (PMAs backed by `frontendFDMemmapFile`) can't be serialized by stateify. Modified `InvalidateUnsavable()` to drop these PMAs before save. After restore, when the application touches that memory, gVisor's page fault handler re-creates the PMA through `frontendFD.Translate()` on demand.

**Serialization annotations** (`object.go`) — Two nvproxy structs were missing `+stateify savable` annotations, causing nil pointer panics during serialization. Added annotations and marked host-specific fields (`pinnedRanges`, `m`, `len`) as `nosave`.

**SaveRestoreExec binary** (`gvisor-gpu-ckpt`) — Go binary that gVisor calls via `--save-restore-exec-argv` during checkpoint/restore. On save: Lock → Checkpoint. On restore: Restore → Unlock. Uses `dlopen("libcuda.so.1")` at runtime — no compile-time NVIDIA dependency. Same binary handles 1 GPU or 8.

**Discovered the nvproxy routing constraint** — The cuda-checkpoint API must be called from **inside the sandbox**, not from the host. The sentry creates GPU contexts via raw ioctl forwarding without loading libcuda.so, so `cuCheckpointProcessLock(sentryPID)` from the host returns `CUDA_ERROR_NOT_INITIALIZED`. From inside, the ioctls route through nvproxy to the real driver, and it works.

## Checkpoint/Restore Flow

**Save:**
1. gVisor invokes `gvisor-gpu-ckpt` inside the sandbox with `MODE=save`
2. `cuCheckpointProcessLock(1)` — atomically locks all GPU contexts
3. `cuCheckpointProcessCheckpoint(1)` — snapshots all GPU state (VRAM, CUDA contexts, streams)
4. gVisor's stateify serializes the container (nvproxy state, process memory, kernel state)
5. `InvalidateUnsavable()` drops GPU-backed PMAs before serialization

**Restore:**
1. gVisor deserializes the checkpoint
2. `frontendFD.afterLoadImpl()` reopens `/dev/nvidia0`, `/dev/nvidiactl` on the new host
3. `uvmFD.afterLoadImpl()` reopens `/dev/nvidia-uvm`, re-registers fdnotifier
4. GPU-backed PMAs re-created lazily via page faults
5. gVisor invokes `gvisor-gpu-ckpt` with `MODE=restore`
6. `cuCheckpointProcessRestore(1)` — restores all GPU state
7. `cuCheckpointProcessUnlock(1)` — resumes execution

## Test Results

Tested on Lambda Labs bare-metal A10 GPU, driver 570.148.08, with a CUDA process holding 228 MiB of GPU device memory.

```
# From inside gVisor container (through nvproxy):
cuCheckpointProcessLock(1)       = 0   # SUCCESS
cuCheckpointProcessCheckpoint(1) = 0   # SUCCESS
cuCheckpointProcessRestore(1)    = 0   # SUCCESS
cuCheckpointProcessUnlock(1)     = 0   # SUCCESS

# State transitions: running → locked → checkpointed → running
# Process survived with 228 MiB GPU memory intact

# gVisor checkpoint: 604KB kernel state + 9.2MB process memory
# gVisor restore: state deserialized in 31ms, tasks resumed
```

## Repository Structure

```
cmd/gvisor-gpu-ckpt/                         # Standalone binary (compiles independently)
  main.go                                    # Entry point, MODE dispatch
  cuda.go                                    # cgo wrappers for cuCheckpointProcess*

pkg/sentry/devices/nvproxy/                  # gVisor changes (apply to gVisor tree)
  save_restore_impl.go                       # FD lifecycle for checkpoint/restore
  object.go                                  # stateify annotations

pkg/sentry/mm/                               # gVisor changes (apply to gVisor tree)
  save_restore.go                            # PMA invalidation for GPU memory

gvisor-nvproxy-checkpoint.patch              # All gVisor changes as git patch
```

## Usage

```bash
# Build the binary
cd cmd/gvisor-gpu-ckpt && go build -o gvisor-gpu-ckpt .

# Apply gVisor patch
cd /path/to/gvisor && git apply /path/to/gvisor-nvproxy-checkpoint.patch

# Checkpoint a GPU container
runsc checkpoint --save-restore-exec-argv=/usr/local/bin/gvisor-gpu-ckpt \
  --image-path=/tmp/checkpoint $CONTAINER_ID

# Restore
runsc restore --save-restore-exec-argv=/usr/local/bin/gvisor-gpu-ckpt \
  --image-path=/tmp/checkpoint $CONTAINER_ID
```

## Requirements

- NVIDIA driver 570+ (cuda-checkpoint API)
- gVisor with nvproxy enabled
- Linux
