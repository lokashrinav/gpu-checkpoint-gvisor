# Multi-GPU Checkpoint/Restore for gVisor

Checkpoint and restore GPU containers in gVisor without losing GPU state. Supports multiple GPUs with active NCCL communication. All GPUs lock atomically. No deadlocks.

## The Multi-GPU Problem

Multiple GPUs in a container communicate constantly through NCCL (NVIDIA's multi-GPU communication library). GPU 0 sends data to GPU 1. GPU 1 sends to GPU 2. These transfers run continuously during training and inference.

To checkpoint a GPU, you first lock the GPU. Locking freezes all active work so the state is consistent. If you lock GPUs one at a time, you deadlock. You lock GPU 0. GPU 1 is receiving a transfer from GPU 0. GPU 1 refuses to lock until the transfer completes. The transfer never completes because GPU 0 is frozen. Neither GPU makes progress.

With 8 GPUs all communicating with each other, this deadlock is near-certain.

You need all GPUs frozen at the same instant before you checkpoint any of them.

## The Solution

gVisor runs one sentry process per container. The sentry process owns every GPU the container uses. A container with 8 H100s has one sentry PID holding all 8 GPU contexts.

NVIDIA's `cuCheckpointProcessLock` takes a PID and locks every GPU context owned by the PID in a single atomic operation:

```
cuCheckpointProcessLock(sentryPID)
  locks GPU 0, GPU 1, GPU 2, ... GPU 7 at once
  no window where some GPUs are locked and others are not
  NCCL transfers do not deadlock because nothing is partially frozen
```

One call. All GPUs. Atomic. Single-GPU is the trivial case of the same mechanism.

## Background

GPU programs use CUDA to talk to the GPU. Under the hood, CUDA opens device files (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) and sends commands to the NVIDIA kernel driver through `ioctl` system calls. The driver talks to the GPU hardware. The program never touches the GPU directly.

gVisor sandboxes containers by intercepting all system calls. The container talks to gVisor's userspace kernel (the "sentry") instead of the real Linux kernel. For GPU access, a component called nvproxy intercepts GPU-related ioctls from the container and forwards them to the real NVIDIA driver on the host. The container thinks the GPU is local. Every command is proxied.

gVisor already checkpoints CPU containers. GPU containers crash immediately. Every GPU-related save/restore method in nvproxy was `panic("not implemented")`. Three types of GPU state cause the crash:

1. Host file descriptors. nvproxy holds open file descriptors to `/dev/nvidia0` on the host. These handles belong to the host kernel. They are gone after checkpoint.

2. GPU memory mappings. GPU device memory is mapped into the container's address space through special memory regions. gVisor's serializer only handles regular RAM.

3. GPU-side state. Model weights, CUDA contexts, and execution state live on the GPU hardware. The sentry has no way to read them.

## Full Checkpoint/Restore Flow

### Saving a Checkpoint

Step 1: Freeze GPU state. gVisor invokes a helper binary (`gvisor-gpu-ckpt`) inside the container. The binary calls `cuCheckpointProcessLock` then `cuCheckpointProcessCheckpoint`. The NVIDIA driver snapshots all GPU state (memory contents, CUDA contexts, streams) for every GPU the process owns. All GPUs freeze atomically.

Step 2: Drop GPU memory mappings. gVisor walks the container's memory map and removes memory regions backed by GPU device memory. These regions are not serializable. The GPU memory contents were already captured in step 1. The mappings are rebuilt after restore.

Step 3: Serialize everything else. gVisor's serialization framework writes the sentry's state to disk: process memory, file descriptor tables, nvproxy's internal bookkeeping, and the GPU state snapshot from step 1.

### Restoring from a Checkpoint

Step 1: Deserialize sentry state. gVisor reads the checkpoint file and rebuilds the sentry's in-memory state. The file descriptor table says "fd 3 was /dev/nvidia0" but there is no connection to any GPU. The old host's kernel objects are gone.

Step 2: Reopen device files. nvproxy's restore code opens `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm` on the new host. This produces fresh file descriptors backed by fresh kernel objects. The code re-registers for event notifications and updates memory-mapping handles to point at the new devices.

Step 3: Restore GPU state. gVisor invokes the helper binary again. The binary calls `cuCheckpointProcessRestore` and `cuCheckpointProcessUnlock`. The NVIDIA driver loads the GPU state snapshot back onto every GPU. Memory contents, CUDA contexts, and execution state are restored.

Step 4: Rebuild GPU memory mappings on demand. When the container resumes and touches GPU memory, the access triggers a page fault. gVisor's page fault handler maps the memory against the newly opened device file. The container continues running.

## The nvproxy Routing Constraint

The cuda-checkpoint API must be called from inside the gVisor sandbox, not from the host. This was not documented anywhere.

The sentry creates GPU contexts through raw ioctl forwarding. The sentry never loads NVIDIA's userspace library (`libcuda.so`). From the host's perspective, the sentry PID has no CUDA contexts. Calling `cuCheckpointProcessLock(sentryPID)` from the host fails with `CUDA_ERROR_NOT_INITIALIZED`.

From inside the sandbox, ioctls route through nvproxy to the real driver. The driver resolves contexts correctly through this path. The helper binary runs inside the container and targets PID 1 (the container's init process).

## Files Changed

| File | What the file does |
|------|-------------|
| `pkg/sentry/devices/nvproxy/save_restore_impl.go` | Replaces 6 panic stubs with FD reopen, event re-registration, and memory handle updates |
| `pkg/sentry/mm/save_restore.go` | Drops GPU-backed memory regions before serialization |
| `pkg/sentry/devices/nvproxy/object.go` | Adds serialization annotations, marks host-specific fields as non-savable |
| `cmd/gvisor-gpu-ckpt/main.go` | Helper binary entry point and mode dispatch |
| `cmd/gvisor-gpu-ckpt/cuda.go` | Runtime bindings to NVIDIA's cuda-checkpoint API via `dlopen` |

## Usage

```bash
# Build the helper binary
cd cmd/gvisor-gpu-ckpt && go build -o gvisor-gpu-ckpt .

# Apply the gVisor patch
cd /path/to/gvisor && git apply /path/to/gvisor-nvproxy-checkpoint.patch

# Checkpoint
runsc checkpoint \
  --save-restore-exec-argv=/usr/local/bin/gvisor-gpu-ckpt \
  --image-path=/tmp/checkpoint \
  $CONTAINER_ID

# Restore
runsc restore \
  --save-restore-exec-argv=/usr/local/bin/gvisor-gpu-ckpt \
  --image-path=/tmp/checkpoint \
  $CONTAINER_ID
```

## Test Results

Tested on Lambda Labs bare-metal A10 GPU, driver 570.148.08, with a CUDA process holding 228 MiB of GPU device memory.

- Checkpoint: 604 KB kernel state + 9.2 MB process memory serialized
- Restore: State deserialized in 31ms, tasks resumed, GPU memory intact
- cuda-checkpoint API: All four calls (lock, checkpoint, restore, unlock) returned success from inside the gVisor sandbox through nvproxy

## Requirements

- NVIDIA driver 570+ (cuda-checkpoint API support)
- gVisor with nvproxy enabled
- Linux

PR: [google/gvisor#13230](https://github.com/google/gvisor/pull/13230)
