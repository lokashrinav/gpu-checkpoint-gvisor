# Multi-GPU Checkpoint/Restore for gVisor

Atomically checkpoint and restore multi-GPU containers running under gVisor — 1 GPU or 8, with active NCCL communication, no deadlocks.

## The Multi-GPU Problem

When a container uses multiple GPUs, they communicate with each other constantly through NCCL (NVIDIA's multi-GPU communication library). GPU 0 is mid-transfer to GPU 1, GPU 1 is sending to GPU 2, and so on. To checkpoint a GPU, you have to lock it — freeze all its active work so the state is consistent.

If you lock GPUs one at a time, you deadlock. You lock GPU 0, but GPU 1 is in the middle of receiving a transfer from GPU 0. GPU 1 can't be locked until that transfer completes. But the transfer can't complete because GPU 0 is already frozen. Neither GPU can make progress. With 8 GPUs all communicating with each other, this deadlock is almost guaranteed.

You need all GPUs frozen at the same instant before you checkpoint any of them. But how do you atomically lock 8 independent GPUs without hitting exactly this problem?

## The Solution

In gVisor, **one sentry process owns all GPUs in a container**. A container with 8 H100s has one sentry PID holding all 8 GPU contexts — not 8 processes with 1 GPU each.

NVIDIA's `cuCheckpointProcessLock` takes a PID and locks **every GPU context that PID owns** in a single atomic call:

```
cuCheckpointProcessLock(sentryPID)
  → atomically locks GPU 0, GPU 1, GPU 2, ... GPU 7
  → no window where some GPUs are locked and others aren't
  → NCCL transfers can't deadlock because nothing is partially frozen
```

One call. All GPUs. Atomic. Single-GPU is just the trivial case.

## Background

To understand the rest of the implementation, you need to know how GPU access works inside gVisor.

**How a normal program talks to a GPU:** A program uses CUDA to interact with the GPU. Under the hood, CUDA opens device files (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) and sends commands to the NVIDIA kernel driver through `ioctl` system calls. The kernel driver talks to the GPU hardware. The program never touches the GPU directly — everything goes through file descriptors and ioctls.

**How gVisor changes this:** gVisor sandboxes containers by intercepting all system calls. The container doesn't talk to the real Linux kernel — it talks to gVisor's userspace kernel (the "sentry"). For GPU access, a component called **nvproxy** intercepts GPU-related ioctls from the container and forwards them to the real NVIDIA driver on the host. The container thinks it's talking directly to the GPU, but every command is proxied.

**Why checkpoint was broken:** gVisor can checkpoint CPU containers, but GPU containers crash immediately — every GPU-related save/restore method in nvproxy was `panic("not implemented")`. GPU state spans multiple layers the sentry doesn't fully control:

1. **Host file descriptors.** nvproxy holds open file descriptors to the real `/dev/nvidia0` on the host. These handles are managed by the host kernel — they're gone after checkpoint.

2. **GPU memory mappings.** GPU device memory is mapped into the container's address space through special memory regions. gVisor's serializer only handles regular RAM, not GPU-backed memory.

3. **GPU-side state.** Model weights, CUDA contexts, execution state — all of this lives on the GPU hardware. The sentry can't reach in and pull it out.

## Full Checkpoint/Restore Flow

### Saving a Checkpoint

**Step 1: Freeze GPU state.** gVisor invokes a helper binary (`gvisor-gpu-ckpt`) inside the container. This binary calls `cuCheckpointProcessLock` followed by `cuCheckpointProcessCheckpoint`, which tells the NVIDIA driver to snapshot all GPU state (memory contents, CUDA contexts, streams) for every GPU the process owns — atomically.

**Step 2: Drop GPU memory mappings.** gVisor walks the container's memory map and removes any memory regions backed by GPU device memory. These regions can't be serialized, but they don't need to be — the GPU memory contents were already captured in step 1, and the mappings can be rebuilt after restore.

**Step 3: Serialize everything else.** gVisor's serialization framework writes the rest of the sentry's state to disk: process memory, file descriptor tables, nvproxy's internal bookkeeping, and the GPU state snapshot from step 1.

### Restoring from a Checkpoint

**Step 1: Deserialize sentry state.** gVisor reads the checkpoint file and rebuilds the sentry's in-memory state. At this point, the file descriptor table says "fd 3 was /dev/nvidia0" — but there's no actual connection to any GPU. The old host's kernel objects are gone.

**Step 2: Reopen device files.** nvproxy's restore code opens `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm` on the new host, getting fresh file descriptors backed by fresh kernel objects. It re-registers for event notifications and updates memory-mapping handles to point at the new devices.

**Step 3: Restore GPU state.** gVisor invokes the helper binary again. It calls `cuCheckpointProcessRestore` and `cuCheckpointProcessUnlock`, which tells the NVIDIA driver to load the GPU state snapshot back onto the GPU — restoring memory contents, CUDA contexts, and execution state across all GPUs.

**Step 4: Rebuild GPU memory mappings on demand.** When the container resumes and touches GPU memory, the access triggers a page fault. gVisor's page fault handler detects that this is GPU-backed memory, maps it against the newly opened device file, and the container continues as if nothing happened.

## The nvproxy Routing Constraint

The cuda-checkpoint API must be called from **inside the gVisor sandbox**, not from the host. This wasn't documented anywhere.

The sentry creates GPU contexts through raw ioctl forwarding — it never loads NVIDIA's userspace library (`libcuda.so`). So from the host's perspective, the sentry PID doesn't have CUDA contexts. Calling `cuCheckpointProcessLock(sentryPID)` from the host fails with `CUDA_ERROR_NOT_INITIALIZED`.

From inside the sandbox, ioctls route through nvproxy to the real driver, and the driver resolves contexts correctly. So the helper binary runs inside the container and targets PID 1 (the container's init process).

## Files Changed

| File | What it does |
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

- **Checkpoint:** 604 KB kernel state + 9.2 MB process memory serialized
- **Restore:** State deserialized in 31ms, tasks resumed, GPU memory intact
- **cuda-checkpoint API:** All four calls (lock, checkpoint, restore, unlock) returned success from inside the gVisor sandbox through nvproxy

## Requirements

- NVIDIA driver 570+ (cuda-checkpoint API support)
- gVisor with nvproxy enabled
- Linux

PR: [google/gvisor#13230](https://github.com/google/gvisor/pull/13230)
