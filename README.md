# Multi-GPU Checkpoint/Restore for gVisor

## Background

gVisor is a container runtime that runs a user-space kernel (called the "sentry") between your application and the host OS. Instead of your container talking directly to the Linux kernel, it talks to gVisor's sentry, which intercepts every system call. This gives you strong isolation — the container never touches the real kernel.

GPU access works through a component called **nvproxy**. When an application inside a gVisor container opens a GPU device (like `/dev/nvidia0`), nvproxy intercepts the request and opens the *real* device on the host on the container's behalf. Every GPU operation — memory allocation, kernel launches, CUDA context creation — goes through nvproxy as a proxy layer. The application thinks it's talking to a GPU directly, but nvproxy is mediating every interaction.

**Checkpoint/restore** means saving a running container's entire state to disk and later resuming it, possibly on a different machine. This is how platforms like Modal offer instant cold starts — instead of booting a container from scratch and reloading a model (which can take 60+ seconds), they restore from a snapshot in under a second.

## The Problem

### Problem 1: gVisor crashes on GPU checkpoint

gVisor has no support for checkpointing containers that use GPUs. Every checkpoint-related method in nvproxy is implemented as `panic("not implemented")` — meaning if you try to save a GPU container, gVisor crashes immediately.

The reason this is hard: when your application opens `/dev/nvidia0`, nvproxy opens the real `/dev/nvidia0` on the host and holds that file descriptor internally. When gVisor's serializer tries to save the container state, it encounters this host file descriptor and doesn't know what to do with it. The file descriptor number (say, FD 47) is just a number that refers to an open file on *this specific host*. It's meaningless on a different machine. The same problem applies to memory-mapped GPU regions and event notification handles — they're all tied to the physical hardware on the current host.

On top of that, two internal data structures are missing the annotations that tell gVisor's serializer how to handle them, so the serializer hits nil pointer errors before it even gets to the file descriptor problems.

### Problem 2: Multi-GPU deadlocks

Even if you fix the serialization crash, containers with multiple GPUs introduce a coordination problem.

When a container uses multiple GPUs for training or inference, the GPUs communicate with each other through NCCL (NVIDIA's multi-GPU communication library). NCCL sets up peer-to-peer channels between GPUs — GPU 0 sends data to GPU 1, GPU 1 sends to GPU 2, and so on. These transfers happen continuously during operation.

To checkpoint a GPU, you first have to lock it — freeze all its active work so the state is consistent. If you try to lock GPUs one at a time, you deadlock. Here's why: you lock GPU 0, but GPU 1 is in the middle of receiving a transfer from GPU 0. GPU 1 can't be locked until that transfer completes. But the transfer can't complete because GPU 0 is already frozen. Neither GPU can make progress. With 8 GPUs all communicating with each other, this deadlock is almost guaranteed.

The only way to avoid this is to lock **all GPUs at the same instant**, before any of them start checkpointing. But atomically locking 8 independent GPUs without hitting the same deadlock seems like a chicken-and-egg problem.

## The Solution

It turns out gVisor's architecture makes this solvable. In gVisor, there is one sentry process per container, and that single process owns every GPU the container uses. If a container has 8 H100s, all 8 GPU contexts belong to one process ID — the sentry's PID.

NVIDIA provides an API called `cuCheckpointProcessLock` that takes a process ID and locks **every GPU context that process owns** in a single atomic operation. Because all 8 GPUs belong to one sentry PID, one call locks all of them simultaneously:

```
cuCheckpointProcessLock(sentryPID)
  → atomically locks GPU 0, GPU 1, GPU 2, ... GPU 7
  → no window where some GPUs are locked and others aren't
  → NCCL transfers can't deadlock because nothing is partially frozen
```

Single-GPU checkpoint is just the 1-GPU case of the same mechanism. There's no special single-GPU path — the same code handles both.

## What I Built

### Fixing the serialization crash (3 files in gVisor)

**File descriptor lifecycle** — When gVisor restores a container on a new host, the old file descriptors for GPU devices are meaningless. I wrote restore logic that reopens `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm` on the new host, re-registers them with gVisor's event notification system, and updates internal references to point at the new devices. This replaces the 6 `panic("not implemented")` methods that previously existed.

**GPU memory handling** — GPU device memory can't be saved by gVisor's serializer because the serializer only knows how to handle regular system memory. I modified the checkpoint process to drop GPU memory regions before saving. After restore, when the application accesses GPU memory again, gVisor's page fault handler detects the missing region and transparently re-creates it by mapping against the reopened device file. The application never knows anything happened.

**Missing annotations** — Two data structures in nvproxy were missing the metadata that tells gVisor's serializer how to process them. Without these annotations, the serializer crashes with nil pointer errors. I added the annotations and marked fields that contain host-specific values (raw memory addresses that won't be valid on a different machine) as non-serializable.

**Restore bug fix** — Found a bug where restoring a container caused an infinite hang. gVisor serializes a linked list that tracks event listeners. On restore, the list is already populated from the checkpoint data. The original code tried to re-insert entries into the list, creating a cycle — any traversal of the list would loop forever. The fix: update the callback pointer but don't re-insert the entry that's already there.

### The GPU checkpoint binary

A standalone Go binary (`gvisor-gpu-ckpt`) that gVisor calls automatically during checkpoint and restore via a command-line hook (`--save-restore-exec-argv`). The binary runs **inside the container's sandbox** and targets PID 1 (the container's init process).

On checkpoint: it calls `cuCheckpointProcessLock` (freeze all GPUs) then `cuCheckpointProcessCheckpoint` (snapshot GPU state).
On restore: it calls `cuCheckpointProcessRestore` (reload GPU state) then `cuCheckpointProcessUnlock` (resume execution).

It loads NVIDIA's CUDA library at runtime using `dlopen`, so there's no compile-time dependency on the NVIDIA driver. The same binary works for 1 GPU or 8 — all GPU contexts route through the single sentry process.

### A non-obvious discovery

The cuda-checkpoint API has to be called from **inside** the gVisor sandbox, not from the host. This wasn't documented anywhere and took significant debugging to figure out.

The reason: the sentry process creates GPU contexts by forwarding raw ioctl system calls to the NVIDIA driver. It never loads NVIDIA's CUDA library (`libcuda.so`) or calls `cuInit`. From the host's perspective, the sentry has no CUDA state — so calling `cuCheckpointProcessLock(sentryPID)` from the host returns an error because the NVIDIA driver can't find any GPU contexts for that process.

But from inside the sandbox, the API calls go through nvproxy, which forwards them to the real NVIDIA driver. The driver resolves the contexts correctly through this path, and all four checkpoint operations succeed.

## Full Checkpoint/Restore Sequence

**Saving a container:**
1. `runsc checkpoint` is called with `--save-restore-exec-argv=gvisor-gpu-ckpt`
2. gVisor runs `gvisor-gpu-ckpt` inside the sandbox with `MODE=save`
3. The binary calls `cuCheckpointProcessLock(1)` — all GPU contexts freeze atomically
4. The binary calls `cuCheckpointProcessCheckpoint(1)` — GPU state is snapshotted
5. gVisor's serializer drops GPU memory regions that can't be saved
6. gVisor writes the full container state to disk (kernel state, process memory, nvproxy objects)

**Restoring a container:**
1. gVisor reads the checkpoint from disk and deserializes the container state
2. Restore callbacks reopen GPU device files on the new host and reconnect internal references
3. GPU memory regions are left empty — they'll be re-created on demand when accessed
4. gVisor runs `gvisor-gpu-ckpt` with `MODE=restore`
5. The binary calls `cuCheckpointProcessRestore(1)` — GPU state is reloaded
6. The binary calls `cuCheckpointProcessUnlock(1)` — execution resumes
7. The container is running with full GPU state, as if nothing happened

## Test Results

Tested on a Lambda Labs bare-metal instance with an A10 GPU, driver 570.148.08. The test ran a CUDA program inside a gVisor container with an active CUDA context and 228 MiB of allocated GPU device memory.

```
# cuda-checkpoint API (from inside gVisor container):
cuCheckpointProcessLock(1)       = 0   # success
cuCheckpointProcessCheckpoint(1) = 0   # success
cuCheckpointProcessRestore(1)    = 0   # success
cuCheckpointProcessUnlock(1)     = 0   # success
# Process survived with 228 MiB GPU memory intact

# gVisor checkpoint: 604KB kernel state + 9.2MB process memory serialized
# gVisor restore: deserialized in 31ms, tasks resumed
```

## Repository Structure

```
cmd/gvisor-gpu-ckpt/                   # Standalone binary (compiles independently)
  main.go                              # Entry point, dispatches on MODE env var
  cuda.go                              # cgo wrappers for cuCheckpointProcess* via dlopen

pkg/sentry/devices/nvproxy/            # gVisor source changes (apply to gVisor tree)
  save_restore_impl.go                 # FD lifecycle — replaces 6 panic() with real logic
  object.go                            # Serialization annotations for nvproxy structs

pkg/sentry/mm/                         # gVisor source changes (apply to gVisor tree)
  save_restore.go                      # Drops GPU memory regions before checkpoint

gvisor-nvproxy-checkpoint.patch        # All gVisor changes as a single git patch
```

## Usage

```bash
# Build the binary (requires Go 1.21+, Linux, libcuda.so.1 at runtime)
cd cmd/gvisor-gpu-ckpt && go build -o gvisor-gpu-ckpt .

# Apply the gVisor patch
cd /path/to/gvisor && git apply /path/to/gvisor-nvproxy-checkpoint.patch

# Checkpoint a GPU container
runsc checkpoint --save-restore-exec-argv=/usr/local/bin/gvisor-gpu-ckpt \
  --image-path=/tmp/checkpoint $CONTAINER_ID

# Restore
runsc restore --save-restore-exec-argv=/usr/local/bin/gvisor-gpu-ckpt \
  --image-path=/tmp/checkpoint $CONTAINER_ID
```

## Requirements

- NVIDIA driver 570+ (cuda-checkpoint API support)
- gVisor with nvproxy enabled
- Linux

PR: [google/gvisor#13230](https://github.com/google/gvisor/pull/13230)
