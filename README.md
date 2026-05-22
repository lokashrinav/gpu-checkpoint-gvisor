# GPU Checkpoint/Restore for gVisor

Checkpoint and restore GPU containers running under gVisor, supporting single and multi-GPU configurations. This solves two problems that currently prevent GPU checkpoint in gVisor:

1. **gVisor's nvproxy panics on any GPU checkpoint.** Every save/restore method in nvproxy is `panic("not implemented")`. When gVisor tries to serialize a container that has GPU device files open, it hits host file descriptors it can't serialize and crashes.

2. **Multi-GPU checkpoint deadlocks.** If a container has multiple GPUs with active NCCL peer-to-peer channels, locking GPUs sequentially causes deadlocks — GPU 0 freezes mid-transfer to GPU 1, GPU 1 can't complete because GPU 0 is locked.

PR: [google/gvisor#13230](https://github.com/google/gvisor/pull/13230)

## How It Works

### The Multi-GPU Insight

In gVisor, one sentry process owns all GPUs in a container. A container with 8 H100s has one sentry PID holding all 8 GPU contexts. NVIDIA's `cuCheckpointProcessLock` takes a PID and locks every GPU context that PID owns in a single atomic call. No sequential locking, no deadlock window.

### Checkpoint Flow

1. gVisor calls the SaveRestoreExec binary (`gvisor-gpu-ckpt`) inside the sandbox
2. The binary calls `cuCheckpointProcessLock(1)` — locks all GPU contexts atomically
3. The binary calls `cuCheckpointProcessCheckpoint(1)` — snapshots all GPU state
4. gVisor's stateify serializes the container state (nvproxy objects, process memory, kernel state)
5. `InvalidateUnsavable()` drops GPU-backed PMAs that can't be serialized — they get re-created lazily via page faults after restore

### Restore Flow

1. gVisor deserializes the checkpoint state
2. `frontendFD.afterLoadImpl()` reopens `/dev/nvidia0`, `/dev/nvidiactl` on the new host
3. `uvmFD.afterLoadImpl()` reopens `/dev/nvidia-uvm`
4. fdnotifier re-registers, memmapFile updates to new FDs
5. gVisor calls the SaveRestoreExec binary
6. The binary calls `cuCheckpointProcessRestore(1)` — restores GPU state
7. The binary calls `cuCheckpointProcessUnlock(1)` — resumes execution

## Repository Structure

```
cmd/gvisor-gpu-ckpt/          # SaveRestoreExec binary (standalone, compiles independently)
  main.go                     # Entry point — reads MODE env var, dispatches save/restore
  cuda.go                     # cgo wrappers for cuCheckpointProcess* via dlopen

pkg/sentry/devices/nvproxy/   # gVisor changes (apply to gVisor source tree)
  save_restore_impl.go        # FD lifecycle — replaces 6 panic() methods with real implementations
  object.go                   # Added +stateify savable annotations, nosave for host-specific fields

pkg/sentry/mm/                # gVisor changes (apply to gVisor source tree)
  save_restore.go             # PMA invalidation — drops GPU-backed PMAs before save

gvisor-nvproxy-checkpoint.patch  # All changes as a single patch file (git apply)
```

## Usage

### Build the SaveRestoreExec binary

```bash
# Requires Go 1.21+ and a Linux system with libcuda.so.1 at runtime
cd cmd/gvisor-gpu-ckpt
go build -o gvisor-gpu-ckpt .
```

### Apply the gVisor patch

```bash
cd /path/to/gvisor
git apply /path/to/gvisor-nvproxy-checkpoint.patch
```

### Run a GPU checkpoint

```bash
# Start a GPU container with gVisor
docker run --runtime=runsc --gpus all -v /path/to/gvisor-gpu-ckpt:/usr/local/bin/gvisor-gpu-ckpt:ro ...

# Checkpoint
runsc checkpoint \
  --save-restore-exec-argv=/usr/local/bin/gvisor-gpu-ckpt \
  --image-path=/tmp/gpu_checkpoint \
  $CONTAINER_ID

# Restore
runsc restore \
  --save-restore-exec-argv=/usr/local/bin/gvisor-gpu-ckpt \
  --image-path=/tmp/gpu_checkpoint \
  $CONTAINER_ID
```

## Key Technical Details

**Why cuda-checkpoint must run inside the sandbox:** The sentry creates GPU contexts via raw ioctl forwarding without loading libcuda.so. From the host, `cuCheckpointProcessLock(sentryPID)` returns `CUDA_ERROR_NOT_INITIALIZED` (rc=3) because the CUDA driver can't find the sentry's per-process state. From inside the sandbox, the checkpoint ioctls route through nvproxy to the real NVIDIA driver, which resolves the PID mapping correctly.

**Waiter queue fix:** The waiter `Entry` is serialized in the checkpoint. On restore, it's already in the queue's linked list. The original code would re-register it, creating a cycle that causes infinite loops. Fix: call `Init()` to update the callback pointer but skip re-registration.

**GPU memory re-creation:** GPU device memory (PMAs backed by `frontendFDMemmapFile`) can't be serialized by stateify. `InvalidateUnsavable()` drops these PMAs before save. After restore, when the application touches GPU memory, gVisor's page fault handler calls `frontendFD.Translate()`, which mmaps against the reopened device file and re-creates the PMA on demand.

## Test Results

Tested on Lambda Labs bare-metal A10 GPU, driver 570.148.08.

```
# CUDA checkpoint API (from inside gVisor container):
cuCheckpointProcessLock(1)       = 0   # SUCCESS
cuCheckpointProcessCheckpoint(1) = 0   # SUCCESS
cuCheckpointProcessRestore(1)    = 0   # SUCCESS
cuCheckpointProcessUnlock(1)     = 0   # SUCCESS
# Process survived with 228 MiB GPU memory intact

# gVisor checkpoint:
# checkpoint.img: 604KB, pages.img: 9.2MB
# Restore: state deserialized in 31ms, tasks resumed
```

## Requirements

- NVIDIA driver 570+ (cuda-checkpoint API support)
- gVisor with nvproxy enabled
- Linux (the binary uses cgo + dlopen)
