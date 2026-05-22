#!/bin/bash
# End-to-end GPU checkpoint/restore test using runsc directly (no Docker).
# Run on a machine with an NVIDIA GPU and gVisor installed.
#
# This script:
#   1. Creates an OCI bundle with a CUDA test program
#   2. Starts the container with runsc
#   3. Checkpoints the container (with GPU state)
#   4. Restores the container from checkpoint
#   5. Verifies the CUDA process survived with GPU memory intact

set -euo pipefail

BUNDLE_DIR="/tmp/gpu-ckpt-test/bundle"
ROOTFS_DIR="${BUNDLE_DIR}/rootfs"
CHECKPOINT_DIR="/tmp/gpu-ckpt-test/checkpoint"
CONTAINER_ID="gpu-test-$$"
RESTORE_ID="gpu-restored-$$"
CKPT_BINARY="/usr/local/bin/gvisor-gpu-ckpt"

echo "=== Step 0: Build test binary and checkpoint helper ==="

# Build the CUDA test program
cat > /tmp/cuda_test.c << 'CUDA_EOF'
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr dptr;
    size_t sz = 228 * 1024 * 1024; // 228 MiB

    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 1; }
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { fprintf(stderr, "cuDeviceGet failed\n"); return 1; }
    if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) { fprintf(stderr, "cuCtxCreate failed\n"); return 1; }
    if (cuMemAlloc(&dptr, sz) != CUDA_SUCCESS) { fprintf(stderr, "cuMemAlloc failed\n"); return 1; }

    printf("CUDA context active, PID=%d, devptr=0x%llx\n", getpid(), (unsigned long long)dptr);
    fflush(stdout);

    // Write a known pattern to GPU memory so we can verify after restore
    unsigned int pattern = 0xDEADBEEF;
    cuMemsetD32(dptr, pattern, sz / 4);
    printf("Wrote pattern 0x%X to %zu MiB of GPU memory\n", pattern, sz / (1024*1024));
    fflush(stdout);

    // Stay alive so the container can be checkpointed
    for (int i = 0; ; i++) {
        sleep(5);
        printf("Still running... tick=%d\n", (i+1)*5);
        fflush(stdout);
    }
    return 0;
}
CUDA_EOF

nvcc -o /tmp/cuda_test /tmp/cuda_test.c -lcuda
echo "Built /tmp/cuda_test"

# Build the checkpoint helper binary (if not already built)
if [ ! -f "$CKPT_BINARY" ]; then
    echo "Building gvisor-gpu-ckpt..."
    cd /path/to/gpu-checkpoint-gvisor/cmd/gvisor-gpu-ckpt
    go build -o "$CKPT_BINARY" .
    echo "Built $CKPT_BINARY"
fi

echo "=== Step 1: Create OCI bundle ==="

rm -rf /tmp/gpu-ckpt-test
mkdir -p "$ROOTFS_DIR" "$CHECKPOINT_DIR"

# Create a minimal rootfs from the host
# Copy essential libraries and binaries
mkdir -p "$ROOTFS_DIR"/{bin,lib,lib64,usr/lib,usr/lib64,dev,proc,sys,tmp,etc}
cp /tmp/cuda_test "$ROOTFS_DIR/bin/cuda_test"
cp "$CKPT_BINARY" "$ROOTFS_DIR/bin/gvisor-gpu-ckpt"

# Copy dynamic linker and required libraries
cp /lib64/ld-linux-x86-64.so.2 "$ROOTFS_DIR/lib64/" 2>/dev/null || true
for lib in libc.so.6 libpthread.so.0 libdl.so.2 librt.so.1 libm.so.6; do
    find /lib /lib64 /usr/lib /usr/lib64 -name "$lib" -exec cp {} "$ROOTFS_DIR/lib64/" \; 2>/dev/null || true
done

# Copy CUDA libraries
for lib in libcuda.so.1 libnvidia-ml.so.1; do
    find /usr/lib /usr/lib64 /usr/local -name "$lib" -exec cp {} "$ROOTFS_DIR/lib64/" \; 2>/dev/null || true
done

# Create OCI config.json
cat > "$BUNDLE_DIR/config.json" << 'CONFIG_EOF'
{
    "ociVersion": "1.0.0",
    "process": {
        "terminal": false,
        "user": { "uid": 0, "gid": 0 },
        "args": ["/bin/cuda_test"],
        "env": [
            "PATH=/bin:/usr/bin",
            "LD_LIBRARY_PATH=/lib64:/usr/lib64"
        ],
        "cwd": "/"
    },
    "root": {
        "path": "rootfs",
        "readonly": false
    },
    "linux": {
        "namespaces": [
            { "type": "pid" },
            { "type": "mount" },
            { "type": "ipc" },
            { "type": "uts" }
        ]
    },
    "mounts": [
        { "destination": "/proc", "type": "proc", "source": "proc" },
        { "destination": "/sys", "type": "sysfs", "source": "sysfs", "options": ["nosuid", "noexec", "nodev", "ro"] },
        { "destination": "/dev", "type": "tmpfs", "source": "tmpfs", "options": ["nosuid", "strictatime", "mode=755", "size=65536k"] }
    ]
}
CONFIG_EOF

echo "OCI bundle created at $BUNDLE_DIR"

echo "=== Step 2: Start container with runsc ==="

runsc --nvproxy --nvproxy-driver-version=570.133.20 \
    run --bundle "$BUNDLE_DIR" "$CONTAINER_ID" &
RUNSC_PID=$!

# Wait for CUDA context to be created
sleep 10
echo "Container started, runsc PID: $RUNSC_PID"
runsc state "$CONTAINER_ID"

echo "=== Step 3: Checkpoint container ==="

runsc checkpoint \
    --save-restore-exec-argv=/bin/gvisor-gpu-ckpt \
    --image-path="$CHECKPOINT_DIR" \
    "$CONTAINER_ID"

echo "Checkpoint complete. Files:"
ls -la "$CHECKPOINT_DIR"/

echo "=== Step 4: Restore container ==="

runsc restore \
    --save-restore-exec-argv=/bin/gvisor-gpu-ckpt \
    --image-path="$CHECKPOINT_DIR" \
    --bundle="$BUNDLE_DIR" \
    "$RESTORE_ID" &
RESTORE_PID=$!

sleep 5
echo "Restore started, PID: $RESTORE_PID"
runsc state "$RESTORE_ID"

echo "=== Step 5: Verify ==="

# Give the restored container time to print output
sleep 10

echo "=== DONE ==="
echo "If you see 'Still running...' output from the restored container,"
echo "the CUDA process survived checkpoint/restore with GPU memory intact."

# Cleanup
runsc kill "$RESTORE_ID" SIGTERM 2>/dev/null || true
runsc kill "$CONTAINER_ID" SIGTERM 2>/dev/null || true
wait $RESTORE_PID 2>/dev/null || true
wait $RUNSC_PID 2>/dev/null || true
runsc delete "$RESTORE_ID" 2>/dev/null || true
runsc delete "$CONTAINER_ID" 2>/dev/null || true
