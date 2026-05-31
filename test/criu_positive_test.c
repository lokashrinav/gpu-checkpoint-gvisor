/*
 * CRIU + CUDA checkpoint/restore positive test.
 * Proves that CRIU-style process reconstruction satisfies the NVIDIA
 * driver's restore-thread requirement.
 *
 * Tested on 2x H100 SXM5, driver 580.105.08, CRIU 4.2.
 * Result: RESTORE=0, GPU memory 0xDEADBEEF verified after cold restore.
 *
 * Usage:
 *   sudo nvidia-smi -pm 1
 *   gcc criu_positive_test.c -o criu_test -ldl -lnvidia-ml
 *   sudo ./criu_test &
 *   PID=$!
 *   # Wait for "CHECKPOINTED" in output
 *   sudo mkdir -p /tmp/criu_imgs
 *   sudo criu dump --tree $PID --images-dir /tmp/criu_imgs --shell-job
 *   sudo criu restore --images-dir /tmp/criu_imgs --shell-job -d
 *   sudo kill -USR1 $PID
 *   # Output: RESTORE=0, val=0xDEADBEEF, match=YES, CRIU_RESTORE_SUCCESS
 *
 * Build: gcc criu_positive_test.c -o criu_test -ldl -lnvidia-ml
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <nvml.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef CUresult (*Fn)(int, void*);

static CUresult (*p_cuInit)(unsigned);
static CUresult (*p_cuDeviceGetCount)(int*);
static CUresult (*p_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuMemAlloc)(CUdeviceptr*, size_t);
static CUresult (*p_cuMemsetD32)(CUdeviceptr, unsigned, size_t);
static CUresult (*p_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
static Fn p_lock, p_ckpt, p_restore, p_unlock;

static volatile sig_atomic_t do_restore = 0;
static CUdeviceptr saved_dptr;

void on_sigusr1(int s) { do_restore = 1; }

int main() {
    signal(SIGUSR1, on_sigusr1);
    unsetenv("CUDA_VISIBLE_DEVICES");

    nvmlInit();
    nvmlDevice_t nvd; nvmlEnableState_t pm;
    nvmlDeviceGetHandleByIndex(0, &nvd);
    nvmlDeviceGetPersistenceMode(nvd, &pm);

    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    p_cuInit = dlsym(h, "cuInit");
    p_cuDeviceGetCount = dlsym(h, "cuDeviceGetCount");
    p_cuDevicePrimaryCtxRetain = dlsym(h, "cuDevicePrimaryCtxRetain");
    p_cuCtxSetCurrent = dlsym(h, "cuCtxSetCurrent");
    p_cuMemAlloc = dlsym(h, "cuMemAlloc_v2");
    p_cuMemsetD32 = dlsym(h, "cuMemsetD32_v2");
    p_cuMemcpyDtoH = dlsym(h, "cuMemcpyDtoH_v2");
    p_lock = dlsym(h, "cuCheckpointProcessLock");
    p_ckpt = dlsym(h, "cuCheckpointProcessCheckpoint");
    p_restore = dlsym(h, "cuCheckpointProcessRestore");
    p_unlock = dlsym(h, "cuCheckpointProcessUnlock");

    CUresult r = p_cuInit(0);
    fprintf(stderr, "cuInit=%d pid=%d tid=%d persistence=%s\n",
            r, getpid(), (int)syscall(SYS_gettid), pm==1?"ON":"OFF");
    if (r) return 1;

    int ndev; p_cuDeviceGetCount(&ndev);
    fprintf(stderr, "gpus=%d\n", ndev);

    CUcontext ctx;
    p_cuDevicePrimaryCtxRetain(&ctx, 0);
    p_cuCtxSetCurrent(ctx);
    p_cuMemAlloc(&saved_dptr, 4*1024*1024);
    p_cuMemsetD32(saved_dptr, 0xDEADBEEF, 1024*1024);
    fprintf(stderr, "alloc+pattern done, dptr=%llx\n", (unsigned long long)saved_dptr);

    char a[64] = {0};
    r = p_lock(getpid(), a);
    fprintf(stderr, "lock=%d\n", r);
    memset(a, 0, 64);
    r = p_ckpt(getpid(), a);
    fprintf(stderr, "ckpt=%d\n", r);

    printf("CHECKPOINTED\n");
    fflush(stdout);

    fprintf(stderr, "waiting for CRIU dump + restore + SIGUSR1...\n");
    while (!do_restore) pause();

    fprintf(stderr, "RESTORED! pid=%d tid=%d\n", getpid(), (int)syscall(SYS_gettid));

    memset(a, 0, 64);
    fprintf(stderr, "restore(%d)...\n", getpid());
    r = p_restore(getpid(), a);
    fprintf(stderr, "RESTORE=%d\n", r);

    if (r == 0) {
        memset(a, 0, 64);
        r = p_unlock(getpid(), a);
        fprintf(stderr, "unlock=%d\n", r);

        unsigned int val = 0;
        r = p_cuMemcpyDtoH(&val, saved_dptr, sizeof(unsigned int));
        fprintf(stderr, "memcpy=%d val=0x%X expected=0xDEADBEEF match=%s\n",
                r, val, val == 0xDEADBEEF ? "YES" : "NO");
        if (val == 0xDEADBEEF) printf("CRIU_RESTORE_SUCCESS\n");
    } else {
        printf("CRIU_RESTORE_FAILED_%d\n", r);
    }

    return r;
}
