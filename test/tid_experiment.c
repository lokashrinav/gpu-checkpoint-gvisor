/*
 * TID experiment: can we restore CUDA checkpoint from a new process
 * by calling restore from a thread with the same TID as the original
 * restore thread?
 *
 * Process A: checkpoint, save restore_thread_id, exit
 * Process B: create thread at that TID, call restore from it
 *
 * Build: gcc tid_experiment.c -o tid_experiment -ldl -lnvidia-ml -lpthread
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <nvml.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGetCount_t)(int*);
typedef CUresult (*cuDevicePrimaryCtxRetain_t)(CUcontext*, CUdevice);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemsetD32_t)(CUdeviceptr, unsigned int, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuCheckpointProcessLock_t)(int, void*);
typedef CUresult (*cuCheckpointProcessCheckpoint_t)(int, void*);
typedef CUresult (*cuCheckpointProcessRestore_t)(int, void*);
typedef CUresult (*cuCheckpointProcessUnlock_t)(int, void*);
/* 580 API for getting restore thread ID */
typedef CUresult (*cuCheckpointProcessGetRestoreThreadId_t)(int, int*);

static void *cuda_handle;
static cuInit_t p_cuInit;
static cuDeviceGetCount_t p_cuDeviceGetCount;
static cuDevicePrimaryCtxRetain_t p_cuDevicePrimaryCtxRetain;
static cuCtxSetCurrent_t p_cuCtxSetCurrent;
static cuMemAlloc_t p_cuMemAlloc;
static cuMemsetD32_t p_cuMemsetD32;
static cuMemcpyDtoH_t p_cuMemcpyDtoH;
static cuCheckpointProcessLock_t p_lock;
static cuCheckpointProcessCheckpoint_t p_ckpt;
static cuCheckpointProcessRestore_t p_restore;
static cuCheckpointProcessUnlock_t p_unlock;
static cuCheckpointProcessGetRestoreThreadId_t p_getRestoreTid;

static void load_cuda() {
    cuda_handle = dlopen("libcuda.so.1", RTLD_NOW);
    p_cuInit = dlsym(cuda_handle, "cuInit");
    p_cuDeviceGetCount = dlsym(cuda_handle, "cuDeviceGetCount");
    p_cuDevicePrimaryCtxRetain = dlsym(cuda_handle, "cuDevicePrimaryCtxRetain");
    p_cuCtxSetCurrent = dlsym(cuda_handle, "cuCtxSetCurrent");
    p_cuMemAlloc = dlsym(cuda_handle, "cuMemAlloc_v2");
    p_cuMemsetD32 = dlsym(cuda_handle, "cuMemsetD32_v2");
    p_cuMemcpyDtoH = dlsym(cuda_handle, "cuMemcpyDtoH_v2");
    p_lock = dlsym(cuda_handle, "cuCheckpointProcessLock");
    p_ckpt = dlsym(cuda_handle, "cuCheckpointProcessCheckpoint");
    p_restore = dlsym(cuda_handle, "cuCheckpointProcessRestore");
    p_unlock = dlsym(cuda_handle, "cuCheckpointProcessUnlock");
    p_getRestoreTid = dlsym(cuda_handle, "cuCheckpointProcessGetRestoreThreadId");
}

struct restore_thread_args {
    int target_tid;
    int result;
};

static void *restore_thread_fn(void *arg) {
    struct restore_thread_args *a = arg;
    pid_t my_tid = syscall(SYS_gettid);
    fprintf(stderr, "restore thread: my_tid=%d, target_tid=%d, match=%s\n",
            my_tid, a->target_tid, my_tid == a->target_tid ? "YES" : "NO");

    /* Try restore from this thread */
    char args[64]; memset(args, 0, 64);
    pid_t self = getpid();
    fprintf(stderr, "restore thread: calling cuCheckpointProcessRestore(pid=%d)...\n", self);
    fflush(stderr);
    a->result = p_restore(self, args);
    fprintf(stderr, "restore thread: restore=%d\n", a->result);

    if (a->result == 0) {
        char ua[64]; memset(ua, 0, 64);
        int ur = p_unlock(self, ua);
        fprintf(stderr, "restore thread: unlock=%d\n", ur);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s save|restore\n", argv[0]); return 1; }

    unsetenv("CUDA_VISIBLE_DEVICES");
    unsetenv("CUDA_DEVICE_ORDER");

    /* Enable persistence mode */
    nvmlInit();
    nvmlDevice_t nvd; nvmlEnableState_t pm;
    nvmlDeviceGetHandleByIndex(0, &nvd);
    nvmlDeviceGetPersistenceMode(nvd, &pm);
    printf("persistence=%s\n", pm == 1 ? "ON" : "OFF");

    load_cuda();

    CUresult r = p_cuInit(0);
    printf("cuInit=%d pid=%d tid=%d\n", r, getpid(), (int)syscall(SYS_gettid));
    if (r) return 1;

    if (strcmp(argv[1], "save") == 0) {
        CUcontext ctx;
        p_cuDevicePrimaryCtxRetain(&ctx, 0);
        p_cuCtxSetCurrent(ctx);
        CUdeviceptr dptr;
        p_cuMemAlloc(&dptr, 4*1024*1024);
        p_cuMemsetD32(dptr, 0xDEADBEEF, 1024*1024);
        printf("alloc on GPU 0, dptr=%llx\n", (unsigned long long)dptr);

        char la[64]; memset(la, 0, 64);
        char ca[64]; memset(ca, 0, 64);
        pid_t self = getpid();

        printf("lock(%d)...\n", self);
        r = p_lock(self, la);
        printf("lock=%d\n", r);

        printf("checkpoint(%d)...\n", self);
        r = p_ckpt(self, ca);
        printf("checkpoint=%d\n", r);

        /* Get restore thread ID */
        int restore_tid = -1;
        if (p_getRestoreTid) {
            r = p_getRestoreTid(self, &restore_tid);
            printf("getRestoreThreadId=%d tid=%d\n", r, restore_tid);
        } else {
            printf("cuCheckpointProcessGetRestoreThreadId not found in driver\n");
        }

        /* Save restore TID to file */
        FILE *f = fopen("/tmp/.restore_tid", "w");
        if (f) { fprintf(f, "%d\n", restore_tid); fclose(f); }

        printf("EXIT (locked+checkpointed, restore_tid=%d)\n", restore_tid);
        return 0;

    } else if (strcmp(argv[1], "restore") == 0) {
        /* Read saved restore TID */
        int saved_tid = -1;
        FILE *f = fopen("/tmp/.restore_tid", "r");
        if (f) { fscanf(f, "%d", &saved_tid); fclose(f); }
        printf("saved restore_tid=%d\n", saved_tid);

        if (saved_tid <= 0) {
            printf("No valid restore TID, trying from main thread...\n");
            char ra[64]; memset(ra, 0, 64);
            r = p_restore(getpid(), ra);
            printf("restore(main)=%d\n", r);
            return r;
        }

        /* Try 1: call restore from main thread (expected to fail/hang) */
        printf("=== Attempt 1: restore from main thread (tid=%d) ===\n",
               (int)syscall(SYS_gettid));
        fflush(stdout);

        /* Skip this — we know it hangs. Go straight to TID matching. */

        /* Try 2: spawn a thread and call restore from it */
        printf("=== Attempt 2: restore from spawned thread ===\n");
        fflush(stdout);
        struct restore_thread_args targs = { .target_tid = saved_tid, .result = -1 };
        pthread_t t;
        pthread_create(&t, NULL, restore_thread_fn, &targs);

        /* Wait with timeout */
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 10;
        int join_result = pthread_timedjoin_np(t, NULL, &ts);

        if (join_result == 0) {
            printf("Thread completed, restore=%d\n", targs.result);
            if (targs.result == 0) {
                printf("RESTORE_FROM_THREAD_OK\n");
            }
        } else {
            printf("Thread timed out (restore hung)\n");
            pthread_cancel(t);
        }

        return targs.result;
    }
    return 1;
}
