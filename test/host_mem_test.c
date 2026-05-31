/*
 * Test: does cuCheckpointProcessCheckpoint move GPU state to host memory?
 * If so, can cuCheckpointProcessRestore read it from a new process that
 * has the same memory contents (simulating gVisor restore)?
 *
 * Test 1: Same process — lock, checkpoint, restore, unlock, verify.
 *         (Already works — this is our --leave-running test)
 *
 * Test 2: Check process memory size before/after checkpoint.
 *         If checkpoint copies GPU VRAM to host, RSS should increase.
 *
 * Test 3: Fork after checkpoint (child has same memory).
 *         Can the CHILD call restore? (Same memory, different PID)
 *
 * Build: gcc host_mem_restore.c -o host_mem_restore -ldl -lnvidia-ml
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <nvml.h>

typedef int R;typedef int Dev;typedef void*Ctx;typedef unsigned long long Dp;typedef R(*Fn)(int,void*);

static R(*p_init)(unsigned);
static R(*p_dgc)(int*);
static R(*p_dpcr)(Ctx*,int);
static R(*p_csc)(Ctx);
static R(*p_ma)(Dp*,size_t);
static R(*p_ms)(Dp,unsigned,size_t);
static R(*p_md)(void*,Dp,size_t);
static Fn p_lock,p_ckpt,p_restore,p_unlock;

static long get_rss_kb() {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long rss = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

int main() {
    unsetenv("CUDA_VISIBLE_DEVICES");
    nvmlInit();

    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    p_init = dlsym(h,"cuInit"); p_dgc = dlsym(h,"cuDeviceGetCount");
    p_dpcr = dlsym(h,"cuDevicePrimaryCtxRetain"); p_csc = dlsym(h,"cuCtxSetCurrent");
    p_ma = dlsym(h,"cuMemAlloc_v2"); p_ms = dlsym(h,"cuMemsetD32_v2");
    p_md = dlsym(h,"cuMemcpyDtoH_v2");
    p_lock = dlsym(h,"cuCheckpointProcessLock"); p_ckpt = dlsym(h,"cuCheckpointProcessCheckpoint");
    p_restore = dlsym(h,"cuCheckpointProcessRestore"); p_unlock = dlsym(h,"cuCheckpointProcessUnlock");

    p_init(0);
    int ng; p_dgc(&ng);
    printf("pid=%d gpus=%d\n", getpid(), ng);

    Ctx ctx; Dev d;
    R(*dg)(Dev*,int) = dlsym(h,"cuDeviceGet");
    dg(&d, 0); p_dpcr(&ctx, 0); p_csc(ctx);

    /* Allocate 64MB on GPU */
    Dp dptr;
    p_ma(&dptr, 16*1024*1024*sizeof(unsigned int));
    p_ms(dptr, 0xDEADBEEF, 16*1024*1024);
    printf("Allocated 64MB on GPU, pattern=0xDEADBEEF\n");

    long rss_before = get_rss_kb();
    printf("RSS before checkpoint: %ld KB\n", rss_before);

    /* Lock + Checkpoint */
    char a[64]={0};
    R r = p_lock(getpid(), a);
    printf("lock=%d\n", r);
    memset(a,0,64);
    r = p_ckpt(getpid(), a);
    printf("checkpoint=%d\n", r);

    long rss_after = get_rss_kb();
    printf("RSS after checkpoint: %ld KB (delta=%+ld KB)\n", rss_after, rss_after - rss_before);

    if (rss_after - rss_before > 50000) {
        printf("GPU_STATE_IN_HOST_MEMORY: YES (RSS grew by %ld KB ~= %ld MB)\n",
               rss_after - rss_before, (rss_after - rss_before) / 1024);
    } else {
        printf("GPU_STATE_IN_HOST_MEMORY: UNCLEAR (RSS delta small)\n");
    }

    /* Test: same-process restore */
    printf("\n=== Same-process restore ===\n");
    memset(a,0,64);
    r = p_restore(getpid(), a);
    printf("restore=%d\n", r);
    if (r == 0) {
        memset(a,0,64);
        p_unlock(getpid(), a);
        unsigned int val;
        p_md(&val, dptr, sizeof(unsigned int));
        printf("verify: val=0x%X match=%s\n", val, val==0xDEADBEEF ? "YES" : "NO");
    }

    /* Re-checkpoint for fork test */
    printf("\n=== Re-checkpoint for fork test ===\n");
    memset(a,0,64);
    r = p_lock(getpid(), a);
    printf("lock=%d\n", r);
    memset(a,0,64);
    r = p_ckpt(getpid(), a);
    printf("checkpoint=%d\n", r);

    /* Fork — child has same memory as parent */
    printf("\n=== Fork test (child has same host memory) ===\n");
    fflush(stdout); fflush(stderr);
    pid_t child = fork();
    if (child == 0) {
        /* Child process — same memory, different PID */
        printf("child pid=%d (parent was %d)\n", getpid(), getppid());
        /* Try restore from child */
        memset(a,0,64);
        r = p_restore(getpid(), a);
        printf("child restore=%d\n", r);
        if (r == 0) {
            memset(a,0,64);
            p_unlock(getpid(), a);
            unsigned int val;
            p_md(&val, dptr, sizeof(unsigned int));
            printf("child verify: val=0x%X match=%s\n", val, val==0xDEADBEEF ? "YES" : "NO");
            if (val == 0xDEADBEEF) {
                printf("FORK_RESTORE_SUCCESS!\n");
            }
        } else {
            printf("FORK_RESTORE_FAILED: %d\n", r);
        }
        _exit(r);
    } else {
        int status;
        waitpid(child, &status, 0);
        printf("child exited: %d\n", WEXITSTATUS(status));

        /* Parent restores too */
        memset(a,0,64);
        r = p_restore(getpid(), a);
        printf("parent restore=%d\n", r);
        if (r == 0) {
            memset(a,0,64);
            p_unlock(getpid(), a);
        }
    }

    return 0;
}
