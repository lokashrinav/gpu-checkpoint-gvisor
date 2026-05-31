// Multi-GPU checkpoint/restore test binary.
// Uses CUDA driver API (no runtime) for gVisor nvproxy compatibility.
//
// Signal-based checkpoint lifecycle:
//   SIGUSR1 (save):  DMA copy all GPU buffers to host mirrors, destroy CUDA contexts
//   SIGUSR2 (restore): re-initialize CUDA, re-allocate GPU memory, copy host mirrors back
//
// Build: nvcc -Wno-deprecated-gpu-targets -o multi_gpu_test multi_gpu_test.cu -lcuda
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <cuda.h>

#define MAX_GPUS 8
#define ELEMS (4 * 1024 * 1024)
#define BYTES (ELEMS * sizeof(unsigned int))

static int ng;
static CUcontext ctx[MAX_GPUS];
static CUdeviceptr d_buf[MAX_GPUS];
static unsigned int *h_buf[MAX_GPUS];
static volatile sig_atomic_t do_save = 0, do_restore = 0;

void on_save(int s) { do_save = 1; }
void on_restore(int s) { do_restore = 1; }

static int reinit_gpu() {
    fprintf(stderr, "REINIT: re-initializing CUDA on %d GPUs\n", ng);
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "REINIT: cuInit=%d\n", r); return -1; }

    for (int i = 0; i < ng; i++) {
        CUdevice dev;
        cuDeviceGet(&dev, i);
        cuCtxCreate(&ctx[i], 0, dev);
        cuMemAlloc(&d_buf[i], BYTES);
        cuMemcpyHtoD(d_buf[i], h_buf[i], BYTES);
        fprintf(stderr, "REINIT: GPU %d restored from host mirror (0x%X)\n", i, h_buf[i][0]);
    }
    fprintf(stderr, "REINIT: done\n");
    return 0;
}

int main() {
    signal(SIGUSR1, on_save);
    signal(SIGUSR2, on_restore);

    cuInit(0);
    cuDeviceGetCount(&ng);
    if (ng > MAX_GPUS) ng = MAX_GPUS;
    printf("Found %d GPUs\n", ng);
    if (ng < 2) { fprintf(stderr, "Need 2+ GPUs\n"); return 1; }

    for (int i = 0; i < ng; i++) {
        CUdevice dev;
        cuDeviceGet(&dev, i);
        cuCtxCreate(&ctx[i], 0, dev);
        cuMemAlloc(&d_buf[i], BYTES);

        h_buf[i] = (unsigned int*)malloc(BYTES);
        unsigned int pat = 0xCAFE0000 | i;
        for (size_t j = 0; j < ELEMS; j++) h_buf[i][j] = pat;
        cuMemcpyHtoD(d_buf[i], h_buf[i], BYTES);
        printf("GPU %d: 0x%X to %zu MiB\n", i, pat, BYTES/(1024*1024));
    }

    printf("PID=%d\nMULTI_GPU_READY\n", getpid());
    fflush(stdout);

    for (int tick = 1; ; tick++) {
        sleep(2);

        if (do_save) {
            do_save = 0;
            fprintf(stderr, "SAVE: DMA %d GPUs -> host\n", ng);
            for (int i = 0; i < ng; i++) {
                cuCtxSetCurrent(ctx[i]);
                cuCtxSynchronize();
                cuMemcpyDtoH(h_buf[i], d_buf[i], BYTES);
                fprintf(stderr, "SAVE: GPU %d -> 0x%X\n", i, h_buf[i][0]);
            }
            for (int i = ng - 1; i >= 0; i--) {
                cuCtxDestroy(ctx[i]);
                ctx[i] = NULL;
                d_buf[i] = 0;
            }
            fprintf(stderr, "SAVE: contexts destroyed\n");
            FILE *f = fopen("/tmp/.gpu_ckpt_done", "w");
            if (f) { fprintf(f, "0\n"); fclose(f); }
            fprintf(stderr, "SAVE: done\n");
        }

        if (do_restore) {
            do_restore = 0;
            fprintf(stderr, "RESTORE: triggered by SIGUSR2\n");
            if (reinit_gpu() != 0) {
                fprintf(stderr, "REINIT failed\n");
                return 1;
            }
            FILE *f = fopen("/tmp/.gpu_ckpt_done", "w");
            if (f) { fprintf(f, "0\n"); fclose(f); }
        }

        if (ctx[0] == NULL) {
            printf("tick=%d waiting_for_checkpoint\n", tick * 2);
            fflush(stdout);
            continue;
        }

        int ok = 1;
        for (int i = 0; i < ng; i++) {
            cuCtxSetCurrent(ctx[i]);
            unsigned int exp = 0xCAFE0000 | i, val;
            CUresult cr = cuMemcpyDtoH(&val, d_buf[i], sizeof(unsigned int));
            if (cr != CUDA_SUCCESS) {
                printf("GPU %d: cuMemcpyDtoH=%d\n", i, cr);
                ok = 0;
            } else if (val != exp) {
                printf("GPU %d: MISMATCH 0x%X!=0x%X\n", i, val, exp);
                ok = 0;
            }
        }
        printf("tick=%d all_gpus_ok=%d\n", tick * 2, ok);
        fflush(stdout);
    }
}
