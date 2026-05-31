#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <cuda.h>

#define MAX_GPUS 8
#define ELEMS (4*1024*1024)
#define BYTES (ELEMS*sizeof(unsigned int))

static int ng;
static CUcontext ctx[MAX_GPUS];
static CUdeviceptr d_buf[MAX_GPUS];
static unsigned int *h_buf[MAX_GPUS];
static volatile sig_atomic_t do_save = 0;
static int gpu_active = 0;

void on_save(int s) { do_save = 1; }
void on_reexec(int s) {
    // Save mirrors to file, then exec self with --restore flag
    FILE *f = fopen("/tmp/.gpu_mirror_info", "w");
    if (f) { fprintf(f, "%d\n", ng); fclose(f); }
    for (int i = 0; i < ng; i++) {
        char path[64]; snprintf(path, sizeof(path), "/tmp/.gpu_mirror_%d", i);
        FILE *mf = fopen(path, "wb");
        if (mf) { fwrite(h_buf[i], 1, BYTES, mf); fclose(mf); }
    }
    // Signal completion THEN exec self
    f = fopen("/tmp/.gpu_ckpt_done", "w");
    if (f) { fprintf(f, "0\n"); fclose(f); }
    // exec self with --restore
    execl("/bin/multi_gpu_test", "multi_gpu_test", "--restore", NULL);
    _exit(1); // exec failed
}

int main(int argc, char **argv) {
    int restoring = (argc > 1 && strcmp(argv[1], "--restore") == 0);

    signal(SIGUSR1, on_save);
    signal(SIGUSR2, on_reexec);

    cuInit(0); cuDeviceGetCount(&ng);
    if (ng > MAX_GPUS) ng = MAX_GPUS;

    if (restoring) {
        // Read ng from file
        FILE *f = fopen("/tmp/.gpu_mirror_info", "r");
        if (f) { fscanf(f, "%d", &ng); fclose(f); }
        fprintf(stderr, "RESTORE: re-exec, ng=%d\n", ng);
    }

    printf("Found %d GPUs%s\n", ng, restoring ? " (restored)" : "");
    if (ng < 2) return 1;

    for (int i = 0; i < ng; i++) {
        CUdevice dev; cuDeviceGet(&dev, i);
        CUresult r = cuCtxCreate(&ctx[i], 0, dev);
        fprintf(stderr, "ctxCreate(%d)=%d\n", i, r);
        if (r) return 1;
        cuMemAlloc(&d_buf[i], BYTES);
        h_buf[i] = (unsigned int*)malloc(BYTES);
        if (restoring) {
            char path[64]; snprintf(path, sizeof(path), "/tmp/.gpu_mirror_%d", i);
            FILE *mf = fopen(path, "rb");
            if (mf) { fread(h_buf[i], 1, BYTES, mf); fclose(mf); }
            fprintf(stderr, "RESTORE: GPU %d from file 0x%X\n", i, h_buf[i][0]);
        } else {
            unsigned int pat = 0xCAFE0000 | i;
            for (size_t j = 0; j < ELEMS; j++) h_buf[i][j] = pat;
        }
        cuMemcpyHtoD(d_buf[i], h_buf[i], BYTES);
        printf("GPU %d: 0x%X\n", i, h_buf[i][0]);
    }
    gpu_active = 1;
    if (!restoring) printf("MULTI_GPU_READY\n");
    else printf("RESTORE_READY\n");
    fflush(stdout);

    for (int tick = 1; ; tick++) {
        sleep(2);
        if (do_save) {
            do_save = 0;
            for (int i = 0; i < ng; i++) {
                cuCtxSetCurrent(ctx[i]); cuCtxSynchronize();
                cuMemcpyDtoH(h_buf[i], d_buf[i], BYTES);
            }
            for (int i = ng-1; i >= 0; i--) { cuCtxDestroy(ctx[i]); ctx[i]=0; d_buf[i]=0; }
            gpu_active = 0;
            FILE *f = fopen("/tmp/.gpu_ckpt_done","w"); if(f){fprintf(f,"0\n");fclose(f);}
        }
        if (!gpu_active) { printf("tick=%d waiting\n", tick*2); fflush(stdout); continue; }
        int ok = 1;
        for (int i = 0; i < ng; i++) {
            cuCtxSetCurrent(ctx[i]);
            unsigned int exp = 0xCAFE0000|i, val;
            CUresult cr = cuMemcpyDtoH(&val, d_buf[i], sizeof(unsigned int));
            if (cr || val != exp) { printf("GPU %d: r=%d v=0x%X\n", i, cr, val); ok = 0; }
        }
        printf("tick=%d all_gpus_ok=%d\n", tick*2, ok); fflush(stdout);
    }
}
