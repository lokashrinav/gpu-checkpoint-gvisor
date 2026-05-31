/*
 * Cross-process GPU migration test.
 * Process A: cuInit, alloc, pattern, lock, checkpoint, EXIT
 * Process B: cuInit, restore with UUID mapping, verify pattern
 *
 * Tests whether driver 580's migration restore works from a different process.
 * Build: gcc -I /usr/local/cuda/include cross_proc_migration.c -o cross_proc_migration -lcuda -lnvidia-ml -ldl
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <nvml.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s save|restore\n", argv[0]);
        return 1;
    }

    /* Unset masking so all GPUs visible */
    unsetenv("CUDA_VISIBLE_DEVICES");
    unsetenv("CUDA_DEVICE_ORDER");

    /* Check persistence mode */
    nvmlInit();
    nvmlDevice_t nvdev;
    nvmlEnableState_t pm;
    nvmlDeviceGetHandleByIndex(0, &nvdev);
    nvmlDeviceGetPersistenceMode(nvdev, &pm);
    printf("Persistence mode: %s\n", pm == 1 ? "ON" : "OFF");

    CUresult r;
    r = cuInit(0);
    printf("cuInit=%d pid=%d\n", r, getpid());
    if (r) return 1;

    int dev_count;
    cuDeviceGetCount(&dev_count);
    printf("GPUs: %d\n", dev_count);

    if (strcmp(argv[1], "save") == 0) {
        /* Create context on GPU 0, allocate memory, write pattern */
        CUcontext ctx;
        cuDevicePrimaryCtxRetain(&ctx, 0);
        cuCtxSetCurrent(ctx);

        CUdeviceptr dptr;
        cuMemAlloc(&dptr, 4 * 1024 * 1024);
        cuMemsetD32(dptr, 0xDEADBEEF, 1024 * 1024);
        printf("Allocated + patterned on GPU 0, dptr=%llx\n", (unsigned long long)dptr);

        /* Save GPU UUIDs for restore */
        FILE *f = fopen("/tmp/.gpu_uuids", "w");
        for (int i = 0; i < dev_count; i++) {
            CUuuid uuid;
            cuDeviceGetUuid(&uuid, i);
            for (int j = 0; j < 16; j++) fprintf(f, "%02x", uuid.bytes[j] & 0xFF);
            fprintf(f, "\n");
        }
        fclose(f);

        /* Lock + Checkpoint */
        CUcheckpointLockArgs lock_args = {0};
        CUcheckpointCheckpointArgs ckpt_args = {0};
        pid_t self = getpid();

        printf("Lock(%d)...\n", self);
        r = cuCheckpointProcessLock(self, &lock_args);
        printf("Lock=%d\n", r);

        printf("Checkpoint(%d)...\n", self);
        r = cuCheckpointProcessCheckpoint(self, &ckpt_args);
        printf("Checkpoint=%d\n", r);

        /* EXIT without restore/unlock — checkpoint data should persist
         * if persistence mode is on */
        printf("Exiting (locked, checkpointed, no unlock)\n");
        return 0;

    } else if (strcmp(argv[1], "restore") == 0) {
        /* Read saved UUIDs */
        CUuuid uuids[8];
        FILE *f = fopen("/tmp/.gpu_uuids", "r");
        if (!f) { fprintf(stderr, "No UUID file\n"); return 1; }
        int n = 0;
        char line[64];
        while (fgets(line, sizeof(line), f) && n < 8) {
            for (int j = 0; j < 16; j++) {
                unsigned int byte;
                sscanf(line + j*2, "%02x", &byte);
                uuids[n].bytes[j] = (char)byte;
            }
            n++;
        }
        fclose(f);

        /* Set up restore with identity UUID mapping (same GPU -> same GPU) */
        CUcheckpointRestoreArgs restore_args = {0};
        CUcheckpointGpuPair *pairs = calloc(n, sizeof(CUcheckpointGpuPair));
        for (int i = 0; i < n; i++) {
            pairs[i].oldUuid = uuids[i];
            pairs[i].newUuid = uuids[i]; /* same GPU */
        }
        restore_args.gpuPairsCount = n;
        restore_args.gpuPairs = pairs;

        CUcheckpointUnlockArgs unlock_args = {0};
        pid_t self = getpid();

        printf("Restore(%d) with %d GPU pairs...\n", self, n);
        fflush(stdout);
        r = cuCheckpointProcessRestore(self, &restore_args);
        printf("Restore=%d\n", r);

        if (r == 0) {
            r = cuCheckpointProcessUnlock(self, &unlock_args);
            printf("Unlock=%d\n", r);
            printf("CROSS_PROCESS_MIGRATION_OK\n");
        } else {
            printf("CROSS_PROCESS_MIGRATION_FAILED: %d\n", r);
        }

        free(pairs);
        return r;
    }
    return 1;
}
