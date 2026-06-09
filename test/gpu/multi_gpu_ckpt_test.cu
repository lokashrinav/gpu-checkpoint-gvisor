// Multi-GPU checkpoint/restore test (no NCCL).
// Allocates VRAM on ALL available GPUs, writes unique patterns, verifies in a loop.
// cuCheckpointProcess* is called externally by gvisor-gpu-ckpt (direct mode).

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

#define N_ELEMENTS (1024 * 1024)

static uint32_t gpu_pattern(int gpu) {
    uint32_t patterns[] = {0xDEADBEEFu, 0xCAFEBABEu, 0xBEEF1234u, 0xFACEFEEDu, 0xC0DE1337u, 0xBADC0FFEu, 0xD00D1E55u, 0xDECAF000u};
    return patterns[gpu % 8];
}

int main() {
    int nGPUs;
    CUDA_CHECK(cudaGetDeviceCount(&nGPUs));
    fprintf(stderr, "multi_gpu_ckpt_test: %d GPUs\n", nGPUs);

    uint32_t** d_buf = (uint32_t**)malloc(nGPUs * sizeof(uint32_t*));

    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_buf[i], N_ELEMENTS * sizeof(uint32_t)));
        uint32_t pat = gpu_pattern(i);
        uint32_t* h = (uint32_t*)malloc(N_ELEMENTS * sizeof(uint32_t));
        for (int j = 0; j < N_ELEMENTS; j++) h[j] = pat;
        CUDA_CHECK(cudaMemcpy(d_buf[i], h, N_ELEMENTS * sizeof(uint32_t), cudaMemcpyHostToDevice));
        free(h);
        fprintf(stderr, "GPU%d: wrote pattern 0x%X (%d MB)\n", i, pat, (int)(N_ELEMENTS * sizeof(uint32_t) / (1024 * 1024)));
    }

    fprintf(stderr, "READY\n");

    int tick = 0;
    while (1) {
        if (tick % 2 == 0) {
            for (int i = 0; i < nGPUs; i++) {
                CUDA_CHECK(cudaSetDevice(i));
                uint32_t val;
                CUDA_CHECK(cudaMemcpy(&val, d_buf[i], sizeof(uint32_t), cudaMemcpyDeviceToHost));
                uint32_t expected = gpu_pattern(i);
                if (val != expected) {
                    fprintf(stderr, "tick=%d GPU%d MISMATCH: got=0x%X expected=0x%X\n", tick, i, val, expected);
                    return 1;
                }
                fprintf(stderr, "tick=%d GPU%d val=0x%X OK\n", tick, i, val);
            }
        }
        tick++;
        usleep(500000);
    }
    return 0;
}
