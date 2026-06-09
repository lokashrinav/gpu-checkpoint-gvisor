#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define NCCL_CHECK(x) do { ncclResult_t e = (x); if (e != ncclSuccess) { fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(e)); exit(1); } } while(0)

#define N_ELEMENTS (1024 * 1024)
#define PATTERN_A 0xDEADBEEFu
#define PATTERN_B 0xCAFEBABEu

int main() {
    int nGPUs;
    CUDA_CHECK(cudaGetDeviceCount(&nGPUs));
    fprintf(stderr, "nccl_ckpt_test: %d GPUs, mode=direct (no app-side checkpoint handling)\n", nGPUs);

    if (nGPUs < 2) {
        fprintf(stderr, "Need at least 2 GPUs\n");
        return 1;
    }

    ncclComm_t comms[nGPUs];
    float* d_send[nGPUs];
    float* d_recv[nGPUs];
    uint32_t* d_pattern[nGPUs];
    cudaStream_t streams[nGPUs];
    int devs[nGPUs];

    for (int i = 0; i < nGPUs; i++) devs[i] = i;

    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_send[i], N_ELEMENTS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_recv[i], N_ELEMENTS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pattern[i], N_ELEMENTS * sizeof(uint32_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        uint32_t pat = (i == 0) ? PATTERN_A : PATTERN_B;
        uint32_t* h = (uint32_t*)malloc(N_ELEMENTS * sizeof(uint32_t));
        for (int j = 0; j < N_ELEMENTS; j++) h[j] = pat;
        CUDA_CHECK(cudaMemcpy(d_pattern[i], h, N_ELEMENTS * sizeof(uint32_t), cudaMemcpyHostToDevice));
        free(h);
    }

    NCCL_CHECK(ncclCommInitAll(comms, nGPUs, devs));
    fprintf(stderr, "NCCL initialized: %d GPUs\n", nGPUs);
    fprintf(stderr, "READY\n");

    int tick = 0;
    while (1) {
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < nGPUs; i++) {
            NCCL_CHECK(ncclAllReduce(d_send[i], d_recv[i], 256, ncclFloat, ncclSum, comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());

        for (int i = 0; i < nGPUs; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }

        if (tick % 2 == 0) {
            for (int i = 0; i < nGPUs; i++) {
                CUDA_CHECK(cudaSetDevice(i));
                uint32_t val;
                CUDA_CHECK(cudaMemcpy(&val, d_pattern[i], sizeof(uint32_t), cudaMemcpyDeviceToHost));
                uint32_t expected = (i == 0) ? PATTERN_A : PATTERN_B;
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
