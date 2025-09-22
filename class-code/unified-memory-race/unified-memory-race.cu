#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <thread> 

// Macro to wrap CUDA API calls with error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel: add 1 to each element
__global__ void add_one(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

void launch_kernel(int N) {

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    int *d_data;
    CUDA_CHECK(cudaMallocManaged(&d_data, N * sizeof(int)));
    for (int i = 0; i < N; i++) {
      d_data[i] = 0;
    }

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    add_one<<<blocks, threads>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());

    // kernel is running asynchronously at this point
    // read unified memory while kernel is running
    unsigned zeroes = 0;
    for (int i = 0; i < N; i++) {
      if (0 == d_data[i]) {
        zeroes++;
      }
    }
    double perc = ((double) zeroes) / N;
    printf("Found %u / %u (%.2f%%) zeroes.\n", zeroes, N, perc);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Serial version time: %.3f ms\n\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char *argv[]) {
    if (argc != 1) {
        printf("Usage: %s \n", argv[0]);
        return 1;
    }

    launch_kernel(1024);

    return 0;
}
