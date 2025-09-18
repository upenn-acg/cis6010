#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

void serial_version(int N, int K) {
    printf("=== Serial Version (default stream) ===\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int k = 0; k < K; k++) {
        int *d_data;
        int *h_data = new int[N];

        // Initialize host array
        for (int i = 0; i < N; i++) {
            h_data[i] = i;
        }

        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

        // Copy host -> device
        CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

        // Launch kernel
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        add_one<<<blocks, threads>>>(d_data, N);
        CUDA_CHECK(cudaGetLastError());

        // Copy device -> host
        CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

        //printf("Array %d: h_data[0] = %d, h_data[N-1] = %d\n", k, h_data[0], h_data[N - 1]);

        CUDA_CHECK(cudaFree(d_data));
        delete[] h_data;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Serial version time: %.3f ms\n\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void serial_unified_memory_version(int N, int K) {
    printf("=== Serial Version (default stream, unified memory) ===\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int k = 0; k < K; k++) {
        int *d_data;
        int *h_data = new int[N];

        // Initialize host array
        for (int i = 0; i < N; i++) {
            h_data[i] = i;
        }

        CUDA_CHECK(cudaMallocManaged(&d_data, N * sizeof(int)));

        // Launch kernel
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        add_one<<<blocks, threads>>>(d_data, N);
        CUDA_CHECK(cudaGetLastError());

        //printf("Array %d: h_data[0] = %d, h_data[N-1] = %d\n", k, h_data[0], h_data[N - 1]);

        CUDA_CHECK(cudaFree(d_data));
        delete[] h_data;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Serial version time: %.3f ms\n\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void streams_version(int N, int K) {
    printf("=== Streams Version (parallel, explicit copies) ===\n");

    int **d_data = new int *[K];
    int **h_data = new int *[K];
    cudaStream_t *streams = new cudaStream_t[K];

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int k = 0; k < K; k++) {
        CUDA_CHECK(cudaMalloc(&d_data[k], N * sizeof(int)));
        h_data[k] = new int[N];
        for (int i = 0; i < N; i++) {
            h_data[k][i] = i;
        }
        CUDA_CHECK(cudaStreamCreate(&streams[k]));
    }

    CUDA_CHECK(cudaEventRecord(start));

    for (int k = 0; k < K; k++) {
        // Copy host -> device
        CUDA_CHECK(cudaMemcpyAsync(d_data[k], h_data[k],
                                   N * sizeof(int), cudaMemcpyHostToDevice,
                                   streams[k]));

        // Kernel
        add_one<<<blocks, threads, 0, streams[k]>>>(d_data[k], N);
        CUDA_CHECK(cudaGetLastError());

        // Copy device -> host
        CUDA_CHECK(cudaMemcpyAsync(h_data[k], d_data[k],
                                   N * sizeof(int), cudaMemcpyDeviceToHost,
                                   streams[k]));
    }

    for (int k = 0; k < K; k++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[k]));
        //printf("Array %d: h_data[0] = %d, h_data[N-1] = %d\n", k, h_data[k][0], h_data[k][N - 1]);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Streams version time: %.3f ms\n\n", ms);

    for (int k = 0; k < K; k++) {
        CUDA_CHECK(cudaFree(d_data[k]));
        delete[] h_data[k];
        CUDA_CHECK(cudaStreamDestroy(streams[k]));
    }
    delete[] d_data;
    delete[] h_data;
    delete[] streams;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void unified_memory_version(int N, int K) {
    printf("=== Unified Memory Version (parallel, managed memory) ===\n");

    int **data = new int *[K];
    cudaStream_t *streams = new cudaStream_t[K];

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Allocate & initialize
    for (int k = 0; k < K; k++) {
        CUDA_CHECK(cudaMallocManaged(&data[k], N * sizeof(int)));
        for (int i = 0; i < N; i++) {
            data[k][i] = i;
        }
        CUDA_CHECK(cudaStreamCreate(&streams[k]));

        // Prefetch to GPU before kernel
        //CUDA_CHECK(cudaMemPrefetchAsync(data[k], N * sizeof(int), device, streams[k]));
    }

    CUDA_CHECK(cudaEventRecord(start));

    for (int k = 0; k < K; k++) {
        add_one<<<blocks, threads, 0, streams[k]>>>(data[k], N);
        CUDA_CHECK(cudaGetLastError());
    }

    // Synchronize before host access
    for (int k = 0; k < K; k++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[k]));
        //printf("Array %d: data[0] = %d, data[N-1] = %d\n", k, data[k][0], data[k][N - 1]);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Unified memory version time: %.3f ms\n\n", ms);

    for (int k = 0; k < K; k++) {
        CUDA_CHECK(cudaFree(data[k]));
        CUDA_CHECK(cudaStreamDestroy(streams[k]));
    }
    delete[] data;
    delete[] streams;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <N: array size> <K: number of arrays/streams>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    printf("Running with array of %d elements and K = %d\n\n", 1<<N, K);

    serial_version(1<<N, K);
    streams_version(1<<N, K);
    serial_unified_memory_version(1<<N, K);
    unified_memory_version(1<<N, K);

    return 0;
}
