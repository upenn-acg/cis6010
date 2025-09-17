#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// ---------------------------------------------------------------------
// Naive transpose: one element per thread, direct global memory access
// ---------------------------------------------------------------------
__global__ void transposeNaive(const int* __restrict__ in, int* __restrict__ out, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        out[col * N + row] = in[row * N + col];
    }
}

// ---------------------------------------------------------------------
// Optimized transpose: use shared memory for coalesced access
// ---------------------------------------------------------------------
const int TILE_DIM = 32;      // tile size
const int BLOCK_ROWS = 4;     // number of rows each thread block transposes

__global__ void transposeShared(const int* __restrict__ in, int* __restrict__ out, int N) {
    __shared__ int tile[TILE_DIM][TILE_DIM+1]; // NB: +1 avoids shared memory bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // load tile from global to shared
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < N) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
        }
    }
    __syncthreads();

    // transpose block offset
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // write tile to global transposed
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < N) {
            out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ---------------------------------------------------------------------
// Helper: print a small portion of matrix
// ---------------------------------------------------------------------
void printMatrixPortion(const std::vector<int>& mat, int N, int maxRows=6, int maxCols=6) {
    int rows = std::min(N, maxRows);
    int cols = std::min(N, maxCols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i * N + j] << "\t";
        }
        if (cols < N) std::cout << "...";
        std::cout << "\n";
    }
    if (rows < N) std::cout << "...\n";
}

int main(int argc, char** argv) {
    int N = 1024; // default matrix size
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    std::cout << "Matrix size: " << N << "x" << N << std::endl;

    size_t bytes = N * N * sizeof(int);

    // Host allocation
    std::vector<int> h_in(N * N);
    std::vector<int> h_out(N * N, 0);

    // Fill with random integers
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < N * N; i++) {
        h_in[i] = std::rand() % 100;
    }

    // Device allocation
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------
    // Run naive kernel
    // -----------------------------------------------------------------
    cudaEvent_t start, stop;
    float msNaive = 0.0f, msShared = 0.0f;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // -----------------------------------------------------------------
    // Kernel launch
    // -----------------------------------------------------------------
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);
    transposeNaive<<<blocks, threads>>>(d_in, d_out, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msNaive, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    std::cout << "\nNaive transpose time: " << msNaive << " ms\n";

    std::cout << "\nOriginal matrix (portion):\n";
    printMatrixPortion(h_in, N);
    std::cout << "\nNaive transpose (portion):\n";
    printMatrixPortion(h_out, N);

    // -----------------------------------------------------------------
    // Run shared memory optimized kernel
    // -----------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));

    // -----------------------------------------------------------------
    // Kernel launch
    // -----------------------------------------------------------------
    dim3 threadsShared(TILE_DIM, BLOCK_ROWS);
    //dim3 threadsShared(TILE_DIM, TILE_DIM);
    dim3 blocksShared((N + TILE_DIM - 1) / TILE_DIM,
                      (N + TILE_DIM - 1) / TILE_DIM);
    transposeShared<<<blocksShared, threadsShared>>>(d_in, d_out, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msShared, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    std::cout << "\nShared-memory transpose time: " << msShared << " ms\n";
    std::cout << "\nShared-memory transpose (portion):\n";
    printMatrixPortion(h_out, N);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
