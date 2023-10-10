#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// from https://github.com/jarro2783/cxxopts
#include "cxxopts.hpp"

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))
#define ROUND_UP_TO_NEAREST(M, N) (((M) + (N)-1) / (N))

enum Algo
{
    cublas = 0,
    basic,
    gmem_coalesced,
    smem,
    smem_multioutput,
    numAlgos
};

const char *algo2str(Algo a)
{
    switch (a)
    {
    case cublas:
        return "cublas";
    case basic:
        return "basic";
    case gmem_coalesced:
        return "gmem_coalesced";
    case smem:
        return "sharedmem";
    case smem_multioutput:
        return "sharedmem_multioutput";
    default:
        return "INVALID";
    }
}

void cudaErrorCheck(cudaError_t error, const char *file, int line);
void cublasErrorCheck(cublasStatus_t status, const char *file, int line);
void randomize_matrix(float *mat, int N);
void const_init_matrix(float *mat, int N, float F);
bool verify_matrix(float *expected, float *actual, int M, int N);
void print_matrix(const float *A, int M, int N, std::ostream &outs);
void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

const std::string errLogFile = "gemmValidationFailure.txt";

// NB: must use a single generator to avoid duplicates
std::default_random_engine generator(2);
std::uniform_real_distribution<float> distribution(0, 1);

int main(int argc, char **argv)
{
    // command-line flags
    cxxopts::Options options("gemm.cu", "CUDA GEMM kernels");
    options.add_options()("size", "matrix size (N x N)", cxxopts::value<uint16_t>()->default_value("128"))                //
        ("reps", "repeat GEMM this many times", cxxopts::value<uint16_t>()->default_value("1"))                           //
        ("algo", "GEMM algorithm to use, a number in [0,4], 0 is cuBLAS", cxxopts::value<uint16_t>()->default_value("0")) //
        ("validate", "Validate output against cuBLAS", cxxopts::value<bool>()->default_value("true"))                     //
        ("rngseed", "PRNG seed", cxxopts::value<uint>()->default_value("2"))                     //
        ("h,help", "Print usage");

    auto clFlags = options.parse(argc, argv);
    if (clFlags.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    const uint16_t SIZE = clFlags["size"].as<uint16_t>();
    if (SIZE % 32 != 0)
    {
        //std::cout << "--size must be a multiple of 32" << std::endl;
        //exit(EXIT_FAILURE);
    }
    const uint16_t REPS = clFlags["reps"].as<uint16_t>();
    const Algo ALGO = static_cast<Algo>(clFlags["algo"].as<uint16_t>());
    if (ALGO >= numAlgos)
    {
        printf("Invalid algorithm: %d\n", ALGO);
        exit(EXIT_FAILURE);
    }

    const bool VALIDATE = clFlags["validate"].as<bool>();
    const uint SEED = clFlags["rngseed"].as<uint>();
    generator.seed(SEED);
    printf("Multiplying two %u x %u matrices with %u trials using %s algorithm\n", SIZE, SIZE, REPS, algo2str(ALGO));

    cudaCheck(cudaSetDevice(0));

    // Setup cublas
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    uint16_t m = SIZE, n = SIZE, k = SIZE;

    // GEMM computes C = α*AB+β*C

    // just do pure A*B (for simpler debugging)
    float alpha = 1.0, beta = 1.0, initC = 1.0;

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;     // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * SIZE * SIZE);
    B = (float *)malloc(sizeof(float) * SIZE * SIZE);
    C = (float *)malloc(sizeof(float) * SIZE * SIZE);
    C_ref = (float *)malloc(sizeof(float) * SIZE * SIZE);

    randomize_matrix(A, SIZE * SIZE);
    randomize_matrix(B, SIZE * SIZE);
    randomize_matrix(C, SIZE * SIZE);

    const_init_matrix(C, SIZE * SIZE, initC);
    // print_matrix(A, SIZE, SIZE, std::cout);
    // print_matrix(B, SIZE, SIZE, std::cout);
    // print_matrix(C, SIZE, SIZE, std::cout);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * SIZE * SIZE));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));

    printf("dimensions(m=n=k) %u, alpha: %f, beta: %f\n", m, alpha, beta);

    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (!VALIDATE)
    {
        printf("disabled validation\n");
    }
    else
    {
        // run cublas to get correct answer in dC_ref
        runCublas(handle, m, n, k, alpha, dA, dB, beta, dC_ref);

        // run user's algorithm, filling in dC
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC);

        cudaCheck(cudaDeviceSynchronize());

        // copy both results back to host
        cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        if (verify_matrix(C_ref, C, n, m))
        {
            printf("Validated successfully!\n");
        }
        else
        {
            printf("Failed validation against NVIDIA cuBLAS.\n");
            std::cout << " Logging faulty output into " << errLogFile << "\n";
            std::ofstream fs;
            fs.open(errLogFile, std::ios::out | std::ios::trunc);
            fs << "α=" << alpha << " β=" << beta << std::endl;
            fs << "C matrix initialized to " << initC << std::endl << std::endl;
            fs << "A:" << std::endl;
            print_matrix(A, m, n, fs);
            fs << "B:" << std::endl;
            print_matrix(B, m, n, fs);
            fs << "C:" << std::endl;
            print_matrix(C, m, n, fs);
            fs << "Expected:" << std::endl;
            print_matrix(C_ref, m, n, fs);
            fs.close();
            exit(EXIT_FAILURE);
        }
    }

    // timing run(s)
    cudaEventRecord(beg);
    for (int j = 0; j < REPS; j++)
    {
        // We don't reset dC between runs to save time
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC);
        cudaCheck(cudaDeviceSynchronize());
    }

    // TODO: measure timing without memory transfers?
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = (double)2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: (%u).\n",
        elapsed_time / REPS,
        (REPS * flops * 1e-9) / elapsed_time,
        m);

    // free CPU and GPU memory
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dB));
    cudaCheck(cudaFree(dC));
    cudaCheck(cudaFree(dC_ref));
    cublasCheck(cublasDestroy(handle));

    return 0;
}

/** Function to check for errors in CUDA API calls */
void cudaErrorCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line,
               cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void cublasErrorCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[CUDA ERROR] at file %s:%d:\n %s: %s\n", file, line,
               cublasGetStatusName(status), cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

/** Initialize the given matrix `mat` which has `N` contiguous values. Contents of `mat` are set to random values. */
void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = distribution(generator);
    }
}

void const_init_matrix(float *mat, int N, float F)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = F;
    }
}

/** Print the given MxN matrix `mat` to the provided output stream. */
void print_matrix(const float *A, int M, int N, std::ostream &outs)
{
    outs << "[";
    for (int i = 0; i < M * N; i++)
    {
        if ((i + 1) % N == 0)
        {
            outs << std::fixed << std::setprecision(3) << A[i];
        }
        else
        {
            outs << std::fixed << std::setprecision(3) << A[i] << ", ";
        }
        if ((i + 1) % N == 0)
        {
            if (i + 1 < M * N)
                outs << ";" << std::endl;
        }
    }
    outs << "]" << std::endl << std::endl;
}

bool verify_matrix(float *expected, float *actual, int M, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            float fexp = (expected[(i * N) + j]);
            float fact = (actual[(i * N) + j]);
            double diff = std::fabs(fexp - fact);
            if (diff > 0.002)
            {
                printf("Divergence! Should be %5.3f, is %5.3f (diff %5.3f) at [%d,%d]\n",
                       fexp, fact, diff, i, j);
                return false;
            }
        }
    }
    return true;
}

void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha,
               float *A, float *B, float beta, float *C)
{
    // cuBLAS uses *column-major* order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // cublasStatus_t ok = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
    //                                  N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, /*CUBLAS_COMPUTE_16F*/ CUBLAS_COMPUTE_16F_PEDANTIC,
    //                                  CUBLAS_GEMM_DEFAULT);
    cublasStatus_t ok = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    cublasCheck(ok);
}

__global__ void runBasic(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float tmp = 0.0;
        // C = α*(AxB)+β*C
        for (int i = 0; i < K; ++i)
        {
            // tmp += __A__[x][i] * __B__[i][y]
            tmp += A[(x * K) + i] * B[(i * N) + y];
        }
        // __C__[x][y]
        C[(x * N) + y] = (alpha * tmp) + (beta * C[x * N + y]);
    }
}

__global__ void runGmemCoalesced(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // HW1 TODO: copy runBasic() code here and update to avoid uncoalesced accesses to global memory.
    // Note, you are also free to change the grid dimensions in the kernel launch below.

}

const uint F = 32;

__global__ void runSharedMem(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // HW2 TODO: Use shared memory to cache square FxF tiles of the A and B matrices in shared memory 
    // (SA and SB, respectively, provided below). Each thread should compute the result for one cell 
    // of the output matrix C.

    // Note, you will also need to change the grid dimensions in the kernel launch below to take into account the value
    // of F (which is a constant, defined above). You should experiment with different values of F to see how it 
    // affects performance.

    __shared__ float SA[F][F];
    __shared__ float SB[F][F];

}

const uint G = 4;

__global__ void runSharedMemMultiOutput(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // HW3 TODO: Copy your runSharedMem() code here and update it so that each thread computes the result for GxG cells 
    // of the output matrix C. Each thread should accumulate temporary results in the local LC matrix, provided below,
    // before writing them to C in global memory.

    // Note, you will also need to change the grid dimensions in the kernel launch below. You should experiment 
    // with different values of F and G to see how they affect performance.

    __shared__ float SA[F][F];
    __shared__ float SB[F][F];

    float LC[G][G] = {0.0};

}

void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha,
             float *A, float *B, float beta, float *C)
{
    switch (algo)
    {
    case cublas:
        runCublas(handle, M, N, K, alpha, A, B, beta, C);
        break;
    case basic:
    {
        dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
        dim3 blockDim(32, 32);
        runBasic<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case gmem_coalesced:
    {
        dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
        dim3 blockDim(32, 32);
        runGmemCoalesced<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case smem:
    {
        assert(0 == M % F);
        assert(0 == N % F);
        assert(0 == K % F);
        // TODO: update your grid here
        dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
        dim3 blockDim(32, 32);
        runSharedMem<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case smem_multioutput:
    {
        assert(0 == M % F);
        assert(0 == N % F);
        assert(0 == K % F);
        assert(0 == F % G);
        assert((F*F) / (G*G) >= F);
        // TODO: update your grid here
        dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
        dim3 blockDim(32, 32);
        runSharedMemMultiOutput<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish
    cudaCheck(cudaGetLastError());      // check for errors from kernel run
}
