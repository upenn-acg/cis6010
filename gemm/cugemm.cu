#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// from https://github.com/jarro2783/cxxopts
#include "cxxopts.hpp"

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))
#define ROUND_UP_TO_NEAREST(M, N) (((M) + (N)-1) / (N))

enum Algo
{
    cublas = 0,
    basic,
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
    default:
        return "INVALID";
    }
}

void cudaErrorCheck(cudaError_t error, const char *file, int line);
void cublasErrorCheck(cublasStatus_t status, const char *file, int line);
void randomize_matrix(float *mat, int N);
void const_init_matrix(float *mat, int N, float F);
bool verify_matrix(float *expected, float *actual, int N);
void print_matrix(const float *A, int M, int N, std::ostream &outs);
void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha,
             float *A, float *B, float beta, float *C);
void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C);

const std::string errLogFile = "gemmValidationFailure.txt";

int main(int argc, char **argv)
{
    // command-line flags
    cxxopts::Options options("gemm.cu", "CUDA GEMM kernels");
    options.add_options()("size", "matrix size (N x N)", cxxopts::value<uint16_t>()->default_value("128"))                //
        ("reps", "repeat GEMM this many times", cxxopts::value<uint16_t>()->default_value("1"))                           //
        ("algo", "GEMM algorithm to use, a number in [0,5], 0 is cuBLAS", cxxopts::value<uint16_t>()->default_value("0")) //
        ("validate", "Validate output against cuBLAS", cxxopts::value<bool>()->default_value("true"))                     //
        ("h,help", "Print usage");

    auto clFlags = options.parse(argc, argv);
    if (clFlags.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    const uint16_t SIZE = clFlags["size"].as<uint16_t>();
    const uint16_t REPS = clFlags["reps"].as<uint16_t>();
    const Algo ALGO = static_cast<Algo>(clFlags["algo"].as<uint16_t>());
    if (ALGO >= numAlgos)
    {
        printf("Invalid algorithm: %d\n", ALGO);
        exit(EXIT_FAILURE);
    }

    const bool VALIDATE = clFlags["validate"].as<bool>();
    printf("Multiplying two %u x %u matrices with %u trials using %s algorithm\n", SIZE, SIZE, REPS, algo2str(ALGO));

    cudaCheck(cudaSetDevice(0));

    // TODO: compute max compute throughput

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

    float alpha = 0.5, beta = 3.0;

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;     // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * SIZE * SIZE);
    B = (float *)malloc(sizeof(float) * SIZE * SIZE);
    C = (float *)malloc(sizeof(float) * SIZE * SIZE);
    C_ref = (float *)malloc(sizeof(float) * SIZE * SIZE);

    randomize_matrix(A, SIZE * SIZE);
    randomize_matrix(B, SIZE * SIZE);
    randomize_matrix(C, SIZE * SIZE);

    // just do pure A*B (for debugging)
    // alpha = 1.0, beta = 1.0;
    // const_init_matrix(C, SIZE * SIZE, 1.0);

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
    if (VALIDATE)
    {
        // run cublas to get correct answer in dC_ref
        runCublasFP32(handle, m, n, k, alpha, dA, dB, beta, dC_ref);

        // run user's algorithm, filling in dC
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC);

        // copy both results back to host
        cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        if (verify_matrix(C_ref, C, m * n))
        {
            printf("Validated successfully!\n");
        }
        else
        {
            printf("Failed validation against NVIDIA cuBLAS.\n");
            std::cout << " Logging faulty output into " << errLogFile << "\n";
            std::ofstream fs;
            fs.open(errLogFile);
            fs << "A:\n";
            print_matrix(A, m, n, fs);
            fs << "B:\n";
            print_matrix(B, m, n, fs);
            fs << "C:\n";
            print_matrix(C, m, n, fs);
            fs << "Should:\n";
            print_matrix(C_ref, m, n, fs);
            exit(EXIT_FAILURE);
        }
    }

    // timing run(s)
    cudaEventRecord(beg);
    for (int j = 0; j < REPS; j++)
    {
        // We don't reset dC between runs to save time
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC);
    }

    // measure timing without memory transfers?
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
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10, 10);
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
    printf("[");
    for (int i = 0; i < M * N; i++)
    {
        if ((i + 1) % N == 0)
            printf("%5.2f", A[i]); // Set field width and print the value
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % N == 0)
        {
            if (i + 1 < M * N)
                printf(";\n");
        }
    }
    printf("]\n\n");
}

bool verify_matrix(float *expected, float *actual, int N)
{
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++)
    {
        diff = std::fabs(expected[i] - actual[i]);
        if (diff > 0.01)
        {
            // print divergence in 2D coords instead?
            printf("Divergence! Should be %5.2f, is %5.2f (Diff %5.2f) at %d\n",
                   expected[i], actual[i], diff, i);
            return false;
        }
    }
    return true;
}

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C)
{
    // cuBLAS uses *column-major* order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // This runs cuBLAS in full fp32 mode
    cublasStatus_t ok = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                                     N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasCheck(ok);
}

__global__ void runBasic(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // TODO: add your matrix multiply implementation here!
}

void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha,
             float *A, float *B, float beta, float *C)
{
    switch (algo)
    {
    case cublas:
        runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
        break;
    case basic:
        // TODO: launch runBasic() kernel here
        break;
    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish
    cudaCheck(cudaGetLastError());      // check for errors from kernel run
}