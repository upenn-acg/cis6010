
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>

cudaEvent_t start, stop;
void startTimer() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

/** Return elapsed time (in ms) since startTime() was called */
float stopTimer() {
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	return time;
}

const int ROWS = 32;
const int COLUMNS = 32;
int hostMatrix[ROWS][COLUMNS];

__global__ void transpose(int* in, int* out) {
	int i = (blockDim.x * threadIdx.y) + threadIdx.x;
	int o = (blockDim.y * threadIdx.x) + threadIdx.y;

	int tmp = in[i];
	out[o] = tmp;
}

__global__ void transposeShared(int* in, int* out) {
	int i = (blockDim.x * threadIdx.y) + threadIdx.x;
	int o = (blockDim.y * threadIdx.x) + threadIdx.y;

	__shared__ int s[ROWS][COLUMNS+1];

	int tmp = in[i];
	
	s[threadIdx.y][threadIdx.x] = tmp;

	__syncthreads();

	out[i] = s[threadIdx.x][threadIdx.y];
}

void printMatrix(int* mat, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%d ", mat[(i*cols) + j]);
		}
		printf("\r\n");
	}
	printf("\r\n");
}

int main() {

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			hostMatrix[i][j] = (i * COLUMNS) + j;
		}
	}

	int* dev_in = NULL;
	int* dev_out = NULL;

    cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkCudaErrors(cudaStatus);

	const int matrixSizeBytes = ROWS * COLUMNS * sizeof(int);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_in, matrixSizeBytes);
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMalloc((void**)&dev_out, matrixSizeBytes);
	checkCudaErrors(cudaStatus);

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_in, hostMatrix, matrixSizeBytes, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);

	//printMatrix((int*)hostMatrix, ROWS, COLUMNS);

	const int TRIALS = 1000;
	float elapsed;
	
	startTimer();
	for (int i = 0; i < TRIALS; i++) {
		// Launch a kernel on the GPU with one thread for each element.
		dim3 blockOrg;
		blockOrg.x = COLUMNS;
		blockOrg.y = ROWS;
		transpose <<<1, blockOrg >>> (dev_in, dev_out);
		cudaStatus = cudaDeviceSynchronize();
		checkCudaErrors(cudaStatus);
	}
	elapsed = stopTimer();
	printf("%d transpose() trials took %f ms, %f ms/trial\r\n", TRIALS, elapsed, elapsed/TRIALS);
	

	printf("\r\n");
	startTimer();
	for (int i = 0; i < TRIALS; i++) {
		// Launch a kernel on the GPU with one thread for each element.
		dim3 blockOrg;
		blockOrg.x = COLUMNS;
		blockOrg.y = ROWS;
		transposeShared <<<1, blockOrg >>> (dev_in, dev_out);
		cudaStatus = cudaDeviceSynchronize();
		checkCudaErrors(cudaStatus);
	}
	elapsed = stopTimer();
	printf("%d transposeShared() trials took %f ms, %f ms/trial\r\n", TRIALS, elapsed, elapsed / TRIALS);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);

	// Copy output vectors from GPU to host
	cudaStatus = cudaMemcpy(hostMatrix, dev_out, matrixSizeBytes, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);

	//printMatrix((int*)hostMatrix, COLUMNS, ROWS);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
	checkCudaErrors(cudaStatus);

	cudaFree(dev_in);
	cudaFree(dev_out);

    return 0;
}
