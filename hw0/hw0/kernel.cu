
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>

__global__ void helloKernel(char* string) {
    int i = threadIdx.x;
	printf("%c", string[i]);
}

int main() {
	const char* msg = "Hello, CUDA world!";
	const int msglen = strlen(msg);
	char* dev_msg = NULL;

    cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkCudaErrors(cudaStatus);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_msg, msglen);
	checkCudaErrors(cudaStatus);

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_msg, msg, msglen, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);

	// Launch a kernel on the GPU with one thread for each element.
	helloKernel <<<1, msglen>>>(dev_msg);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
	checkCudaErrors(cudaStatus);

	cudaFree(dev_msg);

    return 0;
}
