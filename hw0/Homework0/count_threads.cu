
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <assert.h>
#include <windows.h>


__global__ void getTid(int* myTid, int* maxTidPerBlock) {
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	// write my own tid
	myTid[tid] = tid;

	// TODO: fill myTid with max tid from each block

	// TODO: write max tid from block into maxTidPerBlock

}

__global__ void getMaxTid(int* maxTids) {
	// TODO: find max tid from among all blocks
}

void __cdecl odprintf(const char *format, ...) {
	char    buf[4096], *p = buf;
	va_list args;
	int     n;

	va_start(args, format);
	n = _vsnprintf(p, sizeof buf - 3, format, args); // buf-3 is room for CR/LF/NUL
	va_end(args);

	p += (n < 0) ? sizeof buf - 3 : n;

	while (p > buf  &&  isspace(p[-1]))
		*--p = '\0';

	*p++ = '\r';
	*p++ = '\n';
	*p = '\0';

	OutputDebugString(buf);
}

int main()
{
	const int BLOCKS = 4;
	const int THREADS_PER_BLOCK = 16;

	cudaError_t cudaStatus;

	int* mtpb = (int*)malloc(BLOCKS * sizeof(int));
	assert(NULL != mtpb);
	
	// malloc, and zero, device buffers
	int *d_myTid, *d_mtpb;
	cudaStatus = cudaMalloc(&d_myTid, BLOCKS * THREADS_PER_BLOCK * sizeof(int));
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaMemset(d_myTid, 0, BLOCKS * THREADS_PER_BLOCK * sizeof(int));
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMalloc(&d_mtpb, BLOCKS * sizeof(int));
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaMemset(d_mtpb, 0, BLOCKS * sizeof(int));
	checkCudaErrors(cudaStatus);

	// launch first kernel
	getTid<<<BLOCKS, THREADS_PER_BLOCK>>>(d_myTid, d_mtpb);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);

	// launch second kernel
	getMaxTid<<<1, BLOCKS>>>(d_mtpb);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMemcpy(mtpb, d_mtpb, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);

	for (int i = 0; i < BLOCKS; i++) {
		odprintf("%d ", mtpb[i]);
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
	checkCudaErrors(cudaStatus);

    return 0;
}