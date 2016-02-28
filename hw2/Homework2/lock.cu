#include "lock.cuh"
#include "barrier.cuh"
#include "timer.cuh"

#include <stdio.h>

const unsigned ITERATIONS = 2;
const unsigned NUM_BLOCKS = 16;
const unsigned WARPS_PER_BLOCK = 32; // max of 32
const unsigned WARP_SIZE = 32;

__device__ WarpLevelLock* d_WLLock = NULL;
__device__ ThreadLevelLock* d_TLLock = NULL;

__device__ __managed__ unsigned d_counter = 0;

__global__ void initLock() {
	assert(blockIdx.x == 0 && threadIdx.x == 0);
	d_WLLock = new WarpLevelLock();
	d_TLLock = new ThreadLevelLock();
}

__global__ void destroyLock() {
	assert(blockIdx.x == 0 && threadIdx.x == 0);
	delete d_WLLock;
	delete d_TLLock;
}

__global__ void incrementCounterWarpsOnly() {
	// NB: only one thread per warp runs this code to avoid branch divergence issues
	if (threadIdx.x % warpSize != 0) { return; }

	for (unsigned i = 0; i < ITERATIONS; i++) {
		d_WLLock->lock();
		d_counter++;
		d_WLLock->unlock();
	}
}

__device__ void incr() {
	d_counter++;
}

__global__ void incrementCounterAllThreads() {
	// NB: all threads run this code!

	for (unsigned i = 0; i < ITERATIONS; i++) {
		d_TLLock->criticalSection(incr);
	}
}

enum LockFlavor { LOCK_PER_WARP, LOCK_PER_THREAD };

void lockTest(const LockFlavor flavor) {
	cudaError_t cudaStatus;
	CudaTimer timer;

	// ALLOCATE DEVICE MEMORY
	timer.start();

	initLock<<<1, 1>>>();
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);

	printf("Setup device memory:  %3.1f ms \n", timer.stop());

	// LAUNCH KERNELS

	timer.start();
	d_counter = 0;
	if (flavor == LOCK_PER_WARP) {
		incrementCounterWarpsOnly<<<NUM_BLOCKS, WARPS_PER_BLOCK * WARP_SIZE>>>();
	} else if (flavor == LOCK_PER_THREAD) {
		incrementCounterAllThreads<<<NUM_BLOCKS, WARPS_PER_BLOCK * WARP_SIZE>>>();
	} else {
		assert(false);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);

	printf("Lock kernel time:  %3.1f ms \n", timer.stop());

	// CHECK COUNTER VALUE IS CORRECT
	unsigned expected = 0;
	if (flavor == LOCK_PER_WARP) {
		expected = NUM_BLOCKS * WARPS_PER_BLOCK * ITERATIONS;
	} else if (flavor == LOCK_PER_THREAD) {
		expected = NUM_BLOCKS * WARPS_PER_BLOCK * WARP_SIZE * ITERATIONS;
	}
	if (d_counter != expected) {
		printf("Expected counter value %u BUT GOT %u INSTEAD :-(\n", expected, d_counter);
	} else {
		printf("Counter has expected value of %u\n", expected);
	}

	// CLEANUP

	destroyLock<<<1, 1>>>();
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);
}

int main() {
	cudaError_t cudaStatus;

	// ensure kernel timeout is enabled
	int kernelTimeout;
	cudaStatus = cudaDeviceGetAttribute(&kernelTimeout, cudaDevAttrKernelExecTimeout, 0/*device*/);
	checkCudaErrors(cudaStatus);
	if (kernelTimeout == 0) {
		printf("WARNING: kernel timeout is NOT enabled!\nAre you sure you want to continue (y/n)?", kernelTimeout);
		char c;
		scanf(" %c", &c);
		if (c != 'y') return 0;
	}

	lockTest(LOCK_PER_WARP);

	printf("\n\n");

	lockTest(LOCK_PER_THREAD);

	printf("\n\n");

	barrierTest(SPIN_BARRIER);

	printf("\n\n");

	barrierTest(TWO_LEVEL_BARRIER);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	checkCudaErrors(cudaStatus);

    return 0;
}