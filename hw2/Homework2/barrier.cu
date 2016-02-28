#include "lock.cuh"
#include "barrier.cuh"
#include "timer.cuh"

#include <stdio.h>
#include <stdint.h>

#define NUM_BLOCKS 10
#define WARPS_PER_BLOCK 10 // max of 32
#define NUM_WARPS (NUM_BLOCKS * WARPS_PER_BLOCK)
const unsigned WARP_SIZE = 32;

/** the interface to a barrier */
/** The interface to a global (device-wide) barrier. */
class IBarrier {
protected:
	const unsigned m_expected;
	unsigned arrived;
	bool sense;

public:
	/** Initialize the barrier */
	__device__ IBarrier(const unsigned count) : m_expected(count) {
		arrived = 0;
		sense = true;
	}

	/** Each calling thread waits at this call until the barrier's count 
	has been reached. No thread leaves the barrier until all threads have 
	arrived. */
	__device__ virtual void wait() = 0;
};

/** A sense-reversing centralized global (device-wide) barrier. This 
barrier can only be called by one thread from each warp. Branch 
divergence issues must be handled by the caller. */
class SpinBarrier : public IBarrier {
protected:
	WarpLevelLock warpLock;

public:
	__device__ SpinBarrier(const unsigned count) : IBarrier(count) {}

	__device__ virtual void wait() {
		// TODO: your code here
	}
};

/** A sense-reversing two-level global (device-wide) barrier. This barrier 
performs block-level barrier before coordinating across blocks. This barrier 
can safely be called by every thread within a warp. */
class TwoLevelBarrier : public SpinBarrier {
public:
	__device__ TwoLevelBarrier(const unsigned count) : SpinBarrier(count) {}

	__device__ virtual void wait() {
		// TODO: your code here
	}
};

__device__ SpinBarrier* d_SpinBar = NULL;
__device__ TwoLevelBarrier* d_2LBar = NULL;

__global__ void initBarriers() {
	assert(blockIdx.x == 0 && threadIdx.x == 0);
	d_SpinBar = new SpinBarrier(NUM_WARPS);
	d_2LBar = new TwoLevelBarrier(NUM_BLOCKS);
}

__global__ void destroyBarriers() {
	assert(blockIdx.x == 0 && threadIdx.x == 0);
	delete d_SpinBar;
	delete d_2LBar;
}

__global__ void rotateRows(const BarrierFlavor flavor, int* array, const int arrayDim) {
	// NB: only one thread per warp runs this code to avoid branch divergence issues
	if (threadIdx.x % warpSize != 0) { return; }

	int in_c = ((blockDim.x * blockIdx.x) + threadIdx.x) / warpSize;
	int out_c = (in_c + 1) % arrayDim;

	for (int r = 0; r < arrayDim-1; r++) {
		// copy (row r, column c) to (row r+1, column c+1)
		array[((r+1) * arrayDim) + out_c] = array[(r * arrayDim) + in_c];
		if (flavor == SPIN_BARRIER) {
			d_SpinBar->wait();
		} else if (flavor == TWO_LEVEL_BARRIER) {
			d_2LBar->wait();
		} else {
			assert(false);
		}
	}
}

void barrierTest(const BarrierFlavor flavor) {
	cudaError_t cudaStatus;
	CudaTimer timer;

	const unsigned numWarps = NUM_BLOCKS * WARPS_PER_BLOCK;
	const unsigned arraySizeBytes = numWarps * numWarps * sizeof(int);

	// ALLOCATE DEVICE MEMORY
	timer.start();

	int* h_array = new int[numWarps * numWarps];
	memset(h_array, 0, arraySizeBytes);

	uint64_t rowSum = 0;
	for (int i = 0; i < numWarps; i++) { // initialize first row with random values
		h_array[i] = rand();
		rowSum += h_array[i];
		//printf("%d ", h_array[i]);
	}
	//printf(" // initialized row with sum of %llu\n", rowSum);
	int* d_array;
	cudaStatus = cudaMalloc(&d_array, arraySizeBytes);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaMemcpy(d_array, h_array, arraySizeBytes, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);

	initBarriers<<<1, 1>>>();
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);

	printf("Setup device memory:  %3.1f ms \n", timer.stop());

	// LAUNCH KERNEL

	timer.start();
	rotateRows<<<NUM_BLOCKS, WARPS_PER_BLOCK * WARP_SIZE>>>(flavor, d_array, numWarps);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);

	printf("Barrier kernel time:  %3.1f ms \n", timer.stop());

	// COPY ARRAY BACK TO HOST
	timer.start();
	cudaStatus = cudaMemcpy(h_array, d_array, arraySizeBytes, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);

	printf("Copy from device:  %3.1f ms \n", timer.stop());

	// CHECK ARRAY VALUES ARE CORRECT
	// each row and column should have the same sum
	const uint64_t expected = rowSum;
	bool foundError = false;

	// check rows
	for (int i = 0; i < numWarps; i++) {
		uint64_t sum = 0;
		for (int j = 0; j < numWarps; j++) {
			sum += h_array[(i * numWarps) + j];
		}
		if (sum != expected) {
			printf("Expected row %d to have sum of %llu BUT GOT %llu INSTEAD :-(\n", i, expected, sum);
			foundError = true;
		}
	}
	// check columns
	for (int i = 0; i < numWarps; i++) {
		uint64_t sum = 0;
		for (int j = 0; j < numWarps; j++) {
			sum += h_array[(j * numWarps) + i];
		}
		if (sum != expected) {
			printf("Expected column %d to have sum of %llu BUT GOT %llu INSTEAD :-(\n", i, expected, sum);
			foundError = true;
		}
	}

	/* print h_array for debugging
	for (int i = 0; i < numWarps; i++) {
		for (int j = 0; j < numWarps; j++) {
			printf("%d ", h_array[(i * numWarps) + j]);
		}
		printf("\n");
	}
	*/
	
	if (!foundError) {
		printf("Each row/column has expected sum of %llu\n", expected);
	}

	// CLEANUP

	destroyBarriers<<<1, 1>>>();
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaFree(d_array);
	checkCudaErrors(cudaStatus);
	delete h_array;
}