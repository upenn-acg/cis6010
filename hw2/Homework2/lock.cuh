#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <assert.h>

#define LOCK_FREE 0
#define LOCK_HELD 1

/** A lock that can be called from just one thread within each warp. 
Branch divergence issues should be handled by the caller. This lock 
can be used across thread blocks within a single device. */
class WarpLevelLock {
private:
	/** NB: This must be volatile to ensure it is never cached in the L1. */
	volatile unsigned theLock;

public:
	__device__ WarpLevelLock() {
		theLock = LOCK_FREE;
	}

	/** Acquires this lock for the calling warp. Blocks until the lock has been acquired, then returns. */
	__device__ void lock() {
		// TODO: your code here
	}

	/** Releases this lock. Assumes the lock is held by the calling warp. */
	__device__ void unlock() {
		// TODO: your code here
	}
};

/** A lock that can be acquired by every thread within a warp. This lock 
can be used across thread blocks within a single device. */
class ThreadLevelLock {
private:
	WarpLevelLock warpLock;

public:
	__device__ ThreadLevelLock() {}
	
	/** fun points to a function that implements the desired critical section code. 
	Executes this critical section for each thread within the calling warp before returning. */
	__device__ void criticalSection(void(*fun)()) {
		// TODO: your code here
	}
};