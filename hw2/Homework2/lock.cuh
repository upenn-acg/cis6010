#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <assert.h>

#define LOCK_FREE 0
#define LOCK_HELD 1

class WarpLevelLock {
private:
	volatile unsigned theLock;

public:
	__device__ WarpLevelLock() {
		theLock = LOCK_FREE;
	}

	__device__ void lock() {
		// TODO: your code here
	}

	__device__ void unlock() {
		// TODO: your code here
	}
};

class ThreadLevelLock {
private:
	WarpLevelLock warpLock;

public:
	__device__ ThreadLevelLock() {}
	
	__device__ void criticalSection(void(*fun)()) {
		// TODO: your code here
	}
};