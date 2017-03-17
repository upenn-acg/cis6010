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

	__device__ void waitFor(unsigned int until) {
		clock_t start = clock64();
		clock_t now;
		for (;;) {
			now = clock64();
			clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
			if (cycles >= until) {
				break;
			}
		}
	}

	__device__ void lock() {
		// TODO: PART 1
	}

	__device__ void backoffLock() {
		// TODO: PART 2
		unsigned int period = 100;
	}

	__device__ void unlock() {
		// TODO: PART 1

	}
};