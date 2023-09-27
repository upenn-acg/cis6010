#pragma once

#include "cuda_runtime.h"

class CudaTimer {
private:
	cudaEvent_t m_start, m_stop;

public:
	CudaTimer() {}

	void start() {
		cudaEventCreate(&m_start);
		cudaEventCreate(&m_stop);
		cudaEventRecord(m_start, 0);
	}

	/** Return elapsed time (in ms) since start() was called */
	float stop() {
		float time;
		cudaEventRecord(m_stop, 0);
		cudaEventSynchronize(m_stop);
		cudaEventElapsedTime(&time, m_start, m_stop);
		return time;
	}
};