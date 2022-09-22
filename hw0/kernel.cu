
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <cstdint>
#include <iostream>
#include <random>

using namespace std;

__global__ void sumKernel(const uint32_t* array, uint32_t numElements) {
  if (0 == threadIdx.x) {
    uint64_t sum = 0;
    for (int i = 0; i < numElements; i++) {
      sum += array[i];
    }
    printf("[device] sum of %d elements = %lu\n", numElements, sum);
  }
}

int main() {
  const uint32_t NUM_ELEMENTS = 1024;
  uint32_t* d_array = nullptr;
  const uint32_t arraySizeBytes = NUM_ELEMENTS * sizeof(d_array[0]);
  uint32_t* h_array = (uint32_t*) malloc(arraySizeBytes);

  random_device rd;
  uint64_t h_sum = 0;
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    h_array[i] = rd();
    h_sum += h_array[i];
  }
  printf("[host]   sum of %d elements = %lu\n", NUM_ELEMENTS, h_sum);
  
  cudaError_t cudaStatus;

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  // Allocate GPU buffers
  cudaStatus = cudaMalloc((void**)&d_array, arraySizeBytes);
  checkCudaErrors(cudaStatus);
  
  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(d_array, h_array, arraySizeBytes, cudaMemcpyHostToDevice);
  checkCudaErrors(cudaStatus);
  
  // Launch a kernel on the GPU with one thread for each element.
  sumKernel <<<1, NUM_ELEMENTS>>>(d_array, NUM_ELEMENTS);

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
  
  cudaFree(d_array);
        
  return 0;
}
