#include <stdio.h>

__global__ void getTid(int* myTid, int* maxTidPerBlock) {
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

  // write my own tid
  myTid[tid] = tid;

  // TODO: fill myTid with max tid from each block

  // TODO: write max tid from block into maxTidPerBlock

}

__global__ void getMaxTid(int* maxTids) {
  // TODO: find max among all blocks
}

int main(void) {

  const int BLOCKS = 32;
  const int THREADS_PER_BLOCK = 64;

  int* mtpb = (int*) malloc( BLOCKS * sizeof(int) );
  int *d_myTid, *d_mtpb;
  cudaMalloc( &d_myTid, BLOCKS * THREADS_PER_BLOCK * sizeof(int) );
  cudaMalloc( &d_mtpb, BLOCKS * sizeof(int) );

  // launch kernels
  getTid<<<BLOCKS, THREADS_PER_BLOCK>>>(d_myTid, d_mtpb);

  cudaMemcpy(mtpb, d_mtpb, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < BLOCKS; i++) {
    printf( "%d ", mtpb[i] );
  }
  printf("\n");

}
