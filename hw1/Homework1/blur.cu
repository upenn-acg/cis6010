
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <atlimage.h>

const double PI = 3.14159265358979323846;

const int BLOCKDIM = 32;

// Note: this must be an odd number
__device__ const int FILTER_SIZE = 15;
__device__ const int FILTER_RADIUS = FILTER_SIZE >> 1;

__device__ const int BLUE_MASK = 0x00ff0000;
__device__ const int GREEN_MASK = 0x0000ff00;
__device__ const int RED_MASK = 0x000000ff;

/** compute index into an int* for the pixel (x,y) in the given 2D pitched allocation */
__device__ int index(int x, int y, const cudaPitchedPtr& cpp) {
	// divide by 4 because each pixel is 4B and cpp.pitch is in bytes, but we need to return an index
	return (y * (cpp.pitch / 4)) + x;
}

/** Clamp the given value to the interval [0,bound) */
__device__ int clamp(int value, int bound) {
	if (value < 0) {
		return 0;
	}
	if (value < bound) {
		return value;
	}
	return bound - 1;
}

/** Compute a Gaussian blur of src image and place into dst. Use only global memory. */
__global__ void blurGlobal(cudaPitchedPtr src, cudaPitchedPtr dst, float* gaussian) {

	int x = (blockDim.x * blockIdx.x) + threadIdx.y;
	int y = (blockDim.y * blockIdx.y) + threadIdx.x;

	float r = 0.0, g = 0.0, b = 0.0;

	for (int ky = 0; ky < FILTER_SIZE; ky++) {
		for (int kx = 0; kx < FILTER_SIZE; kx++) {
			// this replicates border pixels
			int i = index(clamp(x + kx - FILTER_RADIUS, src.xsize / 4),
				clamp(y + ky - FILTER_RADIUS, src.ysize), src);
			unsigned int pixel = ((int*)src.ptr)[i];
			// convolute each channel separately
			const float k = gaussian[(ky * FILTER_SIZE) + kx];
			b += (float)((pixel & BLUE_MASK) >> 16) * k;
			g += (float)((pixel & GREEN_MASK) >> 8) * k;
			r += (float)((pixel & RED_MASK)) * k;
		}
	}
	// Re-assemble destination pixel
	unsigned int dpixel = 0x00000000
		| ((((int)b) << 16) & BLUE_MASK)
		| ((((int)g) << 8) & GREEN_MASK)
		| (((int)r) & RED_MASK);
	((int*)dst.ptr)[index(x, y, dst)] = dpixel;
}



void setupGaussian(float** d_gaussian) {
	// calculate gaussian blur filter
	float gaussian[FILTER_SIZE][FILTER_SIZE];
	double sigma = 5.0;
	double mean = FILTER_SIZE / 2;
	for (int x = 0; x < FILTER_SIZE; ++x) {
		for (int y = 0; y < FILTER_SIZE; ++y) {
			double g = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (2 * PI * sigma * sigma);
			gaussian[y][x] = (float)g;
		}
	}
	// normalize the filter
	float sum = 0.0;
	for (int x = 0; x < FILTER_SIZE; ++x) {
		for (int y = 0; y < FILTER_SIZE; ++y) {
			sum += gaussian[y][x];
		}
	}
	for (int x = 0; x < FILTER_SIZE; ++x) {
		for (int y = 0; y < FILTER_SIZE; ++y) {
			gaussian[y][x] /= sum;
		}
	}

	// copy gaussian to device memory
	cudaError_t cudaStatus = cudaMalloc(d_gaussian, FILTER_SIZE * FILTER_SIZE * sizeof(float));
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMemcpy(*d_gaussian, &gaussian[0], FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);
}

// used by startTimer() and stopTimer()
cudaEvent_t start, stop;

void startTimer() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

/** Return elapsed time (in ms) since startTime() was called */
float stopTimer() {
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	return time;
}

int main() {

	// LOAD IMAGE FROM FILE
	CImage img;
	img.Load("C:\\Users\\Administrator\\Source\\Repos\\cis601\\hw1\\steel_wool_large.jpg");

	// ensure that image dimensions are a multiple of the block size
	if (img.GetHeight() % BLOCKDIM != 0) {
		printf("ERROR: image height (%d) must be a multiple of the block size (%d)\n", img.GetHeight(), BLOCKDIM);
		return 1;
	}
	if (img.GetWidth() % BLOCKDIM != 0) {
		printf("ERROR: image width (%d) must be a multiple of the block size (%d)\n", img.GetWidth(), BLOCKDIM);
		return 1;
	}

	const int IMG_WIDTH_BYTES = img.GetWidth() * 4;

	// COLORREF is a 32-bit int of the form 0x00bbggrr (8 bits for each of the blue, green, and red channels)
	COLORREF* h_buf = new COLORREF[img.GetWidth() * img.GetHeight()];

	// fill up h_buf
	for (int y = 0; y < img.GetHeight(); y++) {
		for (int x = 0; x < img.GetWidth(); x++) {
			COLORREF c = img.GetPixel(x, y);
			h_buf[(y * img.GetWidth()) + x] = c;
		}
	}

	cudaError_t cudaStatus;

	// use 48KB for shared memory, and 16KB for L1D$
	cudaStatus = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	checkCudaErrors(cudaStatus);

	// ensure kernel timeout is disabled
	int kernelTimeout;
	cudaStatus = cudaDeviceGetAttribute(&kernelTimeout, cudaDevAttrKernelExecTimeout, 0/*device*/);
	checkCudaErrors(cudaStatus);
	if (kernelTimeout != 0) {
		printf("WARNING: kernel timeout is enabled!\n", kernelTimeout);
	}

	// COPY IMAGE BUFFERS AND FILTER TO DEVICE
	startTimer();
	cudaExtent extent = make_cudaExtent(IMG_WIDTH_BYTES, img.GetHeight(), 1);
	cudaPitchedPtr d_src, d_dst;
	cudaStatus = cudaMalloc3D(&d_src, extent);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaMemcpy2D(d_src.ptr, d_src.pitch,
		h_buf, IMG_WIDTH_BYTES, IMG_WIDTH_BYTES, img.GetHeight(),
		cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMalloc3D(&d_dst, extent);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaMemset2D(d_dst.ptr, d_dst.pitch, 0, IMG_WIDTH_BYTES, img.GetHeight());
	checkCudaErrors(cudaStatus);

	float* d_gaussian;
	setupGaussian(&d_gaussian);

	printf("Copy to device:  %3.1f ms \n", stopTimer());

	// LAUNCH KERNEL

	for (int i = 0; i < 5; i++) {
		startTimer();
		dim3 blocksInGrid(img.GetWidth() / BLOCKDIM, img.GetHeight() / BLOCKDIM);
		dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
		blurGlobal<<<blocksInGrid, threadsPerBlock>>>(d_src, d_dst, d_gaussian);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		checkCudaErrors(cudaStatus);

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		checkCudaErrors(cudaStatus);

		printf("Kernel time:  %3.1f ms \n", stopTimer());
	}

	// COPY  OUTPUT IMAGE BACK TO HOST
	startTimer();
	cudaStatus = cudaMemcpy2D(h_buf, IMG_WIDTH_BYTES,
		d_dst.ptr, d_dst.pitch, IMG_WIDTH_BYTES, d_dst.ysize,
		cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);

	printf("Copy from device:  %3.1f ms \n", stopTimer());

	// WRITE OUT UPDATED IMAGE
	for (int y = 0; y < img.GetHeight(); y++) {
		for (int x = 0; x < img.GetWidth(); x++) {
			COLORREF c = h_buf[(y * img.GetWidth()) + x];
			img.SetPixel(x, y, c);
		}
	}
	img.Save("C:\\Users\\Administrator\\Source\\Repos\\cis601\\hw1\\out.jpg");

	// CLEANUP

	cudaStatus = cudaFree(d_src.ptr);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaFree(d_dst.ptr);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaFree(d_gaussian);
	checkCudaErrors(cudaStatus);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	checkCudaErrors(cudaStatus);

	return 0;
}
