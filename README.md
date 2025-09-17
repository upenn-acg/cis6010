# CUDA GEMM Optimization

In this homework series, you'll optimize a CUDA implementation of 
[General Matrix Multiply aka GEMM](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3). 
Note that GEMM is slightly more involved than just matrix multiply: it also uses some constant scaling factors 
and adds the results to the existing values in the output matrix.

## Recommended Lambda.ai setup

TBD

## Recommended CETS Virtual PC Lab setup

TBD

## [Deprecated] Recommended EC2 setup

Use an EC2 `g4dn.xlarge` instance (currently the cheapest Nvidia GPU instance) with the `ami-05c3e698bd0cffe7e` AMI (an official Ubuntu 20.04 image with Nvidia GPU tools & PyTorch installed). Other AMIs can sometimes have fees associated with them.

You can use the cheapest storage (magnetic HDD) as disk performance doesn't matter for us. 
I recommend setting up an Elastic IP Address so that you have a consistent DNS name for your instance; it makes 
it much easier to connect to your instance via SSH and VSCode.

I also recommend using VSCode to write your code. Some key extensions to install are `Nsight Visual Studio Code Edition`, 
`Remote - SSH` and `C/C++ Extension Pack`. This allows you to connect to your instance as a "remote" and write
code on your local machine. It also provides integration with the `cuda-gdb` debugger which is very helpful.

Finally, install the [Nvidia Compute Insight profiler](https://developer.nvidia.com/tools-overview/nsight-compute/get-started) 
on your local machine (it's pre-installed on your instance) to allow you to peruse profiling reports easily. Note that you 
don't need to have an Nvidia GPU to view profiling data. You'll *generate* a profiling report on the EC2 instance and then *view* it on your local machine.

## Code overview

Our GEMM algorithms will operate on matrices with 32-bit float elements, which is the `float` datatype in CUDA.

At a high level, the code provided in `cugemm.cu` does the following:
1. allocates input and output square matrices of the requested size
2. initializes the input matrices with random values
3. runs the requested GEMM algorithm (more details below) for the requested number of repetitions
4. (optionally) validates the GEMM result

The matrix size, validation, repetition count and algorithm can all be controlled via command-line flags.

To begin with, only two GEMM algorithms are available: a naive version in `runBasic` 
and a super-optimized version from Nvidia's [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) library in `runCublas`. 
cuBLAS is the reference point for validation: if validation is requested then we run cuBLAS to get the correct answer
and compare the other algorithm's output to it.

## Build & profile

Build & profile the `runBasic` code as follows:

```
git checkout ...
cd <repo-working-copy>/gemm/
make -j3 all
./cugemm.bin --size=2048 --reps=1 --algo=1
```
This will build 3 versions of the code: an optimized version, an optimized version with some debugging information for profiling,
and one without optimizations and extra debugging symbols. 
When you run the optimized version `cugemm.bin` it should report a performance of around 60 GFLOPS, which is far below what the GPU can provide.

Next, we'll profile our kernel to see why it is so slow:
```
sudo /usr/local/cuda-11.8/bin/ncu -o profile-basic --set full ./cugemm-profile.bin --size=4096 --reps=1 --algo=1 --validate=false
```
> Note: you can follow [these instructions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#AllUsersTag) to avoid the need for `sudo` when profiling.

Because we used `--set full` to collect a full set of profiling data, it will take a couple minutes to run. The results 
are best viewed with the Nvidia Compute Insight profiler running in a graphical environment (i.e., not command-line) on a local machine.

Profiling will reveal an absurd number of uncoalesced global memory accesses. 

## Debug

Nvidia ships a number of "compute sanitizers" that check for common memory safety (e.g., out-of-bounds accesses) and concurrency errors. 
You should run them on your `debug` binaries to get better reporting of where errors are in your source code. They are an easy way to get
some clues about where to start when your code isn't passing validation.

```
compute-sanitizer --tool memcheck ./cugemm-debug.bin ...
compute-sanitizer --tool racecheck ./cugemm-debug.bin ...
```

## HW1: Fix uncoalesced memory accesses

Your first task is to fix the uncoalesced global memory accesses in `runBasic`. Copy the `runBasic` code to `runGmemCoalesced` and edit it there. Resolving the issues should result in a significant speedup (~550 GFLOPS on 2048<sup>2</sup> input matrices).

## HW2: Use shared memory

Cache tiles of the input matrices into shared memory, to avoid redundant loads to global memory. This should result in another significant speedup to ~1 TFLOPS.

## HW3: Multiple results per thread

Have each thread compute multiple cells of the output matrix C, instead of just one. This improves arithmetic intensity and should lift performance further to about ~3 TFLOPS. For reference, cuBLAS was reaching about 7.1 TFLOPS on my instance (with the T4's hardware limit being 8.1 TFLOPS), so we're over 40% of that optimal performance - not too shabby!
