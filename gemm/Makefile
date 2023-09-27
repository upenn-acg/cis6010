all: cugemm.bin cugemm-debug.bin cugemm-profile.bin

SOURCE_FILE=cugemm.cu

# optimized binary
cugemm.bin: $(SOURCE_FILE)
	nvcc -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# debug binary without optimizations
cugemm-debug.bin: $(SOURCE_FILE)
	nvcc -g -G -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# optimized binary with line number information for profiling
cugemm-profile.bin: $(SOURCE_FILE)
	nvcc -g --generate-line-info -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

clean:
	rm -f cugemm*.bin
