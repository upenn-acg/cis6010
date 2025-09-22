umr.bin: unified-memory-race.cu
	nvcc --generate-code=arch=compute_52,code=[compute_52,sm_52] $^ -o $@

run:
	./umr.bin
