streams.bin: streams.cu
	nvcc --generate-code=arch=compute_52,code=[compute_52,sm_52] $^ -o $@

run:
	./streams.bin 10 4

profile:
	nsys profile --trace=cuda,cudart ./streams.bin 10 4
