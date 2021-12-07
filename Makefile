


# -g for debugging, remove for performance evaluation 
CFLAGS=-g

.PHONY:	all

all:	main shared notshared kernels pipeline

main:	main.c 
	cc $(CFLAGS) -o $@ $< -lm


shared: shared.cu
	nvcc -o cudaShared shared.cu

notshared: not_shared.cu
	nvcc -o cudaNShared not_shared.cu

kernels: shared_twoKernels.cu
	nvcc -o cuda2Kernels shared_twoKernels.cu


pipeline: not_shared_pipeline.cu
	nvcc -o cudaPipeline not_shared_pipeline.cu

#gcc main.c -o main



.PHONY:	clean
clean:
	rm -f main out.ppm
	rm -f cudaShared outShared.ppm
	rm -f cudaNShared outNotShared.ppm
	rm -f cuda2Kernels outKernels.ppm
	rm -f cudaPipeline outPipeline.ppm
