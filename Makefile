


# -g for debugging, remove for performance evaluation 
CFLAGS=-g

.PHONY:	all

all:	main

main:	main.c 
	cc $(CFLAGS) -o $@ $< -lm


shared: shared.cu
	nvcc -o cudaShared shared.cu

notshared: not_shared.cu
	nvcc -o cudaNShared not_shared.cu

kernels: shared_twoKernels.cu
	nvcc -o cuda2Kernels shared_twoKernels.cu

#gcc main.c -o main



.PHONY:	clean
clean:
	rm -f main out.ppm
