


# -g for debugging, remove for performance evaluation 
CFLAGS=-g

.PHONY:	all

all:	main

main:	main.c 
	cc $(CFLAGS) -o $@ $< -lm

#gcc main.c -o main

#nvcc -o cudaShared shared.cu
#nvcc -o cudaNShared not_shared.cu
#nvcc -o cuda2Kernels shared_twoKernels.cu


.PHONY:	clean
clean:
	rm -f main out.ppm
