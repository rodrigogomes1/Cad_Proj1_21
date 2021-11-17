
# -g for debugging, remove for performance evaluation 
CFLAGS=-g

.PHONY:	all

all:	main

main:	main.c 
	cc $(CFLAGS) -o $@ $< -lm


.PHONY:	clean
clean:
	rm -f main out.ppm
