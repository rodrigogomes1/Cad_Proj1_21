//
// Created by vad on 16/10/21.
// DI FCTUNL
//

#ifndef LAB2_VSIZE_H
#define LAB2_VSIZE_H

// Vector Size
#define TILE_W (BLOCK_W-(2*R))
#define TILE_H (BLOCK_H-(2*R))
#define R 1
#define D (R*2+1)
#define S (D*D)
#define BLOCK_W 8
#define BLOCK_H 8

// when using threads use this thread group size by default
#define GROUPSIZE 1024

#endif //LAB2_VSIZE_H
