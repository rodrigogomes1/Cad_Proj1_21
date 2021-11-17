//
// Created by vad on 16/10/21.
// DI FCTUNL
//

#ifndef LAB2_VSIZE_H
#define LAB2_VSIZE_H

// Vector Size
#define TILE_W 8
#define TILE_H 8
#define R 1
#define D (R*2+1)
#define S (D*D)
#define BLOCK_W (TILE_W+(2*R))
#define BLOCK_H (TILE_H+(2*R))

// when using threads use this thread group size by default
#define GROUPSIZE 1024

#endif //LAB2_VSIZE_H
