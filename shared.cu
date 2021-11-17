//
// Created by Ruben on 27/10/2021.
//

/*
 * lab3 CAD 2021/2022 FCT/UNL
 * vad
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <cuda.h>

#include "vsize.h"


/* read_ppm - read a PPM image ascii file
 *   returns pointer to data, dimensions and max colors (from PPM header)
 *   data format: sequence of width x height of 3 ints for R,G and B
 *   aborts on errors
 */
void read_ppm(FILE *f, int **img, int *width, int *height, int *maxcolors) {
    int count=0;
    char ppm[10];
    int c;
    // header
    while ( (c = fgetc(f))!=EOF && count<4 ) {
        if (isspace(c)) continue;
        if (c=='#') {
            while (fgetc(f) != '\n')
                ;
            continue;
        }
        ungetc(c,f);
        switch (count) {
            case 0: count += fscanf(f, "%2s", ppm); break;
            case 1: count += fscanf(f, "%d%d%d", width, height, maxcolors); break;
            case 2: count += fscanf(f, "%d%d", height, maxcolors); break;
            case 3: count += fscanf(f, "%d", maxcolors);
        }
    }
    assert(c!=EOF);
    assert(strcmp("P3", ppm)==0);
    // data
    int *data= *img = (int*)malloc(3*(*width)*(*height)*sizeof(int));
    assert(img!=NULL);
    int r,g,b, pos=0;
    while ( fscanf(f,"%d%d%d", &r, &g, &b)==3) {
        data[pos++] = r;
        data[pos++] = g;
        data[pos++] = b;
    }
    assert(pos==3*(*width)*(*height));
}

/* write_ppm - write a PPM image ascii file
 */
void write_ppm(FILE *f, int *img, int width, int height, int maxcolors) {
    // header
    fprintf(f, "P3\n%d %d %d\n", width, height, maxcolors);
    // data
    for (int l = 0; l < height; l++) {
        for (int c = 0; c < width; c++) {
            int p = 3 * (l * width + c);
            fprintf(f, "%d %d %d  ", img[p], img[p + 1], img[p + 2]);
        }
        fputc('\n',f);
    }
}

/* printImg - print to screen the content of img
 */
void printImg(int imgh, int imgw, const int *img) {
    for (int j=0; j < imgh; j++) {
        for (int i=0; i<imgw; i++) {
            int x= 3*(i+j*imgw);
            printf("%d,%d,%d  ", img[x], img[x+1], img[x+2]);
        }
        putchar('\n');
    }
}


/* averageImg - do the average of one point (line,col) with its 8 neighbours
 */
__global__ void averageImg(int*out, int*img, int width, int height) {
    __shared__ int red[blockDim.x+2)*(blockDim.y+2)];
    __shared__ int green[blockDim.x+2)*(blockDim.y+2)];
    __shared__ int blue[blockDim.x+2)*(blockDim.y+2)];
    int r=0,g=0,b=0, n=0;
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;



    //int lindex=threadIdx.x+(blockDim.x*threadIdx.y);
    int lindex = (threadIdx.x+1)+((blockDim.x+2)*(threadIdx.y+1))
    int idx = 3*(y*width+x);
    red[lindex] = img[idx];
    green[lindex] = img[idx+1];
    blue[lindex] = img[idx+2];

    if(threadIdx.x==0){
        //esuerda
           if(x==0){
               red[lindex-1]=img[idx];
               green[lindex-1]=img[idx+1];
               blue[lindex-1]=img[idx+2];
           }else{
               red[lindex-1]=img[idx-3];
               green[lindex-1]=img[idx-2];
               blue[lindex-1]=img[idx-1];
           }
         //esquerda cima
         //TODO CHECKAR CANTO ESQUERDO, COMPUTAR PIXEL EM CIMA DO BLOCO ANTERIOR, E COMPUTAR LOCAL INDEX DO PIXEL ACIMA NA MEMORIA PARTILHADA

         if(threadIdx.y==0){
             int globalabove = (blockIdx.y-1)*blockDim.y+threadIdx.y;
             int above = (blockIdx.y)*blockDim.y+threadIdx.y;
             if(y==0){
                 red[threadIdx.x+1]=img[idx];
                 green[threadIdx.x+1]=img[idx+1];
                 blue[threadIdx.x+1]=img[idx+2];
                 red[threadIdx.x]=img[idx];
                 green[threadIdx.x]=img[idx+1];
                 blue[threadIdx.x]=img[idx+2];
             }else{
                 red[threadIdx.x+1]=img[globalabove];
                 green[threadIdx.x+1]=img[globalabove+1];
                 blue[threadIdx.x+1]=img[globalabove+2];
                 red[threadIdx.x]=img[globalabove];
                 green[threadIdx.x]=img[globalabove+1];
                 blue[threadIdx.x]=img[globalabove+2];
             }
         }


    }


    __syncthreads();
    //unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x
    //unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y

    for (int l=y-1; l<y+2 && l<height; l++)
        for (int c=x-1; c<x+2 && c<width; c++)
            if (l>=0 && c>=0 ) {
                int idx = (l*width+c);
                r+=red[idx]; g+=green[idx]; b+=blue[idx];
                n++;
            }
    int idx = 3*(y*width+x);
    out[idx]=r/n;
    out[idx+1]=g/n;
    out[idx+2]=b/n;

}



int main(int argc, char *argv[]) {
    int imgh, imgw, imgc;
    int *img;
    if (argc!=2) {
        fprintf(stderr,"usage: %s img.ppm\n", argv[0]);
        return EXIT_FAILURE;
    }
    FILE *f=fopen(argv[1],"r");
    if (f==NULL) {
        fprintf(stderr,"can't read file %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    read_ppm(f, &img, &imgw, &imgh, &imgc);
    printf("PPM image %dx%dx%d\n", imgw, imgh, imgc);
    // printImg(imgh, imgw, img);

    int *out = (int*)malloc((3*imgw*imgh)*sizeof(int));
    assert(out!=NULL);


    int *d_in;
    cudaMalloc(&d_in, (imgh*imgw*3)*sizeof(int));
    int *d_out;
    cudaMalloc(&d_out, (imgh*imgw*3)*sizeof(int));
    if ( d_in==NULL || d_out==NULL) {
        fprintf(stderr,"No GPU mem!\n");
        return EXIT_FAILURE;
    }
    cudaMemcpy(d_in,img, (imgh*imgw*3)*sizeof(int),cudaMemcpyHostToDevice);




    clock_t t = clock();
    dim3 nb(imgw/8,imgh/8);
    dim3 th(8,8);

    averageImg<<<nb, th>>>(d_out,d_in, imgw, imgh);
    
    cudaMemcpy(out, d_out, (imgh*imgw*3)*sizeof(int),cudaMemcpyDeviceToHost);
    t = clock()-t;
    printf("time %f ms\n", t/(double)(CLOCKS_PER_SEC/1000));
    

    // printImg(imgh, imgw, out);
    FILE *g=fopen("out.ppm", "w");
    write_ppm(g, out, imgw, imgh, imgc);
    fclose(g);
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return EXIT_SUCCESS;
}


