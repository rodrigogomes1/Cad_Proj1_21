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
    int count = 0;
    char ppm[10];
    int c;
    // header
    while ((c = fgetc(f)) != EOF && count < 4) {
        if (isspace(c)) continue;
        if (c == '#') {
            while (fgetc(f) != '\n');
            continue;
        }
        ungetc(c, f);
        switch (count) {
            case 0:
                count += fscanf(f, "%2s", ppm);
                break;
            case 1:
                count += fscanf(f, "%d%d%d", width, height, maxcolors);
                break;
            case 2:
                count += fscanf(f, "%d%d", height, maxcolors);
                break;
            case 3:
                count += fscanf(f, "%d", maxcolors);

        }
    }
    assert(c != EOF);
    assert(strcmp("P3", ppm) == 0);
    // data
    int *data = *img = (int *) malloc(3 * (*width) * (*height) * sizeof(int));
    assert(img != NULL);
    int r, g, b, pos = 0;
    while (fscanf(f, "%d%d%d", &r, &g, &b) == 3) {
        data[pos++] = r;
        data[pos++] = g;
        data[pos++] = b;
    }
    assert(pos == 3 * (*width) * (*height));
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
        fputc('\n', f);
    }
}

/* printImg - print to screen the content of img
 */
void printImg(int imgh, int imgw, const int *img) {
    for (int j = 0; j < imgh; j++) {
        for (int i = 0; i < imgw; i++) {
            int x = 3 * (i + j * imgw);
            printf("%d,%d,%d  ", img[x], img[x + 1], img[x + 2]);
        }
        putchar('\n');
    }
}

/* averageImg - do the average of one point (line,col) with its 8 neighbours
 */
__global__ void averageImg(int *out, int *img,int lineSum,int line, int width, int height, int *filter, float alpha) {

    int r = 0, g = 0, b = 0;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + line;

    if (x < width && y < lineSum) {
        int n = 0;
        int f_pos = 0;
        for (int l = y - 1; l < y + 2 && l < height; l++) {
            for (int c = x - 1; c < x + 2 && c < width; c++) {
                if (l >= 0 && c >= 0) {
                    int idx = 3 * (l * width + c);
                    int scale = filter[f_pos];
                    r += img[idx] * scale;
                    g += img[idx + 1] * scale;
                    b += img[idx + 2] * scale;
                    n = n + scale;
                    f_pos = f_pos + 1;
                }
            }
        }

        if (n == 0) {
            n = 1;
        }
        if (r < 0) {
            r = 0;
        }
        if (b < 0) {
            b = 0;
        }
        if (g < 0) {
            g = 0;
        }

        r = r / n;
        g = g / n;
        b = b / n;
        float grey = alpha * (0.3 * r + 0.59 * g + 0.11 * b);

        int idx = 3 * (y * width + x);
        out[idx] = (1 - alpha) * r + grey;
        out[idx + 1] = (1 - alpha) * g + grey;
        out[idx + 2] = (1 - alpha) * b + grey;
    }
}
int * split(int nstreams, int height){
    int *res= (int*)malloc((nstreams) * sizeof(int));
    if (height % nstreams == 0) {
        for (int i = 0; i < nstreams; i++)
            res[i]=height/nstreams;
    } else {
        int zp = nstreams - (height % nstreams);
        int pp = height / nstreams;
        for (int i = 0; i < nstreams; i++) {
            if (i >= zp)
                res[i]= (pp + 1) ;
            else
                res[i]= pp;
        }
    }
    return res;
}


int main(int argc, char *argv[]) {
    int imgh, imgw, imgc;
    int *img;
    float alpha = 0.5;

    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s img.ppm [alpha]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 3) {
        alpha = atof(argv[2]);
        if (alpha > 1)
            alpha = 1;
        if (alpha < 0)
            alpha = 0;
    }
    FILE *f = fopen(argv[1], "r");
    if (f == NULL) {
        fprintf(stderr, "can't read file %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    read_ppm(f, &img, &imgw, &imgh, &imgc);
    printf("PPM image %dx%dx%d\n", imgw, imgh, imgc);
    // printImg(imgh, imgw, img);


    int *out = (int *) malloc((3 * imgw * imgh) * sizeof(int));
    assert(out != NULL);


    int *d_in;
    cudaMallocHost(&d_in, (imgh * imgw * 3) * sizeof(int));
    int *d_out;
    cudaMallocHost(&d_out, (imgh * imgw * 3) * sizeof(int));
    if (d_in == NULL || d_out == NULL) {
        fprintf(stderr, "No GPU mem!\n");
        return EXIT_FAILURE;
    }

    int filter[3][3] = {{1, 2, 1},  // gaussian filter
                        {2, 4, 2},
                        {1, 2, 1}};

    int *d_filter;
    cudaMallocHost(&d_filter, (3 * 3) * sizeof(int));
    clock_t t = clock();
    cudaMemcpy(d_filter, filter, (3 * 3) * sizeof(int), cudaMemcpyHostToDevice);
    //---------------------------------------

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }


    int *fill=split(NUM_STREAMS,imgh);

    int lineSum=0;
    int line = imgw * 3 * sizeof(int);
    for (int i = 0; i < NUM_STREAMS; i++) {
        int streamBytes = fill[i]*line;
        int offset = line * (lineSum) / sizeof(int);
        int startLine = lineSum;
        lineSum+=fill[i];


        //adicionar linha de cima para o filtro

        /*
        if (offset!=0){
            offset=offset-line;
            streamBytes+=line;
        }*/
        //adicionar linha de baixo para o filtro
        /*if (offset+streamBytes < line*imgh){
            streamBytes+=line;
        }*/
        printf("offset = %d\n", offset );
        printf("fill = %d\n", fill[i] );
        printf("StreamBytes = %d\n", streamBytes );
        cudaMemcpyAsync(&d_in[offset], &img[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);

        dim3 nb(imgw + (BLOCK_W - 1) / BLOCK_W, (fill[i]) + (BLOCK_H - 1) / BLOCK_H);
        dim3 th(BLOCK_W, BLOCK_H);

        averageImg<<<nb, th, 0, streams[i]>>>(d_out, d_in, lineSum ,startLine , imgw, imgh, d_filter, alpha);


        if (offset==0){
            cudaMemcpyAsync(&out[offset], &d_out[offset], streamBytes,cudaMemcpyDeviceToHost, streams[i]);
        }else{
            //TODO Refazer condição
            if (offset+streamBytes == line*imgh){
                //cudaMemcpyAsync(&out[offset], &d_out[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
            } else{
                cudaMemcpyAsync(&out[offset], &d_out[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
            }
        }

    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    t = clock() - t;
    printf("time %f ms\n", t / (double) (CLOCKS_PER_SEC / 1000));


    // printImg(imgh, imgw, out);
    FILE *g = fopen("outPipeline.ppm", "w");
    write_ppm(g, out, imgw, imgh, imgc);
    fclose(g);

    /*
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_filter);
*/
    return EXIT_SUCCESS;
}


