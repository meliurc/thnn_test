

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <TH/TH.h>
#include "THNN.h"

#define Real Double
#define real double


void initW(real *data, int size){
    for(int i=0; i<size; i++){
        *data = i;
        data++;
    }
    data = data - size;
}

int main(){
    int sizeInput = 2; int rowsInput = 2; int colsInput = 1;
    int sizeW1 = 4; int rowsW1 = 2; int colsW1 = 2;
    int sizeW2 = 4; int rowsW2 = 2; int colsW2 = 2;
    int sizeW3 = 2; int rowsW3 = 2; int colsW3 = 1;
    int sizeOutput = 1; int rowsOutput = 1; int colsOutput = 1;

    real *input = malloc(sizeInput*sizeof(real));
    real *W1 = malloc(sizeW1*sizeof(real));
    real *W2 = malloc(sizeW2*sizeof(real));
    real *W3 = malloc(sizeW3*sizeof(real));
    real *output = malloc(sizeOutput*sizeof(real));

    initW(input, sizeInput);
    initW(W1, sizeW1);
    initW(W2, sizeW2);
    initW(W3, sizeW3);

    real yReal = 1;

    THStorage* storageInput = THStorage_(newWithData)(input, (ptrdiff_t)sizeInput);
    THTensor* tensorInput = THTensor_(newWithStorage1d)(storageInput, 0, sizeInput, 1);

    THStorage* storageW1 = THStorage_(newWithData)(W1, (ptrdiff_t)sizeW1);
    THTensor* tensorW1 = THTensor_(newWithStorage2d)(storageW1, 0, rowsW1, colsW1, colsW1, 1);

    THStorage* storageW2 = THStorage_(newWithData)(W2, (ptrdiff_t)sizeW2);
    THTensor* tensorW2 = THTensor_(newWithStorage2d)(storageW2, 0, rowsW2, colsW2, colsW2, 1);

    THStorage* storageW3 = THStorage_(newWithData)(W3, (ptrdiff_t)sizeW3);
    THTensor* tensorW3 = THTensor_(newWithStorage1d)(storageW2, 0, sizeW3, 1);

    THStorage* storageOutput = THStorage_(newWithData)(output, (ptrdiff_t)sizeW3);
    THTensor* tensorOutput = THTensor_(newWithStorage1d)(storageW2, 0, sizeW3, 1);

    real *Z1 = THBlas_(gemm)('n', 't', m, n, k, al, a, lda, b, ldb, be, c, ldc);

}