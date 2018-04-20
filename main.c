

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
    // initialize tensors, set tensor size and dimensions
    int sizeInput = 2; int rowsInput = 2; int colsInput = 1;
    int sizeW1 = 6; int rowsW1 = 2; int colsW1 = 3;
    int sizeW2 = 6; int rowsW2 = 3; int colsW2 = 2;
    int sizeW3 = 2; int rowsW3 = 2; int colsW3 = 1;
    int sizeLabel = 1; int rowsLabel = 1; int colsLabel = 1;

    real *input = malloc(sizeInput*sizeof(real));
    real *W1 = malloc(sizeW1*sizeof(real));
//    real *Z1 = malloc(rowsW2*sizeof(real));
//    real *A1 = malloc(rowsW2*sizeof(real));
    real *W2 = malloc(sizeW2*sizeof(real));
//    real *Z2 = malloc(rowsW3*sizeof(real));
//    real *A2 = malloc(rowsW3*sizeof(real));
    real *W3 = malloc(sizeW3*sizeof(real));
//    real *Z3 = malloc(sizeLabel*sizeof(real));
//    real *A3 = malloc(sizeLabel*sizeof(real));
    real *label = malloc(sizeLabel*sizeof(real));

    // initialize tensors, assign some "random" value
    initW(input, sizeInput);
    initW(W1, sizeW1);
    initW(W2, sizeW2);
    initW(W3, sizeW3);
    *label = 1;

    // initialize tensors, define THTensors
    THStorage* storageInput = THStorage_(newWithData)(input, (ptrdiff_t)sizeInput);
    THTensor* tensorInput = THTensor_(newWithStorage1d)(storageInput, 0, sizeInput, 1);

    THStorage* storageW1 = THStorage_(newWithData)(W1, (ptrdiff_t)sizeW1);
    THTensor* tensorW1 = THTensor_(newWithStorage2d)(storageW1, 0, rowsW1, colsW1, colsW1, 1);
    THStorage* storageZ1 = THStorage_(newWithSize)(rowsW2);
    THTensor* tensorZ1 = THTensor_(newWithStorage1d)(storageZ1, 0, rowsW2, 1);
    THStorage* storageA1 = THStorage_(newWithSize)(rowsW2);
    THTensor* tensorA1 = THTensor_(newWithStorage1d)(storageA1, 0, rowsW2, 1);

    THStorage* storageW2 = THStorage_(newWithData)(W2, (ptrdiff_t)sizeW2);
    THTensor* tensorW2 = THTensor_(newWithStorage2d)(storageW2, 0, rowsW2, colsW2, colsW2, 1);
    THStorage* storageZ2 = THStorage_(newWithSize)(rowsW3);
    THTensor* tensorZ2 = THTensor_(newWithStorage1d)(storageZ2, 0, rowsW3, 1);
    THStorage* storageA2= THStorage_(newWithSize)(rowsW3);
    THTensor* tensorA2 = THTensor_(newWithStorage1d)(storageA2, 0, rowsW3, 1);

    THStorage* storageW3 = THStorage_(newWithData)(W3, (ptrdiff_t)sizeW3);
    THTensor* tensorW3 = THTensor_(newWithStorage1d)(storageW3, 0, sizeW3, 1);
    THStorage* storageZ3 = THStorage_(newWithSize)(sizeLabel);
    THTensor* tensorZ3 = THTensor_(newWithStorage1d)(storageZ3, 0, sizeLabel, 1);
    THStorage* storageA3= THStorage_(newWithSize)(sizeLabel);
    THTensor* tensorA3 = THTensor_(newWithStorage1d)(storageA3, 0, sizeLabel, 1);

    THStorage* storageLabel = THStorage_(newWithData)(label, (ptrdiff_t)sizeLabel);
    THTensor* tensorLabel = THTensor_(newWithStorage1d)(storageLabel, 0, sizeLabel, 1);

//    real *Z1 = THBlas_(gemm)('t', 'n', rowsW1, colsW1, rowsW2, 1, a, lda, b, ldb, be, c, ldc);

}