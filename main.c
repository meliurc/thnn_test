//
// Created by jiguang on 2018/4/12.
//

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <TH/TH.h>
#include "THNN.h"

#define Real Double
#define real double

int main(){
    int size = 100;
    double *data = malloc(size*sizeof(real));
    for(int i=0; i<size; i++){
        *data = i;
        data++;
    }
    data = data - 100;

    THStorage* storage_input = THStorage_(newWithData)(data, (ptrdiff_t)100);
    THTensor* tensor_input = THTensor_(newWithStorage1d)(storage_input, 0, 100, 1);

    THStorage* storage_output = THStorage_(newWithSize1)(100);
    THTensor* tensor_output = THTensor_(newWithStorage1d)(storage_output, 0, 100, 1);

    THNNState* state;
    THNN_(Sigmoid_updateOutput)(state, tensor_input, tensor_output);

    for (int i=0; i<100; i++){
        printf("tensor_output->storage->data[%d] is %f \n", i, tensor_output->storage->data[i]);
    }

}