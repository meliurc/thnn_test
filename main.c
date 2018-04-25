

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <TH/TH.h>
#include "THNN.h"

#define Real Double
#define real double


void initStorageData(real *data, int size){
    for(int i=0; i<size; i++){
        *data = i;
        data++;
    }
    data = data - size;
}

int main(){
    int sizeInput = 3*32*32; int layersInput = 3; int heightInput = 32; int widthInput = 32;
    real *input = malloc(sizeInput*sizeof(real));
    initStorageData(input, sizeInput);
    THStorage* storageInput = THStorage_(newWithData)(input, (ptrdiff_t)sizeInput);
    THTensor* tensorInput = THTensor_(newWithStorage3d)(storageInput, 0,
                                                        layersInput, widthInput*heightInput,
                                                        heightInput, widthInput,
                                                        widthInput, 1);

    int sizeKernel = 3*3; int heightKernel = 3; int widthKernel = 3;
    real *kernel = malloc(sizeKernel*sizeof(real));
    initStorageData(kernel, sizeKernel);
    THStorage* storageKernel = THStorage_(newWithData)(kernel, (ptrdiff_t)sizeKernel);
    THTensor* tensorKernel = THTensor_(newWithStorage2d)(storageKernel, 0,
                                                         heightKernel, widthKernel,
                                                         widthKernel, 1);

    THTensor* tensorOutput = THTensor_(new)();
    THTensor* tensorFInput = THTensor_(new)();
    THTensor* tensorFGradInput = THTensor_(new)();

    THNNState* state = NULL;


    THNN_(SpatialConvolutionMM_updateOutput)(
            state,
            tensorInput,
            tensorOutput,
            tensorKernel,
            NULL,
            tensorFInput,
            tensorFGradInput,
            widthKernel,
            heightKernel,
            1,
            1,
            0,
            0);

}