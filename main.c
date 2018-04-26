

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <TH/TH.h>
#include "THNN.h"

#define Real Double
#define real double

#include "utils/helperFunctions.h"

static THTensor* THNN_(view_weight_MM2d)(THTensor *weight) {
    weight = THTensor_(newContiguous)(weight);
    if (weight->nDimension == 4) {
        long s1 = weight->size[0];
        long s2 = weight->size[1] * weight->size[2] * weight->size[3];
        THTensor *old_weight = weight;
        weight = THTensor_(newWithStorage2d)(weight->storage, weight->storageOffset,
                                             s1, -1, s2, -1);
        THTensor_(free)(old_weight);
    }
    return weight;
}

int main(){
    int layersInput = 1; int heightInput = 3; int widthInput = 3;
    int sizeInput = layersInput*heightInput*widthInput;
    real *input = malloc(sizeInput*sizeof(real));
    initStorageData(input, sizeInput);
    THStorage* storageInput = THStorage_(newWithData)(input, (ptrdiff_t)sizeInput);
    THTensor* tensorInput = THTensor_(newWithStorage3d)(storageInput, 0,
                                                        layersInput, -1,
                                                        heightInput, -1,
                                                        widthInput,  -1);

    int nInputPlane = 1; int nOutputPlane = 1; int heightKernel = 2; int widthKernel = 2;
    int sizeKernel = nInputPlane*nOutputPlane*heightKernel*widthKernel;
    real *kernel = malloc(sizeKernel*sizeof(real));
    initKernelData(kernel, sizeKernel);
    THStorage* storageKernel = THStorage_(newWithData)(kernel, (ptrdiff_t)sizeKernel);
    THTensor* tensorKernel = THTensor_(newWithStorage4d)(storageKernel, 0,
                                                         nOutputPlane, -1,
                                                         nInputPlane,  -1,
                                                         heightKernel, -1,
                                                         widthKernel,  -1);

    THTensor* tensorOutput = THTensor_(new)();
    THTensor* tensorFInput = THTensor_(new)();
    THTensor* tensorFGradInput = THTensor_(new)();

    THNNState* state = NULL;

    tensorKernel = THNN_(view_weight_MM2d)(tensorKernel);

//    input = THTensor_(newContiguous)(input);
//    int ndim = input->nDimension;
//    int dimf = 0;
//    int dimh = 1;
//    int dimw = 2;
//
//    if (ndim == 4) {
//        dimf++;
//        dimh++;
//        dimw++;
//    }
//
//    THNN_(SpatialConvolutionMM_updateOutput)(
//            state,
//            tensorInput,
//            tensorOutput,
//            tensorKernel,
//            NULL,
//            tensorFInput,
//            NULL,
//            widthKernel,
//            heightKernel,
//            1,
//            1,
//            0,
//            0);
//
//    printTensorData(tensorOutput);
}