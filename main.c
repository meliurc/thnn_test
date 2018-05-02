

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <TH/TH.h>
#include "THNN.h"

#define Real Double
#define real double
#define accreal double

#include "utils/helperFunctions.h"

#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)

void THNN_(Linear_updateAddBuffer)(
        THNNState *state,
        THTensor *input,
        THTensor *addBuffer)
{
    long nframe = THTensor_(size)(input,0);
    long nElement = THTensor_(nElement)(addBuffer);
    if (nElement != nframe) {
        THTensor_(resize1d)(addBuffer,nframe);
        THTensor_(fill)(addBuffer,1.0);
    }
}

void THNN_(Linear_updateGradInput)(
        THNNState *state,
        THTensor *input,
        THTensor *gradOutput,
        THTensor *gradInput,
        THTensor *weight)
{
    if (gradInput) {
        long nElement = THTensor_(nElement)(gradInput);
        THTensor_(resizeAs)(gradInput,input);
        if (THTensor_(nElement)(gradInput) != nElement) {
            THTensor_(zero)(gradInput);
        }

        long dim = THTensor_(nDimension)(input);
        if (dim == 1) {
            THTensor *tweight = THTensor_(new)();
            THTensor_(transpose)(tweight,weight,0,1);
            THTensor_(addmv)(gradInput,0,gradInput,1,tweight,gradOutput);
            THTensor_(free)(tweight);
        }
        else if (dim == 2) {
            THTensor_(addmm)(gradInput,0,gradInput,1,gradOutput,weight);
        }
    }
}

void THNN_(Linear_accGradParameters)(
        THNNState *state,
        THTensor *input,
        THTensor *gradOutput,
        THTensor *gradInput,
        THTensor *weight,
        THTensor *bias,
        THTensor *gradWeight,
        THTensor *gradBias,
        THTensor *addBuffer,
        accreal scale_)
{
    real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
    long dim = THTensor_(nDimension)(input);
    if (dim == 1) {
        THTensor_(addr)(gradWeight,1,gradWeight,scale,gradOutput,input);
        if (bias) {
            THTensor_(cadd)(gradBias,gradBias,scale,gradOutput);
        }
    }
    else if (dim == 2) {
        THTensor *tgradOutput = THTensor_(new)();
        THTensor_(transpose)(tgradOutput,gradOutput,0,1);
        THTensor_(addmm)(gradWeight,1,gradWeight,scale,tgradOutput,input);
        if (bias) {
            THNN_(Linear_updateAddBuffer)(state,input,addBuffer);
            THTensor_(addmv)(gradBias,1,gradBias,scale,tgradOutput,addBuffer);
        }
        THTensor_(free)(tgradOutput);
    }
}

int main(){
    int layersInput = 2; int heightInput = 3; int widthInput = 3;
    int sizeInput = layersInput*heightInput*widthInput;
    real *input = malloc(sizeInput*sizeof(real));
    initStorageData(input, sizeInput);
    THStorage* storageCnnInput = THStorage_(newWithData)(input, (ptrdiff_t)sizeInput);
    THTensor* tensorCnnInput = THTensor_(newWithStorage3d)(storageCnnInput, 0,
                                                        layersInput, -1,
                                                        heightInput, -1,
                                                        widthInput,  -1);

    int nInputPlane = 2; int nOutputPlane = 2; int heightKernel = 2; int widthKernel = 2;
    int sizeKernel = nInputPlane*nOutputPlane*heightKernel*widthKernel;
    real *kernel = malloc(sizeKernel*sizeof(real));
    initKernelData(kernel, sizeKernel);
    THStorage* storageKernel = THStorage_(newWithData)(kernel, (ptrdiff_t)sizeKernel);
    THTensor* tensorKernel = THTensor_(newWithStorage4d)(storageKernel, 0,
                                                         nOutputPlane, -1,
                                                         nInputPlane,  -1,
                                                         heightKernel, -1,
                                                         widthKernel,  -1);

    int nLinearWeight = 1; int sizeLinearWeight = nOutputPlane*heightKernel*widthKernel;
    real *linearWeight = malloc(sizeLinearWeight*sizeof(real));
    initKernelData(linearWeight, sizeLinearWeight);
    THStorage* storageLinearWeight = THStorage_(newWithData)(linearWeight, (ptrdiff_t)sizeLinearWeight);
    THTensor* tensorLinearWeight = THTensor_(newWithStorage2d)(storageLinearWeight, 0,
                                                               nLinearWeight, -1,
                                                               sizeLinearWeight, -1);

    int sizeLinearBias = nOutputPlane*heightKernel*widthKernel;
    real *linearBias = malloc(sizeLinearBias*sizeof(real));
    initKernelData(linearBias, sizeLinearBias);
    THStorage* storageLinearBias = THStorage_(newWithData)(linearBias, (ptrdiff_t)sizeLinearBias);
    THTensor* tensorLinearBias = THTensor_(newWithStorage1d)(storageLinearBias, 0, nLinearWeight, -1);

    THTensor* tensorCnnOutput = THTensor_(new)();
    THTensor* tensorCnnFInput = THTensor_(new)();
    THTensor* tensorCnnFGradInput = THTensor_(new)();

    THNNState* state = NULL;

    THNN_(SpatialConvolutionMM_updateOutput)(
        state,
        tensorCnnInput,
        tensorCnnOutput,
        tensorKernel,
        NULL,
        tensorCnnFInput,
        NULL,
        widthKernel,
        heightKernel,
        1,
        1,
        0,
        0);

    THTensor *tensorLinearInput = THTensor_(newWithStorage2d)(tensorCnnOutput->storage, 0,
                                                              nLinearWeight, -1,
                                                              sizeLinearWeight, -1);
    THTensor *tensorLinearOutput = THTensor_(new)();
    THTensor *tensorLinearAddBuffer = THTensor_(new)();

    THNN_(Linear_updateOutput)(
            state,
            tensorLinearInput,
            tensorLinearOutput,
            tensorLinearWeight,
            tensorLinearBias,
            tensorLinearAddBuffer);

    THTensor *tensorLinearGradInput = THTensor_(new)();


    real *gradOutputOnes = malloc(nLinearWeight*sizeof(real));
    initKernelData(gradOutputOnes, nLinearWeight);
    THStorage* storageGradOutputOnes = THStorage_(newWithData)(gradOutputOnes, (ptrdiff_t)nLinearWeight);
    THTensor* tensorGradOutputOnes = THTensor_(newWithStorage2d)(storageGradOutputOnes, 0,
                                                                 1, -1,
                                                                 nLinearWeight, -1);

    THTensor *tensorLinearGradOutput = tensorGradOutputOnes;

    THNN_(Linear_updateGradInput)(
            state,
            tensorLinearInput,
            tensorLinearGradOutput,
            tensorLinearGradInput,
            tensorLinearWeight);

    printTensorData(tensorLinearGradInput);

    THTensor *tensorLinearGradWeight = THTensor_(newWithSize2d)(tensorLinearWeight->size[0], tensorLinearWeight->size[1]);
    THTensor *tensorLinearGradBias = THTensor_(newWithSize1d)(tensorLinearBias->size[0]);


    THNN_(Linear_accGradParameters)(
            state,
            tensorLinearInput,
            tensorLinearGradOutput,
            tensorLinearGradInput,
            tensorLinearWeight,
            tensorLinearBias,
            tensorLinearGradWeight,
            tensorLinearGradBias,
            tensorLinearAddBuffer,
            1.);

    printTensorData(tensorLinearGradWeight);

    // updateGradInput(input, gradOutput) + accGradParameters(input,gradOutput,scale)

//    THNN_(Linear_updateGradInput)(
//            THNNState *state,
//            THTensor *input,
//            THTensor *gradOutput,
//            THTensor *gradInput,
//            THTensor *weight)

//    THNN_(Linear_accGradParameters)(
//            THNNState *state,
//            THTensor *input,
//            THTensor *gradOutput,
//            THTensor *gradInput,
//            THTensor *weight,
//            THTensor *bias,
//            THTensor *gradWeight,
//            THTensor *gradBias,
//            THTensor *addBuffer,
//    accreal scale_)
}