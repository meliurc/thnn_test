

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <TH/TH.h>
#include "THNN.h"

#define Real Double
#define real double

#include "utils/helperFunctions.h"

void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *m1, THTensor *m2)
{
    char transpose_r, transpose_m1, transpose_m2;
    THTensor *r__, *m1_, *m2_;

    if( (m1->nDimension != 2) || (m2->nDimension != 2))
        THError("matrices expected, got %dD, %dD tensors", m1->nDimension, m2->nDimension);

    if(m1->size[1] != m2->size[0]) {
        THDescBuff bm1 = THTensor_(sizeDesc)(m1);
        THDescBuff bm2 = THTensor_(sizeDesc)(m2);
        THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
    }

    if( t->nDimension != 2 )
        THError("matrix expected, got %dD tensor for t", t->nDimension);

    if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) ) {
        THDescBuff bt  = THTensor_(sizeDesc)(t);
        THDescBuff bm1 = THTensor_(sizeDesc)(m1);
        THDescBuff bm2 = THTensor_(sizeDesc)(m2);
        THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
    }

    if(t != r_)
    {
        THTensor_(resizeAs)(r_, t);
        if (beta != 0.0) {
            THTensor_(copy)(r_, t);
        }
    }

    /* r_ */
    if(r_->stride[0] == 1 &&
       r_->stride[1] != 0)
    {
        transpose_r = 'n';
        r__ = r_;
    }
    else if(r_->stride[1] == 1 &&
            r_->stride[0] != 0)
    {
        THTensor *swap = m2;
        m2 = m1;
        m1 = swap;
        transpose_r = 't';
        r__ = r_;
    }
    else
    {
        transpose_r = 'n';

        THTensor *transp_r_ = THTensor_(newTranspose)(r_, 0, 1);
        r__ = THTensor_(newClone)(transp_r_);
        THTensor_(free)(transp_r_);
        THTensor_(transpose)(r__, NULL, 0, 1);
    }

    /* m1 */
    if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
       m1->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
    {
        transpose_m1 = 'n';
        m1_ = m1;
    }
    else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
            m1->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
    {
        transpose_m1 = 't';
        m1_ = m1;
    }
    else
    {
        transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
        m1_ = THTensor_(newContiguous)(m1);
    }

    /* m2 */
    if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
       m2->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
    {
        transpose_m2 = 'n';
        m2_ = m2;
    }
    else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
            m2->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
    {
        transpose_m2 = 't';
        m2_ = m2;
    }
    else
    {
        transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
        m2_ = THTensor_(newContiguous)(m2);
    }

#pragma omp critical(blasgemm)
    /* do the operation */
    THBlas_(gemm)(transpose_m1,
                  transpose_m2,
                  r__->size[(transpose_r == 'n' ? 0 : 1)],
                  r__->size[(transpose_r == 'n' ? 1 : 0)],
                  m1_->size[(transpose_r == 'n' ? 1 : 0)],
                  alpha,
                  THTensor_(data)(m1_),
                  (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  THTensor_(data)(m2_),
                  (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  beta,
                  THTensor_(data)(r__),
                  r__->stride[(transpose_r == 'n' ? 1 : 0)]);

    /* free intermediate variables */
    if(m1_ != m1)
        THTensor_(free)(m1_);

    if(m2_ != m2)
        THTensor_(free)(m2_);

    if(r__ != r_)
        THTensor_(freeCopyTo)(r__, r_);
}

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

static void THNN_(SpatialConvolutionMM_updateOutput_frame)(
        THTensor *input,
        THTensor *output,
        THTensor *weight,
        THTensor *bias,
        THTensor *finput,
        int kW,
        int kH,
        int dW,
        int dH,
        int padW,
        int padH,
        long nInputPlane,
        long inputWidth,
        long inputHeight,
        long nOutputPlane,
        long outputWidth,
        long outputHeight)
{
    long i;
    THTensor *output2d;

    printTensorData(input);

    THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH,
                         nInputPlane, inputWidth, inputHeight,
                         outputWidth, outputHeight);

    output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset,
                                           nOutputPlane, -1,
                                           outputHeight*outputWidth, -1);
    if (bias) {
        for(i = 0; i < nOutputPlane; i++)
            THVector_(fill)
                    (output->storage->data + output->storageOffset + output->stride[0] * i,
                     THTensor_(get1d)(bias, i), outputHeight*outputWidth);
    } else {
        THTensor_(zero)(output);
    }

    printTensorData(weight);
    printTensorData(finput);

//    void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *m1, THTensor *m2)
    THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);

    printTensorData(output);

    THTensor_(free)(output2d);
}

int main(){
    int layersInput = 2; int heightInput = 3; int widthInput = 3;
    int sizeInput = layersInput*heightInput*widthInput;
    real *input = malloc(sizeInput*sizeof(real));
    initStorageData(input, sizeInput);
    THStorage* storageInput = THStorage_(newWithData)(input, (ptrdiff_t)sizeInput);
    THTensor* tensorInput = THTensor_(newWithStorage3d)(storageInput, 0,
                                                        layersInput, -1,
                                                        heightInput, -1,
                                                        widthInput,  -1);

    int nInputPlane = 2; int nOutputPlane = 1; int heightKernel = 2; int widthKernel = 2;
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
    tensorInput = THTensor_(newContiguous)(tensorInput);
    int ndim = tensorInput->nDimension;
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4) {
        dimf++;
        dimh++;
        dimw++;
    }

    nInputPlane  = tensorInput->size[dimf];
    long inputHeight  = tensorInput->size[dimh];
    long inputWidth   = tensorInput->size[dimw];
    nOutputPlane = tensorKernel->size[0];
    long outputHeight = (inputHeight + 2*0 - heightKernel) / 1 + 1;
    long outputWidth  = (inputWidth + 2*0 - widthKernel) / 1 + 1;

    THTensor_(resize2d)(tensorFInput, widthKernel*heightKernel*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(tensorOutput, nOutputPlane, outputHeight, outputWidth);

    THNN_(SpatialConvolutionMM_updateOutput_frame)
            (tensorInput, tensorOutput, tensorKernel, NULL, tensorFInput,
             widthKernel, heightKernel, 1, 1, 0, 0,
             nInputPlane, inputWidth, inputHeight,
             nOutputPlane, outputWidth, outputHeight);

//    printTensorData(tensorOutput);


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