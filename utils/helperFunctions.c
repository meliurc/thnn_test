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

void initKernelData(real *kernel, int size){
    for(int i=0; i<size; i++){
        *kernel = 1;
        kernel++;
    }
    kernel = kernel - size;
}

void printTensorData(THTensor* tensor){
    if (tensor->nDimension == 1){
        long H = tensor->size[0];

        for (int i=0; i < tensor->size[0]; i++) {
            printf("%.2f", tensor->storage->data[i]);
            if (i < tensor->size[0]-1)
                printf(", ");
        }
        printf("\n");
    }

    if (tensor->nDimension == 2){
        long H = tensor->size[0];
        long W = tensor->size[1];

        for (int j=0; j<tensor->size[0]; j++){
            for (int i=0; i < tensor->size[1]; i++) {
                printf("%.2f", tensor->storage->data[j*W + i]);
                if (i < tensor->size[1]-1)
                    printf(", ");
            }
            printf("\n");
        }
        printf("\n");
    }

    if (tensor->nDimension == 3){
        long M = tensor->size[0];
        long H = tensor->size[1];
        long W = tensor->size[2];

        for (int k=0; k<tensor->size[0]; k++){
            printf("(%d, :, :) \n", k);
            for (int j=0; j<tensor->size[1]; j++){
                for (int i=0; i < tensor->size[2]; i++) {
                    printf("%.2f", tensor->storage->data[k*H*W + j*H + i]);
                    if (i < tensor->size[2]-1)
                        printf(", ");
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    if (tensor->nDimension == 4){
        long M = tensor->size[0];
        long N = tensor->size[1];
        long H = tensor->size[2];
        long W = tensor->size[3];

        for(int m=0; m<tensor->size[0]; m++){
            printf("(%d, :, :, :) \n", m);
            for (int k=0; k<tensor->size[1]; k++){
                printf("(%d, %d, :, :) \n", m, k);
                for (int j=0; j<tensor->size[2]; j++){
                    for (int i=0; i < tensor->size[3]; i++) {
                        printf("%.2f", tensor->storage->data[k*H*W + j*H + i]);
                        if (i < tensor->size[3]-1)
                            printf(", ");
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}