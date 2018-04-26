//
// Created by jiguang on 2018/4/26.
//

#ifndef THNN_TEST_HELPOERFUNCTIONS_H
#define THNN_TEST_HELPOERFUNCTIONS_H

//#define Real Double
//#define real double

void initStorageData(real *data, int size);
void initKernelData(real *kernel, int size);
void printTensorData(THTensor* tensor);

#endif //THNN_TEST_HELPOERFUNCTIONS_H
