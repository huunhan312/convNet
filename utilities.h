/***************************************************************
* DSDL - Soongsil University                                   *
* Huu Nhan - Dec 01, 2016                                      *
* This code is for implement convolutional neural network      *
****************************************************************/
#include "cnpy.h"
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <string>

#include "parameters.h"

float** dsdl_loadWeight_2d(const char* filename, int* filter_infor);
float* dsdl_loadWeight_1d(const char* filename);

float** dsdl_loadWeightfc_2d(const char* filename, int* w_infor);

float* dsdl_flatten3d(float*** inArr, int* i_infor);

float*** dsdl_loadOutconv(const char* filename, int* o_infor);