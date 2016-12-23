/***************************************************************
* DSDL - Soongsil University                                   *
* Huu Nhan - Dec 01, 2016                                      *
* This code is for implement convolutional neural network      *
****************************************************************/
#include <iostream>
#include <stdio.h>
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>

/**
**  this structure will hold the parameter of a convolutional layer
**
struct dsdl_conv_param {
	int f_infor[4]; // filter infor
  	int i_infor[3]; // input infor
  	const char* w_name; // weight file
  	const char* b_name;// bias name
  	float** filter; //hold filter weight
  	float* Bias; //hold bias
  	float*** o_conv; //hold conv result
};

/** In the beginning these functions for ReLU function 
*** But now, it is not useful, becuz reLU is integrated to conv2d
**/


float*** dsdl_conv2d_in(uchar* i_input,int* i_infor, float** f_value, int* f_infor,float* bias);
float*** dsdl_conv2d_hid(float*** i_input, int* i_infor, float** f_value, int* f_infor, float* bias);
float* dsdl_fullyConnected(float* inArr, int inArr_len, float** w_arr, int* w_infor, float* bias,bool brelu);
int dsdl_softmax1d(float* inArr,int inlength);

//float*** dsdl_ReLU(float*** inRelu,int row, int col, int depth);
float*** dsdl_concatenate(float*** firstArr, int* st_infor, float*** secondArr, int* nd_infor, float*** thirdArr, int* rd_infor);


float* getMaxArray(float*** inArr,int toplx,int toply,int rowpool, int colpool,int* i_infor);
float*** dsdl_pooling(float*** inArr,int* i_infor,int rowpool, int colpool);

