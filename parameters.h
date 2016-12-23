/***************************************************************
* DSDL - Soongsil University                                   *
* Huu Nhan - Dec 01, 2016                                      *
* This code is for implement convolutional neural network      *
****************************************************************/

/*
This file declares the parameters of each layer in the network.
The saved weight is extracted from tensorflow check-point and 
saving each layer 
*/

#ifndef PARAMS
#define PARAMS

#include <string>


/********** FLAG DEFINE ***************/ 
#define SHOW_LOAD_W 0
#define SHOW_LOAD_B 0
#define SHOW_LOAD_FC 0
#define SHOW_LOAD_OUT 0

/*   parameter for conv layer 1 */
const int I_HEIGHT1				= 64; //x
const int I_WIDTH1           	= 64;
const int I_CH1 				=3;

const int F_WIDTH1				= 1;
const int F_HEIGHT1				= 3;
const int F_CH_IN1 				= 3;
const int F_CH_OUT1				= 32;

/*   parameter for conv layer 2 */
const int I_HEIGHT2            	= 64;
const int I_WIDTH2 				= 64;
const int I_CH2 				=32;

const int F_WIDTH2				= 1;
const int F_HEIGHT2				= 3;
const int F_CH_IN2 				= 32;
const int F_CH_OUT2				= 32;


/********************** weight location *****************/
const std::string W_LOC("/home/nhan/workplace/functionTest/weightData3/");
/************************* weight ***********************
B1B1C1W = Block 1  branch 1 convolution layer 1 weight  */
#define B1B1C1W "JhNetDeep_Block1_branch1_conv1_weights.npy"
#define B1B1C2W "JhNetDeep_Block1_branch1_conv2_weights.npy"
#define B1B2C1W "JhNetDeep_Block1_branch2_conv1_weights.npy"
#define B1B2C2W "JhNetDeep_Block1_branch2_conv2_weights.npy"

#define B2B1C1W "JhNetDeep_Block2_branch1_conv1_weights.npy"
#define B2B1C2W "JhNetDeep_Block2_branch1_conv2_weights.npy"
#define B2B2C1W "JhNetDeep_Block2_branch2_conv1_weights.npy"
#define B2B2C2W "JhNetDeep_Block2_branch2_conv2_weights.npy"

#define B3B1C1W "JhNetDeep_Block3_branch1_conv1_weights.npy"
#define B3B1C2W "JhNetDeep_Block3_branch1_conv2_weights.npy"
#define B3B2C1W "JhNetDeep_Block3_branch2_conv1_weights.npy"
#define B3B2C2W "JhNetDeep_Block3_branch2_conv2_weights.npy"
#define FC1W 	"JhNetDeep_fc1_weights.npy"
#define FC2W 	"JhNetDeep_fc2_weights.npy"
#define LOGICW 	"JhNetDeep_logits_weights.npy"


#define B1B1C1B "JhNetDeep_Block1_branch1_conv1_biases.npy"
#define B1B1C2B "JhNetDeep_Block1_branch1_conv2_biases.npy"
#define B1B2C1B "JhNetDeep_Block1_branch2_conv1_biases.npy"
#define B1B2C2B "JhNetDeep_Block1_branch2_conv2_biases.npy"

#define B2B1C1B "JhNetDeep_Block2_branch1_conv1_biases.npy"
#define B2B1C2B "JhNetDeep_Block2_branch1_conv2_biases.npy"
#define B2B2C1B "JhNetDeep_Block2_branch2_conv1_biases.npy"
#define B2B2C2B "JhNetDeep_Block2_branch2_conv2_biases.npy"

#define B3B1C1B "JhNetDeep_Block3_branch1_conv1_biases.npy"
#define B3B1C2B "JhNetDeep_Block3_branch1_conv2_biases.npy"
#define B3B2C1B "JhNetDeep_Block3_branch2_conv1_biases.npy"
#define B3B2C2B "JhNetDeep_Block3_branch2_conv2_biases.npy"
#define FC1B 	"JhNetDeep_fc1_biases.npy"
#define FC2B 	"JhNetDeep_fc2_biases.npy"
#define LOGICB 	"JhNetDeep_logits_biases.npy"



/********************** Point extract location *****************/
const std::string OUT_LOC("/home/nhan/workplace/functionTest/extractPoint/");
/************************* weight ***********************/

#define CONCAT1 "B1Cat1.npy"

#define POOL1 	"B1Pool1.npy"

#define B211 "b211OutRelu.npy"
#define B212 "b212OutRelu.npy"
#define B221 "b221OutRelu.npy"
#define B222 "b222OutRelu.npy"

#define POOL2   "Pool2.npy"

#define B311 "b311OutRelu.npy"
#define B312 "b312OutRelu.npy"
#define B321 "b321OutRelu.npy"
#define B322 "b322OutRelu.npy"

#define POOL3   "Pool3.npy"

#define FLATTEN   "p5flat.npy"

#endif 
