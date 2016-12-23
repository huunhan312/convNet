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
#include <opencv2/opencv.hpp>

using namespace std;

/* this function is to load 2d numpy array to 2d c++ array
** example: conv weight
*/
float** dsdl_loadWeight_2d(const char* filename, int* filter_infor){
	/********************* weight loading *****************/
   cnpy::NpyArray arr = cnpy::npy_load(W_LOC+filename);
  float* loaded_data = reinterpret_cast<float*>(arr.data);


  /*  show information of numpy package  */
  if(SHOW_LOAD_W){
	  cout << "\nShape of arr "<< filename <<" : "<< arr.shape.size() << endl;
	  for(int j=0;j<arr.shape.size();j++){
	  		filter_infor[j]=arr.shape[j];
	      cout << "Dimension "<<j<<": "<< arr.shape[j] << endl;
	  }
	  cout << "word size: "<< arr.word_size << endl;   
  }else{
  	for(int j=0;j<arr.shape.size();j++){
	  		filter_infor[j]=arr.shape[j];	  
	}
  }

  /*  end show information of numpy package  */

   /********************* weight loading *****************/
    int f_width = arr.shape[0]*arr.shape[1]*arr.shape[2];
    float**  filterArr = new float*[arr.shape[3]];
    for(int i=0;i<arr.shape[3];++i){
          filterArr[i] = new float[f_width];          
    }/* allocate the memory for storing the weight  */

   /// load weight into a number of filters 
   for(int h=0;h<arr.shape[0];h++){
    for(int i=0;i<arr.shape[1];i++){
          for(int j=0;j<arr.shape[2];j++){
            for(int k=0;k<arr.shape[3];k++){
                //printf("%f ",loaded_data[h*3*3*32+i*3*32+j*32+k]);
    filterArr[k][h*arr.shape[1]*arr.shape[2]+i*arr.shape[2]+j] 
    =loaded_data[h*arr.shape[1]*arr.shape[2]*arr.shape[3]+i*arr.shape[2]*arr.shape[3]+j*arr.shape[3]+k];

            }
          }
      }
   }
  /***************************************************/

   return filterArr;
}


/** this function load the 1d array only
*** example: Bias
**/
float* dsdl_loadWeight_1d(const char* filename){
	  /*              loading bias               */
  cnpy::NpyArray arrBias = cnpy::npy_load(W_LOC+filename);
  float* loaded_Bias = reinterpret_cast<float*>(arrBias.data);
   /*  end bias  */

  if(SHOW_LOAD_B){
	  cout << "\nLoaded bias: "<< filename <<" - Shape: "<< arrBias.shape.size() << endl;
	  for(int j=0;j<arrBias.shape.size();j++){
	      cout << "Dimension "<<j<<": "<< arrBias.shape[j] << endl;
	  }
	  cout << "word size: "<< arrBias.word_size << endl;   
	  /*  end show information of numpy package  */
  }
  return loaded_Bias;
}


/** this function load the 2d array only
*** example: fully connected weight
**/
float** dsdl_loadWeightfc_2d(const char* filename, int* w_infor){
	  /*              loading bias               */
  cnpy::NpyArray arrfc = cnpy::npy_load(W_LOC+filename);
  float* loaded_fc = reinterpret_cast<float*>(arrfc.data);
   /*  end bias  */

   if(SHOW_LOAD_FC){
	  /*  show information of numpy package SHOW_LOAD_FC */ 
	   cout << "\nLoaded fc weight: "<< filename <<" --> shape "<< arrfc.shape.size() << endl;
	  
	  for(int j=0;j<arrfc.shape.size();j++){
	  		w_infor[j]=arrfc.shape[j];
	      cout << "Dimension "<<j<<": "<< arrfc.shape[j] << endl;
	  }
	  cout << "word size: "<< arrfc.word_size << endl;   
	  /*  end show information of numpy package  */
	}else{
		for(int j=0;j<arrfc.shape.size();j++){
	  		w_infor[j]=arrfc.shape[j];
	  }
	}

   /********************* weight loading *****************/
    float**  filterArr = new float*[arrfc.shape[0]];
    for(int i=0;i<arrfc.shape[0];++i){
          filterArr[i] = new float[arrfc.shape[1]];  
          memcpy(filterArr[i], loaded_fc+i*arrfc.shape[1], arrfc.shape[1]*sizeof(float));        
    }/* allocate the memory for storing the weight  */

   /* load weight into a number of filters *
   for(int h=0;h<arr.shape[0];h++){
    for(int i=0;i<arr.shape[1];i++){
          for(int j=0;j<arr.shape[2];j++){
            for(int k=0;k<arr.shape[3];k++){
                //printf("%f ",loaded_data[h*3*3*32+i*3*32+j*32+k]);
    filterArr[k][h*arr.shape[1]*arr.shape[2]+i*arr.shape[2]+j] 
    =loaded_data[h*arr.shape[1]*arr.shape[2]*arr.shape[3]+i*arr.shape[2]*arr.shape[3]+j*arr.shape[3]+k];

            }
          }
      }
   }
  /***************************************************/

   return filterArr;

}


/*
**  This function is for flattening the 3D matrix which will be feed into FC layer
**  i_infor[3]={8,8,451};
*/
float* dsdl_flatten3d(float*** inArr, int* i_infor){
	/********************************************/
	int olen=i_infor[0]*i_infor[1]*i_infor[2];

	//cout << "Features: "<< olen << endl;

	float* oArr= new float[olen];

	for(int i=0;i<i_infor[0];i++){
		for(int j=0;j<i_infor[1];j++){

memcpy(oArr + (i*i_infor[1]*i_infor[2]+j*i_infor[2]), inArr[i][j], i_infor[2]*sizeof(float));

		}
	}
	return oArr;
}

/*
**  Load the output from convolution net for testing purpose
*/

float*** dsdl_loadOutconv(const char* filename, int* o_infor){
  cnpy::NpyArray arr = cnpy::npy_load(OUT_LOC+filename);
  float* loaded_fc = reinterpret_cast<float*>(arr.data);
   /*  end bias  */

   if(SHOW_LOAD_OUT){
	  /*  show information of numpy package SHOW_LOAD_FC */ 
	   cout << "\nLoaded fc weight: "<< filename <<" --> shape "<< arr.shape.size() << endl;
	  
	  for(int j=0;j<arr.shape.size();j++){
	  	  o_infor[j]=arr.shape[j];
	      cout << "Dimension "<<j<<": "<< arr.shape[j] << endl;
	  }
	  cout << "word size: "<< arr.word_size << endl;   
	  /*  end show information of numpy package  */
	}else{
		for(int j=0;j<arr.shape.size();j++){
	  		o_infor[j]=arr.shape[j];
	  }
	}

   /********************* weight loading *****************/
    float***  OutArr = new float**[arr.shape[1]];
    for(int i=0;i<arr.shape[1];++i){
          OutArr[i] = new float*[arr.shape[2]];  
          for(int j=0;j<arr.shape[2];++j){
          OutArr[i][j]= new float[arr.shape[3]];
          memcpy(OutArr[i][j], loaded_fc+i*arr.shape[2]*arr.shape[3]+j*arr.shape[3], arr.shape[3]*sizeof(float));
          }        
    }/* allocate the memory for storing the weight  */

    return OutArr;
}

/*
** Image preprocessing before feeding to classifying process
*/
