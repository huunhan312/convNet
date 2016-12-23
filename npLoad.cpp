/***************************************************************
* DSDL - Soongsil University                                   *
* Huu Nhan - Dec 01, 2016                                      *
* This code is for implement convolutional neural network      *
****************************************************************/

#include "cnpy.h"
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <ctime>
#include <opencv2/opencv.hpp>

//defined functions
#include "coreFunc.h"
#include "utilities.h"
#include "parameters.h"

using namespace std;
using namespace cv;

//// compile command
////  g++ -o conv2d npLoad.cpp -L /usr/local/ -l cnpy -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect 

int main(int argc, char** argv)
{

  istringstream ss(argv[2]);
  int classid;
  if (!(ss >> classid))
	 cerr << "Invalid number " << argv[1] << '\n';
  ifstream infile(argv[1]);//("/home/nhan/workplace/functionTest/listFileTest/lf00.txt");
  string str;

  /********************* load weight and define support array *****************/
  int f111_infor[4];
  int i111_infor[3]={64,64,3};

  int f112_infor[4];
  int i112_infor[3]={64,64,32};

  int f121_infor[4];
  int i121_infor[3]={64,64,3};

  int f122_infor[4];
  int i122_infor[3]={64,64,32};



  int f211_infor[4];
  int i211_infor[3]={32,32,67};

  int f212_infor[4];
  int i212_infor[3]={32,32,64};

  int f221_infor[4];
  int i221_infor[3]={32,32,67};

  int f222_infor[4];
  int i222_infor[3]={32,32,64};



  int f311_infor[4];
  int i311_infor[3]={16,16,195};

  int f312_infor[4];
  int i312_infor[3]={16,16,128};

  int f321_infor[4];
  int i321_infor[3]={16,16,195};

  int f322_infor[4];
  int i322_infor[3]={16,16,128};


  int fcw1_infor[2];

  int fcw2_infor[2];

  int fcw3_infor[2];

  /******************* Stage 1 Branch 1 conv 1  **********************/
  float** filterArr111=dsdl_loadWeight_2d(B1B1C1W,f111_infor);
  float* loaded_Bias111=dsdl_loadWeight_1d(B1B1C1B);

  /********************** Stage 1 Branch 1 conv 2  ********************/
  float** filterArr112=dsdl_loadWeight_2d(B1B1C2W,f112_infor); //load weight
  float* loaded_Bias112=dsdl_loadWeight_1d(B1B1C2B); // load bias

  /******************* Stage 1 Branch 2 conv 1  ********************/
  float** filterArr121=dsdl_loadWeight_2d(B1B2C1W,f121_infor);
  float* loaded_Bias121=dsdl_loadWeight_1d(B1B2C1B);

  /************* Stage 1 Branch 2 conv 2  ***************************/
  float** filterArr122=dsdl_loadWeight_2d(B1B2C2W,f122_infor);
  float* loaded_Bias122=dsdl_loadWeight_1d(B1B2C2B);



  /******************* Stage 2 Branch 1 conv 1  **********************/
  float** filterArr211=dsdl_loadWeight_2d(B2B1C1W,f211_infor);
  float* loaded_Bias211=dsdl_loadWeight_1d(B2B1C1B);

  /******************* Stage 2 Branch 1 conv 1  **********************/
  float** filterArr212=dsdl_loadWeight_2d(B2B1C2W,f212_infor); //load weight
  float* loaded_Bias212=dsdl_loadWeight_1d(B2B1C2B); // load bias

  /******************* Stage 2 Branch 1 conv 1  **********************/
  float** filterArr221=dsdl_loadWeight_2d(B2B2C1W,f221_infor);
  float* loaded_Bias221=dsdl_loadWeight_1d(B2B2C1B);

  /******************* Stage 2 Branch 1 conv 1  **********************/
  float** filterArr222=dsdl_loadWeight_2d(B2B2C2W,f222_infor);
  float* loaded_Bias222=dsdl_loadWeight_1d(B2B2C2B);



  /******************* Stage 3 Branch 1 conv 1  **********************/
  float** filterArr311=dsdl_loadWeight_2d(B3B1C1W,f311_infor);
  float* loaded_Bias311=dsdl_loadWeight_1d(B3B1C1B);

  /******************* Stage 3 Branch 1 conv 1  **********************/
  float** filterArr312=dsdl_loadWeight_2d(B3B1C2W,f312_infor); //load weight
  float* loaded_Bias312=dsdl_loadWeight_1d(B3B1C2B); // load bias

  /******************* Stage 3 Branch 1 conv 1  **********************/
  float** filterArr321=dsdl_loadWeight_2d(B3B2C1W,f321_infor);
  float* loaded_Bias321=dsdl_loadWeight_1d(B3B2C1B);

  /******************* Stage 3 Branch 1 conv 1  **********************/
  float** filterArr322=dsdl_loadWeight_2d(B3B2C2W,f322_infor);
  float* loaded_Bias322=dsdl_loadWeight_1d(B3B2C2B);



  /******************* Stage FC 1  **********************/
  float** fc1=dsdl_loadWeightfc_2d(FC1W,fcw1_infor);
  float* fcBias1=dsdl_loadWeight_1d(FC1B);

  /******************* Stage FC 2  **********************/
  float** fc2=dsdl_loadWeightfc_2d(FC2W,fcw2_infor);
  float* fcBias2=dsdl_loadWeight_1d(FC2B);

  /******************* Stage FC 3  **********************/
  float** fc3=dsdl_loadWeightfc_2d(LOGICW,fcw3_infor);
  float* fcBias3=dsdl_loadWeight_1d(LOGICB);

  /*********************************************************************/


  /**************************** load image ***************************/
  int count=0; // this variable is for counting misclassifying
  while (std::getline(infile, str)) {
  clock_t begin = clock();
  cout << "Loading image: "<< str << endl;

  Mat src1,src;
  //src1 = imread("1.png");
  src1 = imread(str);
  if( !src1.data ){
      printf("Deo load duoc anh\n");
      return -1;
  }
  else cvtColor(src1,src,CV_RGB2BGR);

  int size = src.total() * src.elemSize();

 /* printf("Total number of features: %d\n" 
         "src.rows: %d \n"  
         "src.cols: %d \n",size,src.rows,src.cols);*/

  float* testData= new float[size];

  // convert form uchar to float
  copy(src.data, src.data + size, testData);

  /***************** mean and stddev of image for preprocessing ****************/

  double dmean = 0.0;
  double dstddev = 0.0;

  // Mean standard algorithm
  for (int i = 0; i < size; ++i)
  {
      dmean += testData[i];
  }
  dmean /= size;

  // Standard deviation standard algorithm
  std::vector<double> var(size);
  for (int i = 0; i < size; ++i)
  {
      var[i] = (dmean - testData[i]) * (dmean - testData[i]);
  }
  for (int i = 0; i < size; ++i)
  {
      dstddev += var[i];
  }
  dstddev = sqrt(dstddev / size);

  std::cout << "Mean: " << dmean << "   StdDev: " << dstddev << std::endl;
  for (int i = 0; i < size; ++i)
  {
      testData[i] = (testData[i]-dmean)/dstddev;
  }
  /*************************** restruct image  ****************************/
  /**/
  float***  data = new float**[64];
  for(int i=0;i<64;++i){
      data[i] = new float*[64];
      for(int j=0;j<64;++j){
          data[i][j] = new float[3];
          // copy data to each element
           memcpy(data[i][j],testData+i*64*3+j*3,3*sizeof(float));
        }
  }/* allocate and copy the img to the memory for storing the img  */
  /*************************** restruct image  ****************************/



  /*************************************************************************/
  /************************** Stage 1 *************************/

  float*** o_conv111=
    dsdl_conv2d_hid(data, i111_infor, filterArr111, f111_infor,loaded_Bias111);

  float*** o_conv112 =
    dsdl_conv2d_hid(o_conv111, i112_infor, filterArr112, f112_infor,loaded_Bias112);

  float*** o_conv121 =
    dsdl_conv2d_hid(data, i121_infor, filterArr121, f121_infor,loaded_Bias121);

  float*** o_conv122 =
    dsdl_conv2d_hid(o_conv121, i122_infor, filterArr122, f122_infor,loaded_Bias122);
  

  /******************************** CONCAT STAGE 1 ********************************/
  float*** concat1 = 
      dsdl_concatenate(o_conv112, i112_infor,o_conv122, i122_infor, data, i111_infor);

  /******************************* POOLING STAGE 1 *************************************/
  int ipool1_infor[3]={64,64,67};
  float*** pool1=
      dsdl_pooling(concat1, ipool1_infor,2, 2);

  /****************************************************************************/
  /*********************** Stage 2   ***************************/

  float*** o_conv211=
    dsdl_conv2d_hid(pool1, i211_infor, filterArr211, f211_infor,loaded_Bias211);

  float*** o_conv212 =
    dsdl_conv2d_hid(o_conv211, i212_infor, filterArr212, f212_infor,loaded_Bias212);

  float*** o_conv221 =
    dsdl_conv2d_hid(pool1, i221_infor, filterArr221, f221_infor,loaded_Bias221);

  float*** o_conv222 =
    dsdl_conv2d_hid(o_conv221, i222_infor, filterArr222, f222_infor,loaded_Bias222);

  /******************************** CONCAT STAGE 2 ********************************/
  float*** concat2 = 
      dsdl_concatenate(o_conv212, i212_infor,o_conv222, i222_infor, pool1, i211_infor);

  /******************************* POOLING STAGE 2 **************************/
  int ipool2_infor[3]={32,32,195};
  float*** pool2=
      dsdl_pooling(concat2, ipool2_infor,2, 2);
 
  /****************************************************************************/
  /**************************    Stage 3   ***************************/

  float*** o_conv311=
    dsdl_conv2d_hid(pool2, i311_infor, filterArr311, f311_infor,loaded_Bias311);

  float*** o_conv312 =
    dsdl_conv2d_hid(o_conv311, i312_infor, filterArr312, f312_infor,loaded_Bias312);

  float*** o_conv321 =
    dsdl_conv2d_hid(pool2, i321_infor, filterArr321, f321_infor,loaded_Bias321);

  float*** o_conv322 =
    dsdl_conv2d_hid(o_conv321, i322_infor, filterArr322, f322_infor,loaded_Bias322);

  /******************************** CONCAT STAGE 3 ********************************/
  float*** concat3 = 
      dsdl_concatenate(o_conv312, i312_infor,o_conv322, i322_infor, pool2, i311_infor);
  
  /******************************* POOLING STAGE 3 ***********************/
  int ipool3_infor[3]={16,16,451};
  float*** pool3=
      dsdl_pooling(concat3, ipool3_infor,2, 2);
  

 
  /****************************************************************/
  /*********************  flatten 3d  *********************************/
  int ifc1_infor[3]={8,8,451};
  float* flattenArr=dsdl_flatten3d(pool3, ifc1_infor);
  
   /***** test the output of softmax calculation ********
  cout << endl << endl;
  for(int i = 0; i < 28864; i++){
    cout << flattenArr[i] <<"  ";
  }
  cout << endl<< endl<< endl;  /***/

  /**************************  FC 1  ***************************/
  int inArr1_len=8*8*451;
  float* fcOut1=dsdl_fullyConnected(flattenArr, inArr1_len, fc1, fcw1_infor, fcBias1,true);

  /***** test the output of softmax calculation ********
  cout << endl << endl;
  for(int i = 0; i < 1024; i++){
    cout << fcOut1[i] <<"  ";
  }
  cout << endl<< endl<< endl;  /***/
  /**************************  FC 2  ***************************/
  int inArr2_len=1024;
  float* fcOut2=dsdl_fullyConnected(fcOut1, inArr2_len, fc2, fcw2_infor, fcBias2, true);

  /**************************  FC 3  ***************************/
  int inArr3_len=128;
  float* fcOut3=dsdl_fullyConnected(fcOut2, inArr3_len, fc3, fcw3_infor, fcBias3, false);

 
 /***** test the output of softmax calculation ********
  cout << endl << endl;
  for(int i = 0; i < 43; i++){
    cout << fcOut3[i] <<"  ";
  }
  cout << endl<< endl<< endl;  /***/

  /****************** Softmax calculation  ******************/
 // int thisClass=(int)(argv[2][0]-'0');
  cout<<"Classified: " << dsdl_softmax1d(fcOut3,43);

  if(dsdl_softmax1d(fcOut3,43) != classid) count++;

  cout << " --> Miss: " << count;
  /***** test the output of softmax calculation ********/

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "   Elapsed time: " << elapsed_secs << " sec" << endl;
}
  return 0;
}
