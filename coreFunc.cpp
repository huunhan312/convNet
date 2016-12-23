/***************************************************************
* DSDL - Soongsil University                                   *
* Huu Nhan - Dec 01, 2016                                      *
* This code is for implement convolutional neural network      *
****************************************************************/
#include <iostream>
#include <stdio.h>
#include <string>
#include <cstdlib>
#include <pthread.h>
#include <thread>
#include <math.h> 
#include <opencv2/opencv.hpp>


using namespace std; 


/*
manipulate the conv2d as in tensorflow
this function  does support padding the input
This function support 
*/
float*** dsdl_conv2d_in(uchar* i_input, int* i_infor, float** f_value, int* f_infor,float* bias){
	
	/****************************** conv2d  ************************************/
   int xcount = 64/1; //number windows with respect to row, stride = 1
   int ycount = 64/1; //number windows with respect to col


  int xmar = f_infor[0]/2; //margin x 
  int ymar = f_infor[1]/2; //margin y direction
  

  char epty[5000]={0};
  int newRow= i_infor[0]+2*xmar; //padded size
  int newCol= i_infor[1]+2*ymar;


   /*************************** padding image  ****************************/
  /**/
  uchar***  data = new uchar**[newRow];
  for(int i=0;i<newRow;++i){
        data[i] = new uchar*[newCol];
        for(int j=0;j<newCol;++j){
            data[i][j] = new uchar[i_infor[2]];
            if(i<xmar || i>=(i_infor[0]+xmar) || j>=(i_infor[1]+ymar) || j<ymar)
              memcpy(data[i][j],epty,i_infor[2]*sizeof(uchar));
            else
              // copy data to each element
              memcpy(data[i][j],i_input+(i-xmar)*i_infor[1]*i_infor[2]+(j-ymar)*i_infor[2],i_infor[2]*sizeof(uchar));
          }
  }/* allocate and copy the img to the memory for storing the img  */



 
   /* allocate memory and copy the img for storing receptive field */
   int toplx = 0;
   int toply = 0;
   int featureTotal=f_infor[0]*f_infor[1]*f_infor[2];
   uchar** inqueue = new uchar *[xcount*ycount];
   for(int i=0;i<xcount;++i){ //number of receptive fields
      for(int j=0;j<ycount;++j){
          toplx=i; //update toplx
          toply=j;
          inqueue[i*ycount+j] = new uchar[featureTotal];
          for(int k=0;k<f_infor[0];++k){ //size of filter
             for(int l=0;l<f_infor[1];++l){
                int idx=toplx+k;
                int idy=toply+l;
                int temp = (k*f_infor[1]+l)*f_infor[2]; //index in queue 
                //inqueue[i*ycount+j][temp+0]=data[idx][idy][0];
                //inqueue[i*ycount+j][temp+1]=data[idx][idy][1];
                //inqueue[i*ycount+j][temp+2]=data[idx][idy][2];
                memcpy(inqueue[i*ycount+j]+temp,data[idx][idy],i_infor[2]*sizeof(uchar));
             }
          }
      }
   }
   /* Check create receptive field right or not 
   for(int k=0;k<ycount;++k){
      for(int l=0;l<9;++l){
         printf("%d ",inqueue[k][l]);
      }
      printf("\n");
   } /**/
   
   /* Doing the convolution */
    float*** conv= new float**[xcount];
	for(int i=0;i<xcount;++i){//xcount x-directon
		conv[i] = new float*[ycount];
    	for(int j=0;j<ycount;++j){//ycount
    		conv[i][j] = new float[f_infor[3]];
    		for(int k=0;k<f_infor[3];++k){
           		conv[i][j][k]=0.0;
           		for(int l=0;l<9;++l){
              		conv[i][j][k]+=inqueue[i*ycount+j][l]*(f_value[k][l]);
             	}
              	//// time for adding bias
				conv[i][j][k]+= bias[k];
				if (conv[i][j][k]<0)
						conv[i][j][k]=0.0;
         	}
      	}
    }

   /* print conv for testing 
   for(int i=0;i<2;++i){//xcount
      for(int j=0;j<2;++j){ //ycount
         for(int k=0;k<f_infor[3];++k)
            printf("%f ",conv[i][j][k]);
         printf("\n");
      } 
    } */

    return conv;
}

/*
Same with dsdconv2d_in but for hidden layers
*/
float*** dsdl_conv2d_hid(float*** i_input,int* i_infor, float** f_value, int* f_infor,float* bias){
	/****************************** conv2d  ************************************/

	int xmar = f_infor[0]/2; //margin x 
	int ymar = f_infor[1]/2;  // margin y direction

	int xcount = i_infor[0]/1; //number windows with respect to row
	int ycount = i_infor[1]/1; //number windows with respect to col

	char epty[5000]={0};
	int newRow= i_infor[0]+2*xmar; //padded size
  	int newCol= i_infor[1]+2*ymar;

  	int featureTotal=f_infor[0]*f_infor[1]*f_infor[2];

  	/*************************** padding image  ****************************/
  	float***  data = new float**[newRow];
  	//float*** data = new float[newRow][newCol][i_infor[2]];
	for(int i=0;i<newRow;++i){
	    data[i] = new float*[newCol];
	    for(int j=0;j<newCol;++j){
	        data[i][j] = new float[i_infor[2]];
	        if(i<xmar || i>=(i_infor[0]+xmar) || j>=(i_infor[1]+ymar) || j<ymar)
	          memcpy(data[i][j],epty,i_infor[2]*sizeof(float));
	        else
	          // copy data to each element
	          memcpy(data[i][j],i_input[i-xmar][j-ymar],i_infor[2]*sizeof(float));
	      }
	}/* allocate and copy the img to the memory for storing the img  */

	/*****  print the image value for testing *****
    for(int i=0;i<2;++i){//newRow
      printf("\n Test padding img\n Row %d :",i);
      for(int j=0;j<2;++j){ //newCol
         printf("\n Col %d :",j);
         for(int k=0;k<i_infor[2];++k){
             printf("%f ",data[i][j][k]);
         }
      }
     printf("\n");
    }
    printf("\n"); 
    /*******   end testing ******/


   /*      create the array of receptive field     */
   int toplx = 0;
   int toply = 0;
   float** inqueue = new float *[xcount*ycount];
   for(int i=0;i<xcount;++i){ //number of receptive fields
      for(int j=0;j<ycount;++j){
      	  inqueue[i*ycount+j] = new float[featureTotal];
	      toplx=i; //update toplx
	      toply=j;
	      for(int k=0;k<f_infor[0];++k){ //size of filter
             for(int l=0;l<f_infor[1];++l){
                int idx=toplx+k;
                int idy=toply+l;
                int temp = (k*f_infor[1]+l)*f_infor[2]; //index in queue 
                //inqueue[i*ycount+j][temp+0]=data[idx][idy][0];
                //inqueue[i*ycount+j][temp+1]=data[idx][idy][1];
                //inqueue[i*ycount+j][temp+2]=data[idx][idy][2];
                memcpy(inqueue[i*ycount+j]+temp,data[idx][idy],i_infor[2]*sizeof(float));
             }
          }
      }
   }/* end of array of receptive field */

   /* Check create receptive field right or not *
   for(int k=0;k<2;++k){//ycount
      for(int l=0;l<featureTotal;++l){
         printf("%f ",inqueue[k][l]);
      }
      printf("\n");
   } /**/
   
	/* Doing the convolution */
    /**********  this function can be implemented in multi-threading style *************/
    float*** conv= new float**[xcount];   
	for(int i=0;i<xcount;++i){//xcount x-directon
		conv[i] = new float*[ycount];
		for(int j=0;j<ycount;++j){//ycount
			conv[i][j] = new float[f_infor[3]];
			for(int k=0;k<f_infor[3];++k){
				conv[i][j][k]=0.0;
				for(int l=0;l<featureTotal;++l){
					conv[i][j][k]+=inqueue[i*ycount+j][l]*(f_value[k][l]);
					//printf("%d %f result:%f \n",inqueue[i*ycount+j][l],filterArr[k][l],conv[i][j][k]);
					}
					//// time for adding bias
					conv[i][j][k]+= bias[k];
					if (conv[i][j][k]<0)
						conv[i][j][k]=0.0;
			}

		}
	}
	/* end of convolution */

   /* print conv for testing 
   for(int i=0;i<2;++i){//xcount
      for(int j=0;j<2;++j){ //ycount
         for(int k=0;k<f_infor[3];++k)
            printf("%f ",conv[i][j][k]);
         printf("\n");
      } 
    } /* end of conv out testing */
    /*************************** padding image release ****************************
	for(int i=0;i<newRow;++i){
	    for(int j=0;j<newCol;++j){
	        delete [] data[i][j];
	      }
	      delete [] data[i];
	}
	delete [] data;
    /*************************** padding image release ****************************/

    /*************************** padding image release ****************************
	for(int i=0;i<xcount;++i) //number of receptive fields
      for(int j=0;j<ycount;++j){
      	  delete [] inqueue[i*ycount+j];
	}
	delete [] inqueue;
    /*************************** padding image release ****************************/

    return conv;
}

/**
***  this function use to concatenate the depth of neural network
***  firstArr  -->  left-branch
***  secondArr -->  right-branch
***  thirdArr  -->  input
**/
float*** dsdl_concatenate(float*** firstArr, int* st_infor, float*** secondArr, int* nd_infor, float*** thirdArr, int* rd_infor){
	/************************/
	if((st_infor[0] != nd_infor[0]) || (st_infor[0] != rd_infor[0]) || 
	   (st_infor[1] != nd_infor[1]) || (st_infor[1] != rd_infor[1])){
		printf("Input dimensions are not matched !\n");
		return NULL;
	}
	//printf("\nConcatenate 3 3D array. %dx%dx%d\n",st_infor[0],st_infor[1],st_infor[2]);

	float*** conArr= new float**[st_infor[0]];
	for(int i=0;i<st_infor[0];++i){
	    conArr[i] = new float*[st_infor[1]];
	    for(int j=0;j<st_infor[1];++j){
	        conArr[i][j] = new float[st_infor[2]+nd_infor[2]+rd_infor[2]];
        	memcpy(conArr[i][j],                        firstArr[i][j],st_infor[2]*sizeof(float));
        	memcpy(conArr[i][j]+st_infor[2],            secondArr[i][j],nd_infor[2]*sizeof(float));
        	memcpy(conArr[i][j]+st_infor[2]+nd_infor[2],thirdArr[i][j],rd_infor[2]*sizeof(float));
	    }
	}
	return conArr;
}

/**
** this function is built for max-pooling function
*/
float* getMaxArray(float*** inArr,int toplx,int toply,int rowpool, int colpool,int* i_infor){
	/*******************/
	//printf("\n\n\n New depth %dx%dx%d \n ",toplx,toply,i_infor[2]);
	float* maxArray=new float[i_infor[2]];
	for(int k=0;k<i_infor[2];k++){		

		float temMax=inArr[toplx][toply][k];
		for(int i=toplx;i<toplx+rowpool;i++)
			for(int j=toply;j<toply+colpool;j++){

				if(inArr[i][j][k]>temMax)
					temMax=inArr[i][j][k];
				//printf("%f ",inArr[i][j][k]);

			}
		maxArray[k]=temMax;
		//printf("\n ");
	}
	return maxArray;
}
float*** dsdl_pooling(float*** inArr,int* i_infor,int rowpool, int colpool){
	/********************************/
	int newRow=i_infor[0]/2;
	int newCol=i_infor[1]/2;
	/********************************/
	int toplx = 0;
    int toply = 0;

	float*** outArr = new float**[newRow];
	for(int i=0;i<newRow;++i){
	    outArr[i] = new float*[newCol];
	    for(int j=0;j<newCol;++j){
	        outArr[i][j] = new float[i_infor[2]];
	        toplx=i*rowpool; //update toplx
	      	toply=j*colpool;
	        outArr[i][j]=getMaxArray(inArr,toplx,toply, rowpool, colpool,i_infor);
	      }
	}/* allocate and implement max-pooling the img to the memory */

	return outArr;
}

/********   this function is only for testing the matrix multiplication
******/
void multiply1d(float* inArr, float** w_arr,float* bias,int col){
	float tem=0.0;

	for(int i=0;i<28864;i++){
		tem +=  inArr[i]*w_arr[i][col];
		printf("%f+=%f*%f\n",tem,inArr[i],w_arr[i][col]);
	}
	tem+=bias[col];
	cout << "tem: " << tem << endl;
	cout << "bias: " << bias[col] << endl;


//	if(data->relu==true && data->outArr[data->col]<0)
//		data->outArr[data->col]=0;
}
/** this function is for implementing matrix multiplication  
*** still not testing this function
**/
struct params {
    int col; /* the output node in fc layer */
	float* inArr;
	int inArr_len; /* length of input array*/
	float** w_arr;
	float* bias;
	float* outArr;
	bool relu; /** using relu function or not, not in the last layer***/
};
void *runner(void* param) {
	struct params* data = (struct params*) param; // the structure that holds our data

	/************ for 3D fully connected layer *****************
	for(int i=0;i<data->i_infor[0];i++){
		for(int j=0;j<data->i_infor[1];j++){
			for(int k=0;k<data->i_infor[2];k++){
				data->outArr[data->col]+=
				data->inMatrix[i][j][k]*data->w_arr[i*data->i_infor[1]*data->i_infor[2]+j*data->i_infor[2]+j][data->col];
			}
		}
	} /** for 3d fully connected layer */
	data->outArr[data->col]=0.0;

	for(int i=0;i<data->inArr_len;i++){
		data->outArr[data->col]+=  data->inArr[i]*data->w_arr[i][data->col];
		//printf("%f+=%f*%f\n",data->outArr[data->col],data->inArr[i],data->w_arr[i][data->col]);
	}
	data->outArr[data->col]+=data->bias[data->col];
	//printf("%f+= bias %f\n",data->outArr[data->col],data->bias[data->col]);

	if(data->relu==true && data->outArr[data->col]<0)
		data->outArr[data->col]=0;

	//cout << "Index: row col dep =" <<data->row << " "<<data->col<<" "<<data->dep;
   //Exit the thread
   pthread_exit(0);
}

/**
** this function is for implementing matrix multiplication 
** Input: 3D float array and 2D weight 
** Output: flat float array
*/
float* dsdl_fullyConnected(float* inArr, int inArr_len, float** w_arr, int* w_infor, float* bias, bool brelu){
	/*****************************************/
	//int inArr_len=i_infor[0]*i_infor[1]*i_infor[2];

	float* outArr= new float[w_infor[1]];

	//cout << "Number of output: " <<w_infor[1] <<endl;

	for(int i = 0; i < w_infor[1] ; i++){ //w_infor[1]

		//multiply1d(inArr,w_arr,bias,i);
		
		params* data = (params *) malloc(sizeof(params));

      	data->col = i;

      	data->inArr = inArr;
      	data->inArr_len = inArr_len;
      	
      	data->w_arr=w_arr;
      	data->bias=bias;

      	data->outArr=outArr;

      	data->relu=brelu;

         /* Now create the thread passing it data as a parameter */
        pthread_t tid;       //Thread ID
        pthread_attr_t attr; //Set of thread attributes
        pthread_attr_init(&attr);//Get the default attributes
        pthread_create(&tid, &attr,runner, data);//Create the thread

        //Make sure the parent waits for all thread to complete
	    pthread_join(tid, NULL);
	    /**/
	}

	return outArr;
}


/**
** this function is for implementing softmax multiplication 
** in log scale (Or normalization in log)
*/
float dsdl_maximum(float* a, int numberOfElements){
	// moved code from main() to here
	int mymaximum = a[0];

	for(int i= 1; i < numberOfElements; i++){
	    if(a[i] > mymaximum){
	        mymaximum = a[i];
	    }
	}
	return mymaximum;
}

int dsdl_softmax1d(float* inArr,int inlength){
	/******************************************/
	float maxVal =dsdl_maximum(inArr,inlength);

	float sumexp=0.0;
	for(int i=0;i<inlength;i++){
		sumexp=sumexp+exp(inArr[i]-maxVal);
		//cout << "element "<<i<<": " <<exp(inArr[i]-maxVal) <<endl;
	}
	//cout << "sumexp: " <<sumexp <<endl;
	

	float* outArr=new float[inlength];
	float maxProb=0.0;
	int indx;
	for(int i=0;i<inlength;i++){		
		outArr[i]=exp(inArr[i]-maxVal-log(sumexp));
		if(outArr[i]>maxProb){
			maxProb=outArr[i];
			indx=i;
		}
	}

	return indx;
}