#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//avoid calling for "struct TrainingData"
typedef struct TrainingData TrainingData;

//struct with raw inputData
typedef struct TrainingData
{
   double xvalue;  // the x value of training point
   double yvalue;   // the y value of training point
   double expout;  // expected output for this training point
   double bias;   //bias for decision boundary

}inputTraningDataptr;

/** @funtion:Normalize
 *  @brief:for big values to avoid too big peaks
 *
 */
TrainingData* Normalize(TrainingData store[1000],double maximum,size_t* count)
{
    double max=maximum;
    for(size_t i=0;i<*count;i++)
    {
        store[i].xvalue=store[i].xvalue/(fabs(max));
        store[i].yvalue=store[i].yvalue/fabs(max);
    }
    return store;
}

//input data
double ReadData(TrainingData Store[1000],size_t* count, double global_maximum)
{
    char* input = (char*)malloc(sizeof(char)*32);
    char* delimiter;
    char endstring[]= "0,0,0";

    size_t j=0;

    while(scanf("%s",input)!= EOF)
    {
            if(strcmp(input,endstring)==0)break;
            else
            Store[j].xvalue=strtod(input,&delimiter);
            if(fabs(Store[j].xvalue)>fabs(global_maximum))global_maximum=fabs(Store[j].xvalue);

            Store[j].yvalue=strtod(delimiter+1,&delimiter);
            if(fabs(Store[j].yvalue)>fabs(global_maximum))global_maximum=fabs(Store[j].yvalue);

            Store[j].expout=strtod(delimiter+1,&delimiter);
            Store[j].bias=-1;
            j++;
    }
    *count=j;
    Store=Normalize(Store,global_maximum,count);
    return global_maximum;
}


void TrainNeuron(TrainingData* store,size_t* count, double* weightlist, double max)
{
    double weightx=-0.8;
    double weighty=0.9;
    double weightbias=1;
    double learning_rate= 0.7;
    size_t u=10;
    size_t epoche=0;
    while(epoche!=100)
    {

        //printf("Training beginnt...");
        for(size_t i=0;i<*count;i++)
        {

            double out=weightx*(store[i].xvalue)+weighty*(store[i].yvalue)+weightbias*(store[i].bias);

            //learning_rate = pow((store[i].expout-((double)tanh(out))),2)/2;

            //weight update x
            weightx+=learning_rate*2.0*(store[i].expout-((double)tanh(out)))*(1.0/(double)pow((double)cosh(out),2))*store[i].xvalue;
            //weight update y
            weighty+=learning_rate*2.0*(store[i].expout-((double)tanh(out)))*(1.0/(double)pow((double)cosh(out),2))*store[i].yvalue;
            //weight update bias
            weightbias+=learning_rate*2.0*(store[i].expout-((double)tanh(out)))*(1.0/(double)pow((double)cosh(out),2))*store[i].bias;


            //printf("Training Step %d: weights: X: %f , Y: %f , bias: %f \n",i+1,weightx,weighty,weightbias);
        }
        epoche++;
    }
    //store weights into a vector
    weightlist[0]=weightx;
    weightlist[1]=weighty;
    weightlist[2]=weightbias;
}
char TestData(double* weights)
{
    // applies the test data on produced perceptron/ classifies data point in 1 or -1
    char* input = (char*)malloc(sizeof(char)*20);
    while(scanf("%s",input)!=EOF)
    {
        char* delimiter;
        char positive = '+';
        char negative = '-';
        double output = weights[0]*strtod(input,&delimiter);
        output+= weights[1]*strtod(delimiter+1,&delimiter);
        output-=weights[2];
        output=tanh(output);
        //printf("%f\n",output);
        if((0<(double)output)&&(output<=(double)1))
        {
            printf("%c1\n",positive);
        }
        if(((double)-1<=output)&&(output<(double)0))
        {
            printf("%c1\n",negative);
        }
     }
return (0);
}

int main(void)
{
    double global_maximum=0;
    TrainingData store[1000];
    size_t count=0;
    size_t* data_count=&count;
    double weightlist[3];

    global_maximum=ReadData(store,data_count,global_maximum);
    TrainNeuron(store,data_count,weightlist,global_maximum);
    TestData(weightlist);
    return(0);
}


