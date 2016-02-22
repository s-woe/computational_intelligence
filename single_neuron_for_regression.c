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
   double xvalue;
   double expout;
   double bias;

}inputTraningDataptr;

double error=2;

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
    }
    return store;
}


//read input data, store into a array of struct Trainingdata elements
double ReadData(TrainingData Store[1000],size_t* count, double global_maximum)
{
    char* input = (char*)malloc(sizeof(char)*32);
    char* delimiter;
    char endstring[]= "0,0";


    size_t j=0;

    while(scanf("%s",input)!= EOF)
    {
	    // training end?
            if(strcmp(input,endstring)==0)break;
            else
	    // store(converted string into double)
            Store[j].xvalue=strtod(input,&delimiter);
            if(fabs(Store[j].xvalue)>fabs(global_maximum))global_maximum=fabs(Store[j].xvalue);

            Store[j].expout=strtod(delimiter+1,&delimiter);
            //bias input always -1
            Store[j].bias=-1;
            j++;
    }
    *count=j;
    Store=Normalize(Store,global_maximum,count);
    
    return global_maximum;//this value is needed for later normalization
}

// train neuron by weightupdate
void TrainNeuron(TrainingData* store,size_t* count, double* weightlist, double max)
{
    double weightx=-0.8;
    double weightbias=1;
    double learning_rate= 0.07;
    size_t epoche=0;
    while(epoche!=150)
    {

        //printf("Training beginnt...");
        for(size_t i=0;i<*count;i++)
        {
            // compute neuron output (y value)
            double out=weightx*(store[i].xvalue)+weightbias*(store[i].bias);

	    // error
            error = pow((store[i].expout-(out)),2)/2;

            //weight update x
            weightx+=learning_rate*2.0*(store[i].expout-(out))*store[i].xvalue;
            

            weightbias+=learning_rate*2.0*(store[i].expout-(out))*store[i].bias;


            //printf("%f\n", err);
        }
        epoche++;
    }
    //store weights into a vector
    weightlist[0]=weightx;
    weightlist[1]=weightbias;
}
char TestData(double* weights, double global_maximum)
{
    char* input = (char*)malloc(sizeof(char)*20);
    while(scanf("%s",input)!=EOF)
    {
        char* delimiter;
	// compute normalized output
        double output = weights[0]*strtod(input,&delimiter)/fabs(global_maximum);
        // subtract bias
        output-=weights[1];
        // print out
        printf("%f\n",output);

     }
    free(input);
return (0);
}

int main(void)
{
    double global_maximum=0;
    TrainingData store[1000];
    size_t count=0;
    size_t* data_count=&count;
    double weightlist[2];

    global_maximum=ReadData(store,data_count,global_maximum);
    TrainNeuron(store,data_count,weightlist,global_maximum);
    TestData(weightlist,global_maximum);
    return(0);
}


