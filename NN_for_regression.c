#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define INPUTUNITS (int)1
#define ENDSTRING "0,0"
#define OUTPUTUNITS (int)1
// structure of hidden layer units
char HIDDENLAYERunits[] = "4,8,8,4";
#define LEARNINGRATE (double)0.1
#define EPOCHES (int)40000
#define RANDOMWEIGHTDIVISOR 3(double)
#define LAYERCOUNT (int)6
#define BIAS (int)1
#define NETWORK_max_units (int)8
#define ADAPTABLE_LEARNINGRATE 2
#define LEARNINGRATEINCR (double)1.03
#define LEARNINGRATEDECR (double)0.7
#define PRINTDEBUG 0
#define PRINTDEBUGERROR 0
#define NORMALIZE 1
#define OUTPUTNORMALIZE 1
#define ADDDIVISOR 10000
#define SIM_ANNEALING 0
#define PI 3.14159265359



typedef struct TrainingData TrainingData;
//struct with raw inputData
typedef struct TrainingData
{
   double input[INPUTUNITS];
   double expout;
   double bias;

}inputTraningDataptr;

typedef struct neuron neuron;
typedef struct neuron
{
    double weights[NETWORK_max_units];
    double weightsbuf[NETWORK_max_units];
    double net;
    double out;
    double delta;
}neuronptr;

typedef struct layer layer;
typedef struct layer
{
    int number;
    neuron units[NETWORK_max_units];
    neuron bias;
}layerptr;

double INPUT_data_maximum=0;
double INPUT_out_maximum=0;
int INPUT_data_count;
TrainingData Store[1000];
int NETWORK_units_per_layer[LAYERCOUNT];
layer LAYERS[LAYERCOUNT];
double mainerror;
double error_before;


void Normalize()
{
    // Normalization for unknown input space
    double max=INPUT_data_maximum;
    double outmax=INPUT_out_maximum;
    for(int i=0;i<INPUT_data_count;i++)
    {
        for(int unit=0; unit<INPUTUNITS;unit++)
        {
            Store[i].input[unit]=Store[i].input[unit]/(fabs(max));
        }
        if(OUTPUTNORMALIZE)Store[i].expout=Store[i].expout/fabs(outmax);
    }
}


void INPUT_read_data()
{
    char* input = (char*)malloc(sizeof(char)*32);
    char* delimiter;
    char endstring[]= ENDSTRING;
    int j=0;

    while(scanf("%s",input)!= EOF)
    {
            //reached end of training data?
            if(strcmp(input,endstring)==0)break;
            else
            Store[j].input[0]=strtod(input,&delimiter);
            //find global maximum
            if(fabs(Store[j].input[0])>fabs(INPUT_data_maximum))INPUT_data_maximum=fabs(Store[j].input[0]);
            if(INPUTUNITS>1)
            {
                for(int unit=1;unit<INPUTUNITS;unit++)
                {
                    Store[j].input[unit]=strtod(delimiter+1,&delimiter);
                    if(fabs(Store[j].input[unit])>fabs(INPUT_data_maximum))INPUT_data_maximum=fabs(Store[j].input[unit]);
                }
            }
            Store[j].expout=strtod(delimiter+1,&delimiter);
            if(fabs(Store[j].expout)>INPUT_out_maximum)INPUT_out_maximum=fabs(Store[j].expout);
            Store[j].bias=BIAS;
            j++;
    }
    INPUT_data_count=j;
    if(NORMALIZE)Normalize();

}
NETWORK_set_units_per_layer()
{
    char* delimiter;
    char* hu = HIDDENLAYERunits;
    NETWORK_units_per_layer[1]=(int)strtol(hu,&delimiter,10);
    delimiter++;

    for(int e=2;e<=LAYERCOUNT-2;e++)
    {
        NETWORK_units_per_layer[e]=(int)strtol(delimiter,&delimiter,10);
        delimiter++;
    }

    NETWORK_units_per_layer[0]=INPUTUNITS;
    NETWORK_units_per_layer[LAYERCOUNT-1]=OUTPUTUNITS;
    for(int layer=0;layer<LAYERCOUNT;layer++)
    {
        if(PRINTDEBUG) printf("Layer %d: %d Unit(s)\n",layer,NETWORK_units_per_layer[layer]);
    }
}

double random_addition()
{
    // compute a little random value
    double r1=(double)rand()/(double) RAND_MAX;
    double r2=(double)rand()/(double) RAND_MAX;
    return sqrt(-2*log(r1))*cos(2*PI*r2)/ADDDIVISOR;
}

double random_weight(int count_prev, int count_aft)
{
    // compute little random init values
    double randdouble=count_aft*(double)pow(-1.0,(int)rand())*(double)rand()/(count_prev*(double)RAND_MAX);
    if(randdouble==0)randdouble=count_aft*(double)pow(-1.0,(int)rand())*(double)rand()/(count_prev*(double)RAND_MAX);
    return randdouble;
}

void redo_weightupdate()
{
    // redo weight update if it was inefficient
    int layer=LAYERCOUNT-1;
    while (layer>=0)
    {
        for(int units=0;units<NETWORK_units_per_layer[layer];units++)
        {
            for(int aft_unit=0;aft_unit<NETWORK_units_per_layer[layer+1];aft_unit++)
            {
                LAYERS[layer].units[units].weights[aft_unit]=LAYERS[layer].units[units].weightsbuf[aft_unit];
                LAYERS[layer].bias.weights[aft_unit]=LAYERS[layer].bias.weightsbuf[aft_unit];
            }
        }
    layer--;
    }
}

// build and initialize network
void NETWORK_build()
{
    for(int i=0; i<LAYERCOUNT-1;i++)
    {
        for(int u=0; u<NETWORK_units_per_layer[i];u++)
        {
            for(int l=0;l<NETWORK_units_per_layer[i+1];l++)
            {
                if(i==0)LAYERS[i].units[u].weights[l]=random_weight(1,NETWORK_units_per_layer[1]);
                else LAYERS[i].units[u].weights[l]=random_weight(NETWORK_units_per_layer[i-1],NETWORK_units_per_layer[i+1]);
                if(PRINTDEBUG)printf("Layer: %d, Unit: %d zu Unit: %d, weight: %f\n",i,u,l,LAYERS[i].units[u].weights[l]);
            }
        }

        for(int w=0;w<NETWORK_units_per_layer[i+1];w++)
        {
            if(i==0)LAYERS[i].bias.weights[w]=random_weight(1,NETWORK_units_per_layer[1]);
            else LAYERS[i].bias.weights[w]=random_weight(NETWORK_units_per_layer[i-1],NETWORK_units_per_layer[i+1]);

            if(PRINTDEBUG)printf("Layer: %d, Unit: Bias, weight: %f zu Unit:%d\n",i,LAYERS[i].bias.weights[w],w);

        }
        LAYERS[i].bias.out=BIAS;

    }
}
void NETWORK_train()
{
    if(PRINTDEBUG)printf("Training...\n");
    int epoches=0;
    float success=0;

    double learning_rate=LEARNINGRATE;
    while(epoches!=EPOCHES)/*((mainerror/INPUT_data_count)<0.2)&&(epoches!=0))*/
    {
        int annealing=0;
        // simulated annealing only after some epoches
        if(((epoches%(EPOCHES/5)==0)&&(epoches!=0))&&(SIM_ANNEALING))annealing=1;
        // compute succes of process
        success=(float)epoches*100.0/(float)EPOCHES;
        mainerror=0;
        for(int k=0; k<INPUT_data_count;k++)
        {
            for(int unit=0;unit<INPUTUNITS;unit++)
            {
                LAYERS[0].units[unit].out=Store[k].input[unit];
                if(PRINTDEBUG)printf("Layer: 0, unit: %d, out: %f\n",unit,LAYERS[0].units[unit].out);
            }
            LAYERS[0].bias.out=Store[k].bias;

            //activation
            for (int layer=1; layer<=LAYERCOUNT-1;layer++)
            {
                for(int unit=0;unit<NETWORK_units_per_layer[layer];unit++)
                {
                    LAYERS[layer].units[unit].net=0;
                    for(int prev_unit=0;prev_unit<NETWORK_units_per_layer[layer-1];prev_unit++)
                    {
                        LAYERS[layer].units[unit].net+=LAYERS[layer-1].units[prev_unit].out*LAYERS[layer-1].units[prev_unit].weights[unit];
                    }
                    LAYERS[layer].units[unit].net+=LAYERS[layer-1].bias.weights[unit];
                    if(PRINTDEBUG)printf("Layer: %d, unit: %d, net: %f\n",layer,unit,LAYERS[layer].units[unit].net);
                    if((layer!=LAYERCOUNT-1))LAYERS[layer].units[unit].out=(double)tanh(LAYERS[layer].units[unit].net);//TODO: falls nichtlinear
                }

            }
            double netout=0.0;
            //if(INPUTUNITS!=1)netout=tanh(LAYERS[LAYERCOUNT-1].units[0].net);
            /*else*/ netout =(LAYERS[LAYERCOUNT-1].units[0].net);

            if(PRINTDEBUG)printf("%f\n",netout);

            // error backpropagation: start from output unit
            double out_next;
            LAYERS[LAYERCOUNT-1].units[0].delta=2*(Store[k].expout-netout);
            int layer=(int)LAYERCOUNT-2;
            while (layer>=0)
            {
                for(int units=0;units<NETWORK_units_per_layer[layer];units++)
                {
                    LAYERS[layer].units[units].delta=0;
                    out_next=LAYERS[layer].units[units].out;
                    for(int aft_unit=0;aft_unit<NETWORK_units_per_layer[layer+1];aft_unit++)
                    {   
                        // error of previous layer
                        double delta_prev=LAYERS[layer+1].units[aft_unit].delta;
                        // update weights of this layer's units
                        LAYERS[layer].units[units].weights[aft_unit]+=learning_rate*delta_prev*out_next;
                        // simulated annealing for local minima
                        if(annealing)LAYERS[layer].units[units].weights[aft_unit]+=random_addition();
                        if(PRINTDEBUG)printf("Layer: %d, unit:%d zu unit:%d -->%f\n",layer,units,aft_unit,LAYERS[layer].units[units].weights[aft_unit]);
                        // just once update the bias of this layer/in case of simulated annealing random addition
                        if(units==0)LAYERS[layer].bias.weights[aft_unit]+=learning_rate*delta_prev*BIAS;
                        if(annealing)LAYERS[layer].bias.weights[aft_unit]+=random_addition();
                        double weight=LAYERS[layer].units[units].weights[aft_unit];
                        // compute deriviative
                        double deriv=1.0-(double)pow(LAYERS[layer].units[units].out,2);
                        // store new error of these units
                        LAYERS[layer].units[units].delta+=delta_prev*weight*deriv;
                    }
                }
            layer--;
            }
            if(PRINTDEBUG)printf("%f\n",netout);
            mainerror+=(double)pow((Store[k].expout-netout),2);
            if(PRINTDEBUG)printf("%f\n",mainerror/k);

            }
        // compute main error
        double error = mainerror/INPUT_data_count;
        if(PRINTDEBUGERROR)printf("success: %f Prozent, mainerror: %f error: %f\n",success,mainerror, error);
        switch(ADAPTABLE_LEARNINGRATE)
        {
	    /*threre are two kinds of learning rate adaptions: case 1: multiply in 
	      dependence of decreasing error/increasing error, case 2: linear falling learning rate*/
	    
            case 1:
            {
                if(error>error_before)
                {
                    double newlearn=learning_rate*LEARNINGRATEDECR;
                    if(!(learning_rate<0.00005))learning_rate=newlearn;
                    if(!(learning_rate>=0.00005))redo_weightupdate();
                    error_before=error;
                }
                else
                {
                    double newlearn=learning_rate*LEARNINGRATEINCR;
                    if(!(learning_rate>3))learning_rate=newlearn;
                    error_before=error;
                }
            }break;
            case 2:
            {

                learning_rate=learning_rate-learning_rate*(double)((epoches)/EPOCHES);

            }break;
        }
        epoches++;
    }
}

void NETWORK_test()
/* forward test inputs through every layer of the network, compute output*/
{
    char* input = (char*)malloc(sizeof(char)*20);
    while(scanf("%s",input)!=EOF)
    {
        char* delimiter;
        char positive = '+';
        char negative = '-';
        double in[INPUTUNITS];

        in[0]=strtod(input,&delimiter);
        for(int co=1;co<INPUTUNITS;co++)
        {
            in[co]=strtod(delimiter+1,&delimiter);
        }

        for(int co=0;co<INPUTUNITS;co++)
        {
            if(NORMALIZE)LAYERS[0].units[co].out=in[co]/INPUT_data_maximum;
            else LAYERS[0].units[co].out=in[co];
        }

        LAYERS[0].bias.out=BIAS;

        //forward through network

        for (int layer=1; layer<LAYERCOUNT;layer++)
        {
            for(int unit=0;unit<NETWORK_units_per_layer[layer];unit++)
            {
                LAYERS[layer].units[unit].net=0;
                for(int prev_unit=0;prev_unit<NETWORK_units_per_layer[layer-1];prev_unit++)
                {
                    LAYERS[layer].units[unit].net+=LAYERS[layer-1].units[prev_unit].out*LAYERS[layer-1].units[prev_unit].weights[unit];
                }
                LAYERS[layer].units[unit].net+=LAYERS[layer-1].bias.weights[unit];
                if((layer!=LAYERCOUNT-1))LAYERS[layer].units[unit].out=(double)tanh(LAYERS[layer].units[unit].net);
            }

        }
        double netout=0.0;
        //if(/*INPUTUNITS!=1*/)netout=tanh(LAYERS[LAYERCOUNT-1].units[0].net);
         netout=INPUT_out_maximum*(LAYERS[LAYERCOUNT-1].units[0].net);

        // classify
        if(PRINTDEBUG)printf("%f\n",netout);
        if(INPUTUNITS==1)printf("%f\n",netout);
        else
        {
            if(0.0<netout)
            {
                printf("%c1\n",positive);
            }
            if(netout<0.0)
            {
                printf("%c1\n",negative);
            }
        }
    }
}



int main(void)
{
    NETWORK_set_units_per_layer();
    NETWORK_build();
    INPUT_read_data();
    NETWORK_train();
    NETWORK_test();

    return 0;
}




