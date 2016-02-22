#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define INPUTUNITS (int)2
#define ENDSTRING "0,0,0"
#define OUTPUTUNITS (int)1
char HIDDENLAYERunits[] = "4,4,4";
#define LEARNINGRATE (double)0.02
#define EPOCHES (int)10000
#define RANDOMWEIGHTDIVISOR (double)1
#define LAYERCOUNT (int)5
#define BIAS (int)1
#define NETWORK_max_units (int)4
#define ADAPTABLE_LEARNINGRATE 0
#define LEARNINGRATEINCR (double)1
#define LEARNINGRATEDECR (double)0.5
#define PRINTDEBUG 0
#define NORMALIZE 1





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

double INPUT_data_maximum;
int INPUT_data_count;
TrainingData Store[1000];
int NETWORK_units_per_layer[LAYERCOUNT];
layer LAYERS[LAYERCOUNT];
double mainerror;
double error_before;

void Normalize()
{
    double max=INPUT_data_maximum;
    for(int i=0;i<INPUT_data_count;i++)
    {
        for(int unit=0; unit<INPUTUNITS;unit++)
        {
            Store[i].input[unit]=Store[i].input[unit]/(fabs(max));
        }
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
            if(strcmp(input,endstring)==0)break;
            else
            Store[j].input[0]=strtod(input,&delimiter);
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


double random_weight(int count_prev, int count_aft)
{
    double randdouble=count_aft*(double)pow(-1.0,(int)rand())*(double)rand()/(count_prev*(double)RAND_MAX);
    if(randdouble==0)randdouble=count_aft*(double)pow(-1.0,(int)rand())*(double)rand()/(count_prev*(double)RAND_MAX);
    return randdouble;
}

void NETWORK_build()
{
    for(int i=0; i<LAYERCOUNT-1;i++)
    {
        for(int u=0; u<NETWORK_units_per_layer[i];u++)
        {

            //printf("%d",NETWORK_units_per_layer[i+1]);
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
                        double delta_prev=LAYERS[layer+1].units[aft_unit].delta;
                        LAYERS[layer].units[units].weights[aft_unit]+=learning_rate*delta_prev*out_next;
                        if(PRINTDEBUG)printf("Layer: %d, unit:%d zu unit:%d -->%f\n",layer,units,aft_unit,LAYERS[layer].units[units].weights[aft_unit]);
                        if(units==0)LAYERS[layer].bias.weights[aft_unit]+=learning_rate*delta_prev*BIAS;

                        double weight=LAYERS[layer].units[units].weights[aft_unit];
                        double deriv=1.0-(double)pow(LAYERS[layer].units[units].out,2);
                        LAYERS[layer].units[units].delta+=delta_prev*weight*deriv;
                    }
                }
            layer--;
            }
            if(PRINTDEBUG)printf("%f",netout);
            mainerror+=(double)pow((Store[k].expout-netout),2);

            }

        double error = mainerror/INPUT_data_count;
        if(PRINTDEBUG)printf("success: %f Prozent, mainerror: %f error: %f\n",success,mainerror, error);
        switch(ADAPTABLE_LEARNINGRATE)
        {
            case 1:
            {
                if(error<error_before)
                {
                    learning_rate=learning_rate*LEARNINGRATEDECR;
                    error_before=error;
                }
                else
                {
                    learning_rate=learning_rate*LEARNINGRATEINCR;
                    error_before=error;
                }
             }break;
            case 2:break;
        }
        epoches++;
    }
}

void NETWORK_test()
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
                if((layer!=LAYERCOUNT-1))LAYERS[layer].units[unit].out=(double)tanh(LAYERS[layer].units[unit].net);//TODO: falls nichtlinear
            }

        }
        double netout=0.0;
        //if(/*INPUTUNITS!=1*/)netout=tanh(LAYERS[LAYERCOUNT-1].units[0].net);
         netout=(LAYERS[LAYERCOUNT-1].units[0].net);

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


