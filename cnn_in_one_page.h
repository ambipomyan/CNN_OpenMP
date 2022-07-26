#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <pthread.h>


typedef enum {
    NET,
    CONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX
}LAYER_TYPE;

typedef enum {
    RELU,
    ELU,
    MAXOUT
}ACT_TYPE;

typedef struct matrix_ {
    int nrows;
    int ncols;
    int nchannels;
    float *vals;
}MATRIX;

typedef struct data_{
    int h;
    int w;
    MATRIX *X;
    MATRIX *y; // labels, one-hot encoded
}DATA;

typedef struct update_args_ { // update parameters
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
}UPDATE_ARGS;

typedef struct load_args_{ // load parameters
    char **paths;
    char *path;
    int n;
    int m;
    const char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int min;
    int max;
    int size;
    float aspect;
    int classes;
    
    DATA *data;
}LOAD_ARGS;

struct network_;
typedef struct network_ NETWORK;

struct layer_;
typedef struct layer_ LAYER;

struct layer_ {
    int layer_type;
    int activation;
    
    int batch;
    int inputs;
    int outputs;
    int groups;
    int nweights;
    int nbiases;
    
    int h;
    int w;
    int c;
    int out_h;
    int out_w;
    int out_c;
    int n;
    int size;
    int stride;
    int padding;
    int truths;
    int softmax;
    
    float temperature;
    float cost;
    
    int *indexes;
    float *biases;
    float *bias_updates;
    float *scales;
    float *scale_updates;
    float *weights;
    float *weight_updates;
    
    float *delta;
    float *output;
    float *loss;
    float truth;
    
    float *x;
    float *x_norm;
    
    float *workspace;
    int workspace_size;
};

struct network_ {
    int n;
    int batch;
    float epoch;
    LAYER **layers; // all layers
    float *output;
    
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    
    int inputs;
    int outputs;
    int h;
    int w;
    int c;
    float aspect;
    
    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int workspace_size;
    
    int seen;
    int t;
    int time_steps;
    int subdivisions;
    int max_batches;
    int truths;
    int train;
    int index;
    float cost;
};
