#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>

#include "omp.h"

#include "helpers_main.h"


/****** forward ******/

void forward_convolutional_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op, int dev_id, int num_dev) {
    // shape
    //int n;
    //int N, M, K;
    //int height, width, channels, ksize, stride, pad, height_col, width_col, channels_col;

    // conv1 shape
    int M = layer->n;
    int K = layer->size*layer->size*layer->c;
    int N = layer->out_w*layer->out_h;

    int height   = layer->h;
    int width    = layer->w;
    int channels = layer->c;
    int ksize    = layer->size;
    int stride   = layer->stride;
    int pad      = layer->padding;

    int height_col   = (height+2*pad-ksize)/stride+1;
    int width_col    = (width+2*pad-ksize)/stride+1;
    int channels_col = channels*ksize*ksize;

    // init
    int n = layer->outputs * layer->batch;
            
    for (int i = 0; i < n; i++) layer->delta[i] = 0;            
    for (int i = 0; i < n; i++) output[i] = 0;

    // inputs:      
    //
    // M, K, N
    // batch            [network->layers[0]->batch]
    // channels_col
    // height_col
    // width_col
    // ksize
    // stride
    // channels
    // height
    // width
    // pad
    // *input           [network->input]
    // *output          [network->layers[0]->output]
    // *weights         [network->layers[0]->weights]
    // *T               tensor, device-only data

    // conv
    conv(layer->batch, M, K, N, channels_col, height_col, width_col, ksize, stride, channels, height, width, pad, input, output, layer->weights, dev_id, num_dev);

    // add bias
    bias(layer->batch, M, N, output, layer->biases, dev_id, num_dev);

    // relu
    if (Op != 0) relu(layer->batch, M, N, output, dev_id, num_dev);

}


void forward_pooling_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op, int dev_id, int num_dev) {
    //int N = layer->out_h*layer->out_w*layer->out_c;
    //int M = layer_->out_h*layer_->out_w*layer_->out_c;	    

    int height     = layer->h;
    int width      = layer->w;
    int channels   = layer->c;
    int ksize      = layer->size;
    int stride     = layer->stride;
    int pad        = layer->padding;

    int height_out   = layer->out_h;
    int width_out    = layer->out_w;

    // init
    int n = layer->outputs*layer->batch;
    for (int i = 0; i < n; i++) layer->delta[i] = 0;

    // max pooling
    max_pool(layer->batch, height_out, width_out, ksize, stride, channels, height, width, pad, input, output, layer->indexes, dev_id, num_dev); 

}

void forward_connected_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op, int dev_id, int num_dev) {
    int K = layer->inputs;
    int N = layer->outputs;

    // init
    int n = layer->outputs*layer->batch;
    for (int i = 0; i < n; i++) layer->delta[i] = 0;
    for (int i = 0; i < n; i++) layer->output[i] = 0;

    // connected
    connect(layer->batch, K, N, input, output, layer->weights, dev_id, num_dev);

    // add bias
    bias(layer->batch, 1, N, output, layer->biases, dev_id, num_dev);
    
    // relu
    if (Op != 0) relu(layer->batch, 1, N, output, dev_id, num_dev);

}

void forward_softmax_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int dev_id, int num_dev) {
    int N = layer->inputs;

    // init 
    int n = layer->outputs*layer->batch;
    for (int i = 0; i < n; i++) layer->delta[i] = 0;

    //printf("N: %d\n", N);
    softmax(layer->batch, N, input, output, dev_id, num_dev);

}


/****** loss function ******/

float compute_loss_function(LAYER *layer, float *network_truth, int training_volume, int training_epoch) {
    int N = layer->batch*layer->inputs;

    // update cost
    for(int i = 0; i < N; i++) {
        int truth_index = i;
        
	float t = network_truth[truth_index];
        //printf("t: %f\n", t);
        float p = layer->output[i];
        //printf("p: %f\n", p);
        // loss
        if (t) {
            layer->loss[i] = -log(p);
        } else {
            layer->loss[i] = 0;
        }
        //printf("-logp: %f\n", network->layers[7]->loss[i]);
        layer->delta[i] = t-p;
        //printf("t-p: %f\n", network->layers[7]->delta[i]);
    }

    float sum_cost = 0;
    for (int i = 0; i < N; i++) {
        sum_cost += layer->loss[i];
        //printf("%f\n", network->layers[7]->loss[i]);
    }

    layer->cost = sum_cost;
    //printf("%f\n", network->layers[7]->cost);

    //printf("network->input size: %d\n", network->layers[7]->batch*network->layers[7]->outputs);

    // calc network cost
    float network_cost = layer->cost/(training_volume*training_epoch);

    return network_cost; 

}


/****** backward ******/

void backward_softmax_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int dev_id, int num_dev) {
    int N = layer->inputs;

    //axpy
    softmax_backward(layer->batch, N, delta_in, delta_out, dev_id, num_dev);
}

void backward_connected_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op, int dev_id, int num_dev) {
    int K = layer->inputs;
    int N = layer->outputs;

    //int n_in  = layer->batch*N;
    //int n_out = layer->batch*K;

    // init
    for(int i = 0; i < N; i++) layer->bias_updates[i] = 0;
    for(int i = 0; i < N*K; i++) layer->weight_updates[i] = 0;

    // gradient array
    if (Op != 0) relu_backward(layer->batch, N, layer->output, delta_in, dev_id, num_dev);

    // backward array
    bias_backward(layer->batch, 1, N, delta_in, layer->bias_updates, dev_id, num_dev);
            
    // backward connected
    connect_backward(layer->batch, K, N, delta_in, layer_->output, layer->weight_updates, layer->weights, delta_out, dev_id, num_dev);

}

void backward_pooling_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op, int dev_id, int num_dev) {
    int N = layer->out_h*layer->out_w*layer->out_c;
    int M = layer_->out_h*layer_->out_w*layer_->out_c;            

    int height     = layer->h;
    int width      = layer->w;
    int channels   = layer->c;
    int ksize      = layer->size;
    int stride     = layer->stride;
    int pad        = layer->padding;

    int height_out   = layer->out_h;
    int width_out    = layer->out_w;

    max_pool_backward(layer->batch, N, M, height_out, width_out, ksize, stride, channels, height, width, pad, layer->indexes, delta_in, delta_out, layer->output, layer_->output, dev_id, num_dev);

}

void backward_convolutional_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op, int dev_id, int num_dev) {
    int M = layer->n;
    int K = layer->size*layer->size*layer->c;
    int N = layer->out_w*layer->out_h;

    int height   = layer->h;
    int width    = layer->w;
    int channels = layer->c;
    int ksize    = layer->size;
    int stride   = layer->stride;
    int pad      = layer->padding;

    int height_col   = (height+2*pad-ksize)/stride+1;
    int width_col    = (width+2*pad-ksize)/stride+1;
    int channels_col = channels*ksize*ksize;

    //int n_in  = layer->batch*M*N;
    //int n_out = layer->batch*height*width*channels;

    // init 
    for(int i = 0; i < M; i++) layer->bias_updates[i] = 0;
    for(int i = 0; i < K*M; i++) layer->weight_updates[i] = 0;

    // gradient array
    if (Op != 0) relu_backward(layer->batch, N*M, layer->output, delta_in, dev_id, num_dev);
            
    // backward bias
    bias_backward(layer->batch, N, M, delta_in, layer->bias_updates, dev_id, num_dev);

    // conv backward
    conv_backward(layer->batch, K, N, M, channels_col, height_col, width_col, ksize, stride, channels, height, width, pad, layer_->output, delta_in, layer->weight_updates, delta_out, layer->weights, dev_id, num_dev);

}
