#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>

#include "helpers_main.h"


void forward_convolutional_layer(LAYER *layer, float *input, float *output, int Op) {
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
    conv(layer->batch, M, K, N, channels_col, height_col, width_col, ksize, stride, channels, height, width, pad, input, output, layer->weights);

    // add bias
    bias(layer->batch, M, N, output, layer->biases);

    // relu
    if (Op != 0) relu(layer->batch, M, N, output);

}


void forward_pooling_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op) {
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
    max_pool(layer->batch, height_out, width_out, ksize, stride, channels, height, width, pad, input, output, layer->indexes); 

}

void forward_connected_layer(LAYER *layer, float *input, float *output, int Op) {
    int K = layer->inputs;
    int N = layer->outputs;

    // init
    int n = layer->outputs*layer->batch;
    for (int i = 0; i < n; i++) layer->delta[i] = 0;
    for (int i = 0; i < n; i++) layer->output[i] = 0;

    // connected
    connect(layer->batch, K, N, input, output, layer->weights);

    // add bias
    bias(layer->batch, 1, N, output, layer->biases);
    
    // relu
    if (Op != 0) relu(layer->batch, 1, N, output);

}

void forward_softmax_layer(LAYER *layer, float *input, float *output) {
    int N = layer->inputs;
    
    // init 
    int n = layer->outputs*layer->batch;
    for (int i = 0; i < n; i++) layer->delta[i] = 0;

    //printf("N: %d\n", N);
    softmax(layer->batch, N, input, output);

}
