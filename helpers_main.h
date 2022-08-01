#include "cnn_in_one_page.h"
#include "conv.h"
#include "connect.h"

void forward_convolutional_layer(LAYER *layer, float *input, float *output, int Op);          // op: 1 for reLU, 0 for no activation

void forward_pooling_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op); // Op: 1 for maxpooling

void forward_connected_layer(LAYER *layer, float *input, float *output, int Op);              // Op: 1 for reLU, 0 for no activation

void forward_softmax_layer(LAYER *layer, float *input, float *output);


