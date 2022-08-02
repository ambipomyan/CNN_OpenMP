#include "cnn_in_one_page.h"
#include "conv.h"
#include "connect.h"

void forward_convolutional_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op); // op: 1 for reLU, 0 for no activation

void forward_pooling_layer(      LAYER *layer_, LAYER *layer, float *input, float *output, int Op); // Op: 1 for maxpooling

void forward_connected_layer(    LAYER *layer_, LAYER *layer, float *input, float *output, int Op); // Op: 1 for reLU, 0 for no activation

void forward_softmax_layer(      LAYER *layer_, LAYER *layer, float *input, float *output        );

float compute_loss_function(LAYER *layer, float *network_truth, int training_volume, int training_epoch);

void backward_softmax_layer(      LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out        );

void backward_connected_layer(    LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op);

void backward_pooling_layer(      LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op);

void backward_convolutional_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op);
