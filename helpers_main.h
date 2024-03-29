#include "cnn_in_one_page.h"
#include "conv.h"
#include "connect.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;


float rand_uniform(float min, float max);

float rand_normal();

void forward_convolutional_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op, int dev_id, int num_dev); // op: 1 for reLU, 0 for no activation

void forward_pooling_layer(      LAYER *layer_, LAYER *layer, float *input, float *output, int Op, int dev_id, int num_dev); // Op: 1 for maxpooling

void forward_connected_layer(    LAYER *layer_, LAYER *layer, float *input, float *output, int Op, int dev_id, int num_dev); // Op: 1 for reLU, 0 for no activation

void forward_softmax_layer(      LAYER *layer_, LAYER *layer, float *input, float *output,         int dev_id, int num_dev);

float compute_loss_function(LAYER *layer, float *network_truth, int training_volume, int training_epoch);

void backward_softmax_layer(      LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out,         int dev_id, int num_dev);

void backward_connected_layer(    LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op, int dev_id, int num_dev);

void backward_pooling_layer(      LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op, int dev_id, int num_dev);

void backward_convolutional_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op, int dev_id, int num_dev);

NETWORK *load_network(int n_layers, int img_h, int img_w, int img_c, int n_classes, int batch);

void add_convolutional_layer(NETWORK *network, int id, int n, int size, int stride, int padding,                        int img_h, int img_w, int img_c, int batch);

void add_pooling_layer(      NETWORK *network, int id,        int size, int stride, int padding, LAYER_TYPE layer_type, int img_h, int img_w, int img_c, int batch);

void add_connected_layer(    NETWORK *network, int id, int l_outputs,                                                   int img_h, int img_w, int img_c, int batch);

void add_softmax_layer(      NETWORK *network, int id, int n_classes,                                                   int img_h, int img_w, int img_c, int batch);

// data
MATRIX *get_data(  int img_m, int img_n, int img_h, int img_w, int img_c, const char *train_data_path, const char *test_data_path, int n_classes, float mean, float std);

MATRIX *get_labels(int img_m, int img_n, int img_h, int img_w, int img_c, const char *train_data_path, const char *test_data_path, int n_classes);
