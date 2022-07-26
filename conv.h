#include "util.h"

void conv(int batch, int M, int K, int N, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, float *weights, int dev_id, int num_dev);

void bias(int batch, int M, int N, float *output, float *biases, int dev_id, int num_dev);

void relu(int batch, int M, int N, float *output, int dev_id, int num_dev);

void max_pool(int batch, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, int *indexes, int dev_id, int num_dev);

void skip_connection(int batch, int M, int N, float *input, float *output);

void softmax(int batch, int N, float *input, float *output, int dev_id, int num_dev);

void softmax_backward(int batch, int N, float *input, float *output, int dev_id, int num_dev);

void relu_backward(int batch, int N, float *output, float *delta, int dev_id, int num_dev);

void bias_backward(int batch, int N, int M, float *input, float *output, int dev_id, int num_dev);

//void max_pool_backward(int batch, int N, int M, int *indexes, float *delta_in, float *delta_out);

void max_pool_backward(int batch, int N, int M, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, int *indexes, float *delta_in, float *delta_out, float *input, float *output, int dev_id, int num_dev);

void conv_backward(int batch, int M, int K, int N, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *delta_in, float *weight_updates, float *delta_out, float *weights, int dev_id, int num_dev);

void conv_update(int nbias, float *biases, float *bias_updates, int nweights, float *weights, float *weight_updates, float p1, float p2, float p3);

