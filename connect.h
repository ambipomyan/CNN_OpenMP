#include "util.h"

void connect(int batch, int K, int N, float *input, float *output, float *weights);

void connect_backward(int batch, int N, int M, float *delta_in, float *input, float *weight_updates, float *weights, float *delta_out);

void connect_update(int nbias, float *biases, float *bias_updates, int nweights, float *weights, float *weight_updates, float p1, float p2, float p3);
