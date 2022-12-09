#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>
                    
#include "connect.h"

void connect(int batch, int K, int N, float *input, float *output, float *weights, int dev_id, int num_dev) {
    int i,j,k;

#pragma omp parallel for private(j,k) shared(input,weights,output)
{
    for (i = 0; i < batch; i++) {
        for (j = 0; j < N; j++) {
            float sum = 0.0;
            for (k = 0; k < K; k++) sum += input[i*K+k]*weights[j*K+k];
            output[i*N+j] += sum;
        }
    }
} // parallel region 1

}

void connect_backward(int batch, int N, int M, float *delta_in, float *input, float *weight_updates, float *weights, float *delta_out, int dev_id, int num_dev) { 
    int i,j,k;

#pragma omp parallel for private(j,k) shared(delta_in,weight_updates,input)
{
    // gemm
    for (i = 0; i < M; i++) {
        for (k = 0; k < batch; k++) {
            float a_part = delta_in[k*M+i];
            for (j = 0; j < N; j++) {
                weight_updates[i*N+j] += a_part*input[k*N+j];
            }
        }
    }
} // parallel region 1

#pragma omp parallel for private(j,k) shared(delta_in,delta_out,weights)
{
    // gemm2	
    for (i = 0; i < batch; i++) {
        for (k = 0; k < M; k++) {
            float a_part = delta_in[i*M+k];
            for (j = 0; j < N; j++) {
                delta_out[i*N+j] += a_part*weights[k*N+j];
            }
        }
    }
} // parallel region 2

}

void connect_update(int nbias, float *biases, float *bias_updates, int nweights, float *weights, float *weight_updates, float p1, float p2, float p3) {
    // axpy
    for (int i = 0; i < nbias; i++) {biases[i] += p1*bias_updates[i];}
    // scale
    for (int i = 0; i < nbias; i++) {bias_updates[i] *= p3;}
    // axpy
    for (int i = 0; i < nweights; i++) {weight_updates[i] += p2*weights[i];}
    // axpy2
    for (int i = 0; i < nweights; i++) {weights[i] += p1*weight_updates[i];}
    // scale
    for (int i = 0; i < nweights; i++) {weight_updates[i] *= p3;}
}
