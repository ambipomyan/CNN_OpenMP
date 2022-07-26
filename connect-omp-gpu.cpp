#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>
                    
#include "connect.h"


void connect(int batch, int K, int N, float *input, float *output, float *weights) {
    int i,j,k;
    
    int HWC_in     = batch*K;
    int HWC_out    = batch*N;
    int HWC_weight = N*K;

#pragma omp target teams distribute parallel for private(j,k) collapse(2) map(to:input[0:HWC_in], weights[0:HWC_weight]) map(tofrom:output[0:HWC_out])
{
    for (i = 0; i < batch; i++) {
        for (j = 0; j < N; j++) {
            float sum = 0.0;
            for (k = 0; k < K; k++) sum += input[i*K+k]*weights[j*K+k];
            output[i*N+j] += sum;
        }
    }
} // target region 1

}

void connect_backward(int batch, int N, int M, float *delta_in, float *input, float *weight_updates, float *weights, float *delta_out) { 
    int i,j,k;

    int HWC_in             = batch*N;
    int HWC_delta_in       = batch*M;
    int HWC_delta_out      = batch*N;
    int HWC_weight         = M*N;
    int HWC_weight_updates = M*N;

    // gemm
#pragma omp target teams distribute private(j,k) collapse(2) map(to:input[0:HWC_in], delta_in[0:HWC_delta_in]) map(tofrom:weight_updates[0:HWC_weight_updates])
{
    for (i = 0; i < M; i++) {
        //for (k = 0; k < batch; k++) {
            //float a_part = delta_in[k*M+i];
        for (j = 0; j < N; j++) {
	    float sum = 0.0;
#pragma omp parallel for reduction(+:sum)
            for (k = 0; k < batch; k++) {
                sum += delta_in[k*M+i]*input[k*N+j];
            }
	    weight_updates[i*N+j] = sum;
        }
    }
} // target region 1

    // gemm2
#pragma omp target teams distribute parallel for private(j,k) collapse(2) map(to:delta_in[0:HWC_delta_in], weights[0:HWC_weight]) map(tofrom:delta_out[0:HWC_delta_out])
{    
    for (i = 0; i < batch; i++) {
        //for (k = 0; k < M; k++) {
            //float a_part = delta_in[i*M+k];
        for (j = 0; j < N; j++) {
	    float sum = 0.0;
//#pragma omp parallel for reduction(+:sum)
	    for (k = 0; k < M; k++) {
                sum += delta_in[i*M+k]*weights[k*N+j];
            }
	    delta_out[i*N+j] = sum;
        }
    }
} // target region 2

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
