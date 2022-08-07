#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "connect.h"


void connect(int batch, int K, int N, float *input, float *output, float *weights) {
    cublasHandle_t handle_;
    cublasCreate(&handle_);

    float *A, *B, *C;

    float alpha = 1.0;
    float beta = 1.0;

    cudaMalloc(&A,       batch*K*sizeof(float)); 
    cudaMemcpy(A, input, batch*K*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&B,         K*N*sizeof(float));
    cudaMemcpy(B, weights, K*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&C,        batch*N*sizeof(float)); 
    cudaMemcpy(C, output, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T, batch, N, K, &alpha, A, K, B, N, &beta, C, batch);

    cudaMemcpy(output, C, batch*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(C);
    cudaFree(B);
    cudaFree(A);

    cublasDestroy(handle_);

}

/*
void connect(int batch, int K, int N, float *input, float *output, float *weights) {
    int i,j,k;
    float sum;

    for (i = 0; i < batch; i++) {
        for (j = 0; j < N; j++) {
            sum = 0;
            for (k = 0; k < K; k++) sum += input[i*K+k]*weights[j*K+k];
            output[i*N+j] += sum;
	}
    }
}
 */

void connect_backward(int batch, int N, int M, float *delta_in, float *input, float *weight_updates, float *weights, float *delta_out) {
    cublasHandle_t handle_;
    cublasCreate(&handle_);

    float *A, *B1, *C1;
    float *B2, *C2;

    float alpha = 1.0;
    float beta = 1.0;

    cudaMalloc(&A,          M*batch*sizeof(float));
    cudaMemcpy(A, delta_in, M*batch*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&B1,       batch*N*sizeof(float));
    cudaMemcpy(B1, input, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&B2,         M*N*sizeof(float));
    cudaMemcpy(B2, weights, M*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&C1,                M*N*sizeof(float));
    cudaMemcpy(C1, weight_updates, M*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&C2,           batch*N*sizeof(float));
    cudaMemcpy(C2, delta_out, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T, M, N, batch, &alpha, A, batch, B1, N, &beta, C1, M);
    cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T, batch, N, M, &alpha, A, M, B2, N, &beta, C2, batch);

    cudaMemcpy(delta_out, C2, batch*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weight_updates, C1, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(C2);
    cudaFree(C1);
    cudaFree(B2);
    cudaFree(B1);
    cudaFree(A);

    cublasDestroy(handle_);
}

/*
void connect_backward(int batch, int N, int M, float *delta_in, float *input, float *weight_updates, float *weights, float *delta_out) { 
    int i,j,k;
    float a_part;

    // gemm
    for (i = 0; i < M; i++) {
        for (k = 0; k < batch; k++) {
            a_part = delta_in[k*M+i];
            for (j = 0; j < N; j++) {
                weight_updates[i*N+j] += a_part*input[k*N+j];
            }
        }
    }

    // gemm2
    for (i = 0; i < batch; i++) {
        for (k = 0; k < M; k++) {
            a_part = delta_in[i*M+k];
            for (j = 0; j < N; j++) {
                delta_out[i*N+j] += a_part*weights[k*N+j];
            }
        }
    }

}
 */

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
