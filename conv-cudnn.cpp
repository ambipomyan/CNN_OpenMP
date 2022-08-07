#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "conv.h"


void conv(int M, int K, int N, int batch, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, float *weights) {
    // code adapted from: https://gist.github.com/odashi/1c20ba90388cf02330e1b95963d78039
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height, width);

    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, M, channels, ksize, ksize);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, M, N);

    size_t ws_size = 0;
    //cudnnGetConvolutionForwardWorkspaceSize(handle_, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &ws_size);

    float *ws_data = NULL;
    //cudaMalloc(&ws_data, ws_size);

    float alpha = 1.0;
    float beta = 0.0;
    
    float *in_data;
    cudaMalloc(&in_data,       batch*channels*height*width*sizeof(float)); 
    cudaMemcpy(in_data, input, batch*channels*height*width*sizeof(float), cudaMemcpyHostToDevice);

    float *filt_data;
    cudaMalloc(&filt_data,         M*channels*ksize*ksize*sizeof(float));
    cudaMemcpy(filt_data, weights, M*channels*ksize*ksize*sizeof(float), cudaMemcpyHostToDevice);

    float *out_data;
    cudaMalloc(&out_data,        batch*M*N*sizeof(float)); 
    cudaMemcpy(out_data, output, batch*M*N*sizeof(float), cudaMemcpyHostToDevice); 

    double tmp, time_conv_fwd;
    tmp = read_timer_ms();

    cudnnConvolutionForward(handle_, &alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, ws_data, ws_size, &beta, out_desc, out_data); 

    time_conv_fwd = read_timer_ms() - tmp;
    printf("conv-forward: %lf\n", time_conv_fwd);

    cudaMemcpy(output, out_data, batch*M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(ws_data);
    cudaFree(out_data);
    cudaFree(filt_data);
    cudaFree(in_data);

    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroy(handle_);

}

/*
void conv(int batch, int M, int K, int N, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, float *weights) {
    int i, j, p, q, c, h, w;
    int w_offset, h_offset, c_im, row, col, col_index, out_index;
    float a_part;

    int HWC_conv_tensor  = height_col*width_col*channels_col;
    float *conv_tensor = (float *)malloc(HWC_conv_tensor*sizeof(float));
    
    //double tmp, time_im2col, time_gemm;

    // conv
    for (i = 0; i < batch; i++) {
        //tmp = read_timer_ms();
	for (c = 0; c < channels_col; c++) {
            w_offset = c%ksize;
            h_offset = (c/ksize)%ksize;
            c_im     = (c/ksize)/ksize;
	    for (h = 0; h < height_col; h++) {
                for (w = 0; w < width_col; w++) {
                    row = h_offset + h*stride;
                    col = w_offset + w*stride;
                    out_index = i*channels*height*width + c_im*height*width + row*width + col;
                    col_index = c*height_col*width_col + h*width_col + w;
                    row -= pad;
                    col -= pad;
                    if (row < 0 || col < 0 || row >= height || col >= width) {
                        conv_tensor[col_index] = 0.0;
                    } else {
                        conv_tensor[col_index] = input[out_index];
                    }
                    //printf("%d, %d\n", out_index, col_index);
                }
            }
	}

        //time_im2col = read_timer_ms() - tmp;
        //printf("conv-1   forward epoch# %d batch# %d device# %d: %lf\n", N, N, N, time_im2col);

        //tmp = read_timer_ms();
	for (p = 0; p < M; p++) {
            for (q = 0; q < K; q++) {
                a_part = weights[p*K+q];
		for (j = 0; j < N; j++) {
	            output[i*M*N+p*N+j] += a_part*conv_tensor[q*N+j];
		    //printf("%f, %f, %f\n", weights[p*K+q], B0[q*N+j], output[i*M*N+p*N+j]);
                }
            }
	}

        //time_gemm = read_timer_ms() - tmp;
        //printf("conv-2   forward epoch# %d batch# %d device# %d: %lf\n", N, N, N, time_gemm);

    }
}
 */

void bias(int batch, int M, int N, float *output, float *biases) {
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    cudnnTensorDescriptor_t aDesc, cDesc;
    cudnnCreateTensorDescriptor(&aDesc);
    cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, M, 1);

    cudnnCreateTensorDescriptor(&cDesc);
    cudnnSetTensor4dDescriptor(cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, M, N);

    float alpha = 1.0;
    float beta  = 1.0;

    float *A;
    cudaMalloc(&A,        M*sizeof(float));
    cudaMemcpy(A, biases, M*sizeof(float), cudaMemcpyHostToDevice);

    float *C;
    cudaMalloc(&C,        batch*M*N*sizeof(float));
    cudaMemcpy(C, output, batch*M*N*sizeof(float), cudaMemcpyHostToDevice);

    cudnnAddTensor(handle_, &alpha, aDesc, A, &beta, cDesc, C);

    cudaMemcpy(output, C, batch*M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(C);
    cudaFree(A);

    cudnnDestroyTensorDescriptor(cDesc);
    cudnnDestroyTensorDescriptor(aDesc);
    cudnnDestroy(handle_);
}

/*
void bias(int batch, int M, int N, float *output, float *biases) {
    int b, p, q;
    
    // # of images
    for (b = 0; b < batch; b++) {
        // # of feature maps per image / otuput channels
        for (p = 0; p < M; p++) {
            // # of pixel per feature map
            for (q = 0; q < N; q++) {
                output[b*M*N+p*N+q] += biases[p];
                //printf("%f\n", output[b*N*M+p*N+q]);
            }
        }
    }

}

*/

void relu(int batch, int M, int N, float *output) {
    // code adapted from: http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    // Describe the activation
    cudnnActivationDescriptor_t act_desc;
    cudnnCreateActivationDescriptor(&act_desc);
    cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, M, N);

    float alpha = 0.0001;
    float beta = 0.0;

    float *out_data;
    cudaMalloc(&out_data,        batch*M*N*sizeof(float));
    cudaMemcpy(out_data, output, batch*M*N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform the forward pass of the activation
    cudnnActivationForward(handle_, act_desc, &alpha, out_desc, out_data, &beta, out_desc, out_data);

    cudaMemcpy(output, out_data, batch*M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(out_data);

    cudnnDestroyTensorDescriptor(out_desc);
    // Release resources
    cudnnDestroyActivationDescriptor(act_desc);
    cudnnDestroy(handle_);
}

/*
void relu(int batch, int M, int N, float *output) {
    int i;
    
    for (i = 0; i < batch*M*N; i++) {
        if (output[i] < 0) output[i] = 0.0001*output[i];
        //printf("%f\n", output[i]);
    }
}
*/

void max_pool(int batch, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, int *indexes) {
    // code adapted from: https://gist.github.com/samskalicky/b9e80e5bbe558329ba2c2b02f6fb43db
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    cudnnPoolingDescriptor_t pool_desc;
    //create descriptor handle
    cudnnCreatePoolingDescriptor(&pool_desc);
    //initialize descriptor
    cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, ksize, ksize, pad, pad, stride, stride);

    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height, width);

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, height_out, width_out);

    float alpha = 1.0;
    float beta = 0.0;

    float *in_data;
    cudaMalloc(&in_data,       batch*channels*height*width*sizeof(float));
    cudaMemcpy(in_data, input, batch*channels*height*width*sizeof(float), cudaMemcpyHostToDevice);

    float *out_data;
    cudaMalloc(&out_data,        batch*height_out*width_out*sizeof(float));
    cudaMemcpy(out_data, output, batch*height_out*width_out*sizeof(float), cudaMemcpyHostToDevice);

    cudnnPoolingForward(handle_, pool_desc, &alpha, in_desc, in_data, &beta, out_desc, out_data);

    cudaMemcpy(output, out_data, batch*height_out*width_out*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(out_data);
    cudaFree(in_data);

    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyPoolingDescriptor(pool_desc);
    cudnnDestroy(handle_);

}

/*
void max_pool(int batch, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, int *indexes) {
    int b, k, i, j, n, m;
    int out_index, col_index, cur_h, cur_w;
    int max_i, valid;
    float max, val;
    
    for (b = 0; b < batch; b++) {
        for (k = 0; k < channels; k++) {
            for (i = 0; i < height_out; i++) {
                for (j = 0; j < width_out; j++) {
                    out_index = b*height_out*width_out*channels + k*height_out*width_out + i*width_out + j;
                    max = -FLT_MAX;
                    max_i = -1;
                    for (n = 0; n < ksize; n++) {
                        for (m = 0; m < ksize; m++) {
                            cur_h = -pad + i*stride + n;
                            cur_w = -pad + j*stride + m;
                            col_index = b*height*width*channels + k*height*width + cur_h*width + cur_w;
                            valid = (cur_h >= 0 && cur_h < height && cur_w >= 0 && cur_w < width);
                            val = -FLT_MAX;
                            if (valid != 0) {val = input[col_index];}
                            if (val > max) {max = val; max_i = col_index;}
                            //printf("%d, %d\n", out_index, col_index);
                            //printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", b, k, i, j, m, n, height, width, channels, cur_h, cur_w);
                        }
                    }
                    output[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }

}
 */

void skip_connection(int batch, int M, int N, float *input, float *output) {
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, M, N);

    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 1, 1); // 1x1 filter

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, M, N);

    size_t ws_size = 0;
    //cudnnGetConvolutionForwardWorkspaceSize(handle_, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &ws_size);

    float *ws_data = NULL;
    //cudaMalloc(&ws_data, ws_size);

    float alpha = 1.0;
    float beta = 0.0;

    float *in_data;
    cudaMalloc(&in_data,       batch*M*N*sizeof(float));
    cudaMemcpy(in_data, input, batch*M*N*sizeof(float), cudaMemcpyHostToDevice);

    float *filt_data;
    cudaMalloc(&filt_data, batch*M*N*sizeof(float));

    float *out_data;
    cudaMalloc(&out_data,        batch*M*N*sizeof(float));
    cudaMemcpy(out_data, output, batch*M*N*sizeof(float), cudaMemcpyHostToDevice);

    cudnnConvolutionForward(handle_, &alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, ws_data, ws_size, &beta, out_desc, out_data);

    cudaMemcpy(output, out_data, batch*M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(ws_data);
    cudaFree(out_data);
    cudaFree(filt_data);
    cudaFree(in_data);

    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroy(handle_);
}

/*
void skip_connection(int batch, int M, int N, float *input, float *output) {
    int i, j, k;	
    int HWC_temp = batch*M*N;

    float *temp;
    temp = (float *)malloc(HWC_temp*sizeof(float));

    for (i = 0; i < batch; i++){
        for (j = 0; j < M; j++) {
            for (k = 0; k < N; k++) {
                temp[i*M*N+j*N+k] = input[i*M/4*N*2+j/4*N*2+k*2];
            }
        }
    }
    
    for (i = 0; i < batch*M*N; i++){
        output[i] += temp[i];
    }

}
 */

void softmax(int batch, int N, float *input, float *output) {
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    cudnnTensorDescriptor_t xDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, 1, N);

    cudnnTensorDescriptor_t yDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, 1, N);

    float alpha = 1.0;
    float beta = 1.0;

    float *x;
    cudaMalloc(&x,       batch*N*sizeof(float));
    cudaMemcpy(x, input, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    float *y;
    cudaMalloc(&y,        batch*N*sizeof(float));
    cudaMemcpy(y, output, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, xDesc, x, &beta, yDesc, y);

    cudaMemcpy(output, y, batch*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(y);
    cudaFree(x);

    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle_);
}

/*
void softmax(int batch, int N, float *input, float *output) {
    int b,i;
    float largest, sum;
    largest = -FLT_MAX;
    
    for (b = 0; b < batch; b++) {
        sum = 0;
        for (i = 0; i < N; i++){
            if(input[b*N+i] > largest) largest = input[b*N+i];
        }
        //printf("largest: %f\n", largest);
        for (i = 0; i < N; i++){
            float e = exp(input[b*N+i]-largest);
            //printf("e: %f\n", e);
            //printf("diff: %f\n", largest - input[b*N+i]);
            output[b*N+i] = e;
            sum += e;
        }
        //printf("sum: %f\n", sum);
        for (i = 0; i < N; i++) {
            output[b*N+i] = output[b*N+i]/sum;
            //printf("%f ", output[b*N+i]);
        }
        //printf("\n");
    }

}
 */

void softmax_backward(int batch, int N, float *input, float *output) {
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    cudnnTensorDescriptor_t xDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, 1, N);

    cudnnTensorDescriptor_t yDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, 1, N);

    float alpha = 1.0;
    float beta = 1.0;

    float *x;
    cudaMalloc(&x,       batch*N*sizeof(float));
    cudaMemcpy(x, input, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    float *y;
    cudaMalloc(&y,        batch*N*sizeof(float));
    cudaMemcpy(y, output, batch*N*sizeof(float), cudaMemcpyHostToDevice);
 
    cudnnSoftmaxBackward(handle_, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, yDesc, y, yDesc, y, &beta, xDesc, x);

    cudaMemcpy(output, y, batch*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(y);
    cudaFree(x);

    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle_);
}

/*
void softmax_backward(int batch, int N, float *input, float *output) {
    int i;
    
    for (i = 0; i < batch*N; i++) {
        output[i] += input[i];
    }

}
*/

void relu_backward(int batch, int N, float *output, float *delta) {
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    // Describe the activation
    cudnnActivationDescriptor_t act_desc;
    cudnnCreateActivationDescriptor(&act_desc);
    cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);

    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, N, 1);

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, N, 1);

    float alpha = 1.0;
    float beta = 1.0;

    float *in_data;
    cudaMalloc(&in_data,        batch*N*sizeof(float));
    cudaMemcpy(in_data, output, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    float *out_data;
    cudaMalloc(&out_data,       batch*N*sizeof(float));
    cudaMemcpy(out_data, delta, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform the forward pass of the activation
    cudnnActivationBackward(handle_, act_desc, &alpha, in_desc, in_data, out_desc, out_data, in_desc, in_data, &beta, out_desc, out_data);

    cudaMemcpy(delta, out_data, batch*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(out_data);
    cudaFree(in_data);

    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    // Release resources
    cudnnDestroyActivationDescriptor(act_desc);
    cudnnDestroy(handle_);
}

/*
void relu_backward(int batch, int N, float *output, float *delta) {
    int i;
    
    for (i = 0; i < batch*N; i++) {
	if (output[i] <= 0) delta[i] = 0;
    }
}
*/

void bias_backward(int batch, int N, int M, float *input, float *output) {
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    cudnnTensorDescriptor_t aDesc, cDesc;
    cudnnCreateTensorDescriptor(&aDesc);
    cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, M, N);

    cudnnCreateTensorDescriptor(&cDesc);
    cudnnSetTensor4dDescriptor(cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, M, 1);

    float alpha = 1.0;
    float beta  = 0.0;

    float *A;
    cudaMalloc(&A,       batch*M*N*sizeof(float));
    cudaMemcpy(A, input, batch*M*N*sizeof(float), cudaMemcpyHostToDevice);

    float *C;
    cudaMalloc(&C,        M*sizeof(float));
    cudaMemcpy(C, output, M*sizeof(float), cudaMemcpyHostToDevice);

    cudnnAddTensor(handle_, &alpha, aDesc, A, &beta, cDesc, C);

    cudaMemcpy(output, C, M*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(C);
    cudaFree(A);

    cudnnDestroyTensorDescriptor(cDesc);
    cudnnDestroyTensorDescriptor(aDesc);
    cudnnDestroy(handle_);

}

/*
void bias_backward(int batch, int N, int M, float *input, float *output) {
    int b, i, j;
    
    for (b = 0; b < batch; b++) {
        for (i = 0; i < N; i++) {
	    for (j = 0; j < M; j++) {
                output[j] += input[b*N*M+i*M+j];
            }
        }
    }

}
 */

void max_pool_backward(int batch, int N, int M, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, int *indexes, float *delta_in, float *delta_out, float *input, float *output) {
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    cudnnPoolingDescriptor_t pool_desc;
    //create descriptor handle
    cudnnCreatePoolingDescriptor(&pool_desc);
    //initialize descriptor
    cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, ksize, ksize, pad, pad, stride, stride);

    cudnnTensorDescriptor_t xDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height_out, width_out);

    cudnnTensorDescriptor_t dxDesc;
    cudnnCreateTensorDescriptor(&dxDesc);
    cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height_out, width_out);

    cudnnTensorDescriptor_t yDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height, width);

    cudnnTensorDescriptor_t dyDesc;
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height, width);

    float alpha = 1.0;
    float beta = 0.0;

    float *x;
    cudaMalloc(&x,       batch*N*sizeof(float));
    cudaMemcpy(x, input, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    float *y;
    cudaMalloc(&y,        batch*M*sizeof(float));
    cudaMemcpy(y, output, batch*M*sizeof(float), cudaMemcpyHostToDevice);

    float *dx;
    cudaMalloc(&dx,          batch*N*sizeof(float));
    cudaMemcpy(dx, delta_in, batch*N*sizeof(float), cudaMemcpyHostToDevice);

    float *dy;
    cudaMalloc(&dy,           batch*M*sizeof(float));
    cudaMemcpy(dy, delta_out, batch*M*sizeof(float), cudaMemcpyHostToDevice);

    cudnnPoolingBackward(handle_, pool_desc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx);

    cudaMemcpy(delta_out, dy, batch*M*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dy);
    cudaFree(dx);
    cudaFree(y);
    cudaFree(x);

    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyTensorDescriptor(dxDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyPoolingDescriptor(pool_desc);
    cudnnDestroy(handle_);

}

/*
void max_pool_backward(int batch, int N, int M, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, int *indexes, float *delta_in, float *delta_out, float *input, float *output) {
    int i, index;
    
    for (i = 0; i < batch*N; i++) {
        index = indexes[i];
        delta_out[index] += delta_in[i];
    }

}
*/

void conv_backward(int batch, int M, int K, int N, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *delta_in, float *weight_updates, float *delta_out, float *weights) {
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    cudnnTensorDescriptor_t xDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height, width);

    cudnnTensorDescriptor_t dxDesc;
    cudnnCreateTensorDescriptor(&dxDesc);
    cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height, width);

    cudnnFilterDescriptor_t dwDesc;
    cudnnCreateFilterDescriptor(&dwDesc);
    cudnnSetFilter4dDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, M, channels, ksize, ksize);

    cudnnFilterDescriptor_t wDesc;
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, M, channels, ksize, ksize);

    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    cudnnTensorDescriptor_t dyDesc;
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, 1, N, K);

    size_t ws_size = 0;
    //cudnnGetConvolutionForwardWorkspaceSize(handle_, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &ws_size);

    float *ws_data = NULL;
    //cudaMalloc(&ws_data, ws_size);

    float alpha = 1.0;
    float beta = 0.0;

    float *x;
    cudaMalloc(&x,       batch*channels*height*width*sizeof(float));
    cudaMemcpy(x, input, batch*channels*height*width*sizeof(float), cudaMemcpyHostToDevice);

    float *dx;
    cudaMalloc(&dx,           batch*channels*height*width*sizeof(float));
    cudaMemcpy(dx, delta_out, batch*channels*height*width*sizeof(float), cudaMemcpyHostToDevice);

    float *dy;
    cudaMalloc(&dy,          batch*N*K*sizeof(float));
    cudaMemcpy(dy, delta_in, batch*N*K*sizeof(float), cudaMemcpyHostToDevice);

    float *dw;
    cudaMalloc(&dw,                K*channels*ksize*ksize*sizeof(float));
    cudaMemcpy(dw, weight_updates, K*channels*ksize*ksize*sizeof(float), cudaMemcpyHostToDevice);

    float *w;
    cudaMalloc(&w,         K*channels*ksize*ksize*sizeof(float));
    cudaMemcpy(w, weights, K*channels*ksize*ksize*sizeof(float), cudaMemcpyHostToDevice);

    double tmp_filt, tmp_data, time_conv_bwd_filt, time_conv_bwd_data;
    tmp_filt = read_timer_ms();

    cudnnConvolutionBackwardFilter(handle_, &alpha, xDesc, x, dyDesc, dy, convDesc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, ws_data, ws_size, &beta, dwDesc, dw);

    time_conv_bwd_filt = read_timer_ms() - tmp_filt;
    printf("conv_bwd_filt: %lf\n", time_conv_bwd_filt);

    tmp_data = read_timer_ms();

    cudnnConvolutionBackwardData(  handle_, &alpha, wDesc, w, dyDesc, dy, convDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,   ws_data, ws_size, &beta, dxDesc, dx);

    time_conv_bwd_data = read_timer_ms() - tmp_data;
    printf("conv_bwd_data: %lf\n", time_conv_bwd_data);

    cudaMemcpy(weight_updates, dw, K*channels*ksize*ksize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_out, dx, batch*channels*height*width*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dy);
    cudaFree(w);
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(x);

    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyFilterDescriptor(dwDesc);
    cudnnDestroyTensorDescriptor(dxDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle_);
}

/*
void conv_backward(int batch, int M, int K, int N, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *delta_in, float *weight_updates, float *delta_out, float *weights) {
    int i, j, k, b, c, h, w;
    int w_offset, h_offset, c_im, row, col, col_index, out_index;
    float sum, a_part;

    int HWC_conv_t1      = height_col*width_col*channels_col;
    int HWC_conv_t2      = M*K*N;
    float *conv_t1     = (float *)malloc(HWC_conv_t1*sizeof(float));
    float *conv_t2     = (float *)malloc(HWC_conv_t2*sizeof(float));

    // conv-bwd-filter
    for (b = 0; b < batch; b++) {
        for (c = 0; c < channels_col; c++) {
            w_offset = c%ksize;
            h_offset = (c/ksize)%ksize;
            c_im = (c/ksize)/ksize;
            for (h = 0; h < height_col; h++) {
                for (w = 0; w < width_col; w++) {
                    row = h_offset + h*stride;
                    col = w_offset + w*stride;
                    out_index = b*height*width*channels + c_im*height*width + row*width + col;
                    col_index = c*height_col*width_col + h*width_col + w;
                    row -= pad;
                    col -= pad;
                    if (row < 0 || col < 0 || row >= height || col >= width) {
                        conv_t1[col_index] = 0.0;
                    } else {
                        conv_t1[col_index] = input[out_index];
                    }
                }
            }
        }

        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                sum= 0;
                for (k = 0; k < K; k++) {
                    sum += delta_in[b*N*K+j*K+k]*conv_t1[i*K+k];
                }
                weight_updates[i*N+j] += sum;
            }
        }
    }

    // conv-bwd-data
    for (b = 0; b < batch; b++) {
        
	for (i = 0; i < HWC_conv_t2; i++) conv_t2[i] = 0.0;    

	for (i = 0; i < N; i++) {
            for (j = 0; j < M; j++) {
                a_part = weights[i*M+j];
                for (k = 0; k < K; k++) {
                    conv_t2[j*N*K+i*K+k] += a_part*delta_in[b*N*K+i*K+k];
                }
            }
        }

        // col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
        for (c = 0; c < channels_col; c++) {
            w_offset = c%ksize;
            h_offset = (c/ksize)%ksize;
            c_im     = c/ksize/ksize;
            for (h = 0; h < height_col; h++) {
                for (w = 0; w < width_col; w++) {
                    row = h_offset + h * stride;
                    col = w_offset + w * stride;
                    out_index = b*height*width*channels + c_im*height*width + row*width + col;
                    col_index = c*height_col*width_col + h*width_col + w;
                    row -= pad;
                    col -= pad;
                    if (!(row < 0 || col < 0 || row >= height || col >= width)) {
                        delta_out[out_index] += conv_t2[col_index];
                    }
                }
            }
        }
    }
}
 */

void conv_update(int nbias, float *biases, float *bias_updates, int nweights, float *weights, float *weight_updates, float p1, float p2, float p3) {
    // axpy
    for (int i = 0; i < nbias; i++) {biases[i] += p1*bias_updates[i]; /*printf("%lf\n", bias_updates[i]);*/}
    // scale
    for (int i = 0; i < nbias; i++) {bias_updates[i] *= p3;}
    // axpy
    for (int i = 0; i < nweights; i++) {weight_updates[i] += p2*weights[i];}
    // axpy2
    for (int i = 0; i < nweights; i++) {weights[i] += p1*weight_updates[i];}
    // scale
    for (int i = 0; i < nweights; i++) {weight_updates[i] *= p3;}
}

