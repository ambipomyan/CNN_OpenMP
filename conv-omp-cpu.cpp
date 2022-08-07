#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>

#include "omp.h"

#include "conv.h"


void conv(int batch, int M, int K, int N, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, float *weights) {
    int i, j, p, q, c, h, w;
    int w_offset, h_offset, c_im, row, col, col_index, out_index;
    float a_part;

    int HWC_conv_tensor  = height_col*width_col*channels_col;

    double tmp, time_conv_fwd;
    tmp = read_timer_ms();

    // conv
#pragma omp parallel private(c,h,w,row,col,col_index,out_index,w_offset,h_offset,c_im,p,q,j,a_part) shared(input,output,weights)
{
    float *conv_tensor = (float *)malloc(HWC_conv_tensor*sizeof(float));
#pragma omp for
{
    for (i = 0; i < batch; i++) {
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

	for (p = 0; p < M; p++) {
            for (q = 0; q < K; q++) {
                a_part = weights[p*K+q];
		for (j = 0; j < N; j++) {
	            output[i*M*N+p*N+j] += a_part*conv_tensor[q*N+j];
		    //printf("%f, %f, %f\n", weights[p*K+q], B0[q*N+j], output[i*M*N+p*N+j]);
                }
            }
	}
    }

} // omp-for
   free(conv_tensor); 
} // parallel region 1

    time_conv_fwd = read_timer_ms() - tmp;
    printf("conv-forward: %lf\n", time_conv_fwd);

}

void bias(int batch, int M, int N, float *output, float *biases) {
    int b, p, q;

#pragma omp parallel for private(p,q) shared(output,biases)
{
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
}// parallel region 1

}

void relu(int batch, int M, int N, float *output) {
    int i;

#pragma omp parallel for shared(output)
{
    for (i = 0; i < batch*M*N; i++) {
        if (output[i] < 0) output[i] = 0.0001*output[i];
        //printf("%f\n", output[i]);
    }
} // parallel region 1

}

void max_pool(int batch, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, int *indexes) {
    int b, k, i, j, n, m;
    int out_index, col_index, cur_h, cur_w;
    int max_i, valid;
    float max, val;

#pragma omp parallel for private(k,i,j,out_index,max,max_i,n,m,cur_h,cur_w,col_index,valid,val) shared(input,output,indexes)
{
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
}// parallel region 1

}

void skip_connection(int batch, int M, int N, float *input, float *output) {
    int i, j, k;	
    int HWC_temp = batch*M*N;

    float *temp;
    temp = (float *)malloc(HWC_temp*sizeof(float));

//#pragma omp parallel for private(j,k)
//{
    for (i = 0; i < batch; i++){
        for (j = 0; j < M; j++) {
            for (k = 0; k < N; k++) {
                temp[i*M*N+j*N+k] = input[i*M/4*N*2+j/4*N*2+k*2];
            }
        }
    }
//} // parallel region 1

//#pragma omp parallel for
//{
    for (i = 0; i < batch*M*N; i++){
        output[i] += temp[i];
    }
//} // parallel region 2

    free(temp);
}

void softmax(int batch, int N, float *input, float *output) {
    int b,i;
    float largest, sum, e;
    
    largest = -FLT_MAX;

#pragma omp parallel for private(i,largest,e) shared(input,output) reduction(+:sum)
{
    for (b = 0; b < batch; b++) {
        sum = 0;
        for (i = 0; i < N; i++){
            if(input[b*N+i] > largest) largest = input[b*N+i];
        }
        //printf("largest: %f\n", largest);
        for (i = 0; i < N; i++){
            e = exp(input[b*N+i]-largest);
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
}// parallel region 1

}

void softmax_backward(int batch, int N, float *input, float *output) {
    int i;

#pragma omp parallel for shared(output,input)
{
    for (i = 0; i < batch*N; i++) {
        output[i] += input[i];
    }
}// parallel region 1

}

void relu_backward(int batch, int N, float *output, float *delta) {
    int i;

#pragma omp parallel for shared(output,delta)
{
    for (i = 0; i < batch*N; i++) {
	if (output[i] <= 0) delta[i] = 0;
    }
} // parallel region 1

}

void bias_backward(int batch, int N, int M, float *input, float *output) {
    int b, i, j;

#pragma omp parallel for private(i,j) shared(output,input)
{
    for (b = 0; b < batch; b++) {
        for (i = 0; i < N; i++) {
	    for (j = 0; j < M; j++) {
                output[j] += input[b*N*M+i*M+j];
            }
        }
    }
} // parallel region 1

}

void max_pool_backward(int batch, int N, int M, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, int *indexes, float *delta_in, float *delta_out, float *input, float *output) {
    int i, index;

#pragma omp parallel for private(index) shared(indexes,delta_out,delta_in)
{
    for (i = 0; i < batch*N; i++) {
        index = indexes[i];
        delta_out[index] += delta_in[i];
    }
} // parallel region 1

}

void conv_backward(int batch, int M, int K, int N, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *delta_in, float *weight_updates, float *delta_out, float *weights) {
    int i, j, k, b, c, h, w;
    int w_offset, h_offset, c_im, row, col, col_index, out_index;
    float sum, a_part;

    int HWC_conv_t1 = height_col*width_col*channels_col;
    int HWC_conv_t2 = M*K*N;

    double tmp_filt, tmp_data, time_conv_bwd_filt, time_conv_bwd_data;
    
    tmp_filt = read_timer_ms();

    // conv-bwd-filter
#pragma omp parallel private(c,h,w,row,col,col_index,out_index,w_offset,h_offset,c_im,i,j,k,sum) shared(delta_in,input,weight_updates)
{
    float *conv_t1 = (float *)malloc(HWC_conv_t1*sizeof(float));
#pragma omp for
{
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

} // omp-for
    free(conv_t1);
} // parallel region 1

    time_conv_bwd_filt = read_timer_ms() - tmp_filt;
    printf("conv_bwd_filt: %lf\n", time_conv_bwd_filt);

    tmp_data = read_timer_ms();

    // conv-bwd-data
#pragma omp parallel private(i,j,k,a_part,c,h,w,row,col,col_index,out_index,w_offset,h_offset,c_im) shared(weights,delta_in,delta_out)
{
    float *conv_t2  = (float *)malloc(HWC_conv_t2*sizeof(float));
#pragma omp for
{
    for (b = 0; b < batch; b++) {
	
	for (i = 0; i < HWC_conv_t2; i++) conv_t2[i] = 0.0;
        
	for (i = 0; i < N; i++) {
            for (j = 0; j < M; j++) {
                a_part = weights[i*M+j];
                for (k = 0; k < K; k++) {
                    conv_t2[i*M*K+j*K+k] += a_part*delta_in[b*N*K+i*K+k];
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
} // omp-for
    free(conv_t2);
} // parallel region 2

    time_conv_bwd_data = read_timer_ms() - tmp_data;
    printf("conv_bwd_data: %lf\n", time_conv_bwd_data);

}

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



