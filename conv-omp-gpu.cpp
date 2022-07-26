#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>

#include "omp.h"

#include "conv.h"


void conv(int batch, int M, int K, int N, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, float *weights) {
    int i, j, p, q, c, h, w, gid;
    int w_offset, h_offset, c_im, row, col, col_index, out_index;

    int HWC_filt = M*K;
    int HWC_in   = batch*height*width*channels;
    int HWC_out  = batch*M*N;

    int n_groups  = 10240;  // set number of group manually
    int n_threads = 256;    // MAX=992
    int n_teams   = n_groups;

    int HWC_conv_tensor  = n_groups*height_col*width_col*channels_col;

    double tmp, time_conv_fwd;
    tmp = read_timer_ms();

    float *conv_tensor = (float *)malloc(HWC_conv_tensor*sizeof(float));

    // conv
#pragma omp target teams distribute private(gid,c,h,w,row,col,col_index,out_index,w_offset,h_offset,c_im,p,q,j) num_teams(n_teams) thread_limit(n_threads) \
	           map(alloc:conv_tensor[0:HWC_conv_tensor])    \
	           map(to:input[0:HWC_in], weights[0:HWC_filt]) \
	           map(tofrom:output[0:HWC_out])
{
    for (i = 0; i < batch; i++) {
	gid = i%n_groups;
#pragma omp parallel for collapse(3)
	for (c = 0; c < channels_col; c++) {
            //w_offset = c%ksize;
            //h_offset = (c/ksize)%ksize;
            //c_im     = (c/ksize)/ksize;
	    for (h = 0; h < height_col; h++) {
                for (w = 0; w < width_col; w++) {
		    w_offset = c%ksize;
                    h_offset = (c/ksize)%ksize;
                    c_im     = (c/ksize)/ksize;
                    row = h_offset + h*stride;
                    col = w_offset + w*stride;
                    out_index = i*channels*height*width + c_im*height*width + row*width + col;
                    col_index = gid*height_col*width_col*channels_col + c*height_col*width_col + h*width_col + w;
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
#pragma omp parallel for collapse(2)
	for (p = 0; p < M; p++) {
            //for (q = 0; q < K; q++) {
                //float a_part = weights[p*K+q];
            for (j = 0; j < N; j++) {
		float sum = 0.0;
		for (q = 0; q < K; q++) {
	            sum += weights[p*K+q]*conv_tensor[gid*K*N+q*N+j];
                    //printf("%f, %f, %f\n", weights[p*K+q], B0[q*N+j], output[i*M*N+p*N+j]);
                }
		output[i*M*N+p*N+j] = sum;
            }
      	}
    }

} // target region 1

    free(conv_tensor);

    time_conv_fwd = read_timer_ms() - tmp;
    printf("conv-forward: %lf\n", time_conv_fwd);

}

void bias(int batch, int M, int N, float *output, float *biases) {
    int b, p, q;
    
    int HWC_bias = M;
    int HWC_out = batch*M*N;

#pragma omp target teams distribute parallel for private(p,q) collapse(3) map(to:biases[0:HWC_bias]) map(tofrom:output[0:HWC_out])
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
}// target region 1

}

void relu(int batch, int M, int N, float *output) {
    int i;
    
    int HWC_out = batch*M*N; 

#pragma omp target teams distribute parallel for map(tofrom:output[0:HWC_out])
{
    for (i = 0; i < batch*M*N; i++) {
        if (output[i] < 0) output[i] = 0.0001*output[i];
        //printf("%f\n", output[i]);
    }
} // target region 1

}

void max_pool(int batch, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *output, int *indexes) {
    int b, k, i, j, n, m;
    int out_index, col_index, cur_h, cur_w;
    int max_i, valid;
    float max, val;

    int HWC_in    = batch*height*width*channels;
    int HWC_out   = batch*height_out*width_out*channels;
    int HWC_index = batch*height_out*width_out*channels;

#pragma omp target teams distribute parallel for private(k,i,j,out_index,max,max_i,n,m,cur_h,cur_w,col_index,valid,val) collapse(4) map(to:input[0:HWC_in]) map(from:indexes[0:HWC_index]) map(tofrom:output[0:HWC_out])
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
}// target region 1

}

void skip_connection(int batch, int M, int N, float *input, float *output) {
    int i, j, k;	
    
    //int HWC_in   = batch*M*N;
    //int HWC_out  = batch*M*N;
    int HWC_temp = batch*M*N;

    float *temp;
    temp = (float *)malloc(HWC_temp*sizeof(float));

//#pragma omp target teams distribute private(j,k) map(alloc:temp[0:HWC_temp]) map(to:input[0:HWC_in]) map(tofrom:output[0:HWC_out])
//{
//#pragma omp parallel for 
//{
    for (i = 0; i < batch; i++){
        for (j = 0; j < M; j++) {
            for (k = 0; k < N; k++) {
                temp[i*M*N+j*N+k] = input[i*M/4*N*2+j/4*N*2+k*2];
            }
        }
    }
//}

//#pragma omp parallel for
//{
    for (i = 0; i < batch*M*N; i++){
        output[i] += temp[i];
    }
//}

//} // target region 1
//
    free(temp);

}

void softmax(int batch, int N, float *input, float *output) {
    int b,i;
    float largest, sum;
    
    largest = -FLT_MAX;
    
    int HWC_in  = batch*N;
    int HWC_out = batch*N;

#pragma omp target teams distribute parallel for private(i) reduction(+:sum) map(to:input[0:HWC_in]) map(tofrom:output[0:HWC_out])
{
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
}// target region 1

}

void softmax_backward(int batch, int N, float *input, float *output) {
    int i;
    
    int HWC_in  = batch*N;
    int HWC_out = batch*N;

#pragma omp target teams distribute parallel for map(to:input[0:HWC_in]) map(tofrom:output[0:HWC_out])
{
    for (i = 0; i < batch*N; i++) {
        output[i] += input[i];
    }
}// target region 1

}

void relu_backward(int batch, int N, float *output, float *delta) {
    int i;
    
    int HWC_in  = batch*N;
    int HWC_out = batch*N;

#pragma omp target teams distribute parallel for map(to:output[0:HWC_in]) map(tofrom:delta[0:HWC_out])
{
    for (i = 0; i < batch*N; i++) {
        if (output[i] <= 0) delta[i] = 0;
    }
}

}

void bias_backward(int batch, int N, int M, float *input, float *output) {
    int b, i, j;
    
    int HWC_in  = batch*M*N;
    int HWC_out = M;

#pragma omp target teams distribute parallel for private(i,j) collapse(3) map(to:input[0:HWC_in]) map(tofrom:output[0:HWC_out])
{
    for (b = 0; b < batch; b++) {
        for (i = 0; i < N; i++) {
	    for (j = 0; j < M; j++) {
                output[j] += input[b*N*M+i*M+j];
            }
        }
    }
}// target region 1

}

void max_pool_backward(int batch, int N, int M, int height_out, int width_out, int ksize, int stride, int channels, int height, int width, int pad, int *indexes, float *delta_in, float *delta_out, float *input, float *output) {
    int i, index;
    
    int HWC_delta_in  = batch*N;
    int HWC_delta_out = batch*M;
    int HWC_index     = batch*N;

#pragma omp target teams distribute parallel for private(index) map(to:delta_in[0:HWC_delta_in], indexes[0:HWC_index]) map(tofrom:delta_out[0:HWC_delta_out])
{
    for (i = 0; i < batch*N; i++) {
        index = indexes[i];
        delta_out[index] += delta_in[i];
    }
} // target region 1

}

void conv_backward(int batch, int M, int K, int N, int channels_col, int height_col, int width_col, int ksize, int stride, int channels, int height, int width, int pad, float *input, float *delta_in, float *weight_updates, float *delta_out, float *weights) {
    int i, j, k, b, c, h, w;
    int w_offset, h_offset, c_im, row, col, col_index, out_index;

    int HWC_filt           = M*N;
    int HWC_weight_updates = M*N;
    int HWC_delta_in       = batch*N*K;
    int HWC_delta_out      = batch*height*width*channels;
    int HWC_in             = batch*height*width*channels;
    
    //int HWC_out            = batch*M*N;

    int HWC_conv_t1 = height_col*width_col*channels_col;
    int HWC_conv_t2 = M*K*N;

    double tmp_filt, tmp_data, time_conv_bwd_filt, time_conv_bwd_data;
    tmp_filt = read_timer_ms();

    float *conv_t1  = (float *)malloc(HWC_conv_t1*sizeof(float));

    // conv-bwd-filter
#pragma omp target teams distribute private(c,h,w,row,col,col_index,out_index,w_offset,h_offset,c_im,i,j,k) \
                   map(alloc:conv_t1[0:HWC_conv_t1])                 \
                   map(to:input[0:HWC_in], delta_in[0:HWC_delta_in]) \
                   map(tofrom:weight_updates[0:HWC_weight_updates])
{
    for (b = 0; b < batch; b++) {
#pragma omp parallel for collapse(3)
	for (c = 0; c < channels_col; c++) {
            //w_offset = c%ksize;
            //h_offset = (c/ksize)%ksize;
            //c_im = (c/ksize)/ksize;
            for (h = 0; h < height_col; h++) {
                for (w = 0; w < width_col; w++) {
                    w_offset = c%ksize;
                    h_offset = (c/ksize)%ksize;
                    c_im = (c/ksize)/ksize;

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
#pragma omp parallel for collapse(2)
	for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                float sum= 0.0;
                for (k = 0; k < K; k++) {
                    sum += delta_in[b*N*K+j*K+k]*conv_t1[i*K+k];
                }
                weight_updates[i*N+j] += sum;
            }
        }
    }
} // target region 1

    free(conv_t1);

    time_conv_bwd_filt = read_timer_ms() - tmp_filt;
    printf("conv_bwd_filt: %lf\n", time_conv_bwd_filt);

    tmp_data = read_timer_ms();
    float *conv_t2  = (float *)malloc(HWC_conv_t2*sizeof(float));
    for (i = 0; i < HWC_conv_t2; i++) conv_t2[i] = 0.0;
    
    // conv-bwd-data
#pragma omp target teams distribute private(i,j,k,c,h,w,row,col,col_index,out_index,w_offset,h_offset,c_im) \
		   map(alloc:conv_t2[0:HWC_conv_t2])                     \
                   map(to:weights[0:HWC_filt], delta_in[0:HWC_delta_in]) \
		   map(tofrom:delta_out[0:HWC_delta_out])
{
    for (b = 0; b < batch; b++) {
#pragma omp parallel for collapse(3)
	for (i = 0; i < N; i++) {
            for (j = 0; j < M; j++) {
                //float a_part = weights[i*M+j];
                for (k = 0; k < K; k++) {
                    conv_t2[i*M*K+j*K+k] += weights[i*M+j]*delta_in[b*N*K+i*K+k];
                }
            }
        }
#pragma omp parallel for collapse(3)
	// col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
        for (c = 0; c < channels_col; c++) {
            //w_offset = c%ksize;
            //h_offset = (c/ksize)%ksize;
            //c_im     = c/ksize/ksize;
            for (h = 0; h < height_col; h++) {
                for (w = 0; w < width_col; w++) {
                    w_offset = c%ksize;
                    h_offset = (c/ksize)%ksize;
                    c_im     = c/ksize/ksize;

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
} // target region 2
    free(conv_t2);

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



