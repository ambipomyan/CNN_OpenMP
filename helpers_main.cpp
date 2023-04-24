#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>
#include <dirent.h>

#include "omp.h"

#include "helpers_main.h"


float rand_uniform(float min, float max) {
    if (max < min){
        float swap = min;
        min = max;
        max = swap;
    }
    return (rand() / (float) RAND_MAX * (max - min)) + min;
}

float rand_normal() {
    static int haveSpare = 0;
    static double rand1, rand2;

    if(haveSpare)
    {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * 2*3.1415926535;

    return sqrt(rand1) * cos(rand2);
}

/****** forward ******/

void forward_convolutional_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op, int dev_id, int num_dev) {
    // shape
    //int n;
    //int N, M, K;
    //int height, width, channels, ksize, stride, pad, height_col, width_col, channels_col;

    // conv1 shape
    int M = layer->n;
    int K = layer->size*layer->size*layer->c;
    int N = layer->out_w*layer->out_h;

    int height   = layer->h;
    int width    = layer->w;
    int channels = layer->c;
    int ksize    = layer->size;
    int stride   = layer->stride;
    int pad      = layer->padding;

    int height_col   = (height+2*pad-ksize)/stride+1;
    int width_col    = (width+2*pad-ksize)/stride+1;
    int channels_col = channels*ksize*ksize;

    // init
    int n = layer->outputs * layer->batch;
            
    for (int i = 0; i < n; i++) layer->delta[i] = 0;            
    for (int i = 0; i < n; i++) output[i] = 0;

    // inputs:      
    //
    // M, K, N
    // batch            [network->layers[0]->batch]
    // channels_col
    // height_col
    // width_col
    // ksize
    // stride
    // channels
    // height
    // width
    // pad
    // *input           [network->input]
    // *output          [network->layers[0]->output]
    // *weights         [network->layers[0]->weights]
    // *T               tensor, device-only data

#pragma omp target data map(tofrom:output[0:n]) device(dev_id)
    {
    // conv
    conv(layer->batch, M, K, N, channels_col, height_col, width_col, ksize, stride, channels, height, width, pad, input, output, layer->weights, dev_id, num_dev);

    // add bias
    bias(layer->batch, M, N, output, layer->biases, dev_id, num_dev);

    // relu
    if (Op != 0) relu(layer->batch, M, N, output, dev_id, num_dev);
    }

}


void forward_pooling_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op, int dev_id, int num_dev) {
    //int N = layer->out_h*layer->out_w*layer->out_c;
    //int M = layer_->out_h*layer_->out_w*layer_->out_c;	    

    int height     = layer->h;
    int width      = layer->w;
    int channels   = layer->c;
    int ksize      = layer->size;
    int stride     = layer->stride;
    int pad        = layer->padding;

    int height_out   = layer->out_h;
    int width_out    = layer->out_w;

    // init
    int n = layer->outputs*layer->batch;
    for (int i = 0; i < n; i++) layer->delta[i] = 0;

    // max pooling
    max_pool(layer->batch, height_out, width_out, ksize, stride, channels, height, width, pad, input, output, layer->indexes, dev_id, num_dev); 

}

void forward_connected_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int Op, int dev_id, int num_dev) {
    int K = layer->inputs;
    int N = layer->outputs;

    // init
    int n = layer->outputs*layer->batch;
    for (int i = 0; i < n; i++) layer->delta[i] = 0;
    for (int i = 0; i < n; i++) layer->output[i] = 0;

#pragma omp target data map(tofrom:output[0:n]) device(dev_id)
    {
    // connected
    connect(layer->batch, K, N, input, output, layer->weights, dev_id, num_dev);

    // add bias
    bias(layer->batch, 1, N, output, layer->biases, dev_id, num_dev);
    
    // relu
    if (Op != 0) relu(layer->batch, 1, N, output, dev_id, num_dev);
    }

}

void forward_softmax_layer(LAYER *layer_, LAYER *layer, float *input, float *output, int dev_id, int num_dev) {
    int N = layer->inputs;

    // init 
    int n = layer->outputs*layer->batch;
    for (int i = 0; i < n; i++) layer->delta[i] = 0;

    //printf("N: %d\n", N);
    softmax(layer->batch, N, input, output, dev_id, num_dev);

}


/****** loss function ******/

float compute_loss_function(LAYER *layer, float *network_truth, int training_volume, int training_epoch) {
    int N = layer->batch*layer->inputs;

    // update cost
    for(int i = 0; i < N; i++) {
        int truth_index = i;
        
	float t = network_truth[truth_index];
        //printf("t: %f\n", t);
        float p = layer->output[i];
        //printf("p: %f\n", p);
        // loss
        if (t) {
            layer->loss[i] = -log(p);
        } else {
            layer->loss[i] = 0;
        }
        //printf("-logp: %f\n", network->layers[7]->loss[i]);
        layer->delta[i] = t-p;
        //printf("t-p: %f\n", network->layers[7]->delta[i]);
    }

    float sum_cost = 0;
    for (int i = 0; i < N; i++) {
        sum_cost += layer->loss[i];
        //printf("%f\n", network->layers[7]->loss[i]);
    }

    layer->cost = sum_cost;
    //printf("%f\n", network->layers[7]->cost);

    //printf("network->input size: %d\n", network->layers[7]->batch*network->layers[7]->outputs);

    // calc network cost
    float network_cost = layer->cost/(training_volume*training_epoch);

    return network_cost; 

}


/****** backward ******/

void backward_softmax_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int dev_id, int num_dev) {
    int N = layer->inputs;

    //axpy
    softmax_backward(layer->batch, N, delta_in, delta_out, dev_id, num_dev);
}

void backward_connected_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op, int dev_id, int num_dev) {
    int K = layer->inputs;
    int N = layer->outputs;

    int n_in  = layer->batch*N;
    int n_out = layer->batch*K;

    // init
    for(int i = 0; i < N; i++) layer->bias_updates[i] = 0;
    for(int i = 0; i < N*K; i++) layer->weight_updates[i] = 0;

#pragma omp target data map(to:delta_in[0:n_in]) map(tofrom:delta_out[0:n_out]) device(dev_id)
    {
    // gradient array
    if (Op != 0) relu_backward(layer->batch, N, layer->output, delta_in, dev_id, num_dev);

    // backward array
    bias_backward(layer->batch, 1, N, delta_in, layer->bias_updates, dev_id, num_dev);
            
    // backward connected
    connect_backward(layer->batch, K, N, delta_in, layer_->output, layer->weight_updates, layer->weights, delta_out, dev_id, num_dev);
    }

}

void backward_pooling_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op, int dev_id, int num_dev) {
    int N = layer->out_h*layer->out_w*layer->out_c;
    int M = layer_->out_h*layer_->out_w*layer_->out_c;            

    int height     = layer->h;
    int width      = layer->w;
    int channels   = layer->c;
    int ksize      = layer->size;
    int stride     = layer->stride;
    int pad        = layer->padding;

    int height_out   = layer->out_h;
    int width_out    = layer->out_w;

    max_pool_backward(layer->batch, N, M, height_out, width_out, ksize, stride, channels, height, width, pad, layer->indexes, delta_in, delta_out, layer->output, layer_->output, dev_id, num_dev);

}

void backward_convolutional_layer(LAYER *layer, LAYER *layer_, float *delta_in, float *delta_out, int Op, int dev_id, int num_dev) {
    int M = layer->n;
    int K = layer->size*layer->size*layer->c;
    int N = layer->out_w*layer->out_h;

    int height   = layer->h;
    int width    = layer->w;
    int channels = layer->c;
    int ksize    = layer->size;
    int stride   = layer->stride;
    int pad      = layer->padding;

    int height_col   = (height+2*pad-ksize)/stride+1;
    int width_col    = (width+2*pad-ksize)/stride+1;
    int channels_col = channels*ksize*ksize;

    int n_in  = layer->batch*M*N;
    int n_out = layer->batch*height*width*channels;

    // init 
    for(int i = 0; i < M; i++) layer->bias_updates[i] = 0;
    for(int i = 0; i < K*M; i++) layer->weight_updates[i] = 0;

#pragma omp target data map(to:delta_in[0:n_in]) map(tofrom:delta_out[0:n_out]) device(dev_id)
    {
    // gradient array
    if (Op != 0) relu_backward(layer->batch, N*M, layer->output, delta_in, dev_id, num_dev);
            
    // backward bias
    bias_backward(layer->batch, N, M, delta_in, layer->bias_updates, dev_id, num_dev);

    // conv backward
    conv_backward(layer->batch, K, N, M, channels_col, height_col, width_col, ksize, stride, channels, height, width, pad, layer_->output, delta_in, layer->weight_updates, delta_out, layer->weights, dev_id, num_dev);
    }

}

//
NETWORK *load_network(int n_layers, int img_h, int img_w, int img_c, int n_classes, int batch) {
    NETWORK *network = (NETWORK *)malloc(sizeof(NETWORK));
    
    network->n      = n_layers;
    network->layers = (LAYER **)malloc(n_layers*sizeof(LAYER *));
    network->layers0= (LAYER *)malloc(sizeof(LAYER *));
    network->cost   = 0;
    
    network->h = img_h;
    network->w = img_w;
    network->c = img_c;
    
    network->input  = (float *)malloc(network->h*network->w*network->c*batch*sizeof(float));
    network->truth  = (float *)malloc(n_classes*batch*sizeof(float));
    network->output = (float *)malloc(n_classes*batch*sizeof(float));
    
    return network;
}

void add_convolutional_layer(NETWORK *network, int id, int n, int size, int stride, int padding, int img_h, int img_w, int img_c, int batch) {
    printf("h: %d, w: %d, c: %d, ", img_h, img_w, img_c);
    printf("number of filter: %d, filter size: %d, stride: %d, padding: %d, activation: RELU\n", n, size, stride, padding);
    
    LAYER *layer0;
    layer0 = (LAYER *)malloc(sizeof(LAYER));
    layer0->layer_type = CONVOLUTIONAL;
    layer0->activation = RELU;
    
    layer0->batch   = batch;
    layer0->h       = img_h;
    layer0->w       = img_w;
    layer0->c       = img_c;
    layer0->n       = n;
    layer0->size    = size;
    layer0->stride  = stride;
    layer0->padding = padding;
    
    layer0->nweights = img_c*n*size*size;
    layer0->nbiases  = n;
    
    layer0->weights        = (float *)malloc(layer0->nweights*sizeof(float));
    layer0->weight_updates = (float *)malloc(layer0->nweights*sizeof(float));
    layer0->biases         = (float *)malloc(layer0->nbiases*sizeof(float));
    layer0->bias_updates   = (float *)malloc(layer0->nbiases*sizeof(float));
    
    float scale = sqrt(2./(layer0->nweights/layer0->nbiases));
    for(int i = 0; i < layer0->nweights; i++) {layer0->weights[i] = scale*rand_normal();}
    //for(int i = 0; i < layer0->nweights; i++) {printf("%f\n", layer0->weights[i]);}
    for(int i = 0; i < layer0->nbiases;  i++) {layer0->biases[i] = 0;}
    
    layer0->out_h   = (layer0->h+2*layer0->padding-layer0->size)/layer0->stride+1;
    layer0->out_w   = (layer0->w+2*layer0->padding-layer0->size)/layer0->stride+1;
    layer0->out_c   = layer0->n;
    layer0->outputs = layer0->out_h*layer0->out_w*layer0->out_c;
    layer0->inputs  = layer0->w*layer0->h*layer0->c;
    
    layer0->output = (float *)malloc(layer0->batch*layer0->outputs*sizeof(float));
    layer0->delta  = (float *)malloc(layer0->batch*layer0->outputs*sizeof(float));
    
    network->layers[id] = layer0;

    // layers0
    if (id==0) {
        LAYER *layers0;
        layers0 = (LAYER *)malloc(sizeof(LAYER));
        layers0->layer_type = CONVOLUTIONAL;
        layers0->activation = RELU;

        layers0->batch   = batch;
        layers0->h       = img_h;
        layers0->w       = img_w;
        layers0->c       = img_c;
        layers0->n       = n;
        layers0->size    = size;
        layers0->stride  = stride;
        layers0->padding = padding;

        layers0->nweights = img_c*n*size*size;
        layers0->nbiases  = n;

        layers0->weights        = (float *)malloc(layer0->nweights*sizeof(float));
        layers0->weight_updates = (float *)malloc(layer0->nweights*sizeof(float));
        layers0->biases         = (float *)malloc(layer0->nbiases*sizeof(float));
        layers0->bias_updates   = (float *)malloc(layer0->nbiases*sizeof(float));

        for(int i = 0; i < layer0->nweights; i++) {layers0->weights[i] = scale*rand_normal();}
        for(int i = 0; i < layer0->nbiases;  i++) {layers0->biases[i] = 0;}

        layers0->out_h   = (layer0->h+2*layer0->padding-layer0->size)/layer0->stride+1;
        layers0->out_w   = (layer0->w+2*layer0->padding-layer0->size)/layer0->stride+1;
        layers0->out_c   = layer0->n;
        layers0->outputs = layer0->out_h*layer0->out_w*layer0->out_c;
        layers0->inputs  = layer0->w*layer0->h*layer0->c;

        layers0->output = (float *)malloc(layer0->batch*layer0->outputs*sizeof(float));
        layers0->delta  = (float *)malloc(layer0->batch*layer0->outputs*sizeof(float));

        network->layers0 = layers0;
    }
}

void add_pooling_layer(NETWORK *network, int id, int size, int stride, int padding, LAYER_TYPE layer_type, int img_h, int img_w, int img_c, int batch) {
    printf("h: %d, w: %d, c: %d, ", img_h, img_w, img_c);
    printf("filter size: %d, stride: %d, padding: %d\n", size, stride, padding);
    
    LAYER *layer1;
    layer1 = (LAYER *)malloc(sizeof(LAYER));
    layer1->layer_type = layer_type;
    
    layer1->batch   = batch;
    layer1->h       = img_h;
    layer1->w       = img_w;
    layer1->c       = img_c;
    layer1->size    = size;
    layer1->stride  = stride;
    layer1->padding = padding;
    
    layer1->out_h   = (layer1->h+2*layer1->padding)/layer1->stride;
    layer1->out_w   = (layer1->w+2*layer1->padding)/layer1->stride;
    layer1->out_c   = layer1->c;
    layer1->outputs = layer1->out_h*layer1->out_w*layer1->out_c;
    layer1->inputs  = layer1->w*layer1->h*layer1->c;

    layer1->indexes = (int   *)malloc(layer1->batch*layer1->outputs*sizeof(int));
    layer1->output  = (float *)malloc(layer1->batch*layer1->outputs*sizeof(float));
    layer1->delta   = (float *)malloc(layer1->batch*layer1->outputs*sizeof(float));
    
    network->layers[id] = layer1;
}

void add_connected_layer(NETWORK *network, int id, int l_outputs, int img_h, int img_w, int img_c, int batch) {
    printf("h: %d, w: %d, c: %d, ", img_h, img_w, img_c);
    printf("output size: %d, activation: RELU\n", l_outputs);
    
    LAYER *layer4;
    layer4 = (LAYER *)malloc(sizeof(LAYER));
    layer4->layer_type = CONNECTED;
    layer4->activation = RELU;
    
    layer4->batch   = batch;
    layer4->h       = img_h;
    layer4->w       = img_w;
    layer4->c       = img_c;
    layer4->inputs  = layer4->h*layer4->w*layer4->c;
    layer4->outputs = l_outputs;

    layer4->output = (float *)malloc(layer4->batch*layer4->outputs*sizeof(float));
    layer4->delta  = (float *)malloc(layer4->batch*layer4->outputs*sizeof(float));

    layer4->weights        = (float *)malloc(layer4->outputs*layer4->inputs*sizeof(float));
    layer4->weight_updates = (float *)malloc(layer4->outputs*layer4->inputs*sizeof(float));
    layer4->biases         = (float *)malloc(layer4->outputs*sizeof(float));
    layer4->bias_updates   = (float *)malloc(layer4->outputs*sizeof(float));
    
    float scale = sqrt(2./layer4->inputs);
    for(int i = 0; i < layer4->outputs*layer4->inputs; i++) {layer4->weights[i] = scale*rand_uniform(-1, 1);}
    //for(int i = 0; i < layer4->outputs*layer4->inputs; i++) {printf("%f\n", layer4->weights[i]);}
    for(int i = 0; i < layer4->outputs; i++)                {layer4->biases[i] = 0;}
    
    network->layers[id] = layer4;
}

void add_softmax_layer(NETWORK *network, int id, int n_classes, int img_h, int img_w, int img_c, int batch) {
    printf("number of classes: %d\n", n_classes);
    
    LAYER *layer;
    layer = (LAYER *)malloc(sizeof(LAYER));
    layer->layer_type = SOFTMAX;
    
    layer->batch   = batch;
    layer->inputs  = n_classes;
    layer->outputs = layer->inputs;
    
    layer->loss   = (float *)malloc(layer->inputs*layer->batch*sizeof(float));
    layer->output = (float *)malloc(layer->inputs*layer->batch*sizeof(float));
    layer->delta  = (float *)malloc(layer->inputs*layer->batch*sizeof(float));
    layer->cost   = 0;
    
    network->outputs = layer->outputs;
    network->truths  = layer->outputs;
    
    network->layers[id] = layer;
}

// data
MATRIX *get_data(int img_m, int img_n, int img_h, int img_w, int img_c, const char *train_data_path, const char *test_data_path, int n_classes, float mean, float std) {
    // get file list
    char img_files[img_n+img_m][64];
    
    DIR *d;
    struct dirent *dir;

    int count;
    const char *train_dir[n_classes], *test_dir[n_classes];
    
    train_dir[0] = "../MNIST/train/0/";
    train_dir[1] = "../MNIST/train/1/";
    train_dir[2] = "../MNIST/train/2/";
    train_dir[3] = "../MNIST/train/3/";
    train_dir[4] = "../MNIST/train/4/";
    train_dir[5] = "../MNIST/train/5/";
    train_dir[6] = "../MNIST/train/6/";
    train_dir[7] = "../MNIST/train/7/";
    train_dir[8] = "../MNIST/train/8/";
    train_dir[9] = "../MNIST/train/9/";

    test_dir[0]  = "../MNIST/test/0/";
    test_dir[1]  = "../MNIST/test/1/";
    test_dir[2]  = "../MNIST/test/2/";
    test_dir[3]  = "../MNIST/test/3/";
    test_dir[4]  = "../MNIST/test/4/";
    test_dir[5]  = "../MNIST/test/5/";
    test_dir[6]  = "../MNIST/test/6/";
    test_dir[7]  = "../MNIST/test/7/";
    test_dir[8]  = "../MNIST/test/8/";
    test_dir[9]  = "../MNIST/test/9/";
    
    MATRIX *X = (MATRIX *)malloc(sizeof(MATRIX));
    
    X->nrows = img_n+img_m;
    X->ncols = img_h*img_w;
    X->nchannels = img_c;
    X->vals = (float *)malloc(X->nrows*X->ncols*X->nchannels*sizeof(float));
    
    count = 0;
    
    // training data
    for (int i = 0; i < n_classes; i++) {
        d = opendir(train_dir[i]); //printf("%s\n", train_dir[i]);
        while ((dir = readdir(d)) != NULL && count < img_n) {
            if (strcmp(dir->d_name, "..") != 0 && strcmp(dir->d_name, ".") != 0) {
                // file list
		strcpy(img_files[count], train_dir[i]);
		strcat(img_files[count], dir->d_name);
		
		count++;
	    }
	}
        closedir(d);
    }

    //printf("%d\n", count);

    // testing data
    for (int i = 0; i < n_classes; i++) {
        d = opendir(test_dir[i]); //printf("%s\n", test_dir[i]);
        while ((dir = readdir(d)) != NULL && count < img_n+img_m) {
	    if (strcmp(dir->d_name, "..") != 0 && strcmp(dir->d_name, ".") != 0) {
                // file list
		strcpy(img_files[count], test_dir[i]);
		strcat(img_files[count], dir->d_name);

		count++;
	    }
	}
        closedir(d);
    }
    
    // read images from file: pixels 0-255
    Mat src;
    for (int i = 0; i < img_n+img_m; i++) {
        src = imread(img_files[i], IMREAD_GRAYSCALE); //printf("%s\n", img_files[i]);
        for (int p = 0; p < img_h; p++) {
            for (int q = 0; q < img_w; q++) {
                X->vals[i*img_h*img_w+p*img_w+q] = (float)src.data[p*img_w+q];
                X->vals[i*img_h*img_w+p*img_w+q] = X->vals[i*img_h*img_w+p*img_w+q]/255; // map to [0,1]
                X->vals[i*img_h*img_w+p*img_w+q] = (X->vals[i*img_h*img_w+p*img_w+q] - mean)/std; // Norm
            }
        }
    }
    
    return X;
}

MATRIX *get_labels(int img_m, int img_n, int img_h, int img_w, int img_c, const char *train_data_path, const char *test_data_path, int n_classes) {
    // get file list
    char img_files[img_n+img_m][64];
    
    DIR *d;
    struct dirent *dir;
    
    int count;
    const char *train_dir[n_classes], *test_dir[n_classes];
    
    train_dir[0] = "../MNIST/train/0/";
    train_dir[1] = "../MNIST/train/1/";
    train_dir[2] = "../MNIST/train/2/";
    train_dir[3] = "../MNIST/train/3/";
    train_dir[4] = "../MNIST/train/4/";
    train_dir[5] = "../MNIST/train/5/";
    train_dir[6] = "../MNIST/train/6/";
    train_dir[7] = "../MNIST/train/7/";
    train_dir[8] = "../MNIST/train/8/";
    train_dir[9] = "../MNIST/train/9/";

    test_dir[0]  = "../MNIST/test/0/";
    test_dir[1]  = "../MNIST/test/1/";
    test_dir[2]  = "../MNIST/test/2/";
    test_dir[3]  = "../MNIST/test/3/";
    test_dir[4]  = "../MNIST/test/4/";
    test_dir[5]  = "../MNIST/test/5/";
    test_dir[6]  = "../MNIST/test/6/";
    test_dir[7]  = "../MNIST/test/7/";
    test_dir[8]  = "../MNIST/test/8/";
    test_dir[9]  = "../MNIST/test/9/";
    
    MATRIX *y = (MATRIX *)malloc(sizeof(MATRIX));

    y->nrows = img_n+img_m;
    y->ncols = n_classes;
    y->nchannels = img_c;
    y->vals = (float *)malloc(y->nrows*y->ncols*y->nchannels*sizeof(float));

    count = 0;
    
    // training data
    for (int i = 0; i < n_classes; i++) {
        d = opendir(train_dir[i]); //printf("%s\n", train_dir[i]);
        while ((dir = readdir(d)) != NULL && count < img_n) {
            if (strcmp(dir->d_name, "..") != 0 && strcmp(dir->d_name, ".") != 0) {
                // file list
		strcpy(img_files[count], train_dir[i]);
		strcat(img_files[count], dir->d_name);
		// labels
		for (int j = 0; j < n_classes; j++) {
		    if (j == i) {
		        y->vals[count*n_classes+j] = 1;
		    } else {
			y->vals[count*n_classes+j] = 0;
		    }
		}
		count++;
	    }
	}
        closedir(d);
    }

    //printf("%d\n", count);

    // testing data
    for (int i = 0; i < n_classes; i++) {
        d = opendir(test_dir[i]); //printf("%s\n", test_dir[i]);
        while ((dir = readdir(d)) != NULL && count < img_n+img_m) {
	    if (strcmp(dir->d_name, "..") != 0 && strcmp(dir->d_name, ".") != 0) {
                // file list
		strcpy(img_files[count], test_dir[i]);
		strcat(img_files[count], dir->d_name);
		// labels
		for (int j = 0; j < n_classes; j++) {
                    if (j == i) {
                        y->vals[count*n_classes+j] = 1;
                    } else {
                        y->vals[count*n_classes+j] = 0;
                    }
                }
		count++;
	    }
	}
        closedir(d);
    }
    
    return y;
}
