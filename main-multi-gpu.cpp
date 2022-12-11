#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>
#include <dirent.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "omp.h"

#include "helpers_main.h"

using namespace cv;

float rand_uniform(float min, float max);

float rand_normal();


int main (int argc, char **argv) {
    printf("- run_classifier -\n");
    
    // get input from user
    int training_volume   = atoi(argv[1]);
    int predicting_volume = atoi(argv[2]);
    int training_batch    = atoi(argv[3]);
    int training_epoch    = atoi(argv[4]);
    int num_dev           = atoi(argv[5]); // make it easier for experiments: no worry for conflict between omp targets and CUDA

    int batch = training_volume/training_batch;

    // set parameter for network
    int img_h = 28;
    int img_w = 28;
    int img_c = 1;
    int img_n = training_volume;
    int img_m = predicting_volume;
    int img_total = 70000;
    if (img_n+img_m > img_total) return 0;
    int n_classes = 10; // "0" - "9"
    int n_layers = (1+1)*2+3+1; // conv1->pool1->conv2->pool2->connect1->connect2->connect3->Softmax
    
    // load network
    printf("LOAD NETWORK:\n");
    printf("number of layers: %d, number of classes: %d\n", n_layers, n_classes);
    NETWORK *network;
    network         = (NETWORK *)malloc(sizeof(NETWORK));
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
    
    // init layers
    // conv1
    printf("conv1:    ");
    // filter configs
    int n       = 32;
    int size    = 3;
    int stride  = 1;
    int padding = 1;
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
    
    network->layers[0] = layer0;
    
    img_h = layer0->out_h;
    img_w = layer0->out_w;
    img_c = layer0->out_c;

    // layers0
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
    // layers0
    
    // pool1
    printf("pool1:    ");
    size    = 2;
    stride  = 2;
    padding = 1;
    printf("h: %d, w: %d, c: %d, ", img_h, img_w, img_c);
    printf("filter size: %d, stride: %d, padding: %d\n", size, stride, padding);
    
    LAYER *layer1;
    layer1 = (LAYER *)malloc(sizeof(LAYER));
    layer1->layer_type = MAXPOOL;
    
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
    
    network->layers[1] = layer1;
    
    img_h = layer1->out_h;
    img_w = layer1->out_w;
    img_c = layer1->out_c;
    
    //printf("output size: %d*%d*%d*%d = %d\n", img_h, img_w, img_c, layer1->batch, img_h*img_w*img_c*layer1->batch);
    
    // conv2
    printf("conv2:    ");
    // filter configs
    n       = 64;
    size    = 3;
    stride  = 1;
    padding = 1;
    printf("h: %d, w: %d, c: %d, ", img_h, img_w, img_c);
    printf("number of filter: %d, filter size: %d, stride: %d, padding: %d, activation: RELU\n", n, size, stride, padding);
    
    LAYER *layer2;
    layer2 = (LAYER *)malloc(sizeof(LAYER));
    layer2->layer_type = CONVOLUTIONAL;
    layer2->activation = RELU;
    
    layer2->batch = batch;
    layer2->h       = img_h;
    layer2->w       = img_w;
    layer2->c       = img_c;
    layer2->n       = n;
    layer2->size    = size;
    layer2->stride  = stride;
    layer2->padding = padding;
    
    layer2->nweights = img_c*n*size*size;
    layer2->nbiases  = n;
    
    layer2->weights        = (float *)malloc(layer2->nweights*sizeof(float));
    layer2->weight_updates = (float *)malloc(layer2->nweights*sizeof(float));
    layer2->biases         = (float *)malloc(layer2->nbiases*sizeof(float));
    layer2->bias_updates   = (float *)malloc(layer2->nbiases*sizeof(float));
    
    scale = sqrt(2./(layer2->nweights/layer2->nbiases));
    for(int i = 0; i < layer2->nweights; i++) {layer2->weights[i] = scale*rand_normal();}
    for(int i = 0; i < layer2->nbiases;  i++) {layer2->biases[i] = 0;}
    
    layer2->out_h   = (layer2->h+2*layer2->padding-layer2->size)/layer2->stride+1;
    layer2->out_w   = (layer2->w+2*layer2->padding-layer2->size)/layer2->stride+1;
    layer2->out_c   = layer2->n;
    layer2->outputs = layer2->out_h*layer2->out_w*layer2->out_c;
    layer2->inputs  = layer2->w*layer2->h*layer2->c;
    
    layer2->output = (float *)malloc(layer2->batch*layer2->outputs*sizeof(float));
    layer2->delta  = (float *)malloc(layer2->batch*layer2->outputs*sizeof(float));
    
    network->layers[2] = layer2;
        
    img_h = layer2->out_h;
    img_w = layer2->out_w;
    img_c = layer2->out_c;
    
    //pool2
    printf("pool2:    ");
    size    = 2;
    stride  = 2;
    padding = 1;
    printf("h: %d, w: %d, c: %d, ", img_h, img_w, img_c);
    printf("filter size: %d, stride: %d, padding: %d\n", size, stride, padding);
    
    LAYER *layer3;
    layer3 = (LAYER *)malloc(sizeof(LAYER));
    layer3->layer_type = MAXPOOL;
    
    layer3->batch   = batch;
    layer3->h       = img_h;
    layer3->w       = img_w;
    layer3->c       = img_c;
    layer3->size    = size;
    layer3->stride  = stride;
    layer3->padding = padding;
    
    layer3->out_h   = (layer3->h+2*layer3->padding)/layer3->stride;
    layer3->out_w   = (layer3->w+2*layer3->padding)/layer3->stride;
    layer3->out_c   = layer3->c;
    layer3->outputs = layer3->out_h*layer3->out_w*layer3->out_c;
    layer3->inputs  = layer3->w*layer3->h*layer3->c;

    layer3->indexes = (int   *)malloc(layer3->batch*layer3->outputs*sizeof(int));
    layer3->output  = (float *)malloc(layer3->batch*layer3->outputs*sizeof(float));
    layer3->delta   = (float *)malloc(layer3->batch*layer3->outputs*sizeof(float));
    
    network->layers[3] = layer3;
    
    img_h = 1;
    img_w = 1;
    img_c = layer3->out_h*layer3->out_w*layer3->out_c;
    
    //connect1
    printf("connect1: ");
    int l_outputs = 1024;
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
    
    scale = sqrt(2./layer4->inputs);
    for(int i = 0; i < layer4->outputs*layer4->inputs; i++) {layer4->weights[i] = scale*rand_uniform(-1, 1);}
    //for(int i = 0; i < layer4->outputs*layer4->inputs; i++) {printf("%f\n", layer4->weights[i]);}
    for(int i = 0; i < layer4->outputs; i++)                {layer4->biases[i] = 0;}
    
    layer4->out_h = 1;
    layer4->out_w = 1;
    layer4->out_c = l_outputs;
    
    network->layers[4] = layer4;
    
    img_h = layer4->out_h;
    img_w = layer4->out_w;
    img_c = layer4->out_c;
    
    //connect2
    printf("connect2: ");
    l_outputs = 84;
    printf("h: %d, w: %d, c: %d, ", img_h, img_w, img_c);
    printf("output size: %d, activation: RELU\n", l_outputs);
    
    LAYER *layer5;
    layer5 = (LAYER *)malloc(sizeof(LAYER));
    layer5->layer_type = CONNECTED;
    layer5->activation = RELU;
    
    layer5->batch   = batch;
    layer5->h       = img_h;
    layer5->w       = img_w;
    layer5->c       = img_c;
    layer5->inputs  = layer5->h*layer5->w*layer5->c;
    layer5->outputs = l_outputs;

    layer5->output = (float *)malloc(layer5->batch*layer5->outputs*sizeof(float));
    layer5->delta  = (float *)malloc(layer5->batch*layer5->outputs*sizeof(float));

    layer5->weights        = (float *)malloc(layer5->outputs*layer5->inputs*sizeof(float));
    layer5->weight_updates = (float *)malloc(layer5->outputs*layer5->inputs*sizeof(float));
    layer5->biases         = (float *)malloc(layer5->outputs*sizeof(float));
    layer5->bias_updates   = (float *)malloc(layer5->outputs*sizeof(float));
    
    scale = sqrt(2./layer5->inputs);
    for(int i = 0; i < layer5->outputs*layer5->inputs; i++) {layer5->weights[i] = scale*rand_uniform(-1, 1);}
    for(int i = 0; i < layer5->outputs; i++)                {layer5->biases[i] = 0;}
    
    layer5->out_h = 1;
    layer5->out_w = 1;
    layer5->out_c = l_outputs;
    
    network->layers[5] = layer5;
    
    img_h = layer5->out_h;
    img_w = layer5->out_w;
    img_c = layer5->out_c;
    
    //connect3
    printf("connect3: ");
    l_outputs = n_classes;
    printf("h: %d, w: %d, c: %d, ", img_h, img_w, img_c);
    printf("output size: %d, activation: - \n", l_outputs);
    
    LAYER *layer6;
    layer6 = (LAYER *)malloc(sizeof(LAYER));
    layer6->layer_type = CONNECTED;
    layer6->activation = RELU;
    
    layer6->batch   = batch;
    layer6->h       = img_h;
    layer6->w       = img_w;
    layer6->c       = img_c;
    layer6->inputs  = layer6->h*layer6->w*layer6->c;
    layer6->outputs = l_outputs;

    layer6->output = (float *)malloc(layer6->batch*layer6->outputs*sizeof(float));
    layer6->delta  = (float *)malloc(layer6->batch*layer6->outputs*sizeof(float));

    layer6->weights        = (float *)malloc(layer6->outputs*layer6->inputs*sizeof(float));
    layer6->weight_updates = (float *)malloc(layer6->outputs*layer6->inputs*sizeof(float));
    layer6->biases         = (float *)malloc(layer6->outputs*sizeof(float));
    layer6->bias_updates   = (float *)malloc(layer6->outputs*sizeof(float));
    
    scale = sqrt(2./layer6->inputs);
    for(int i = 0; i < layer6->outputs*layer6->inputs; i++) {layer6->weights[i] = scale*rand_uniform(-1, 1);}
    for(int i = 0; i < layer6->outputs; i++)                {layer6->biases[i] = 0;}
    
    layer6->out_h = 1;
    layer6->out_w = 1;
    layer6->out_c = l_outputs;
    
    network->layers[6] = layer6;
    
    img_h = layer6->out_h;
    img_w = layer6->out_w;
    img_c = layer6->out_c;
    
    //softmax
    printf("softmax:  ");
    printf("number of classes: %d\n", l_outputs);
    
    LAYER *layer;
    layer = (LAYER *)malloc(sizeof(LAYER));
    layer->layer_type = SOFTMAX;
    
    layer->batch   = batch;
    layer->inputs  = l_outputs;
    layer->outputs = layer->inputs;
    
    layer->loss   = (float *)malloc(layer->inputs*layer->batch*sizeof(float));
    layer->output = (float *)malloc(layer->inputs*layer->batch*sizeof(float));
    layer->delta  = (float *)malloc(layer->inputs*layer->batch*sizeof(float));
    layer->cost   = 0;
    
    network->outputs = layer->outputs;
    network->truths  = layer->outputs;
    
    network->layers[7] = layer;
    
    // load data
    printf("LOAD DATA:\n");
    printf("training datasets:   MNIST/train, %d images\n", img_n);
    printf("predicting datasets: MNIST/test,  %d images\n", img_m);
    
    img_h = 28;
    img_w = 28;
    img_c = 1;
    
    // get file list
    char img_files[img_n+img_m][32];
    
    DIR *d;
    struct dirent *dir;
    
    int count;
    const char *train_dir[10], *test_dir[10];
    
    train_dir[0] = "MNIST/train/0/";
    train_dir[1] = "MNIST/train/1/";
    train_dir[2] = "MNIST/train/2/";
    train_dir[3] = "MNIST/train/3/";
    train_dir[4] = "MNIST/train/4/";
    train_dir[5] = "MNIST/train/5/";
    train_dir[6] = "MNIST/train/6/";
    train_dir[7] = "MNIST/train/7/";
    train_dir[8] = "MNIST/train/8/";
    train_dir[9] = "MNIST/train/9/";
    
    test_dir[0] = "MNIST/test/0/";
    test_dir[1] = "MNIST/test/1/";
    test_dir[2] = "MNIST/test/2/";
    test_dir[3] = "MNIST/test/3/";
    test_dir[4] = "MNIST/test/4/";
    test_dir[5] = "MNIST/test/5/";
    test_dir[6] = "MNIST/test/6/";
    test_dir[7] = "MNIST/test/7/";
    test_dir[8] = "MNIST/test/8/";
    test_dir[9] = "MNIST/test/9/";

    MATRIX *X, *y;
    X = (MATRIX *)malloc(sizeof(MATRIX));
    y = (MATRIX *)malloc(sizeof(MATRIX));
    
    X->nrows = img_n+img_m;
    X->ncols = img_h*img_w;
    X->nchannels = img_c;
    X->vals = (float *)malloc(X->nrows*X->ncols*X->nchannels*sizeof(float));

    y->nrows = X->nrows;
    y->ncols = n_classes;
    y->nchannels = img_c;
    y->vals = (float *)malloc(y->nrows*y->ncols*y->nchannels*sizeof(float));

    count = 0;
    
    // training data
    for (int i = 0; i < 10; i++) {
        d = opendir(train_dir[i]);
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
    for (int i = 0; i < 10; i++) {
        d = opendir(test_dir[i]);
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
   
   /* 
    printf("check image list:\n");
    for (int i = 0; i < img_n+img_m; i++) {
        printf("%s\n", img_files[i]);
    }
    */

    // timer
    double tmp_total_epoch;
    double tmp, tmp_forward, tmp_backward, tmp_update, tmp_total_batch;
    
    double time_total_epoch;
    double time_io_read, time_io_write;
    double time_forward, time_backward, time_update, time_total_batch;
    double time_conv1,   time_conv2,   time_connect1,   time_connect2,   time_connect3,   time_pool1,   time_pool2,   time_softmax;
    double time_conv1_2, time_conv2_2, time_connect1_2, time_connect2_2, time_connect3_2, time_pool1_2, time_pool2_2, time_softmax_2;
    double time_conv1_3, time_conv2_3, time_connect1_3, time_connect2_3, time_connect3_3;

    tmp = read_timer_ms();

// INPUT    
    // read images from file: pixels 0-255
    Mat src;
    for (int i = 0; i < img_n+img_m; i++) {
        src = imread(img_files[i], IMREAD_GRAYSCALE);
        for (int p = 0; p < img_h; p++) {
            for (int q = 0; q < img_w; q++) {
                X->vals[i*img_h*img_w+p*img_w+q] = (float)src.data[p*img_w+q];
                X->vals[i*img_h*img_w+p*img_w+q] = X->vals[i*img_h*img_w+p*img_w+q]/255; // map to [0,1]
                X->vals[i*img_h*img_w+p*img_w+q] = (X->vals[i*img_h*img_w+p*img_w+q] - 0.1307)/0.3081; // Norm
            }
        }
    }

    time_io_read = read_timer_ms() - tmp;
    printf("io_read: %lf\n", time_io_read);

   /*
    printf("check imgs: \n");
    for (int i = 0; i < img_n+img_m; i++) {
        printf("#%d\n", i);
        for (int j = 0; j < img_h; j++) {
            for (int k = 0; k < img_w; k++) {
                printf("%.2f ", X->vals[i*img_h*img_w+j*img_w+k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */

    /*
    printf("check labels: \n");
    for (int i = 0; i < img_n+img_m; i++) {
        printf("#%d\n", i);
        for (int j = 0; j < n_classes; j++) {
            printf("%.2f ", y->vals[i*n_classes+j]);
        }
        printf("\n");
    }
    */

// TRAIN
    printf("TRAIN NETWORK:\n");
    if (training_volume%training_batch!=0) return -1;
    
    network->batch         = batch;
    network->learning_rate = 0.0001;
    network->momentum      = 0.9;
    network->decay         = 0.0001;
    
    printf("number of training images: %d, batch: %d, epoch: %d\n", training_volume, training_batch, training_epoch);
    printf("training config: batch size: %d, learning rate: %f, momentum: %f, decay: %f\n", network->batch, network->learning_rate, network->momentum, network->decay);

    // if there is only one device, then this part is of serial processing
    printf("number of devices:%d\n", num_dev);

    // model
    int HWC_conv1_weights = network->layers[0]->n*network->layers[0]->size*network->layers[0]->size*network->layers[0]->c;
    int HWC_conv2_weights = network->layers[2]->n*network->layers[2]->size*network->layers[2]->size*network->layers[2]->c;
    int HWC_connect1_weights = network->layers[4]->inputs*network->layers[4]->outputs;
    int HWC_connect2_weights = network->layers[5]->inputs*network->layers[5]->outputs;
    int HWC_connect3_weights = network->layers[6]->inputs*network->layers[6]->outputs;

    // insert timer here
    time_total_epoch = 0.0;
    
    time_io_read = 0.0; time_io_write = 0.0; time_forward = 0.0; time_backward = 0.0; time_update = 0.0; time_total_batch = 0.0;
    
    time_conv1   = 0.0; time_conv2   = 0.0; time_connect1   = 0.0; time_connect2   = 0.0; time_connect3   = 0.0; time_pool1   = 0.0; time_pool2   = 0.0; time_softmax   = 0.0;
    time_conv1_2 = 0.0; time_conv2_2 = 0.0; time_connect1_2 = 0.0; time_connect2_2 = 0.0; time_connect3_2 = 0.0; time_pool1_2 = 0.0; time_pool2_2 = 0.0; time_softmax_2 = 0.0;
    time_conv1_3 = 0.0; time_conv2_3 = 0.0; time_connect1_3 = 0.0; time_connect2_3 = 0.0; time_connect3_3 = 0.0;

/*
    // get number of devices, teams and threads
    num_dev = omp_get_num_devices(); // num_dev has been initilized before

#pragma omp parallel for num_threads(num_dev)
    for (int i = 0; i < num_dev; i++) {
#pragma omp target device(i)
{
        if (omp_is_initial_device()) {
            printf("Running on host!\n");
        } else {
            int nteams   = omp_get_num_teams();
	    int nthreads = omp_get_num_threads();
	    printf("Running on device %d with %d teams in total and %d threads in each team!\n", i, nteams, nthreads);
        }
}	
    }
 */

    // loop start
    tmp_total_epoch = read_timer_ms();

    // i am lucky
    for (int i_epoch = 0; i_epoch < training_epoch; i_epoch++) {
        //printf("- EPOCH%d -\n", i_epoch);
#pragma omp parallel for num_threads(num_dev)
	for (int i_batch = 0; i_batch < training_batch; i_batch++) {
            int dev_id = i_batch%num_dev;
	    //int dev_id = omp_get_device_num();
	    //printf("- data copy batch%d, device id:%d -\n", i_batch, dev_id);
	    int index = i_batch*batch;
	    
	    network->input = X->vals+index*X->ncols;
            network->truth = y->vals+index*y->ncols;

            /*
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < X->ncols; j++) {
                    if (j%28==0) printf("\n");
                    printf("%.2f ", network->input[i*X->ncols+j]);
                }
                printf("\n");
            }
            printf("\n");
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < n_classes; j++) {
                    printf("%.2f ", network->truth[i*n_classes+j]);
                }
                printf("\n");
            }
            printf("\n");
             */

// BATCH_start
            tmp_total_batch = read_timer_ms();

	    //printf("- training batch%d  -\n", i_batch);

// FORWARD
	    tmp_forward = read_timer_ms();

            //printf("- forward  conv1    -\n");
	    tmp = read_timer_ms();
            
	    forward_convolutional_layer(network->layers[0], network->layers[0], network->input, network->layers[0]->output, 1, dev_id, num_dev);

	    time_conv1 += read_timer_ms() - tmp;
	    printf("conv1    forward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_conv1);

	    //printf("network->input size: %d\n", network->layers[0]->batch*network->layers[0]->outputs);
            
            //printf("- forward  pool1    -\n");
	    tmp = read_timer_ms();
	    
	    //forward_pooling_layer(network->layers[0], network->layers[1], network->layers[0]->output, network->layers[1]->output, 1, dev_id, num_dev);

	    time_pool1 += read_timer_ms() - tmp;
            printf("maxpool1 forward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_pool1);

	    /*
            printf("check imgs: \n");
            for (int i = 0; i < img_n; i++) {
                printf("#%d\n", i);
                for (int j = 0; j < img_h; j++) {
                    for (int k = 0; k < img_w; k++) {
                        printf("%d, %.2f ", i*img_h*img_w+j*img_w+k, network->input[i*img_h*img_w+j*img_w+k]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
             */
            
            //printf("network->input size: %d\n", network->layers[1]->batch*network->layers[1]->outputs);

            //printf("- forward  conv2    -\n");
            tmp = read_timer_ms();
	    
	    //forward_convolutional_layer(network->layers[1], network->layers[2], network->layers[1]->output, network->layers[2]->output, 1, dev_id, num_dev);

	    time_conv2 += read_timer_ms() - tmp;
            printf("conv2    forward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_conv2);

	    //printf("- skip connection   -\n");
	    // skip-connection
            //skip_connection(network->layers[2]->batch, M2, N2, network->layers[0]->output, network->layers[2]->output);

            //printf("network->input size: %d\n", network->layers[2]->batch*network->layers[2]->outputs);

	    //printf("- forward  pool2    -\n");
	    tmp = read_timer_ms();
	    
            //forward_pooling_layer(network->layers[2], network->layers[3], network->layers[2]->output, network->layers[3]->output, 1, dev_id, num_dev);

            time_pool2 += read_timer_ms() - tmp;
            printf("maxpool2 forward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_pool2);

            //printf("network->input size: %d\n", network->layers[3]->batch*network->layers[3]->outputs);
            
            //printf("- forward  connect1 -\n");
	    tmp = read_timer_ms();
	    
	    //forward_connected_layer(network->layers[3], network->layers[4], network->layers[3]->output, network->layers[4]->output, 1, dev_id, num_dev);

	    //printf("network->input size: %d\n", network->layers[4]->batch*network->layers[4]->outputs);
            
	    time_connect1 += read_timer_ms() - tmp;
            printf("connect1 forward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_connect1);

            //printf("- forward  connect2 -\n");
	    tmp = read_timer_ms();

            //forward_connected_layer(network->layers[4], network->layers[5], network->layers[4]->output, network->layers[5]->output, 1, dev_id, num_dev);
	    
	    //printf("network->input size: %d\n", network->layers[5]->batch*network->layers[5]->outputs);
            
	    time_connect2 += read_timer_ms() - tmp;
            printf("connect2 forward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_connect2);

            //printf("- forward  connect3 -\n");
	    tmp = read_timer_ms();

	    //forward_connected_layer(network->layers[5], network->layers[6], network->layers[5]->output, network->layers[6]->output, 0, dev_id, num_dev);

	    time_connect3 += read_timer_ms() - tmp;
            printf("connect3 forward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_connect3);

            //printf("network->input size: %d\n", network->layers[6]->batch*network->layers[6]->outputs);

            //printf("- forward  softmax  -\n");
	    tmp = read_timer_ms();

	    //forward_softmax_layer(network->layers[6], network->layers[7], network->layers[6]->output, network->layers[7]->output, dev_id, num_dev);

	    time_softmax += read_timer_ms() - tmp;
            printf("softmax  forward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_softmax);

	    time_forward += read_timer_ms() - tmp_forward;
            printf("forward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_forward);

// COST 
            network->cost = compute_loss_function(network->layers[7], network->truth, training_volume, training_epoch);

// BACKWARD
	    tmp_backward = read_timer_ms();

            //printf("- backward softmax  -\n");
            // remember to update network->input and network->delta
            //LAYER *prev = network->layers[i-1];
            //network->input = prev->output;
            //network->delta = prev->delta;
            // if i = 0, network->input = network->layers[0]->output
            //
	    tmp = read_timer_ms();
            
            //backward_softmax_layer(network->layers[7], network->layers[6], network->layers[7]->delta, network->layers[6]->delta, dev_id, num_dev);

	    time_softmax_2 += read_timer_ms() - tmp;
            printf("softmax  backward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_softmax_2);
	    
	    //printf("- backward connect3 -\n");
	    tmp = read_timer_ms();

            //backward_connected_layer(network->layers[6], network->layers[5], network->layers[6]->delta, network->layers[5]->delta, 0, dev_id, num_dev);

	    time_connect3_2 += read_timer_ms() - tmp;
            printf("connect3 backward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_connect3_2);

	    //printf("- backward connect2 -\n");
	    tmp = read_timer_ms();

            //backward_connected_layer(network->layers[5], network->layers[4], network->layers[5]->delta, network->layers[4]->delta, 1, dev_id, num_dev);

	    time_connect2_2 += read_timer_ms() - tmp;
            printf("connect2 backward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_connect2_2);
            
	    //printf("- backward connect1 -\n");
	    tmp = read_timer_ms();

            //backward_connected_layer(network->layers[4], network->layers[3], network->layers[4]->delta, network->layers[3]->delta, 1, dev_id, num_dev);

            time_connect1_2 += read_timer_ms() - tmp;
            printf("connect1 backward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_connect1_2);

	    //printf("- backward pool2    -\n");
	    tmp = read_timer_ms();
            
	    //backward_pooling_layer(network->layers[3], network->layers[2], network->layers[3]->delta, network->layers[2]->delta, 1, dev_id, num_dev);

	    time_pool2_2 += read_timer_ms() - tmp;
            printf("pool2    backward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_pool2_2);
 
	    //printf("- backward conv2    -\n");
	    tmp = read_timer_ms();

            //backward_convolutional_layer(network->layers[2], network->layers[1], network->layers[2]->delta, network->layers[1]->delta, 1, dev_id, num_dev);

	    time_conv2_2 += read_timer_ms() - tmp;
            printf("conv2    backward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_conv2_2);

            //printf("- backward pool1    -\n");
	    tmp = read_timer_ms();

	    //backward_pooling_layer(network->layers[1], network->layers[0], network->layers[1]->delta, network->layers[0]->delta, 1, dev_id, num_dev);
       
            time_pool1_2 += read_timer_ms() - tmp;
            printf("pool1    backward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_pool1_2);

            //printf("- backward conv1    -\n");
            tmp = read_timer_ms();

            //backward_convolutional_layer(network->layers[0], network->layers0, network->layers[0]->delta, network->layers0->delta, 1, dev_id, num_dev);

	    time_conv1_2 += read_timer_ms() - tmp;
            printf("conv1    backward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_conv1_2);

	    time_backward += read_timer_ms() - tmp_backward;
            printf("backward epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_backward);
            
// UPDATE
	    tmp_update = read_timer_ms();
	    
	    // update bias and weights
            float p1 = network->learning_rate/network->batch;
            float p2 = -network->decay*network->batch;
            float p3 = network->momentum;
            
	    //printf("- update   conv1    -\n");
	    tmp = read_timer_ms();
	    
	    // conv1 update
	    conv_update(network->layers[0]->n, network->layers[0]->biases, network->layers[0]->bias_updates, network->layers[0]->nweights, network->layers[0]->weights, network->layers[0]->weight_updates, p1, p2, p3);

	    time_conv1_3 += read_timer_ms() - tmp;
            printf("conv1     update epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_conv1_3);

	    //printf("- update   conv2    -\n");
	    tmp = read_timer_ms();

    	    // conv2 update
            conv_update(network->layers[2]->n, network->layers[2]->biases, network->layers[2]->bias_updates, network->layers[2]->nweights, network->layers[2]->weights, network->layers[2]->weight_updates, p1, p2, p3);
            
	    time_conv2_3 += read_timer_ms() - tmp;
            printf("conv2     update epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_conv2_3);

            //printf("- update   connect1 -\n");
            tmp = read_timer_ms();

	    // init
	    n =  network->layers[4]->inputs*network->layers[4]->outputs;
	    
            // connect1 update
	    connect_update(network->layers[4]->outputs, network->layers[4]->biases, network->layers[4]->bias_updates, n, network->layers[4]->weights, network->layers[4]->weight_updates, p1, p2, p3);
           
	    time_connect1_3 += read_timer_ms() - tmp;
            printf("connect1  update epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_connect1_3);
 
            //printf("- update   connect2 -\n");
	    tmp = read_timer_ms(); 

	    // init
	    n =  network->layers[5]->inputs*network->layers[5]->outputs;

	    // connect2 update
            connect_update(network->layers[5]->outputs, network->layers[5]->biases, network->layers[5]->bias_updates, n, network->layers[5]->weights, network->layers[5]->weight_updates, p1, p2, p3);
            
            time_connect2_3 += read_timer_ms() - tmp;
            printf("connect2  update epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_connect2_3);

            //printf("- update   connect3 -\n");
            tmp = read_timer_ms();
            
	    // init
	    n =  network->layers[6]->inputs*network->layers[6]->outputs;

	    // connect3 update
            connect_update(network->layers[6]->outputs, network->layers[6]->biases, network->layers[6]->bias_updates, n, network->layers[6]->weights, network->layers[6]->weight_updates, p1, p2, p3);

	    time_connect3_3 += read_timer_ms() - tmp;
            printf("connect3  update epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_connect3_3);

	    time_update += read_timer_ms() - tmp_update;
            printf("update epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_update);

            time_total_batch += read_timer_ms() - tmp_total_batch;
            printf("total_batch epoch# %d batch# %d device# %d: %lf\n", i_epoch, i_batch, dev_id, time_total_batch);

        }

        printf("error = %f\n", network->cost);
    
    }

    time_total_epoch = read_timer_ms() - tmp_total_epoch;
    printf("total_epoch: %lf\n", time_total_epoch);

// OUTPUT    
    tmp = read_timer_ms();

    // write weights to file
    int i;
    FILE *f;
    f = fopen("weights", "wt");
    for (i = 0; i < HWC_conv1_weights; i++)    fprintf(f, "%lf ", network->layers[0]->weights[i]);
    for (i = 0; i < HWC_conv2_weights; i++)    fprintf(f, "%lf ", network->layers[2]->weights[i]);
    for (i = 0; i < HWC_connect1_weights; i++) fprintf(f, "%lf ", network->layers[4]->weights[i]);
    for (i = 0; i < HWC_connect2_weights; i++) fprintf(f, "%lf ", network->layers[5]->weights[i]);
    for (i = 0; i < HWC_connect3_weights; i++) fprintf(f, "%lf ", network->layers[6]->weights[i]);
    
    time_io_write = read_timer_ms() - tmp_update;
    printf("io_write: %lf\n", time_io_write);

// INFER
    printf("INFER NETWORK:\n");
    int temp_count = 0;
    int predicting_batch = predicting_volume/(network->batch);
    for (int i_batch = 0; i_batch < predicting_batch; i_batch++) {
        // data copy
	int idx = i_batch*batch;
        network->input = X->vals+idx*X->ncols+img_n*X->ncols;
        network->truth = y->vals+idx*y->ncols+img_n*y->ncols;

        // forwarding!
	forward_convolutional_layer(network->layers[0], network->layers[0], network->input,             network->layers[0]->output, 1, 0, num_dev);
	forward_pooling_layer(      network->layers[0], network->layers[1], network->layers[0]->output, network->layers[1]->output, 1, 0, num_dev);
	forward_convolutional_layer(network->layers[1], network->layers[2], network->layers[1]->output, network->layers[2]->output, 1, 0, num_dev);
        forward_pooling_layer(      network->layers[2], network->layers[3], network->layers[2]->output, network->layers[3]->output, 1, 0, num_dev);
	forward_connected_layer(    network->layers[3], network->layers[4], network->layers[3]->output, network->layers[4]->output, 1, 0, num_dev);
	forward_connected_layer(    network->layers[4], network->layers[5], network->layers[4]->output, network->layers[5]->output, 1, 0, num_dev);
        forward_connected_layer(    network->layers[5], network->layers[6], network->layers[5]->output, network->layers[6]->output, 0, 0, num_dev);
        forward_softmax_layer(      network->layers[6], network->layers[7], network->layers[6]->output, network->layers[7]->output,    0, num_dev);

        // recording!
        int N = network->layers[7]->inputs;
        //printf("N: %d\n", N);
	int T[batch];
        //printf("batch: %d\n", batch);
        for (int b = 0; b < network->layers[7]->batch; b++) {
            float temp_val = -FLT_MAX;
            int temp_index;
            //printf("- output#%d: ", i_batch*batch+b);
            for(int i = 0; i < N; i++) {
                if (network->layers[7]->output[b*N+i] > temp_val) {
                    temp_val = network->layers[7]->output[b*N+i];
                    temp_index = i;
                }
                //printf("%f ", network->layers[7]->output[b*N+i]);
            }

	    //printf("[predict: %d; truth: ", temp_index);
	    for (int ii = 0; ii < N; ii++) {
		//printf("%f", network->truth[b*N+ii]);
		if (network->truth[b*N+ii] == 1.0) {
		    //printf("%d", ii);
		    T[b] = ii;
		    break;
		}
	    }
	    //printf("] -\n");

            if (temp_index == T[b]) {
                temp_count++;
            }
        }
    }
    printf("ratio: %f\n", (float)temp_count/img_m);
    
    free(network);
    
    return 0;
}

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
