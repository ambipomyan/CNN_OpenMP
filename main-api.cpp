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
    NETWORK *network = load_network(n_layers, img_h, img_w, img_c, n_classes, batch);
    
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
    printf("training datasets:   ../MNIST/train, %d images\n", img_n);
    printf("predicting datasets: ../MNIST/test,  %d images\n", img_m);

    img_h = 28;
    img_w = 28;
    img_c = 1;
    
    // get file list
    char img_files[img_n+img_m][64];
    
    DIR *d;
    struct dirent *dir;
    
    int count;
    const char *train_dir[10], *test_dir[10];

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
    for (int i = 0; i < 10; i++) {
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


// INPUT    
    // read images from file: pixels 0-255
    Mat src;
    for (int i = 0; i < img_n+img_m; i++) {
        src = imread(img_files[i], IMREAD_GRAYSCALE); //printf("%s\n", img_files[i]);
        for (int p = 0; p < img_h; p++) {
            for (int q = 0; q < img_w; q++) {
                X->vals[i*img_h*img_w+p*img_w+q] = (float)src.data[p*img_w+q];
                X->vals[i*img_h*img_w+p*img_w+q] = X->vals[i*img_h*img_w+p*img_w+q]/255; // map to [0,1]
                X->vals[i*img_h*img_w+p*img_w+q] = (X->vals[i*img_h*img_w+p*img_w+q] - 0.1307)/0.3081; // Norm
            }
        }
    }

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

    // i am lucky
    for (int i_epoch = 0; i_epoch < training_epoch; i_epoch++) {
        //printf("- EPOCH%d -\n", i_epoch);
#pragma omp parallel for num_threads(num_dev)
      for (int dev_id = 0; dev_id < num_dev; dev_id++) {
	int i_1 = training_batch/num_dev*dev_id;
	int i_2 = training_batch/num_dev*(dev_id+1);
	for (int i_batch = i_1; i_batch < i_2; i_batch++) {
            //int dev_id = i_batch%num_dev;
	    //int dev_id = omp_get_device_num();
	    //printf("- data copy batch%d, device id:%d -\n", i_batch, dev_id);
	    int index = i_batch*batch;
	    
	    network->input = X->vals+index*X->ncols;
	    network->truth = y->vals+index*y->ncols;

// BATCH_start

// FORWARD
	    forward_convolutional_layer(network->layers[0], network->layers[0], network->input, network->layers[0]->output, 1, dev_id, num_dev);
	    forward_pooling_layer(network->layers[0], network->layers[1], network->layers[0]->output, network->layers[1]->output, 1, dev_id, num_dev);
	    forward_convolutional_layer(network->layers[1], network->layers[2], network->layers[1]->output, network->layers[2]->output, 1, dev_id, num_dev);
	    forward_pooling_layer(network->layers[2], network->layers[3], network->layers[2]->output, network->layers[3]->output, 1, dev_id, num_dev);
	    forward_connected_layer(network->layers[3], network->layers[4], network->layers[3]->output, network->layers[4]->output, 1, dev_id, num_dev);
	    forward_connected_layer(network->layers[4], network->layers[5], network->layers[4]->output, network->layers[5]->output, 1, dev_id, num_dev);
	    forward_connected_layer(network->layers[5], network->layers[6], network->layers[5]->output, network->layers[6]->output, 0, dev_id, num_dev);
	    forward_softmax_layer(network->layers[6], network->layers[7], network->layers[6]->output, network->layers[7]->output, dev_id, num_dev);

// COST 
            network->cost = compute_loss_function(network->layers[7], network->truth, training_volume, training_epoch);

// BACKWARD 
            backward_softmax_layer(network->layers[7], network->layers[6], network->layers[7]->delta, network->layers[6]->delta, dev_id, num_dev);
            backward_connected_layer(network->layers[6], network->layers[5], network->layers[6]->delta, network->layers[5]->delta, 0, dev_id, num_dev);
            backward_connected_layer(network->layers[5], network->layers[4], network->layers[5]->delta, network->layers[4]->delta, 1, dev_id, num_dev);
            backward_connected_layer(network->layers[4], network->layers[3], network->layers[4]->delta, network->layers[3]->delta, 1, dev_id, num_dev);
            backward_pooling_layer(network->layers[3], network->layers[2], network->layers[3]->delta, network->layers[2]->delta, 1, dev_id, num_dev);
            backward_convolutional_layer(network->layers[2], network->layers[1], network->layers[2]->delta, network->layers[1]->delta, 1, dev_id, num_dev);
            backward_pooling_layer(network->layers[1], network->layers[0], network->layers[1]->delta, network->layers[0]->delta, 1, dev_id, num_dev);
            backward_convolutional_layer(network->layers[0], network->layers0, network->layers[0]->delta, network->layers0->delta, 1, dev_id, num_dev);
            
// UPDATE   
	    // update bias and weights
            float p1 = network->learning_rate/network->batch;
            float p2 = -network->decay*network->batch;
            float p3 = network->momentum;
	    
	    // conv1 update
	    conv_update(network->layers[0]->n, network->layers[0]->biases, network->layers[0]->bias_updates, network->layers[0]->nweights, network->layers[0]->weights, network->layers[0]->weight_updates, p1, p2, p3);

    	    // conv2 update
            conv_update(network->layers[2]->n, network->layers[2]->biases, network->layers[2]->bias_updates, network->layers[2]->nweights, network->layers[2]->weights, network->layers[2]->weight_updates, p1, p2, p3);

	    // init
	    n =  network->layers[4]->inputs*network->layers[4]->outputs;
	    
            // connect1 update
	    connect_update(network->layers[4]->outputs, network->layers[4]->biases, network->layers[4]->bias_updates, n, network->layers[4]->weights, network->layers[4]->weight_updates, p1, p2, p3);
           
	    // init
	    n =  network->layers[5]->inputs*network->layers[5]->outputs;

	    // connect2 update
            connect_update(network->layers[5]->outputs, network->layers[5]->biases, network->layers[5]->bias_updates, n, network->layers[5]->weights, network->layers[5]->weight_updates, p1, p2, p3);
            
	    // init
	    n =  network->layers[6]->inputs*network->layers[6]->outputs;

	    // connect3 update
            connect_update(network->layers[6]->outputs, network->layers[6]->biases, network->layers[6]->bias_updates, n, network->layers[6]->weights, network->layers[6]->weight_updates, p1, p2, p3);

        }

        printf("error = %f\n", network->cost);
      } 
    }

// OUTPUT
    // write weights to file
    int i;
    FILE *f;
    f = fopen("weights", "wt");
    for (i = 0; i < HWC_conv1_weights; i++)    fprintf(f, "%lf ", network->layers[0]->weights[i]);
    for (i = 0; i < HWC_conv2_weights; i++)    fprintf(f, "%lf ", network->layers[2]->weights[i]);
    for (i = 0; i < HWC_connect1_weights; i++) fprintf(f, "%lf ", network->layers[4]->weights[i]);
    for (i = 0; i < HWC_connect2_weights; i++) fprintf(f, "%lf ", network->layers[5]->weights[i]);
    for (i = 0; i < HWC_connect3_weights; i++) fprintf(f, "%lf ", network->layers[6]->weights[i]);

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
