#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>
#include <dirent.h>

#include "omp.h"

#include "helpers_main.h"


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
    
// NETWORK    
    // load network
    printf("LOAD NETWORK:\n");
    printf("number of layers: %d, number of classes: %d\n", n_layers, n_classes);
    NETWORK *network = load_network(n_layers, img_h, img_w, img_c, n_classes, batch);
    
    // init layers
    // conv1
    printf("conv1:    ");
    add_convolutional_layer(network, 0, 32, 3, 1, 1, img_h, img_w, img_c, batch);
    // pool1
    printf("pool1:    ");
    add_pooling_layer(network, 1, 2, 2, 1, MAXPOOL, network->layers[0]->out_h, network->layers[0]->out_w, network->layers[0]->out_c, batch);
    // conv2
    printf("conv2:    ");
    add_convolutional_layer(network, 2, 64, 3, 1, 1, network->layers[1]->out_h, network->layers[1]->out_w, network->layers[1]->out_w, batch);
    //pool2
    printf("pool2:    ");
    add_pooling_layer(network, 3, 2, 2, 1, MAXPOOL, network->layers[2]->out_h, network->layers[2]->out_w, network->layers[2]->out_c, batch);
    //connect1
    printf("connect1: ");
    add_connected_layer(network, 4, 1024, 1, 1, network->layers[3]->out_h*network->layers[3]->out_w*network->layers[3]->out_c, batch);
    //connect2
    printf("connect2: ");
    add_connected_layer(network, 5, 84, 1, 1, 1024, batch);
    //connect3
    printf("connect3: ");
    add_connected_layer(network, 6, n_classes, 1, 1, 84, batch);
    //softmax
    printf("softmax:  ");
    add_softmax_layer(network, 7, n_classes, 1, 1, n_classes, batch);

// DATA 
    // load data
    printf("LOAD DATA:\n");
    printf("training datasets:   ../MNIST/train, %d images\n", img_n);
    printf("predicting datasets: ../MNIST/test,  %d images\n", img_m);
    
    MATRIX *X = get_data(  img_m, img_n, img_h, img_w, img_c, "../MNIST/train/", "../MNIST/test/", n_classes, 0.1307, 0.3081);
    MATRIX *y = get_labels(img_m, img_n, img_h, img_w, img_c, "../MNIST/train/", "../MNIST/test/", n_classes);

// TRAIN
    printf("TRAIN NETWORK:\n");
    if (training_volume%training_batch!=0) return -1;
    
    network->batch         = batch;
    network->learning_rate = 0.0001;
    network->momentum      = 0.9;
    network->decay         = 0.0001;
    float p1 = network->learning_rate/network->batch;
    float p2 = -network->decay*network->batch;
    float p3 = network->momentum;
    
    printf("number of training images: %d, batch: %d, epoch: %d\n", training_volume, training_batch, training_epoch);
    printf("training config: batch size: %d, learning rate: %f, momentum: %f, decay: %f\n", network->batch, network->learning_rate, network->momentum, network->decay);

    // if there is only one device, then this part is of serial processing
    printf("number of devices:%d\n", num_dev);

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
	    // conv1 update
	    conv_update(network->layers[0]->n, network->layers[0]->biases, network->layers[0]->bias_updates, network->layers[0]->nweights, network->layers[0]->weights, network->layers[0]->weight_updates, p1, p2, p3);
    	    // conv2 update
    	    conv_update(network->layers[2]->n, network->layers[2]->biases, network->layers[2]->bias_updates, network->layers[2]->nweights, network->layers[2]->weights, network->layers[2]->weight_updates, p1, p2, p3);
            // connect1 update
	    connect_update(network->layers[4]->outputs, network->layers[4]->biases, network->layers[4]->bias_updates, network->layers[4]->inputs*network->layers[4]->outputs, network->layers[4]->weights, network->layers[4]->weight_updates, p1, p2, p3);
	    // connect2 update
	    connect_update(network->layers[5]->outputs, network->layers[5]->biases, network->layers[5]->bias_updates, network->layers[5]->inputs*network->layers[5]->outputs, network->layers[5]->weights, network->layers[5]->weight_updates, p1, p2, p3);
	    // connect3 update
	    connect_update(network->layers[6]->outputs, network->layers[6]->biases, network->layers[6]->bias_updates, network->layers[6]->inputs*network->layers[6]->outputs, network->layers[6]->weights, network->layers[6]->weight_updates, p1, p2, p3);
        }

        printf("error = %f\n", network->cost);
        
      } 
    }

// OUTPUT
    // model
    int HWC_conv1_weights = network->layers[0]->n*network->layers[0]->size*network->layers[0]->size*network->layers[0]->c;
    int HWC_conv2_weights = network->layers[2]->n*network->layers[2]->size*network->layers[2]->size*network->layers[2]->c;
    int HWC_connect1_weights = network->layers[4]->inputs*network->layers[4]->outputs;
    int HWC_connect2_weights = network->layers[5]->inputs*network->layers[5]->outputs;
    int HWC_connect3_weights = network->layers[6]->inputs*network->layers[6]->outputs;
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
