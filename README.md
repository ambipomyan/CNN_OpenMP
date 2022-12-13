# CNN_OpenMP
Prepared for PMAM'23 which is a workshop of PPoPP'23

#### Environment
```
CPU:         Dual-CPU AMD EPYC Milan 7413 24-Core/48-Threads, 2.55GHz
Memory:      512GB 3200MHz EC REG Memory
GPU:         4 NVIDIA A100 Ampere 40 GB GPU - PCIe 4.0
OS:          Ubuntu 20.04
Compilers:   Clang/LLVM 15.0 with OpenMP GPU offloading support
CUDAToolkit: 11.2
cuDNN:       8.1.1
OpenCV:      4.3.0
```

#### Compile and Run
```
make
export LD_LIBRARY_PATH=/home/kyan2/llvm-15.x-install/lib:$LD_LIBRARY_PATH
make run
```

A sample run could be like:
```
- run_classifier -
LOAD NETWORK:
number of layers: 8, number of classes: 10
conv1:    h: 28, w: 28, c: 1, number of filter: 32, filter size: 3, stride: 1, padding: 1, activation: RELU
pool1:    h: 28, w: 28, c: 32, filter size: 2, stride: 2, padding: 1
conv2:    h: 15, w: 15, c: 32, number of filter: 64, filter size: 3, stride: 1, padding: 1, activation: RELU
pool2:    h: 15, w: 15, c: 64, filter size: 2, stride: 2, padding: 1
connect1: h: 1, w: 1, c: 4096, output size: 1024, activation: RELU
connect2: h: 1, w: 1, c: 1024, output size: 84, activation: RELU
connect3: h: 1, w: 1, c: 84, output size: 10, activation: - 
softmax:  number of classes: 10
LOAD DATA:
training datasets:   MNIST/train, 60000 images
predicting datasets: MNIST/test,  10000 images
io_read: 1257.917725
TRAIN NETWORK:
number of training images: 60000, batch: 60, epoch: 100
training config: batch size: 1000, learning rate: 0.000100, momentum: 0.900000, decay: 0.000100
number of devices:1
conv-forward: 2.644043
conv1    forward epoch# 0 batch# 0 device# 0: 258.643555
maxpool1 forward epoch# 0 batch# 0 device# 0: 34.866211
conv-forward: 13.243896
conv2    forward epoch# 0 batch# 0 device# 0: 46.739502
maxpool2 forward epoch# 0 batch# 0 device# 0: 56.459961
connect1 forward epoch# 0 batch# 0 device# 0: 43.218018
connect2 forward epoch# 0 batch# 0 device# 0: 1.844482
connect3 forward epoch# 0 batch# 0 device# 0: 0.193604
softmax  forward epoch# 0 batch# 0 device# 0: 0.130127
forward epoch# 0 batch# 0 device# 0: 442.171875
softmax  backward epoch# 0 batch# 0 device# 0: 0.072510
connect3 backward epoch# 0 batch# 0 device# 0: 0.323730
connect2 backward epoch# 0 batch# 0 device# 0: 2.673340
connect1 backward epoch# 0 batch# 0 device# 0: 45.306152
pool2    backward epoch# 0 batch# 0 device# 0: 12.332520
conv_bwd_filt: 8.428711
conv_bwd_data: 13.266602
conv2    backward epoch# 0 batch# 0 device# 0: 38.974609
pool1    backward epoch# 0 batch# 0 device# 0: 22.484619
conv_bwd_filt: 2.991699
conv_bwd_data: 1.203125
conv1    backward epoch# 0 batch# 0 device# 0: 25.642578
backward epoch# 0 batch# 0 device# 0: 147.869873
conv1     update epoch# 0 batch# 0 device# 0: 0.001953
conv2     update epoch# 0 batch# 0 device# 0: 0.008057
connect1  update epoch# 0 batch# 0 device# 0: 3.273438
connect2  update epoch# 0 batch# 0 device# 0: 0.042969
connect3  update epoch# 0 batch# 0 device# 0: 0.000732
update epoch# 0 batch# 0 device# 0: 3.343750
total_batch epoch# 0 batch# 0 device# 0: 593.419189
...
```

#### Reproductivity
*Step 0: import data*  
Dataset needs to be imported to the same path of this repo and the file struture looks like:
```
`-- MNIST
    |-- train
    |   |-- 0
    |   |   |-- aaaa.jpg
    |   |   |-- bbbb.jpg
    |   |   |   ...
    |   |   `-- zzzz.jpg
    ...
    `-- test
        |-- 0
        |   |-- AAAA.jpg
        |   |-- BBBB.jpg
        |   |   ...
        |   `-- ZZZZ.jpg
        ...
```

*Step 1: create executable*  
For OpenMP CPU, OpenMP GPU and cuDNN implemented CNN, we type `make omp-cpu`, `make`, and `make cudnn` to generate `main-omp-cpu`, `main` and `main-cudnn`, accordingly.  

*Step 2: run executable*  
The usage for executable is: `./main <num_training_images> <num_test_images> <num_batches_per_epoch> <num_epoches> <num_devices>`.
For experiment of OpenMP GPU version with 100 epoch, 8 batches per epoch and 1 GPU, we type:
```
./main 60000 10000 8 100 1
```
For the evaluations, the `<num_batches_per_epoch>` can be 8, 16, 40, 60 and 80; the `<num_devices>` can be 1, 2 and 4.

*Step 3: run nvprof*   
```
nvprof ./main 60000 10000 8 100 1
```
