# CNN_OpenMP
Prepared for WACCPD'22 which is a workshop of SC'22

#### Environment
```
CPU:         Dual-CPU AMD EPYC Milan 7413 24-Core/48-Threads, 2.55GHz
Memory:      512GB 3200MHz EC REG Memory
GPU:         4 NVIDIA A100 Ampere 40 GB GPU - PCIe 4.0
OS:          Ubuntu 20.04
Compilers:   Clang/LLVM 14.0 with OpenMP GPU offloading support
CUDAToolkit: 11.2
cuDNN:       8.1.1
OpenCV:      4.3.0
```

#### Compile and Run
```
make
export LD_LIBRARY_PATH=/opt/llvm/llvm-14.x-install/lib:$LD_LIBRARY_PATH
make run
```

#### Reproductivity
*Step 0: import data*  
`MNIST` dataset needs to be imported to this repo and the file struture looks like:
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
The usage for executable is: `./main <training images> <testing images> <batches> <epochs> <devices>`.  
Among all of the test cases, we use full MNIST dataset and 100 epochs, then, for instance, for experiment of OpenMP GPU version with batch size of 1k, we type:
```
./main 60000 10000 60 100 1
```
For rest of the tests, with batch sizes of 1k, 2k, 4k, 10k, 15k, 30k, 60k, we change the `<batches>` to be 60, 30, 15, 6, 4, 2, and 1.

*Step 3: run nvprof*  
```
nvprof ./main 60000 10000 60 100 1
```
