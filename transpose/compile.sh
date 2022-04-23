#!/bin/bash

# nvcc -w -std=c++11 -arch=sm_61 demo_kernel.cu -o demo_kernel.out
# nvcc -w -std=c++11 -arch=sm_61 demo_device.cu -o demo_device.out
nvcc -w -std=c++11 -arch=sm_61 demo_matrix.cu -o demo_matrix.out