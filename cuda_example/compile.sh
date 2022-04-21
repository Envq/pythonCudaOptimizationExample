#!/bin/bash

nvcc -w -std=c++11 -arch=sm_61 demo.cu -o demo.out
nvcc -w -std=c++11 -arch=sm_61 demo_transpose.cu -o demo_transpose.out
nvcc -w -std=c++11 -arch=sm_61 demo_kernel.cu -o demo_kernel.out
nvcc -w -std=c++11 -arch=sm_61 demo_overlap.cu -o demo_overlap.out
