#!/bin/bash

nvcc -w -std=c++11 -arch=sm_61 demo_kernel.cu -Iinclude -o demo_kernel.out
# nvcc -w -std=c++11 -arch=sm_61 demo_overlap.cu -Iinclude -o demo_overlap.out