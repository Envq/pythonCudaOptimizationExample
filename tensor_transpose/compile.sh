#!/bin/bash

nvcc -w -std=c++11 -arch=sm_61 demo.cu -o demo.out
# nvcc -w -std=c++11 -arch=sm_61 testbench.cu -o testbench.out