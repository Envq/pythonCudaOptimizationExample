#!/bin/bash


# nvcc -w -std=c++11 -arch=sm_61 src/matrix_transpose.cu -Iinclude


cd build/
cmake ..
make