#!/bin/bash

pip3 install -e . -vvv
# nvcc -w -std=c++11 -arch=sm_61 src/matrix_transpose.cu -Iinclude