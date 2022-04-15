#!/bin/bash


nvcc -w -std=c++11 -arch=sm_61 src/transpose.cu main.cu -Iinclude -o main.out

mkdir -p build
cd build/
cmake ..
make