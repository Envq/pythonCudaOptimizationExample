#!/bin/bash

names=(demo_kernel demo_device demo_matrix)
for name in "${names[@]}"
do
    echo "COMPILE: " $name
    nvcc -w -std=c++11 -arch=sm_61 $name.cu -o $name.out
done
