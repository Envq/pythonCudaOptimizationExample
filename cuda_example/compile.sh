#!/bin/bash

nvcc -w -std=c++11 -arch=sm_61 demo.cu -Iinclude -o demo.out