#!/bin/bash

 nvcc --cubin -w -std=c++11 -arch=sm_61 transpose.cu 