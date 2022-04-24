#!/bin/bash

mkdir -p logs_kernel images_kernel
mkdir -p logs_device images_device
mkdir -p logs_matrix images_matrix

perms=(012 021 102 120 201 210)
for p in "${perms[@]}"
do
    echo "RUN: " $p
    ./demo_kernel.out $p 64 64 64 testbench
    ./demo_kernel.out $p 16 32 512 testbench
    ./demo_kernel.out $p 16 512 32 testbench
    ./demo_kernel.out $p 32 16 512 testbench
    ./demo_kernel.out $p 32 512 16 testbench
    ./demo_kernel.out $p 512 16 32 testbench
    ./demo_kernel.out $p 512 32 16 testbench
    ./demo_kernel.out $p 19 32 16 testbench

    # ./demo_device.out $p 64 64 64 testbench
    # ./demo_device.out $p 16 32 512 testbench
    # ./demo_device.out $p 16 512 32 testbench
    # ./demo_device.out $p 32 16 512 testbench
    # ./demo_device.out $p 32 512 16 testbench
    # ./demo_device.out $p 512 16 32 testbench
    # ./demo_device.out $p 512 32 16 testbench
    # ./demo_device.out $p 19 32 16 testbench
done


# ./demo_matrix.out 1024 1024 testbench
# ./demo_matrix.out 64 1024 testbench
# ./demo_matrix.out 1024 64 testbench
