from math import ceil
import time
import numpy as np
import cupy as cp

test = "2d"
# test = "3d"

if test == "2d":
    ####################################################################
    print('TRANSPOSE 2D')
    dim = (20000, 20000)
    print('dim: ', dim)

    ####################################################################
    print('\nnumpy:')
    start = time.time()
    cpu_array = np.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    end = time.time()
    time_allocate_cpu = (end - start) * 1e3
    start = time.time()
    cpu_res = np.transpose(cpu_array)
    end = time.time()
    time_execute_cpu = (end - start) * 1e3
    print(f'   msec allocate: {time_allocate_cpu}')
    print(f'   msec execute:  {time_execute_cpu}')
    print(f'   msec total:    {time_allocate_cpu+time_execute_cpu}')

    ####################################################################
    print('\ncupy:')
    start = time.time()
    gpu_array = cp.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    end = time.time()
    time_allocate_gpu = (end - start) * 1e3
    start = time.time()
    gpu_res = cp.transpose(gpu_array)
    end = time.time()
    time_execute_gpu = (end - start) * 1e3
    print(f'   msec allocate: {time_allocate_gpu}')
    print(f'   msec execute:  {time_execute_gpu}')
    print(f'   msec total:    {time_allocate_gpu+time_execute_gpu}')

    ####################################################################
    print('\nSpeedup: ')
    print(f'   {time_allocate_cpu/time_allocate_gpu}x')
    print(f'   {time_execute_cpu/time_execute_gpu}x')
    print(f'   {(time_allocate_cpu+time_execute_cpu)/(time_allocate_gpu+time_execute_gpu)}x')



if test == "3d":
    ####################################################################
    print('TRANSPOSE 3D')
    dim = (1000, 1000, 1000)
    print('dim: ', dim)

    ####################################################################
    print('\nnumpy:')
    start = time.time()
    cpu_array = np.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    end = time.time()
    time_allocate_cpu = (end - start) * 1e3
    start = time.time()
    cpu_res = np.transpose(cpu_array, (1, 2, 0))
    end = time.time()
    time_execute_cpu = (end - start) * 1e3
    print(f'   msec allocate: {time_allocate_cpu}')
    print(f'   msec execute:  {time_execute_cpu}')
    print(f'   msec total:    {time_allocate_cpu+time_execute_cpu}')

    ####################################################################
    print('\ncupy:')
    start = time.time()
    gpu_array = cp.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    end = time.time()
    time_allocate_gpu = (end - start) * 1e3
    start = time.time()
    gpu_res = cp.transpose(gpu_array, (1, 2, 0))
    end = time.time()
    time_execute_gpu = (end - start) * 1e3
    print(f'   msec allocate: {time_allocate_gpu}')
    print(f'   msec execute:  {time_execute_gpu}')
    print(f'   msec total:    {time_allocate_gpu+time_execute_gpu}')

    ####################################################################
    print('\nSpeedup: ')
    print(f'   {time_allocate_cpu/time_allocate_gpu}x')
    print(f'   {time_execute_cpu/time_execute_gpu}x')
    print(f'   {(time_allocate_cpu+time_execute_cpu)/(time_allocate_gpu+time_execute_gpu)}x')