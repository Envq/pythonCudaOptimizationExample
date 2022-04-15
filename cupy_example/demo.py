from math import ceil
import time
import numpy as np
import cupy as cp

test_2d = 0
test_3d = 1

if test_2d:
    ####################################################################
    print('TRANSPOSE 2D')
    dim = (20000, 20000)
    print('dim: ', dim)

    ####################################################################
    print('\nnumpy:')
    start = time.time()
    cpu_array = np.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    end = time.time()
    time_allocate = end - start
    start = time.time()
    cpu_res = np.transpose(cpu_array)
    end = time.time()
    time_execute = end - start
    print(f'   msec allocate: {time_allocate * 1e3}')
    print(f'   msec execute:  {time_execute * 1e3}')
    print(f'   msec total:    {(time_allocate+time_execute) * 1e3}')

    ####################################################################
    print('\ncupy:')
    start = time.time()
    gpu_array = cp.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    end = time.time()
    time_allocate = end - start
    start = time.time()
    gpu_res = cp.transpose(gpu_array)
    end = time.time()
    time_execute = end - start
    print(f'   msec allocate: {time_allocate * 1e3}')
    print(f'   msec execute:  {time_execute * 1e3}')
    print(f'   msec total:    {(time_allocate+time_execute) * 1e3}')



if test_3d:
    ####################################################################
    print('TRANSPOSE 3D')
    dim = (1000, 1000, 1000)
    print('dim: ', dim)

    ####################################################################
    print('\nnumpy:')
    start = time.time()
    cpu_array = np.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    end = time.time()
    time_allocate = end - start
    start = time.time()
    cpu_res = np.transpose(cpu_array, (1, 2, 0))
    end = time.time()
    time_execute = end - start
    print(f'   msec allocate: {time_allocate * 1e3}')
    print(f'   msec execute:  {time_execute * 1e3}')
    print(f'   msec total:    {(time_allocate+time_execute) * 1e3}')

    ####################################################################
    print('\ncupy:')
    start = time.time()
    gpu_array = cp.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    end = time.time()
    time_allocate = end - start
    start = time.time()
    gpu_res = cp.transpose(gpu_array, (1, 2, 0))
    end = time.time()
    time_execute = end - start
    print(f'   msec allocate: {time_allocate * 1e3}')
    print(f'   msec execute:  {time_execute * 1e3}')
    print(f'   msec total:    {(time_allocate+time_execute) * 1e3}')