import cuda_accelerations as acc
import numpy as np
import time
from math import ceil 


####################################################################
np.set_printoptions(threshold=1000 ,linewidth=1000)
NUM_REPS  = 100
DIMENSION = 1024
# DIMENSION = round(0.5*np.product((184, 128, 3, 1)))
# DIMENSION = round(0.5*np.product((19, 23, 16)))
# DIMENSION = round(0.5*np.product((38, 23, 16)))
dim = (DIMENSION, DIMENSION)
# input = np.random.uniform(low=0, high=1, size=dim).astype(np.float32)
# input = np.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
# input = np.linspace(0.1, 0.9, np.product(dim), dtype=np.float32).reshape(dim)


#####################################################################
def run_test():
    # -- Generate Input
    input = np.random.uniform(low=0, high=1, size=dim).astype(np.float32)

    # -- Numpy Part
    start = time.time()
    res_numpy2D = np.transpose(input)
    end = time.time()
    numpy_time = (end-start)*1e3

    # -- Cuda Part
    start = time.time()
    input1D = input.flatten()
    end = time.time()
    time_flat = (end-start)*1e3

    res_cuda1D, kernel_time = acc.transpose2D(input1D, input.shape)

    start = time.time()
    res_cuda2D = res_cuda1D.reshape(res_numpy2D.shape)
    end = time.time()
    time_reshape = (end-start)*1e3

    # -- Post-process
    assert np.array_equal(res_numpy2D, res_cuda2D), "wrong result"
    # print("numpy\n", res_numpy2D)
    # print("pycuda\n", res_cuda2D)
    reshaping_time = time_flat + time_reshape
    return numpy_time, kernel_time, reshaping_time



#####################################################################
print('DIMENSION: ', dim)
CROP = ceil(NUM_REPS * 0.1)
print('CROP:     ', CROP)
print('NUM_REPS: ', NUM_REPS)

times_numpy   = list()
times_kernels = list()
times_reshape = list()
for i in range(NUM_REPS+CROP):
    numpy_time, kernel_time, reshaping_time = run_test()
    times_numpy.append(numpy_time)
    times_kernels.append(kernel_time)
    times_reshape.append(reshaping_time)


#####################################################################
print(f'           numpy time (ms): {np.mean(times_numpy[CROP:])}')
print(f'   pycuda kernel time (ms): {np.mean(times_kernels[CROP:])}')
print(f'pycuda reshaping time (ms): {np.mean(times_reshape[CROP:])}')