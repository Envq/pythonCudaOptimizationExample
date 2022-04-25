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
def benchmark_numpy(dim, input, n_run, n_crops):
    times = list()
    for _ in range(n_run+n_crops):
        start = time.time()
        res = np.transpose(input)
        end = time.time()
        time_execute = (end - start) * 1e3
        times.append(time_execute)  # milliseconds
    return res, np.mean(times[n_crops:])
    
def benchmark_cuda(dim, input, n_run, n_crops):
    times_reshaping = list()
    times_kernel = list()
    for _ in range(n_run+n_crops):
        start = time.time()
        input1D = input.flatten()
        end = time.time()
        time_flat = (end-start)*1e3

        res_cuda1D, kernel_time = acc.transpose2D(input1D, input.shape)

        start = time.time()
        res_cuda2D = res_cuda1D.reshape(dim)
        end = time.time()
        time_reshape = (end-start)*1e3

        times_kernel.append(kernel_time)
        times_reshaping.append(time_flat + time_reshape)
    return res_cuda2D, np.mean(times_kernel[n_crops:]), np.mean(times_reshaping[n_crops:])





#####################################################################
print('DIMENSION: ', dim)
NUM_CROP = ceil(NUM_REPS * 0.1)
print('NUM_CROP: ', NUM_CROP)
print('NUM_REPS: ', NUM_REPS)

input = np.arange(0, np.product(dim), dtype=np.float32).reshape(dim)

res_numpy, time_numpy = benchmark_numpy(dim, input, NUM_REPS, NUM_CROP)
res_cuda, time_kernel, time_reshaping = benchmark_cuda(dim, input, NUM_REPS, NUM_CROP)
assert np.array_equal(res_numpy, res_cuda), "wrong result"


#####################################################################
print(f'           numpy time (ms): {time_numpy}')
print(f'   pycuda kernel time (ms): {time_kernel}')
print(f'pycuda reshaping time (ms): {time_reshaping}')




