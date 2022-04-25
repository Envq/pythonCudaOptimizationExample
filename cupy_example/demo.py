from math import ceil
import time
import numpy as np
import cupy as cp


####################################################################
NUM_REPS  = 100
DIMENSION = 1024
# DIMENSION = round(0.5*np.product((184, 128, 3, 1)))
# DIMENSION = round(0.5*np.product((19, 23, 16)))
# DIMENSION = round(0.5*np.product((38, 23, 16)))
dim = (DIMENSION, DIMENSION)


####################################################################
def benchmark_raw(dim, n_run, n_crops):
    mod = cp.RawModule(path="transpose.cubin")
    transposeKernel = mod.get_function("transpose")
    blockDim  = (32, 32, 1)
    gridDim = (ceil(dim[1]/blockDim[0]), ceil(dim[0]/blockDim[1]), 1)
    x = cp.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    y = cp.zeros(dim, dtype=cp.float32)
    times = list()
    for _ in range(n_run+n_crops):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        transposeKernel(gridDim, blockDim, (x, y, dim[1], dim[0]))
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # milliseconds
    return y, np.mean(times[n_crops:])

def benchmark_lib(dim, n_run, n_crops):
    x = cp.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    times = list()
    for _ in range(n_run+n_crops):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        res = cp.transpose(x)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # milliseconds
    return res, np.mean(times[n_crops:])

def benchmark_numpy(dim, n_run, n_crops):
    x = np.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    times = list()
    for _ in range(n_run+n_crops):
        start = time.time()
        res = np.transpose(x)
        end = time.time()
        time_execute = (end - start) * 1e3
        times.append(time_execute)  # milliseconds
    return res, np.mean(times[n_crops:])

def benchmark_lib2(dim, n_run, n_crops):
    x = cp.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
    times = list()
    for _ in range(n_run+n_crops):
        start = time.time()
        res = cp.transpose(x)
        end = time.time()
        time_execute = (end - start) * 1e3
        times.append(time_execute)  # milliseconds
    return res, np.mean(times[n_crops:])


#####################################################################
print('DIMENSION: ', dim)
NUM_CROP = ceil(NUM_REPS * 0.1)
print('NUM_CROP: ', NUM_CROP)
print('NUM_REPS: ', NUM_REPS)

print('\nTest numpy.transpose')
res_cpu, time_cpu = benchmark_numpy(dim, NUM_REPS, NUM_CROP)
print("ms: ", time_cpu)

print('\nTest cupy.transpose with cudaEvent')
res_lib, time_lib = benchmark_lib(dim, NUM_REPS, NUM_CROP)
print("ms: ", time_lib)

print('\nTest cupy.transpose with time')
res_lib2, time_lib2 = benchmark_lib2(dim, NUM_REPS, NUM_CROP)
print("ms: ", time_lib2)

print('\nTest custom kernel')
res_raw, time_raw = benchmark_raw(dim, NUM_REPS, NUM_CROP)
print("ms: ", time_raw)

assert(np.array_equal(res_cpu, cp.asnumpy(res_raw)))
assert(cp.array_equal(res_lib, res_raw))
print('\nCorrectness check: OK')

print('\nSpeedup:')
print(f'numpy / cupy time:        {round(time_cpu/time_lib2, 3)}x')
print(f'numpy / cupy cudaEvent:   {round(time_cpu/time_lib,  3)}x')
print(f'numpy / custom:           {round(time_cpu/time_raw,  3)}x')
print(f'cupy  / custom:           {round(time_lib/time_raw,  3)}x')