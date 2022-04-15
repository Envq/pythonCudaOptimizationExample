from math import ceil
import time
import numpy as np
import cupy as cp


####################################################################
dim = (1000, 1000, 1000)

####################################################################
print('\nnumpy:')
start = time.time()
cpu_array = np.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
cpu_res = np.transpose(cpu_array, (1, 2, 0))
end = time.time()
print(f'   msec: {(end-start)*1e3}')


####################################################################
print('\ncupy:')
start = time.time()
gpu_array = cp.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
gpu_res = cp.transpose(gpu_array, (1, 2, 0))
end = time.time()
print(f'   msec: {(end-start)*1e3}')
