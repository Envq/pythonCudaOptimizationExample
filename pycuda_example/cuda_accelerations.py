from math import ceil
import time
import numpy as np

# --- PyCUDA initialization
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


def transpose2D(h_input, shape):
    # --- Get kernel
    mod = pycuda.driver.module_from_file("transpose.cubin")
    kernel = mod.get_function("transpose")

    # --- Get problem dimension
    M = np.int32(shape[0]) #rows
    N = np.int32(shape[1]) #cols

    # --- Allocate GPU device memory
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_input.nbytes)

    # --- Memcopy from host to device
    cuda.memcpy_htod(d_input, h_input)

    blockDim  = (32, 32, 1)
    gridDim = (ceil(N/blockDim[0]), ceil(M/blockDim[1]), 1)

    # --- Execute kernel
    start = cuda.Event()
    end   = cuda.Event()
    start.record()
    kernel(d_input, d_output, M, N, block = blockDim, grid = gridDim)
    end.record() 
    end.synchronize()
    msec = start.time_till(end)

    # start = time.time()
    # kernel(d_input, d_output, M, N, block = blockDim, grid = gridDim)
    # cuda.Context.synchronize()
    # end = time.time()
    # msec = (end-start)*1e3

    # --- Copy results from device to host
    h_output = np.empty_like(h_input)
    cuda.memcpy_dtoh(h_output, d_output)

    # --- Flush context printf buffer
    cuda.Context.synchronize()

    return h_output, msec