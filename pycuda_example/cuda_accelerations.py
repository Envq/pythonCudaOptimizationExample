from math import ceil
import time
import numpy as np

# --- PyCUDA initialization
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


def printDebug(msg, debug=True):
    if debug:
        print(msg)


def cuda_transpose2D(kernel, h_input, shape, block, grid, debug):
    # --- Get problem dimension
    M = np.int32(shape[0]) #rows
    N = np.int32(shape[1]) #cols
    printDebug(f'M: {M}', debug)
    printDebug(f'N: {N}', debug)

    # --- Allocate GPU device memory
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_input.nbytes)

    # --- Memcopy from host to device
    cuda.memcpy_htod(d_input, h_input)

    blockDim  = (block[0], block[1], 1)
    if grid != None:
        gridDim = (grid[0], grid[1], 1)
    else:
        gridDim = (ceil(N/blockDim[0]), ceil(M/blockDim[1]), 1)
    printDebug(f'blockDim: {blockDim}', debug)
    printDebug(f'gridDim:  {gridDim}', debug)

    # --- Execute kernel
    start = cuda.Event()
    end   = cuda.Event()
    start.record()
    kernel(d_input, d_output, M, N, block = blockDim, grid = gridDim)
    end.record() 
    end.synchronize()
    msec = start.time_till(end)

    # --- Copy results from device to host
    h_output = np.empty_like(h_input)
    cuda.memcpy_dtoh(h_output, d_output)

    # --- Flush context printf buffer
    cuda.Context.synchronize()

    return h_output, msec


def getKernelTranspose(name):
    mod = pycuda.driver.module_from_file("transpose.cubin")
    return mod.get_function(name)