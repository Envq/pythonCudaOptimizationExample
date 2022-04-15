from cuda_accelerations import *

####################################################################
np.set_printoptions(threshold=1000 ,linewidth=1000)
debug_array  = 0
debug_kernel = 0
debug_time   = 1

dim = (1<<14, 1<<14)
# input = np.random.uniform(low=0, high=1, size=dim).astype(np.float32)
input = np.arange(0, np.product(dim), dtype=np.float32).reshape(dim)
# input = np.linspace(0.1, 0.9, np.product(dim), dtype=np.float32).reshape(dim)
printDebug(input, debug_array)


#####################################################################
def test_numpy():
    start = time.time()
    res_numpy2D = np.transpose(input)
    # res_numpy3D = np.transpose(input, (1, 2, 0))
    end = time.time()
    printDebug(f'   msec: {(end-start)*1e3}', debug_time)
    return res_numpy2D


def test_nop():
    k = getKernelTranspose('nop')
    input1D = np.zeros((5,5))
    cuda_transpose2D(k, input1D, input.shape, block=(4, 4), grid=None, debug=debug_kernel)


def test_transpose(kernel_name, blockDim, gridDim=None):
    start = time.time()
    input1D = input.flatten()
    end = time.time()
    time_flat = (end-start)*1e3

    start = time.time()
    k = getKernelTranspose(kernel_name)
    res_cuda1D, kernel_time = cuda_transpose2D(k, input1D, input.shape, block=blockDim, grid=gridDim, debug=debug_kernel)
    end = time.time() 
    time_cuda_transpose = (end-start)*1e3
    printDebug(f'   msec: {kernel_time} [kernel', debug_time)

    start = time.time()
    res_cuda2D = res_cuda1D.reshape(res_numpy2D.shape)
    end = time.time()
    time_reshape= (end-start)*1e3
    # printDebug('Total time: ', time_flat + time_cuda_transpose + time_reshape)
    printDebug(f'   msec: {time_flat + time_reshape} [reshaping]', debug_time)
    return res_cuda2D



#####################################################################
print('\nnumpy:')
res_numpy2D = test_numpy()
# printDebug(res_numpy2D, debug_array)


#####################################################################
printDebug('\npycuda: no operation')
test_nop()


#####################################################################
printDebug('\npycuda: transpose')
res_cuda2D = test_transpose("transpose", blockDim=(32, 32))
printDebug(res_cuda2D, debug_array)
assert np.array_equal(res_numpy2D, res_cuda2D), "wrong result"
printDebug('   result ok')


####################################################################
printDebug('\npycuda: transpose with shared memory')
res_cuda2D = test_transpose("transpose_shm", blockDim=(32, 32))
printDebug(res_cuda2D, debug_array)
assert np.array_equal(res_numpy2D, res_cuda2D), "wrong result"
printDebug('   result ok')


####################################################################
printDebug('\npycuda: transpose with shared memory2')
res_cuda2D = test_transpose("transpose_shm2", blockDim=(32, 8), gridDim=(ceil(dim[0]/32), ceil(dim[1]/32)))
printDebug(res_cuda2D, debug_array)
assert np.array_equal(res_numpy2D, res_cuda2D), "wrong result"
printDebug('   result ok')