import time
import build.cuda_accelerations as acc
import numpy as np


matrix = np.array([
    [0,0,0,0],
    [1,2,0,0],
    [3,4,5,0],
    [6,7,8,9],
], dtype=np.float64)


def transpose_python(matrix):
    size = matrix.shape[0]
    res = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            res[i][j] = matrix[j][i]
    return res

def print_debug(obj, debug=False):
    if debug:
        print('Answer:\n', obj)


####################################################################
print('python:')
start_time = time.perf_counter_ns()
res_python = transpose_python(matrix)
print_debug(res_python)
print('Time:', (time.perf_counter_ns() - start_time) / 1e6, 'msec\n')


####################################################################
print('numpy:')
start_time = time.perf_counter_ns()
res_numpy = np.transpose(matrix)
print_debug(res_numpy)
print('Time:', (time.perf_counter_ns() - start_time) / 1e6, 'msec\n')


#####################################################################
print('C++:')
start_time = time.perf_counter_ns()
res_cpp = acc.transpose_cpp(matrix)
print_debug(res_cpp)
print('Time:', (time.perf_counter_ns() - start_time) / 1e6, 'msec\n')


#####################################################################
print('Cuda:')
start_time = time.perf_counter_ns()
res_cuda = acc.transpose_cuda(matrix)
print_debug(res_cuda)
print('Time:', (time.perf_counter_ns() - start_time) / 1e6, 'msec\n')


#####################################################################
assert np.array_equal(res_python, res_cpp), "Check Error"
assert np.array_equal(res_numpy, res_cpp), "Check Error"
assert np.array_equal(res_cpp, res_cuda), "Check Error"
print('Check OK')
