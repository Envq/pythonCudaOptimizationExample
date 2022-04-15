import time
import build.cuda_accelerations as acc
import numpy as np


####################################################################
def print_debug(obj, debug=False):
    if debug:
        print(obj)


####################################################################
dim = (1<<14, 1<<14)
input = np.arange(np.product(dim), dtype=np.float32).reshape(dim)
# input = np.random.uniform(low=0, high=10, size=dim).astype(np.float32)
print_debug("input:")
print_debug(input)
print_debug("")


####################################################################
test_file = False
if test_file:
    input = np.loadtxt('../input.txt').reshape((19, 23, 16))
    output = np.loadtxt('../output.txt').reshape((23, 16, 19))

    start = time.time()
    heatmap = np.transpose(input, (1, 2, 0))
    end = time.time()
    print(f'time {(end-start)*1e3} msec')
    print("Check: ", np.array_equal(output, heatmap))
    exit(0)


####################################################################
print('numpy:')
start = time.time()
res_numpy = np.transpose(input)
end = time.time() 
print_debug(res_numpy)
print(f'Time: {(end-start)*1e3} msec\n')


#####################################################################
print('C++:')
start = time.time()
res_cuda = acc.transpose_cpp(input)
end = time.time() 
print_debug(res_cuda)
print(f'Time: {(end-start)*1e3} msec')
assert np.array_equal(res_numpy, res_cuda), "Check Error"
print('Check OK\n')


#####################################################################
print('Cuda:')
start = time.time()
res_cuda = acc.transpose_cuda(input)
end = time.time() 
print_debug(res_cuda)
print(f'Time: {(end-start)*1e3} msec')
assert np.array_equal(res_numpy, res_cuda), "Check Error"
print('Check OK\n')
