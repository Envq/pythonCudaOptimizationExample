import time
import build.cuda_accelerations as acc
import numpy as np
from math import ceil


####################################################################
NUM_REPS  = 1
CROP = 0 #ceil(NUM_REPS * 0.1)
def print_debug(obj, debug=False):
    if debug:
        print(obj)


####################################################################
dim = (1024, 1024)
input = np.arange(np.product(dim), dtype=np.float32).reshape(dim)
# input = np.random.uniform(low=0, high=10, size=dim).astype(np.float32)
print_debug("input:")
print_debug(input)
print_debug("")
print(f'num reps: ', NUM_REPS)
print(f'crop: {CROP}\n')



####################################################################
print('numpy:')
timeElapsed = list()
for i in range(NUM_REPS+CROP):
    start = time.time()
    res_numpy = np.transpose(input)
    end = time.time() 
    timeElapsed.append((end-start)*1e3)
print_debug(res_numpy)
print(f'Time: {np.mean(timeElapsed[CROP:])} ms\n')


#####################################################################
print('C++:')
timeElapsed = list()
for i in range(NUM_REPS+CROP):
    start = time.time()
    res_cpp = acc.transpose_cpp(input)
    end = time.time() 
    timeElapsed.append((end-start)*1e3)
    assert np.array_equal(res_numpy, res_cpp), "Check Error"
print_debug(res_cpp)
print(f'Time: {np.mean(timeElapsed[CROP:])} ms\n')


#####################################################################
print('Cuda:')
timeElapsed = list()
for i in range(NUM_REPS+CROP):
    start = time.time()
    res_cuda = acc.transpose_cuda(input)
    end = time.time() 
    timeElapsed.append((end-start)*1e3)
    assert np.array_equal(res_numpy, res_cuda), "Check Error"
print_debug(res_cuda)
print(f'Time: {np.mean(timeElapsed[CROP:])} ms\n')
