import time
import numpy as np
from math import ceil


NUM_REPS  = 100
CROP = ceil(NUM_REPS * 0.1)




dim = (1024, 1024)
print(f'numpy: {dim}')
input = np.arange(np.product(dim), dtype=np.float32).reshape(dim)
timeElapsed = list()
for i in range(NUM_REPS+CROP):
    start = time.time()
    res_numpy = np.transpose(input)
    end = time.time() 
    timeElapsed.append((end-start)*1e3)
print(f'Time: {np.mean(timeElapsed[CROP:])} ms\n')

dim = (10240, 10240)
print(f'numpy: {dim}')
input = np.arange(np.product(dim), dtype=np.float32).reshape(dim)
timeElapsed = list()
for i in range(NUM_REPS+CROP):
    start = time.time()
    res_numpy = np.transpose(input)
    end = time.time() 
    timeElapsed.append((end-start)*1e3)
print(f'Time: {np.mean(timeElapsed[CROP:])} ms\n')

