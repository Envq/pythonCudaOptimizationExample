import time
import numpy as np
from math import ceil


NUM_REPS  = 100
NUM_CROP = ceil(NUM_REPS * 0.1)


def benchmark(dim, input, n_reps, n_crop):
    timeElapsed = list()
    for _ in range(n_reps+n_crop):
        start = time.time()
        res = np.transpose(input)
        end = time.time() 
        timeElapsed.append((end-start)*1e3)
    return np.mean(timeElapsed[n_crop:])


# TEST 1
dim = (100, 100)
input = np.arange(np.product(dim), dtype=np.float32).reshape(dim)
t = benchmark(dim, input, NUM_REPS, NUM_CROP)
print(f'{dim}\n -> {t} ms\n')

# TEST 2
dim = (1000, 1000)
input = np.arange(np.product(dim), dtype=np.float32).reshape(dim)
t = benchmark(dim, input, NUM_REPS, NUM_CROP)
print(f'{dim}\n -> {t} ms\n')

# TEST 3
dim = (10000, 10000)
input = np.arange(np.product(dim), dtype=np.float32).reshape(dim)
t = benchmark(dim, input, NUM_REPS, NUM_CROP)
print(f'{dim}\n -> {t} ms\n')

# TEST 4
dim = (100000, 100000)
input = np.arange(np.product(dim), dtype=np.float32).reshape(dim)
t = benchmark(dim, input, NUM_REPS, NUM_CROP)
print(f'{dim}\n -> {t} ms\n')
