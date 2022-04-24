import numpy as np
from math import ceil

# SETTINGS
print_shmem = 0

dim = (8, 8, 8+1)
# dim = (8, 8+4, 8+1)

# CREATE TENSOR
np.set_printoptions(threshold=2000 ,linewidth=2000)
banks = 32
pattern = np.arange(0, banks).tolist()
size = np.product(dim)
d = size
tensor = list()
while True:
    if (d - banks < 0):
        tensor += pattern[:d]
        break
    tensor += pattern
    d -= banks
tensor = np.array(tensor).reshape(dim)


# CREATE SHARED MEMORY
perms = [
    (0,1,2),
    (0,2,1),
    (1,0,2),
    (1,2,0),
    (2,0,1),
    (2,1,0),
]
for p in perms:
    rows = ceil(size / banks)
    padd = (rows * banks) - size
    shmem = np.transpose(tensor, p).flatten().tolist()
    shmem += [-1] * padd
    shmem = np.array(shmem).reshape((rows, banks))
    conflicts = list()
    for i in range(rows):
        conflicts.append(banks - len(set(shmem[i,:])))
    # assert len(set(conflicts)) == 1, "Variable conflicts"
    print(f'perm: {p}:')
    print(f'banks conflicts avg: {np.mean(conflicts)}')
    if print_shmem:
        print(shmem)
    print()
print("---------------")