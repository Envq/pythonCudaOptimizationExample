import numpy as np
from math import ceil

def check_conflicts(dims, padding, perms, print_shmem, print_conflics):
    banks = 32
    dim = np.array(dims) + np.array(padding)
    # CREATE TENSOR
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
    conflicts_sum = 0
    for p in perms:
        rows = ceil(size / banks)
        padd = (rows * banks) - size
        shmem = np.transpose(tensor, p).flatten().tolist()
        shmem += [-1] * padd
        shmem = np.array(shmem).reshape((rows, banks))
        conflicts = list()
        for i in range(rows):
            conflicts.append(banks - len(set(shmem[i,:])))
        conflicts_mean = np.mean(conflicts)
        conflicts_sum += conflicts_mean
        if print_conflics:
            print(f'conflics of {p}: {conflicts_mean}')
        if print_shmem:
            print(shmem)
    if print_conflics:
        print(f"conflics sum: {conflicts_sum}")


def get_opt_padding(dims, pad_max, perm):
    res = list()
    for x in range(pad_max):
        for y in range(pad_max):
            for z in range(pad_max):
                banks = 32
                dim = np.array(dims) + np.array([z,y,x])
                # CREATE TENSOR
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
                rows = ceil(size / banks)
                padd = (rows * banks) - size
                shmem = np.transpose(tensor, perm).flatten().tolist()
                shmem += [-1] * padd
                shmem = np.array(shmem).reshape((rows, banks))
                conflicts = list()
                for i in range(rows):
                    conflicts.append(banks - len(set(shmem[i,:])))
                # CHECK
                if np.mean(conflicts) == 0.0:
                    # res.append((z,y,x))
                    res.append(dim)
    # FIND BEST
    sum_res = list()
    for e in res:
        sum_res.append(sum(e))
    index = np.argmin(sum_res)
    return res[index].tolist()



if __name__ == '__main__':
    np.set_printoptions(threshold=2000 ,linewidth=2000)
    dims = (8,8,8)
    perms = [
        (0,1,2),
        (0,2,1),
        (1,0,2),
        (1,2,0),
        (2,0,1),
        (2,1,0),
    ]
    for p in perms:
        res = get_opt_padding(dims, 10, p)
        print(f'perm {p} -> tile {res}')
    """
    perm (0, 1, 2) -> padding (0, 0, 0)
    perm (0, 2, 1) -> padding (0, 0, 4)
    perm (1, 0, 2) -> padding (0, 1, 0)
    perm (1, 2, 0) -> padding (0, 2, 2)
    perm (2, 0, 1) -> padding (0, 0, 1)
    perm (2, 1, 0) -> padding (0, 4, 1)

    perm (0, 1, 2) -> tile [8, 8, 8]
    perm (0, 2, 1) -> tile [8, 8, 12]
    perm (1, 0, 2) -> tile [8, 9, 8]
    perm (1, 2, 0) -> tile [8, 10, 10]
    perm (2, 0, 1) -> tile [8, 8, 9]
    perm (2, 1, 0) -> tile [8, 12, 9]

                012, 021, 102, 120, 201, 210
    (0,0,1)  ->  x                   x
    """


    print_shmem    = 0
    print_conflics = 1
    padd = (0,0,1)
    dims = (8,8,8)
    check_conflicts(dims, padd, perms, print_shmem, print_conflics)
    """
    PADDING = (0, 0, 0) =>
    conflics of (0, 1, 2): 0.0
    conflics of (0, 2, 1): 16.0
    conflics of (1, 0, 2): 24.0
    conflics of (1, 2, 0): 28.0
    conflics of (2, 0, 1): 28.0
    conflics of (2, 1, 0): 28.0

    PADDING = (0, 0, 1) =>
    conflics of (0, 1, 2): 0.0
    conflics of (0, 2, 1): 5.333333333333333
    conflics of (1, 0, 2): 5.111111111111111
    conflics of (1, 2, 0): 16.0
    conflics of (2, 0, 1): 0.0
    conflics of (2, 1, 0): 16.0
    """