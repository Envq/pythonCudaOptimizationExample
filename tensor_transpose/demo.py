import numpy as np
import itertools


###############################################################################
dim = (2,3,4)


###############################################################################
def myTranspose3D(input, p):
    iDim = input.shape
    oDim = (iDim[p[0]], iDim[p[1]], iDim[p[2]])
    output = np.zeros(oDim)
    for i in range(iDim[0]):
        for j in range(iDim[1]):
            for k in range(iDim[2]):
                idx = (i, j, k)
                odx = (idx[p[0]], idx[p[1]], idx[p[2]])
                output[odx[0]][odx[1]][odx[2]] = input[idx[0]][idx[1]][idx[2]]
    return output

def run_test3D(tensor3D, perm):
    print("run permutation: ", perm)
    gold = np.transpose(tensor3D, perm)
    result = myTranspose3D(tensor3D, perm)
    if not np.array_equal(gold, result):
        print("Check: FAIL")
        print(gold)
        print(result)
        exit(0)

def myTranspose1D(input, iDim, p):
    oDim = (iDim[p[0]], iDim[p[1]], iDim[p[2]])
    output = np.zeros(input.size)
    for i in range(iDim[0]):
        for j in range(iDim[1]):
            for k in range(iDim[2]):
                idx = (i, j, k)
                odx = (idx[p[0]], idx[p[1]], idx[p[2]])
                iIndex = (idx[0]*iDim[1]*iDim[2])+(idx[1]*iDim[2])+(idx[2])
                oIndex = (odx[0]*oDim[1]*oDim[2])+(odx[1]*oDim[2])+(odx[2])
                output[oIndex] = input[iIndex]
    return output

def run_test1D(tensor3D, perm):
    print("run permutation: ", perm)
    gold = np.transpose(tensor3D, perm).flatten()
    result = myTranspose1D(tensor3D.flatten(), tensor3D.shape, perm)
    if not np.array_equal(gold, result):
        print("Check: FAIL")
        print(gold)
        print(result)
        exit(0)


###############################################################################
tensor3D = np.arange(0, np.product(dim), dtype=np.int32).reshape(dim)
perms = list(itertools.permutations((0,1,2)))
for p in perms:
    run_test3D(tensor3D, p)
print('Check: OK')

for p in perms:
    run_test1D(tensor3D, p)
print('Check: OK')



###############################################################################
print(tensor3D.flatten())
print(np.transpose(tensor3D, (1,2,0)).flatten())
