import numpy as np
import itertools

dim = (2,3,4)

input = np.arange(0, np.product(dim), dtype=np.int32).reshape(dim)
perms = list(itertools.permutations((0,1,2)))


dim_str = str(dim)[1:-1] + '\n'
input_str = str(input.flatten().tolist())[1:-1] + '\n'

with open("testbench.txt", "w") as file:
    for p in perms:
        file.write(dim_str)
        file.write(input_str)
        gold = np.transpose(input, p)
        p_str = str(p)[1:-1] + '\n'
        file.write(p_str)
        gold_str = str(gold.flatten().tolist())[1:-1] + '\n'
        file.write(gold_str)