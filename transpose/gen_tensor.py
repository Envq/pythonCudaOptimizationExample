import numpy as np

np.set_printoptions(threshold=1000 ,linewidth=1000)


dim = (8,8,8)
tensor1d = np.arange(0, np.product(dim))
tensor3d = np.arange(0, np.product(dim)).reshape(dim)

print(f"tensor1d:\n{tensor1d}\n")
# print(f"tensor3d:\n{tensor3d}\n")


perms = [
    (0,1,2),
    (0,2,1),
    (1,0,2),
    (1,2,0),
    (2,0,1),
    (2,1,0),
]
for p in perms:
    transpose3d = tensor3d.transpose(p)
    transpose1d = transpose3d.flatten()

    print(f"transpose1d {p}:\n{transpose1d}\n")
    # print(f"transpose3d {p}:\n{transpose3d}\n")