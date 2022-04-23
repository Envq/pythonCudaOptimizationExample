import matplotlib.pyplot as plt
import numpy as np
import os


logs = sorted(os.listdir("logs_matrix"))
for log in logs:
    plt.figure(figsize=(15,5))
    print(f'Reading {log} ...')
    dim  = log[:-4]
    dims = dim.split("x")
    name      = list()
    check     = list()
    bandwidth = list()
    speedup   = list()
    with open("logs_matrix/"+log, "r") as file:
        for i, info in enumerate(file):
            info = info[:-1] # delete \n
            if i % 4 == 0:
                name.append(info)
            elif i % 4 == 1:
                check.append(bool(info))
            elif i % 4 == 2:
                bandwidth.append(float(info))
            elif i % 4 == 3:
                speedup.append(float(info))
    assert all(check), "Check is False"

    inc = 0.9
    offset = len(name) * inc
    pos = np.arange(0, offset, inc)
    for i in range(len(name)):
        plt.bar(pos[i], bandwidth[i])
        if "2D" in name[i]:
            plt.text(pos[i], -20, "2D", fontsize=11)
        else:
            plt.text(pos[i], -20, "3D", fontsize=11)
    plt.yticks(np.arange(0,325,25)) 
    plt.xticks([])  
    plt.ylabel('Bandwidth (GB/s)')
    plt.xlabel('Test', labelpad=20.0)

    plt.title(f"Dimension: {dim}")
    plt.legend(labels=name, bbox_to_anchor=(1., 1.), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"images_matrix/{dim}.png",dpi=100)
# plt.show()