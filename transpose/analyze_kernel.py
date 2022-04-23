import matplotlib.pyplot as plt
import numpy as np
import os


logs = sorted(os.listdir("logs_kernel"))
while len(logs) > 0:
    # Get all permutations of the same dimension
    target = logs[0].split("_")[0]
    target_list = list()
    for i in range(len(logs)-1, -1, -1):
        if logs[i].split("_")[0] == target:
            target_list.append(logs[i])
            del logs[i]

    # Process bandwith
    plt.figure(figsize=(10,5))
    counter = 0
    bandwidth_max = 0
    for log in target_list:
        print(f'Reading {log} ...')
        perm = log[-7:-4]
        dim  = log[:-8]
        dims = dim.split("x")
        name      = list()
        check     = list()
        bandwidth = list()
        speedup   = list()
        with open("logs_kernel/"+log, "r") as file:
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
        col = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
        offset = len(name) * inc
        pos = np.arange(counter, counter+offset, inc)
        for i in range(len(name)):
            plt.bar(pos[i], bandwidth[i], color=col[i])
        plt.yticks(np.arange(50,650,50)) 
        # plt.yticks(np.arange(0,150,50)) 
        plt.xticks([])  
        plt.text(counter, -30, perm, fontsize=11)
        plt.ylabel('Bandwidth (GB/s)')
        plt.xlabel('permutation', labelpad=20.0)
        counter = counter + offset + inc
        bandwidth_max = max(bandwidth_max,max(bandwidth))

    plt.title(f"Dimension: {dim}")
    plt.legend(labels=name)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"images_kernel/{dim}.png",dpi=100)
    print(f'bandwidth_max: {bandwidth_max}\n')
# plt.show()