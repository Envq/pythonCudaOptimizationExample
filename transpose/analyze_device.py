import matplotlib.pyplot as plt
import numpy as np
import os


logs = sorted(os.listdir("logs_device"))
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
    speedup_max = 0
    for log in target_list:
        print(f'Reading {log} ...')
        perm = log[-7:-4]
        dim  = log[:-8]
        dims = dim.split("x")
        name      = list()
        check     = list()
        speedup   = list()
        with open("logs_device/"+log, "r") as file:
            for i, info in enumerate(file):
                info = info[:-1] # delete \n
                if i % 3 == 0:
                    name.append(info)
                elif i % 3 == 1:
                    check.append(bool(info))
                elif i % 3 == 2:
                    speedup.append(float(info))
        assert all(check), "Check is False"

        inc = 0.9
        col = ['tab:green','tab:red','tab:purple']
        offset = len(name) * inc
        pos = np.arange(counter, counter+offset, inc)
        for i in range(len(name)):
            plt.bar(pos[i], speedup[i], color=col[i])
        # plt.yticks(np.arange(0,22,2))
        plt.yticks(np.arange(0,5,1)) 
        plt.xticks([])  
        plt.text(counter, -2, perm, fontsize=11)
        plt.ylabel('Speedup')
        plt.xlabel('permutation', labelpad=40.0)
        counter = counter + offset + inc
        speedup_max = max(speedup_max,max(speedup))

    plt.title(f"Dimension: {dim}")
    plt.legend(labels=name)
    plt.grid()
    plt.tight_layout()
    # plt.savefig(f"images_device/{dim}.png",dpi=100)
    print(f'speedup_max: {speedup_max}\n')
plt.show()