import matplotlib.pyplot as plt
import numpy as np
import os


# SETTINGS
time_analysis    = 1
speedup_analysis = 1


# TILE ANALYSIS
if time_analysis:
    logs = sorted(os.listdir("logs/logs_device"))
    while len(logs) > 0:
        # Get all permutations of the same dimension
        target = logs[0].split("_")[0]
        target_list = list()
        for i in range(len(logs)-1, -1, -1):
            if logs[i].split("_")[0] == target:
                target_list.append(logs[i])
                del logs[i]

        # Process time
        fig = plt.figure(figsize=(10,8))
        ax = fig.subplots(nrows=2, ncols=1)
        counter = 0
        for log in target_list:
            print(f'Reading {log} ...')
            perm = log[-7:-4]
            dim  = log[:-8]
            dims = dim.split("x")
            host_ms   = 0
            name      = list()
            check     = list()
            device_ms = list()
            speedup   = list()
            with open("logs/logs_device/"+log, "r") as file:
                host_ms = float(file.readline())
                ax[0].bar(counter, host_ms, color='tab:blue')
                ax[0].text(counter, -0.3, perm, fontsize=11)
                for i, info in enumerate(file):
                    info = info[:-1] # delete \n
                    if i % 4 == 0:
                        name.append(f'{chr(65+i//4)} {info}')
                    elif i % 4 == 1:
                        check.append(bool(info))
                    elif i % 4 == 2:
                        device_ms.append(float(info))
                    elif i % 4 == 3:
                        speedup.append(float(info))
            assert all(check), "Check is False"

            inc = 0.9
            col = ['tab:green','tab:red','tab:purple']
            offset = len(name) * inc
            pos = np.arange(counter, counter+offset, inc)
            for i in range(len(name)):
                ax[1].bar(pos[i], device_ms[i], color=col[i])
            ax[1].set_yticks(np.arange(0, 1, 0.1))
            ax[1].set_xticks([])
            ax[1].text(counter, -0.06, perm, fontsize=11)
            ax[1].set_ylabel('Kernel Time (ms)')
            ax[1].set_xlabel('permutation', labelpad=20.0)
            counter = counter + offset + inc

        ax[1].legend(labels=name)
        ax[1].grid()
        ax[0].set_title(f"Dimension: {dim}")
        ax[0].set_xticks([])
        ax[0].set_ylabel('Host Time (ms)')
        ax[0].grid()
        plt.tight_layout()
        plt.savefig(f"images/images_device/time_{dim}.png",dpi=100)
        print()
    print("-----------------\n")


# SPEEDUP ANALYSIS
if speedup_analysis:
    logs = sorted(os.listdir("logs/logs_device"))
    while len(logs) > 0:
        # Get all permutations of the same dimension
        target = logs[0].split("_")[0]
        target_list = list()
        for i in range(len(logs)-1, -1, -1):
            if logs[i].split("_")[0] == target:
                target_list.append(logs[i])
                del logs[i]

        # Process speedup
        plt.figure(figsize=(10,5))
        counter = 0
        for log in target_list:
            print(f'Reading {log} ...')
            perm = log[-7:-4]
            dim  = log[:-8]
            dims = dim.split("x")
            host_ms   = 0
            name      = list()
            check     = list()
            device_ms = list()
            speedup   = list()
            with open("logs/logs_device/"+log, "r") as file:
                host_ms = float(file.readline())
                for i, info in enumerate(file):
                    info = info[:-1] # delete \n
                    if i % 4 == 0:
                        name.append(f'{chr(65+i//4)} {info}')
                    elif i % 4 == 1:
                        check.append(bool(info))
                    elif i % 4 == 2:
                        device_ms.append(float(info))
                    elif i % 4 == 3:
                        speedup.append(float(info))
            assert all(check), "Check is False"

            inc = 0.9
            col = ['tab:green','tab:red','tab:purple']
            offset = len(name) * inc
            pos = np.arange(counter, counter+offset, inc)
            for i in range(len(name)):
                plt.bar(pos[i], speedup[i], color=col[i])
            plt.yticks(np.arange(0,15,1))
            plt.xticks([])  
            plt.text(counter, -1, perm, fontsize=11)
            plt.ylabel('Speedup')
            plt.xlabel('permutation', labelpad=30.0)
            counter = counter + offset + inc

        plt.title(f"Dimension: {dim}")
        plt.legend(labels=name)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"images/images_device/speedup_{dim}.png",dpi=100)
        print()
    print("-----------------\n")

