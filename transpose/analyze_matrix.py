import matplotlib.pyplot as plt
import numpy as np
import os


# SETTINGS
bandwidth_analysis = 1
speedup_analysis   = 1


# BANDWIDTH ANALYSIS
if bandwidth_analysis:
    logs = sorted(os.listdir("logs/logs_matrix"))
    for log in logs:
        plt.figure(figsize=(15,5))
        print(f'Reading {log} ...')
        dim  = log[:-4]
        dims = dim.split("x")
        name      = list()
        check     = list()
        kernel_ms = list()
        bandwidth = list()
        speedup   = list()
        with open("logs/logs_matrix/"+log, "r") as file:
            host_ms = float(file.readline())
            for i, info in enumerate(file):
                info = info[:-1] # delete \n
                if i % 5 == 0:
                    name.append(f'{chr(65+i//5)} {info}')
                elif i % 5 == 1:
                    check.append(info == '1')
                elif i % 5 == 2:
                    kernel_ms.append(float(info))
                elif i % 5 == 3:
                    speedup.append(float(info))
                elif i % 5 == 4:
                    bandwidth.append(float(info))
        assert all(check), "Check is False"

        inc = 0.9
        col = [
            'midnightblue', 'mediumblue', 'royalblue', 'skyblue', \
            'darkgreen', 'seagreen', 'limegreen', 'yellowgreen', \
            'darkgoldenrod', 'darkorange', 'gold', 'yellow' , \
            'maroon', 'brown', 'indianred', 'lightcoral', \
            ]
        offset = len(name) * inc
        pos = np.arange(0, offset, inc)
        for i in range(len(name)):
            plt.bar(pos[i], bandwidth[i], color=col[i])
        plt.yticks(np.arange(0,325,25)) 
        plt.xticks([])  
        plt.ylabel('Bandwidth (GB/s)')
        plt.xlabel('Test', labelpad=20.0)

        plt.title(f"Dimension: {dim}")
        plt.legend(labels=name)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"images/images_matrix/bandwidth_{dim}.png",dpi=100)


# SPEEDUP ANALYSIS
if speedup_analysis:
    logs = sorted(os.listdir("logs/logs_matrix"))
    for log in logs:
        plt.figure(figsize=(15,5))
        print(f'Reading {log} ...')
        dim  = log[:-4]
        dims = dim.split("x")
        name      = list()
        check     = list()
        kernel_ms = list()
        bandwidth = list()
        speedup   = list()
        with open("logs/logs_matrix/"+log, "r") as file:
            host_ms = float(file.readline())
            for i, info in enumerate(file):
                info = info[:-1] # delete \n
                if i % 5 == 0:
                    name.append(f'{chr(65+i//5)} {info}')
                elif i % 5 == 1:
                    check.append(info == '1')
                elif i % 5 == 2:
                    kernel_ms.append(float(info))
                elif i % 5 == 3:
                    speedup.append(float(info))
                elif i % 5 == 4:
                    bandwidth.append(float(info))
        assert all(check), "Check is False"

        inc = 0.9
        col = [
            'midnightblue', 'mediumblue', 'royalblue', 'skyblue', \
            'darkgreen', 'seagreen', 'limegreen', 'yellowgreen', \
            'darkgoldenrod', 'darkorange', 'gold', 'yellow' , \
            'maroon', 'brown', 'indianred', 'lightcoral', \
            ]
        offset = len(name) * inc
        pos = np.arange(0, offset, inc)
        new_lab = list()
        new_i = 0
        for i in range(len(name)):
            if "Copy" not in name[i]:
                plt.bar(pos[new_i], speedup[i], color=col[i])
                new_lab.append(name[i])
                new_i += 1
        plt.yticks(np.arange(0,325,25)) 
        plt.xticks([])  
        plt.ylabel('Speedup')
        plt.xlabel('Test', labelpad=20.0)

        plt.title(f"Dimension: {dim}")
        plt.legend(labels=new_lab)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"images/images_matrix/speedup{dim}.png",dpi=100)