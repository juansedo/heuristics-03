from typing import List
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import math
import os
import time

def get_test_file(id):
    with open(f"data/mtVRP{id}.txt") as f:
        lines = [line.rstrip() for line in f]
        lines = [line.split() for line in lines]
        lines = [[int(x) for x in line] for line in lines]
    return [lines[0], lines[1:]]


def rimraf(folder_path):
    if len(os.listdir(folder_path)) > 0:
        answer = input(f"There are files on the {folder_path} folder, are you sure to delete them? (Y/n): ")
        if answer == "n" or answer == "N":
            print("Files will not be to deleted")
            return
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print("Files deleted successfully")


def find(arr, value, start=0):
    try:
        return arr.index(value, start)
    except:
        return -1
    

#%% DISTANCE MATRIX
def get_distance(n1, n2):
    x1, y1 = n1
    x2, y2 = n2
    return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)


def generate_distance_matrix(data, n):
    distances = np.zeros((n + 1, n + 1))
    demands = np.zeros(n + 1)
    for node in data:
        values = [get_distance(node[1:3], node2[1:3]) for node2 in data]
        distances[node[0]] = values
        demands[node[0]] = node[3]
    return [distances, demands]

#%% LB
class LB:
    mtVRP1 = 384.48
    mtVRP2 = 515.34
    mtVRP3 = 515.34
    mtVRP4 = 515.34
    mtVRP5 = 585.74
    mtVRP6 = 585.74
    mtVRP7 = 596.49
    mtVRP8 = 658.59
    mtVRP9 = 658.59
    mtVRP10 = 336.44
    mtVRP11 = 395.03
    mtVRP12 = 395.03

    def get_by_index(index):
        if index == 1: return LB.mtVRP1
        if index == 2: return LB.mtVRP2
        if index == 3: return LB.mtVRP3
        if index == 4: return LB.mtVRP4
        if index == 5: return LB.mtVRP5
        if index == 6: return LB.mtVRP6
        if index == 7: return LB.mtVRP7
        if index == 8: return LB.mtVRP8
        if index == 9: return LB.mtVRP9
        if index == 10: return LB.mtVRP10
        if index == 11: return LB.mtVRP11
        if index == 12: return LB.mtVRP12
        return LB.mtVRP1
    
    def get_labels():
        return [
            "mtVRP1",
            "mtVRP2",
            "mtVRP3",
            "mtVRP4",
            "mtVRP5",
            "mtVRP6",
            "mtVRP7",
            "mtVRP8",
            "mtVRP9",
            "mtVRP10",
            "mtVRP11",
            "mtVRP12",
        ]

    def get_list():
        return [
            LB.mtVRP1,
            LB.mtVRP2,
            LB.mtVRP3,
            LB.mtVRP4,
            LB.mtVRP5,
            LB.mtVRP6,
            LB.mtVRP7,
            LB.mtVRP8,
            LB.mtVRP9,
            LB.mtVRP10,
            LB.mtVRP11,
            LB.mtVRP12,
        ]

#%% BKS
class BKS:
    mtVRP1 = 546.29
    mtVRP2 = 835.80
    mtVRP3 = 858.58
    mtVRP4 = 866.58
    mtVRP5 = 829.45
    mtVRP6 = 826.14
    mtVRP7 = 1034.61
    mtVRP8 = 1300.02
    mtVRP9 = 1300.62
    mtVRP10 = 1078.64
    mtVRP11 = 845.48
    mtVRP12 = 823.14
    
    def get_by_index(index):
        if index == 1: return BKS.mtVRP1
        if index == 2: return BKS.mtVRP2
        if index == 3: return BKS.mtVRP3
        if index == 4: return BKS.mtVRP4
        if index == 5: return BKS.mtVRP5
        if index == 6: return BKS.mtVRP6
        if index == 7: return BKS.mtVRP7
        if index == 8: return BKS.mtVRP8
        if index == 9: return BKS.mtVRP9
        if index == 10: return BKS.mtVRP10
        if index == 11: return BKS.mtVRP11
        if index == 12: return BKS.mtVRP12
        return BKS.mtVRP1
    
    def get_labels():
        return [
            "mtVRP1",
            "mtVRP2",
            "mtVRP3",
            "mtVRP4",
            "mtVRP5",
            "mtVRP6",
            "mtVRP7",
            "mtVRP8",
            "mtVRP9",
            "mtVRP10",
            "mtVRP11",
            "mtVRP12",
        ]
    
    def get_list():
        return [
            BKS.mtVRP1,
            BKS.mtVRP2,
            BKS.mtVRP3,
            BKS.mtVRP4,
            BKS.mtVRP5,
            BKS.mtVRP6,
            BKS.mtVRP7,
            BKS.mtVRP8,
            BKS.mtVRP9,
            BKS.mtVRP10,
            BKS.mtVRP11,
            BKS.mtVRP12,
        ]

#%% Plots
def save_plot(fig, title):
    if os.path.exists('./outputs') == False:
        os.mkdir('./outputs')
    
    filename = title + ".png"
    path_plot = './outputs/' + filename
    fig.savefig(path_plot, dpi=fig.dpi)
    plt.cla()
    plt.clf()


def compare_plot(method, title, data, initial_sol, final_sol, index):
    paths1, initial_Z, initial_time = initial_sol
    paths2, final_Z, final_time = final_sol
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    fig.suptitle(title, fontsize=16)

    ax1.title.set_text(f'Before ({round(initial_time, 2)} s)')
    ax1.set_xlabel(f'Z: {round(initial_Z, 2)}', fontweight='bold', fontsize=14)

    ax2.title.set_text(f'After ({round(final_time, 2)} s)')
    ax2.set_xlabel(f'Z: {round(final_Z, 2)}', fontweight='bold', fontsize=14)
    
    x = []
    y = []
    for node in data:
        x.append(node[1])
        y.append(node[2])
    ax1.plot(x, y, "o", label="Nodos")
    ax2.plot(x, y, "o", label="Nodos")

    for p in paths1:
        x = []
        y = []
        for node in paths1[p]:
            x.append(data[node][1])
            y.append(data[node][2])
        ax1.plot(x, y, "-", label=f"Camión {p+1}")

    for p in paths2:
        x = []
        y = []
        for node in paths2[p]:
            x.append(data[node][1])
            y.append(data[node][2])
        ax2.plot(x, y, "-", label=f"Camión {p+1}")

    save_plot(fig, f"{method}-mtVRP{index}")


def paths_and_time_plot(title, data, cost, solution, Th, penalized_time, index):
    paths, distances, total_time = solution
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    fig.suptitle(title, fontsize=16)
    ax1.title.set_text(f'Paths ({round(total_time, 2)} s)')
    ax1.set_xlabel(f'Z: {cost}', fontweight='bold', fontsize=13)

    x = []
    y = []
    for node in data:
        x.append(node[1])
        y.append(node[2])
    ax1.plot(x, y, "o", label="Nodos")

    for p in paths:
        x = []
        y = []
        for node in paths[p]:
            x.append(data[node][1])
            y.append(data[node][2])
        ax1.plot(x, y, "-", label=f"Camión {p+1}")
    ax1.legend(loc='upper left', ncols=1)

    y_labels = map(lambda i: f'{i + 1}', range(len(paths)))
    y_pos = np.arange(len(paths))
    ax2.title.set_text(f'Time by path')
    ax2.set_xlabel(f'Penalized time: {penalized_time}', fontweight='bold', fontsize=13)
    ax2.barh(y_pos, distances, align='center')
    ax2.set_ylabel('Camiones')
    ax2.set_yticks(y_pos, labels=y_labels)
    ax2.invert_yaxis()
    ax2.grid(which = "major")
    ax2.grid(which = "minor", alpha = 0.2)
    ax2.xaxis.set_major_locator(FixedLocator([Th]))
    
    fig.set_dpi(90)
    save_plot(fig, f"BRKGA-mtVRP{index}")


def comparison_boxplot(df, test_params):
    filtered = df[df['phase'] == 'GA-VND']

    fig, ax = plt.subplots()
    ax.set_title('BRKGA + VND Gap')

    data = []
    x_labels = []
    for (test_type, gens, p, mr, er) in test_params:
        f = filtered[
            (filtered['test_type'] == test_type) &
            (filtered['gens'] == gens) &
            (filtered['population'] == p) &
            (filtered['mr'] == mr) &
            (filtered['er'] == er)
        ]
        f = f['bks_gap'].values.tolist()
        data.append(f)
        x_labels.append(test_type)
    
    ax.boxplot(data, labels=x_labels)
    fig.set_dpi(90)
    save_plot(fig, f"BRKGA-Boxplot")


def summary_time_plot(filename, labels, total_times = None):
    if (len(labels) == 0 or len(total_times) == 0): return
    
    plt.clf()
    fig = plt.gcf()
    ax = plt.gca()

    size = np.arange(len(labels))
    width = 0.3
    if ('VND' in total_times):
        bar1 = ax.bar(size - width/2, total_times['VND'], width, label='VND')
        ax.bar_label(bar1)
    if ('MS_ILS' in total_times):
        bar2 = ax.bar(size + width/2, total_times['MS_ILS'], width, label='MS_ILS')
        ax.bar_label(bar2)

    ax.set_ylabel('Compute time (seconds)')
    ax.set_title('Algorithms comparison')
    ax.set_xticks(size, labels)
    ax.legend(loc='upper left', ncols=1)
    
    fig.set_size_inches(12.5, 5.5)
    fig.set_dpi(90)
    save_plot(fig, filename)


def summary_plot(filename, data):
    plt.clf()
    fig = plt.gcf()
        
    xitems = []
    yitems = []
    plt.plot(BKS.get_labels(), BKS.get_list(), color="C2")
    plt.fill_between(BKS.get_labels(), BKS.get_list(), color="C2", alpha=0.3)
    for key, value in data.items():
        xitems = [item[0] for item in value]
        yitems = [item[1] for item in value]
        plt.plot(xitems, yitems, label=key, marker='o')
        for i in range(len(xitems)):
            plt.annotate(str(round(yitems[i], 2)), (xitems[i], yitems[i] * 1.05))

    plt.legend(loc='upper left', ncols=1)
    fig.set_size_inches(12.5, 5.5)
    fig.set_dpi(90)
    save_plot(fig, filename)


#%% Decorators
def add_timer(func):
    """Decorator for retrieving the function return
    and the execution time (in seconds) into an array.
    
    Returns:
        [function return, execution time]
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        return (value, round((end - start), 2))
    return wrapper
