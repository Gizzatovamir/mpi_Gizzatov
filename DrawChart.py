import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = "lab_1/"


def acceleration(t_1: np.ndarray, t_2: np.ndarray) -> float:
    return t_1 / t_2


if __name__ == "__main__":
    result = pd.read_csv(path + "Time.csv")
    print(result)
    x_plot = set(result["num_threads"])
    print(x_plot)
    tmp_dict = {}
    for row in result.iterrows():
        print(row[1]["time"])
        if row[1]["num_threads"] not in tmp_dict.keys():
            tmp_dict[row[1]["num_threads"]] = [row[1]["time"]]
        else:
            tmp_dict[row[1]["num_threads"]].append(row[1]["time"])
    x = list(tmp_dict.keys())
    y = [np.mean(el) for el in tmp_dict.values()]
    plt.figure(0)
    plt.plot(x, y)
    plt.xlabel("Number of threads")
    plt.ylabel("Time in milliseconds")
    plt.grid()
    plt.savefig(path + "result_time.png")
    plt.figure(1)
    first_time = np.mean(list(tmp_dict.values())[0])
    acceleration_y = [acceleration(first_time, np.mean(el)) for el in list(tmp_dict.values())[1:]]
    plt.plot(x[1:], acceleration_y)
    plt.xlabel("Number of threads")
    plt.ylabel("acceleration")
    plt.grid()
    plt.savefig(path + "acceleration.png")
    plt.figure(2)
    efficacy_y = [el/x for el,x in zip(acceleration_y,x[1:])]
    plt.plot(x[1:], efficacy_y)
    plt.xlabel("Number of threads")
    plt.ylabel("efficiency coefficient ")
    plt.grid()
    plt.savefig(path + "efficiency.png")
