import numpy as np
import matplotlib.pyplot as plt

def AMPD(data):
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]

column1 = []
column2 = []
column3 = []

file_path = 'D:/pythonProject/AM.txt'

with open(file_path, 'r') as file:

    for line in file:
        data = line.split()


        if len(data) >= 3:
            column1.append(float(data[0]))
            column2.append(float(data[1]))
            column3.append(float(data[2]))
        else:
            print(f"Skipping invalid data: {data}")

def sim_data(x_values):
    x_values = np.asarray(x_values, dtype=float)

    x_values = x_values[~np.isnan(x_values)]

    if len(x_values) == 0:
        raise ValueError("Invalid data: Empty or missing x_values")

    y = 2 * np.cos(2 * np.pi * 300 * x_values) \
        + 5 * np.sin(2 * np.pi * 100 * x_values) \
        + 4 * np.random.randn(len(x_values))
    return y


def vis():

    column1 = []
    column2 = []
    column3 = []

    file_path = 'D:/pythonProject/AM.txt'

    with open(file_path, 'r') as file:

        for line in file:
            data = line.split()
            if len(data) >= 3:
                column1.append(float(data[0]))
                column2.append(float(data[1]))
                column3.append(float(data[2]))
            else:
                print(f"Skipping invalid data: {data}")
    try:
        x_values = np.array(column3)
        y = sim_data(x_values)
        plt.plot(x_values, y)
        px = AMPD(y)
        plt.scatter(x_values[px], y[px], color="red")
        plt.show()
    except ValueError as e:
        print(f"Error: {e}")
vis()