import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class FCM:
    def __init__(self):
        pass

    def compute_dis(self, mat1, k_row, mat2, i_row, n_col, d):
        x = 0
        max_val = 0

        if d == 1:  # 切比雪夫距离
            max_val = 0
            for j in range(n_col):
                x = abs(mat1[k_row][j] - mat2[i_row][j])
                if x > max_val:
                    max_val = x
        elif d == 2:  # 欧几里得距离
            max_val = 0
            for j in range(n_col):
                max_val += (mat1[k_row][j] - mat2[i_row][j]) ** 2
            max_val = np.sqrt(max_val)
        elif d == 3:  # 汉明距离
            max_val = 0
            for j in range(n_col):
                max_val += abs(mat1[k_row][j] - mat2[i_row][j])

        return max_val

    def FCMCenter(self, R, X, N, M, C, Q):
        V = np.zeros((C, M))
        for i in range(C):
            for j in range(M):
                n_sum = 0
                m_sum = 0
                for k in range(N):
                    n_sum += (R[i][k] ** Q) * X[k][j]
                    m_sum += R[i][k] ** Q
                V[i][j] = n_sum / m_sum

        return V

    def modifyR(self, X, V, N, M, C, D, Q):
        R1 = np.zeros((C, N))

        for i in range(C):
            for k in range(N):
                kj_sum = 0
                for j in range(C):
                    if j != i:
                        kj_sum += (self.compute_dis(X, k, V, i, M, D) / self.compute_dis(X, k, V, j, M, D)) ** (2 / (Q - 1))
                R1[i][k] = (kj_sum + 1) ** -1

        return R1

    def display(self, matrix):
        for row in matrix:
            for val in row:
                print(f"{val:.7f}", end="  ")
            print()

    def displayClass(self, R1, N, C):
        CR = np.zeros((C, N))

        for j in range(N):
            max_val = R1[0][j]
            CR[0][j] = 1
            for i in range(1, C):
                if R1[i][j] > max_val:
                    max_val = R1[i][j]
                    CR[i][j] = 1
                    if i == 1:
                        CR[0][j] = 0
                else:
                    CR[i][j] = 0

        return CR

    def ClassFactor(self, R1, N, C):
        classF = np.zeros(2)

        for i in range(C):
            for j in range(N):
                classF[0] += R1[i][j] ** 2
                classF[1] += R1[i][j] * np.log(R1[i][j])

        classF[0] /= N  # 类别因子
        classF[1] = -classF[1] / N  # 平均模糊熵

        return classF


def main():
    # 从CSV文件加载Iris数据集
    iris_df = pd.read_csv("match_2023-wimbledon-1301.csv")
    data = iris_df.iloc[:, -2:-1].values  # 取前四列作为特征
    n = len(data)
    m = len(data[0])
    c = 3  # Iris数据集有3个类别
    e = 0.0001
    q = 3
    d = 1

    x = data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    r0 = np.zeros((c, n))
    for i in range(n):
        _sum = 0
        for j in range(c):
            if j == c - 1:
                r0[j][i] = 1 - _sum
            else:
                r0[j][i] = (1 - _sum) * np.random.random()
                _sum += r0[j][i]

    fcm = FCM()

    times = 0
    # print("            ---------------FCM算法在Python中的应用--------------")
    while True:
        max_val = 0
        vv = fcm.FCMCenter(r0, x, n, m, c, q)
        rr = fcm.modifyR(x, vv, n, m, c, d, q)

        for i in range(c):
            for j in range(n):
                if abs(rr[i][j] - r0[i][j]) > max_val:
                    max_val = abs(rr[i][j] - r0[i][j])

        if max_val < e:
            break

        for i in range(c):
            for j in range(n):
                r0[i][j] = rr[i][j]

        times += 1

    print(" ")
    print("迭代次数:", times)
    print("模糊聚类中心:")
    fcm.display(vv)
    print("模糊隶属矩阵:")
    fcm.display(rr)
    print("确定性聚类矩阵:")
    cr = fcm.displayClass(rr, n, c)
    fcm.display(cr)
    class_factors = fcm.ClassFactor(rr, n, c)
    print("模糊划分熵:", class_factors[0])
    print("平均模糊划分熵:", class_factors[1])

    plt.scatter(x[:, 0], x[:, 1], c=np.argmax(rr, axis=0), cmap='viridis', edgecolors='k', s=50)
    plt.scatter(vv[:, 0], vv[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    plt.title('FCM Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
