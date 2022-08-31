import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, log
from scipy.optimize import minimize
from PrivateUtils import util
from PrivateUtils import plot_util
import pandas as pd
import math
from numpy import sqrt, exp, log, max


# 原书第四章，隐马尔科夫模型与贝叶斯网络，但是例子和符号还是参考李航统计学习方法2版，第十章
class HMM:
    def __init__(self, A, E, Pi, Q, V, x=None, O=None):
        self.A = A
        self.E = E
        self.Pi = Pi
        self.O = O
        if O is None:
            self.T = None
        else:
            self.T = O.size  # 可观测序列长度
        self.N = A.shape[0]  # 隐状态取值个数
        self.Q = Q
        self.V = V
        self.x = x

    def forward(self):
        alpha = -1 * np.ones((self.T, self.N))
        for i in range(self.N):
            alpha[0, i] = self.Pi[i] * self.E[i, self.O[0]]

        for i in range(1, self.T):
            for j in range(self.N):
                alpha[i, j] = sum(alpha[i - 1, :] * self.A[:, j]) * self.E[j, self.O[i]]

        return sum(alpha[-1, :]), alpha

    def backward(self):
        beta = np.ones((self.T, self.N))

        for t in range(self.T - 2, -1, -1):
            for j in range(self.N):
                beta[t, j] = sum(self.A[j, :] * self.E[:, self.O[t + 1]] * beta[t + 1, :])

        return sum(self.Pi * self.E[:, self.O[0]] * beta[0, :]), beta

    def viterbi(self):
        delta = np.zeros((self.T, self.N))
        phi = np.zeros((self.T, self.N))

        delta[0, :] = self.Pi * self.E[:, self.O[0]]
        phi[0, :] = 0

        for t in range(1, self.T):
            for i in range(self.N):
                temp = delta[t - 1, :] * self.A[:, i]
                delta[t, i] = max(temp) * self.E[i, self.O[t]]
                phi[t, i] = np.where(temp == max(temp))[0][0]

        P_star = max(delta[-1, :])
        I_star = np.zeros(self.T)
        I_star[-1] = np.where(delta[-1, :] == P_star)[0][0]

        for t in range(self.T - 2, -1, -1):
            I_star[t] = phi[t + 1, int(I_star[t + 1])]

        if self.V[0] == 1:
            I_star += 1

        return P_star, I_star

    def create_random_O(self, n):
        self.x = -1 * np.ones(n, dtype=np.int32)  # 隐状态
        self.x[0] = util.number_of_certain_probability(self.Q, self.Pi)
        self.O = -1 * np.ones(n, dtype=np.int32)  # 观测值
        self.O[0] = util.number_of_certain_probability(self.V, self.E[self.x[0], :])
        for i in range(1, n):
            self.x[i] = util.number_of_certain_probability(self.Q, self.A[self.x[i - 1], :])
            self.O[i] = util.number_of_certain_probability(self.V, self.E[self.x[i], :])

    # 假设知道隐状态的模型参数估计，不过Pi的估计似乎需要多个序列，太麻烦了，就不估计了
    def know_x(self, x):
        A_hat = -1 * np.ones_like(self.A)
        for i in range(self.N):
            jump_from_i = sum(x[:-1] == i)  # 因为t=1,2,...,L-1
            for j in range(self.N):
                flag1 = (x == i)
                flag2 = (x == j)
                T_ij = sum(flag1[:-1] * flag2[1:])

                A_hat[i, j] = T_ij / jump_from_i

        E_hat = -1 * np.ones_like(self.E)
        for i in range(self.N):
            x_in_i = sum(x == i)
            for k in range(self.E.shape[1]):
                E_hat[i, k] = sum((x == i) * (self.O == self.V[k])) / x_in_i

        return A_hat, E_hat

    def not_know_x(self, iter_times):
        for i in range(iter_times):
            _, predicted_x = self.viterbi()  # 获得当前模型参数下的最可能隐状态
            x = predicted_x

            p_t_ij, p_t_j = self.E_step()
            A_hat, E_hat, Pi_hat = self.M_step(p_t_ij, p_t_j)

            self.A = A_hat
            self.E = E_hat
            self.Pi = Pi_hat

        return self.A, self.E, self.Pi

    def E_step(self):
        _, alpha = self.forward()
        _, beta = self.backward()

        # p_O = np.zeros((self.T-1))
        # for i in range(3):
        #     for j in range(3):
        #         p_O += alpha[0:-1,i] * self.A[i,j] * self.E[j, self.O[1:]] * beta[1:, j]
        p_O = np.sum(alpha[:-1] * beta[:-1], axis=1)

        p_t_ij = -1 * np.ones((self.T - 1, self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                p_t_ij[:, i, j] = (alpha[0:-1, i] * self.A[i, j] * self.E[j, self.O[1:]] * beta[1:, j]) / p_O

        p_t_j = alpha * beta / np.sum(alpha * beta, axis=1)[:, np.newaxis]

        return p_t_ij, p_t_j

    def M_step(self, p_t_ij, p_t_j):
        A_hat = -1 * np.ones_like(self.A)
        for i in range(self.N):
            jump_from_i = 0
            for j in range(self.N):
                jump_from_i += p_t_ij[:, i, j]

            jump_from_i = sum(jump_from_i)
            for j in range(self.N):
                A_hat[i, j] = sum(p_t_ij[:, i, j]) / jump_from_i

        E_hat = -1 * np.ones_like(self.E)
        for j in range(self.N):
            in_q_j = sum(p_t_j[:, j])
            for k in range(self.E.shape[1]):
                E_hat[j, k] = sum(p_t_j[:, j] * (self.O == self.V[k])) / in_q_j

        Pi_hat = p_t_j[0, :]

        return A_hat, E_hat, Pi_hat


def model_example():
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    E = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    Pi = np.array([0.2, 0.4, 0.4])
    O = np.array([0, 1, 0])
    Q = np.array([1, 2, 3])
    V = np.array([1, 2])

    return A, E, Pi, O, Q, V


def model_example2():
    A = np.array([[0, 1, 0], [0.1, 0, 0.9], [0.2, 0.8, 0]])
    E = np.array([[0.9, 0.1], [1, 0], [0.15, 0.85]])
    Pi = np.array([0, 1, 0])
    Q = np.array([0, 1, 2], dtype=np.int32)
    V = np.array([0, 1], dtype=np.int32)

    return A, E, Pi, Q, V


def run_evaluation_decoding(A, E, Pi, O, Q, V):
    print(f'模型的评估问题'.center(100, '*'))

    instance = HMM(A=A, E=E, Pi=Pi, O=O, Q=Q, V=V)

    print('前向算法'.center(50, '*'))
    print(instance.forward())

    print('后向算法'.center(50, '*'))
    print(instance.backward())

    print(f'隐状态推断问题'.center(100, '*'))
    print(instance.viterbi())


def run_learning(A, E, Pi, Q, V, n, iter_times):
    print('模型的学习问题'.center(100, '*'))

    # 利用真实的模型参数模型参数，生成一观测序列与对应的隐状态
    real_instance = HMM(A=A, E=E, Pi=Pi, Q=Q, V=V)
    real_instance.create_random_O(n)

    # 对真实模型参数添加部分噪声作为初始值
    A2 = A + 0.2 * np.random.rand(3, 3)
    E2 = E + 0.2 * np.random.rand(3, 2)
    for i in range(3):
        A2[i, :] /= sum(A2[i, :])
        E2[i, :] /= sum(E2[i, :])
    Pi2 = Pi + 0.2 * np.random.rand(3)
    Pi2 = Pi2 / sum(Pi2)

    print('知道隐状态序列'.center(50, '*'))
    estimate_instance = HMM(A=A2, E=E2, Pi=Pi2, Q=Q, V=V, O=real_instance.O)
    A_hat, E_hat = estimate_instance.know_x(real_instance.x)
    print(f'状态转移矩阵A估计差值: \n {np.around(A - A_hat, 2)}')
    print(f'发射概率矩阵E估计差值: \n {np.around(E - E_hat, 2)}')

    print('不知道隐状态序列'.center(50, '*'))
    estimate_instance2 = HMM(A=A2, E=E2, Pi=Pi2, Q=Q, V=V, O=real_instance.O)
    A_hat, E_hat, Pi_hat = estimate_instance2.not_know_x(iter_times)  # not_know_x方法并更新了类属性
    print(f'状态转移矩阵A估计差值: \n {np.around(A - A_hat, 2)}')
    print(f'发射概率矩阵E估计差值: \n {np.around(E - E_hat, 2)}')
    print(f'发射概率矩阵Pi估计差值: \n {np.around(Pi - Pi_hat, 2)}')


if __name__ == '__main__':
    np.random.seed(29)

    # A, E, Pi, O, Q, V = model_example()
    # run_evaluation_decoding(A, E, Pi, O, Q, V)

    # n = 100
    # iter_times = 200
    # A, E, Pi, Q, V = model_example2()
    # run_learning(A, E, Pi, Q, V, n, iter_times)
