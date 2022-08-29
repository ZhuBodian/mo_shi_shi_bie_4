import matplotlib.pyplot as plt
import mpmath
from scipy.stats import norm
import numpy as np
from numpy import sqrt, pi, exp, log, inf
import tree_class as tc
from sklearn.metrics import log_loss, mean_absolute_error
import scipy
import tree_class
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint, Bounds, LinearConstraint
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.optimize import curve_fit


class Utils:
    # 计算欧式距离
    @staticmethod
    def euc_dist(x1, x2):
        # 方便广播机制
        # if x1.ndim == 1:
        #     x1 = x1[:, np.newaxis]
        # if x2.ndim == 1:
        #     x2 = x2[:, np.newaxis]
        # if x1.shape[1] != x2.shape[1]:
        #     x2 = x2.T

        return sqrt(np.sum((x1 - x2) * (x1 - x2), axis=1))

    # 多分类，positive_label对应阳，其它均为阴。其实sklearn.metrics有accuracy_score, classification_report
    @staticmethod
    def classifier_effect(real_label, cal_label, positive_label):
        TP = sum((real_label == positive_label) * (cal_label == positive_label))
        FN = sum((real_label == positive_label) * (cal_label != positive_label))
        FP = sum((real_label != positive_label) * (cal_label == positive_label))
        TN = sum((real_label != positive_label) * (cal_label != positive_label))
        Sn = TP / (TP + FN)
        Sp = TN / (TN + FP)
        Acc = (TP + TN) / (TP + TN + FP + FN)
        Rec = TP / (TP + FN)
        Pre = TP / (TP + FP)
        F = 2 * Rec * Pre / (Rec + Pre)
        return Sn, Sp, Acc, Rec, Pre, F

    # 给定画布边界，计算决策面（暂用于2维2分类，且有些决策面不是函数，如一个圆，那时候可以考虑画散点图）
    # predict_fun为预测函数，给定输入，输出标签
    @staticmethod
    def cal_decision_curve(left, right, bottom, top, predict_fun, step=200):
        grid = np.meshgrid(np.linspace(left, right, step),
                           np.linspace(bottom, top, step))  # list of array, array为(step,step)
        grid = np.hstack((grid[0].reshape((step * step, 1)),
                          grid[1].reshape((step * step, 1))))  # 转为(step*step, 2)
        temp = predict_fun(grid).reshape(step, step)  # 如果不reshape，直接差分，图像最右边会有一道线
        gap = temp.max() - temp.min()
        idxs = np.vstack((np.diff(temp, axis=0), np.zeros((1, step), dtype=np.int8)))
        idxs = idxs.reshape(step * step)
        return grid[np.where((idxs == gap) + (idxs == -gap))]

    #
    # predict_fun为预测函数，给定输入，输出标签
    @staticmethod
    def cal_decision_area(left, right, bottom, top, predict_fun, step=300):
        grid = np.meshgrid(np.linspace(left, right, step),
                           np.linspace(bottom, top, step))  # list of array, array为(step,step)
        grid = np.hstack((grid[0].reshape((step * step, 1)),
                          grid[1].reshape((step * step, 1))))  # 转为(step*step, 2)
        return grid, predict_fun(grid)

    # 防止下溢的sigmoid函数，（处理多维数组）
    @staticmethod
    def sigmoid(data):
        # lambda a: 1 / (1 + exp(-a)) if a>=0 else lambda a: exp(a) / (1 + exp(a))
        idxs = np.where(data >= 0)
        data[idxs] = 1 / (1 + exp(-data[idxs]))
        idxs = np.where(data < 0)
        data[idxs] = exp(data[idxs]) / (1 + exp(data[idxs]))  # 如果x<0，还是exp(-x)，当x绝对值很大时，如10000，那么exp（10000）溢出
        return data

    # 线性核直接返回两个向量的相乘
    @staticmethod
    def linear_kernal(x1, x2):
        return x1.dot(x2)

    # rbf核
    @staticmethod
    def rbf_kernal(x1, x2, sigma_2=256 * 0.3):
        # 为了方便使用广播机制
        if x1.ndim == 1:
            x1 = x1[:, np.newaxis]
        if x2.ndim == 1:
            x2 = x2[:, np.newaxis]
        if x1.shape[1] != x2.shape[1]:
            x2 = x2.T

        return exp(-Utils().euc_dist(x1, x2) ** 2 / sigma_2)[:, np.newaxis]

    # 根据指定概率生成数字
    @staticmethod
    def number_of_certain_probability(sequence, probability):
        x = np.random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(sequence, probability):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        return item

    @staticmethod
    def plot_bound(data, ratio=1.1):
        left, right = min(data[:, 0]), max(data[:, 0])
        bottom, top = min(data[:, 1]), max(data[:, 1])
        left , right, bottom, top = ratio * left, ratio * right, ratio * bottom, ratio * top

        return left , right, bottom, top




# 原书第三章，概率密度函数的估计
class PDFEstimation:
    # 参数估计，一直密度函数形式，但是其中部分或者全部参数未知
    # 不同于书上显式的做法，这里用数值解法
    class ParmEstimation:
        def __init__(self, fun, data, bnds):
            """
            :param fun: fun(x,data)，匿名函数，已知的函数形式，x为参数，data为用于训练的数据
            :param x:  用于分析的数据
            :param bnds:  参数边界
            """
            self.fun = fun
            self.data = data
            self.bnds = bnds

        def _cal(self, x):
            p_i = self.fun(x=x, data=self.data)
            p_i = log(p_i + 1e-7)
            ans = 0
            for i in range(self.data.size):
                ans = ans + p_i[i]

            return -ans  # 千万不要忘了返回负数

        # 仅仅针对可以用mle方法的概密（函数可求多参数），有些概密本身不可以用mle，没有进行分析；且初始值为了方便，仅是（0,1）的随机数
        def mle(self):
            ans = minimize(fun=self._cal, x0=np.random.rand(len(self.bnds)), bounds=self.bnds)

            return ans.success, ans.x

        # 单参数，假设先验概率也为正态分布
        '''
        不知道贝叶斯估计如何处理（3-33）的连乘式（很多个小于0的数相乘），（mle是取对数，但是那是因为mle有求导操作，所以取对数化练乘为连和，且有估计量不变性；
        而be是积分，没必要求导，且不知道有没有那种不变性）， 且似乎积分运算会十分慢，所以暂且不编
        '''

        def be(self):
            return 0

    # 非参数估计
    class NonParmEstimation:
        def __init__(self, x, y=None, pdf_y=None):
            '''
            :param x: 对分布函数进行采样的横坐标
            :param y: 对分布函数进行采样的纵坐标
            :param pdf_y: 对概密函数进行采样的纵坐标
            '''
            self.x = x  # 不均匀随机采样横坐标

            self.pdf = pdf_y  # 分布函数对应的概密的采样纵坐标
            self.pdf_cal_arr = None  # hist方法中，第一列代表x取值划分为k个块后每个区间的中点坐标；第二列代表在该区间内的样本数目

            self.cdf = y  # 分布函数的采样纵坐标
            self.cdf_cal_arr = None  # self.pdf_cal_arr的维数要比self.cdf_cal_arr多两行（self.x的最小值与最大值）
            self.N = x.size
            self.alg = None
            self.k = None
            self.kn = None
            self.h = None

        def hist(self, k):
            self.alg = 'hist'
            self.k = k

            temp = np.linspace(self.x.min(), self.x.max(), k + 1)  # 将原数据划分为k块（共k+1个点）
            V = (self.x.max() - self.x.min()) / k
            self.pdf_cal_arr = np.zeros((k, 2))
            for i in range(k):
                self.pdf_cal_arr[i, 0] = (temp[i] + temp[i + 1]) / 2  # 用区间的中点代表区间
                self.pdf_cal_arr[i, 1] = sum((self.x > temp[i]) * (self.x < temp[i + 1]))  # 统计当前区间内有多少个样本点
            self.pdf_cal_arr[0, 1] += 1  # 上述不等式少加两个端点
            self.pdf_cal_arr[-1, 1] += 1
            self.pdf_cal_arr[:, 1] /= (self.N * V)

            # 在原来的k个区间中点坐标的基础上，再加上x的最大最小坐标(保证定义域不变)
            self.pdf_cal_arr = np.vstack((np.array([self.x[0], 0]), self.pdf_cal_arr))
            self.pdf_cal_arr = np.vstack((self.pdf_cal_arr, np.array([self.x[-1], 0])))

        # 计算分布函数
        def cdf_cal(self):
            # 两个数组第一列相同（x坐标
            self.cdf_cal_arr = self.pdf_cal_arr[:, 0]
            self.cdf_cal_arr = np.vstack((self.cdf_cal_arr, np.zeros(self.cdf_cal_arr.size).T)).T
            fun = lambda p1, p2: (p1[1] + p2[1]) * (p2[0] - p1[0]) / 2

            for i in range(self.cdf_cal_arr[:, 0].size - 1):
                p1 = [self.pdf_cal_arr[i, 0], self.pdf_cal_arr[i, 1]]
                p2 = [self.pdf_cal_arr[i + 1, 0], self.pdf_cal_arr[i + 1, 1]]
                self.cdf_cal_arr[i + 1, 1] = self.cdf_cal_arr[i, 1] + fun(p1, p2)

        def plot(self):
            # 画出计算所得的概密图像，并将其与理论图像作为对比（如果有的话）；
            # 画出计算所得的概率分布图像，并将其与理论图像进行对比
            self.cdf_cal()

            # 概密图像
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.x, self.pdf, 'r-', label='sampling pdf')
            plt.plot(self.pdf_cal_arr[:, 0], self.pdf_cal_arr[:, 1], 'b-', label='cal pdf')
            plt.legend()
            plt.title('pdf plot')
            if self.alg == 'hist':
                plt.ylabel('k=' + str(self.k) + ', N=' + str(self.N))
            elif self.alg == 'kn_neighbor':
                plt.ylabel('kn=' + str(self.kn) + ', N=' + str(self.N))
            else:
                plt.ylabel('h=' + str(self.h / pi) + '% scale length, N=' + str(self.N))

            # 分布图像
            plt.subplot(1, 2, 2)
            if self.cdf is not None:
                plt.plot(self.x, self.cdf, 'r-', label='sampling CDF')
            plt.plot(self.cdf_cal_arr[:, 0], self.cdf_cal_arr[:, 1], 'b-', label='cal CDF')
            plt.legend()
            plt.title('CDF plot')
            plt.show()

        def kn_neighbor(self, kn):
            self.alg = 'kn_neighbor'
            self.kn = kn

            iter_times = int(self.N / self.kn)
            self.pdf_cal_arr = np.zeros((iter_times, 2))
            for i in range(iter_times):  # 就算是小数，也是向下取整，for循环之后再考虑余数的问题
                start = i * self.kn
                end = (i + 1) * self.kn - 1
                self.pdf_cal_arr[i, 0] = (self.x[start] + self.x[end]) / 2
                self.pdf_cal_arr[i, 1] = self.kn / self.N / (self.x[end] - self.x[start])
            if np.mod(self.N, self.kn) != 0:
                if np.mod(self.N, self.kn) == 1:
                    self.N -= 1  # 余数为1没有区间的概念，抛弃该点
                else:
                    temp_x = (self.x[end + 1] + self.x[-1]) / 2
                    temp_y = (self.N - iter_times * self.kn) / self.N / (self.x[-1] - self.x[end + 1])
                    temp = np.array([temp_x, temp_y])
                    self.pdf_cal_arr = np.vstack((self.pdf_cal_arr, temp))

            # 保证定义域不变
            self.pdf_cal_arr = np.vstack((np.array([self.x[0], 0]), self.pdf_cal_arr))
            self.pdf_cal_arr = np.vstack((self.pdf_cal_arr, np.array([self.x[-1], 0])))

        # 仅实现窗函数
        def parzen(self, h):
            self.alg = 'parzen'
            self.h = h

            # 方窗计数
            rectangular_sum = lambda x, xi: sum(np.abs(x - xi) <= h / 2) / self.h

            self.pdf_cal_arr = np.zeros((self.x.size, 2))
            for i in range(self.x.size):
                self.pdf_cal_arr[i, 0] = self.x[i]
                self.pdf_cal_arr[i, 1] = rectangular_sum(self.x[i], self.x) / self.N


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
        self.x[0] = Utils.number_of_certain_probability(self.Q, self.Pi)
        self.O = -1 * np.ones(n, dtype=np.int32)  # 观测值
        self.O[0] = Utils.number_of_certain_probability(self.V, self.E[self.x[0], :])
        for i in range(1, n):
            self.x[i] = Utils.number_of_certain_probability(self.Q, self.A[self.x[i-1], :])
            self.O[i] = Utils.number_of_certain_probability(self.V, self.E[self.x[i], :])

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
                E_hat[i, k] = sum((x==i) * (self.O==self.V[k])) / x_in_i

        return A_hat, E_hat

    def E_step(self):
        _, alpha = self.forward()
        _, beta = self.backward()

        # p_O = np.zeros((self.T-1))
        # for i in range(3):
        #     for j in range(3):
        #         p_O += alpha[0:-1,i] * self.A[i,j] * self.E[j, self.O[1:]] * beta[1:, j]
        p_O = np.sum(alpha[:-1]*beta[:-1], axis=1)

        p_t_ij = -1 * np.ones((self.T-1, self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                p_t_ij[:, i, j] = (alpha[0:-1, i] * self.A[i, j] * self.E[j, self.O[1:]] * beta[1:, j])/p_O

        p_t_j = alpha*beta / np.sum(alpha*beta, axis=1)[:, np.newaxis]

        return p_t_ij, p_t_j

    def M_step(self, p_t_ij, p_t_j):
        A_hat = -1 * np.ones_like(self.A)
        for i in range(self.N):
            jump_from_i = 0
            for j in range(self.N):
                jump_from_i += p_t_ij[:, i, j]

            jump_from_i = sum(jump_from_i)
            for j in range(self.N):
                A_hat[i, j] = sum(p_t_ij[:, i, j])/jump_from_i

        E_hat = -1 * np.ones_like(self.E)
        for j in range(self.N):
            in_q_j = sum(p_t_j[:,j])
            for k in range(self.E.shape[1]):
                E_hat[j, k] = sum(p_t_j[:, j] * (self.O == self.V[k])) / in_q_j

        Pi_hat = p_t_j[0,:]

        return A_hat, E_hat, Pi_hat

# 原书第五章，线性分类器，均只考虑二维两类
class LinearClassifier:
    def __init__(self, data, label, prior=np.array([0.5, 0.5])):
        self.data = data
        self.N = data.shape[0]
        self.dim = data.shape[1]
        if label.ndim == 1:
            self.label = label[:, np.newaxis]
        else:
            self.label = label
        self.alg = None
        self.prior = prior
        self.decision_function = None  # 二类二维为一条直线的函数（因为形式简单，就写为了显式y=f(x)）
        self._predict_rule = None  # 返回x是否满足式(5-48)大于号的bool数组,要给出包含横纵坐标的向量(注意为列向量)
        self.linear_equation = None

    def fisher_fit(self):
        self.alg = 'fisher'
        data0 = self.data[np.where(self.label == -1)[0], :]
        data1 = self.data[np.where(self.label == 1)[0], :]

        # 一维数组无法直接转置
        mean0 = np.mean(data0, axis=0)[np.newaxis, :]
        mean1 = np.mean(data1, axis=0)[np.newaxis, :]

        S0 = 0
        temp = data0 - mean0
        for i in range(data0.shape[0]):
            temp2 = temp[i, :][:, np.newaxis]  #
            S0 = S0 + temp2.dot(temp2.T)

        S1 = 0
        temp = data1 - mean1
        for i in range(data1.shape[0]):
            temp2 = temp[i, :][:, np.newaxis]  #
            S1 = S1 + temp2.dot(temp2.T)

        Sw = S0 + S1

        omega = np.linalg.inv(Sw).dot((mean0 - mean1).T)

        # label 为0-1变量
        self._predict_rule = lambda x: -1 * np.sign((x - 0.5 * (mean0 + mean1)).dot(omega) - np.log(self.prior[1] / self.prior[0]))

        return omega, mean0.T, mean1.T

    # 绘制决策平面
    def plot(self):
        decision_x = np.linspace(min(self.data[:, 0]), max(self.data[:, 0]), 100)
        plt.figure()
        if min(self.label) == 0:
            plt.plot(self.data[np.where(self.label == 0)[0]][:, 0], self.data[np.where(self.label == 0)[0]][:, 1], 'ro',
                     label='class 0')
        else:
            plt.plot(self.data[np.where(self.label == -1)[0]][:, 0], self.data[np.where(self.label == -1)[0]][:, 1],
                     'ro',
                     label='class -1')
        plt.plot(self.data[np.where(self.label == 1)[0]][:, 0], self.data[np.where(self.label == 1)[0]][:, 1], 'bo',
                 label='class 1')
        plt.plot(decision_x, self._decision_function(decision_x), 'y-', label='decision surface')
        plt.ylabel(self.alg)
        plt.legend()
        plt.show()

    # 预测样本所属类别（主要是处理数据形状）
    def predict(self, x):
        # if x.size == 2:  # 为向量
        #     x.shape = (1, 2)
        # x = x.T
        if self.alg == 'fisher':
            # 这个返回的虽然是一维数组，但是shape是(N,1)，而不是(N,1)，在variable界面中显示的是[[True, False,...]]，
            # 所以第一个sum只会将其变为[1, 0,...]，外面再加一个sum才会得到一个整数
            return self._predict_rule(x)
        elif self.alg == 'perceptron' or self.alg == 'lms':
            x = np.vstack((np.ones(x.shape[1]), x))
            return self._predict_rule(x)
        elif self.alg == 'linearly_separable_svm':
            return self._predict_rule(x)
        elif self.alg == 'kernal_svm':
            return np.array([self._predict_func(x[i, :]) for i in range(x.shape[0])])
        elif self.alg == 'linear_svm':
            decision_x_1 = self._decision_function(x[:,0])
            return -2 * (x[:,1]>decision_x_1) + 1


    # 返回当前方法的准确率,如果不输入数据，就计算经验误差4
    def score(self, data=None, label=None):
        bool_str = self.predict(self.data)
        return sum(bool_str == self.label) / self.N

    # 感知机算法，没带余量（因为也不知道b取多大合适）
    def perceptron_fit(self):
        self.alg = 'perceptron'
        y = np.ones((self.data.shape[0], 1))
        y = np.hstack((y, self.data))
        y[np.where(self.label == 1), :] *= -1

        alpha = np.zeros(y.shape[1])
        err_exist = False  # 当前轮次判断是否有错分样本

        while 1:
            for t in range(self.data.shape[0]):
                if np.dot(alpha, y[t, :]) <= 0:
                    err_exist = True
                    alpha += y[t, :]

            if not err_exist:
                break

            err_exist = False

        # 直线y=f(x)的显式函数式
        self._decision_function = lambda x0: -1 * (alpha[0] + alpha[1] * x0) / alpha[2]
        # 返回布尔型数组，要注意如果写>=，那么ture（1）代表0类，false（0）代表1类
        self._predict_rule = lambda y: np.dot(alpha, y) < 0

        self.linear_equation = 'y=' + str(-alpha[1] / alpha[2]) + '*x+' + str(-alpha[0] / alpha[2])

    # 线性不可分数据集
    def lms_fit(self):
        self.alg = 'lms'
        y = np.ones((self.data.shape[0], 1))
        y = np.hstack((y, self.data))
        y[np.where(self.label == 1), :] *= -1
        b = np.ones((self.data.shape[0], 1))
        alpha = np.dot(np.dot(np.linalg.inv(np.dot(y.T, y)), y.T), b)

        # 直线y=f(x)的显式函数式
        self._decision_function = lambda x0: -1 * (alpha[0] + alpha[1] * x0) / alpha[2]
        # 返回布尔型数组，要注意如果写>=，那么ture（1）代表0类，false（0）代表1类
        self._predict_rule = lambda y: np.dot(alpha.T, y) < 0

        self.linear_equation = 'y=' + str((-alpha[1] / alpha[2])[0]) + '*x+' + str((-alpha[0] / alpha[2])[0])

    # 线性支持向量机
    def linear_svm(self, c, G=None):
        self.alg = 'linear_svm'
        self.kernal = Utils().linear_kernal
        if G is None:
            ub = self.N * [c]
        else:
            ub = np.array(self.N * [c]) * G

        # alpha_i 本身无上界，但是用differential_evolution的话，一方面不能用np.inf，必须用具体值很多alph_i会接近上限，那还不如确定个低上界
        bounds = Bounds(lb=self.N * [0.], ub=ub)
        nlc = NonlinearConstraint(lambda alpha: sum(self.label * alpha[:, np.newaxis])[0], 0, 0)
        # hess = -0.5 * self.label.dot(self.label.T) * self.data.dot(self.data.T)

        res = minimize(self._optimization_function, x0=self.N * [0.], constraints=(nlc), bounds=bounds)

        e = 1e-5
        alpha = res.x[:, np.newaxis]
        omega = sum(alpha * self.label * self.data)

        support_v_idxs = np.where((res.x > e) * (res.x < (c - e)))[0]
        b = 0
        for i in range(len(support_v_idxs)):
            b += 1 / self.label[support_v_idxs[i]] - sum(omega * self.data[support_v_idxs[i], :])
        b /= len(support_v_idxs)
        self._decision_function = lambda x0: (-b - x0 * omega[0]) / omega[1]

        z = 1 - self.label * (self.data.dot(omega[:, np.newaxis]) + b)
        epsilon = np.max(np.hstack((np.zeros_like(z), z)), axis=1)[:, np.newaxis]

        return omega, b

    # 生成svm的优化函数
    def _optimization_function(self, alpha):
        ans = 0
        alpha = alpha[:, np.newaxis]
        for i in range(self.N):
            ans += sum(
                alpha[i] * alpha * self.label[i] * self.label * self.kernal(self.data, self.data[i, :][:, np.newaxis]))
        ans = sum(alpha) - 0.5 * ans
        return -1 * ans[0]  # 注意书上要求的是最大值，但是scipy是求最小值

    # 核支持向量机,目前仅支持rbf核
    def kernal_svm(self, c, kernal=Utils.rbf_kernal):
        self.alg = 'kernal_svm'
        self.kernal = kernal
        bounds = Bounds(lb=self.N * [0], ub=self.N * [c])
        nlc = NonlinearConstraint(lambda alpha: sum(self.label * alpha[:, np.newaxis])[0], 0, 0)  # 0<=x<=0，其实就是等于0
        res = differential_evolution(func=self._optimization_function, constraints=(nlc), bounds=bounds)
        temp = sum(self.label * res.x[:, np.newaxis])

        alpha = res.x[:, np.newaxis]
        e = 1e-5
        support_v_idxs = np.where((res.x > e) * (res.x < (c - e)))[0]
        b = 0
        for i, idx in enumerate(support_v_idxs):
            b += 1 / self.label[idx] - sum(alpha * self.label * kernal(self.data, self.data[idx]))
        b /= len(support_v_idxs)

        z = []
        for i in range(self.N):
            z.append(1 - self.label[i] * (sum(alpha * self.label * kernal(self.data, self.data[i, :])) + b))
        z = np.array(z)
        epsilon = np.max(np.hstack((np.zeros_like(z), z)), axis=1)[:, np.newaxis]

        # 这个函数每次仅能预测一个样例
        self._predict_func = lambda x: np.sign(sum(alpha * self.label * kernal(self.data, x)) + b)
        self._boundary_func = lambda x: np.sign(sum(alpha * self.label * kernal(self.data, x)) + b + 1)
        self._boundary_func2 = lambda x: np.sign(sum(alpha * self.label * kernal(self.data, x)) + b - 1)

    # svm的边界曲线
    def boundary_func(self, x):
        return np.array([self._boundary_func(x[i, :]) for i in range(x.shape[0])])

    # svm的边界曲线2
    def boundary_func2(self, x):
        return np.array([self._boundary_func2(x[i, :]) for i in range(x.shape[0])])

class NonLinearClassifier:
    def __init__(self, x, y, eps):
        self.x = x[:, np.newaxis]
        self.y = y[:, np.newaxis]
        self.l = x.shape[0]
        self.eps = eps

        self.delta = 1e-6

    def _SVR_optimization_function(self, alpha):
        # alpha前self.l个元素为式（6-25）中的alpha*,后self.l个元素为alpha
        eq = np.sum(alpha[:self.l]) - np.sum(alpha[self.l:])
        alpha = alpha[:, np.newaxis]  # 不这样的话，后续广播机制可能会出错

        ans = -self.eps * np.sum(alpha) + np.sum(self.y * (alpha[:self.l] - alpha[self.l:]))
        temp = 0
        for i in range(self.l):
            temp = temp + (alpha[i] - alpha[self.l + i]) * np.sum((alpha[:self.l] - alpha[self.l:]) * self.x[i] * self.x)
        ans = ans - 0.5 * temp

        ans = -1 * ans
        return ans  # 注意书上要求的是最大值，但是scipy是求最小值

    def SVR(self, C):
        # 由于Bounds只能识别一维，所以将alpha*与alpha堆叠在一起
        # bounds = Bo unds(lb=np.zeros((2 * self.l)), ub=C * np.ones((2 * self.l)))
        # nlc = NonlinearConstraint(lambda alpha: np.sum(alpha[:self.l]) - np.sum(alpha[self.l:]), 0, 0)
        # res = differential_evolution(func=self._SVR_optimization_function, constraints=(nlc), bounds=bounds)
        # omega = np.sum((res.x[:self.l] - res.x[self.l:]) * self.x)

        cons = ({'type': 'eq', 'fun': lambda alpha: np.sum(alpha[:self.l]) - np.sum(alpha[self.l:])})
        bounds = [(0, C) for i in range(2 * self.l)]
        x0 = C * np.random.rand(2 * self.l)[:, np.newaxis]
        res = minimize(fun=self._SVR_optimization_function, x0=x0, bounds=bounds, constraints=cons)
        alpha_star = res.x[:self.l][:, np.newaxis]
        alpha = res.x[self.l:][:, np.newaxis]

        omega = np.sum((alpha_star - alpha) * self.x)  # 一定要添加新列！！！
        b_index = np.where((alpha > (0 + self.delta)) * (alpha < (C - self.delta)))[0]
        b = sum(self.y[b_index] + self.eps - omega * self.x[b_index]) / b_index.size

        return omega, b

# 原书第六章，神经网络，用于实现BP算法（反向传播计算梯度用的是sigmoid函数进行推导的）
# 说实话这个算法并不好使，但也懒得改了，所以直接运行肯定会有bug
class NN():
    def __init__(self, neurons):
        '''
        :param activate_fun: 激活函数
        :param neurons: list, 各层的神经元个数，注意n类的样本，输出的节点要为n（利用softmax函数）
        '''
        self.L = len(neurons)
        self.neurons = neurons
        # self.par[l]记录的是l层神经元与l+1层神经元连接的权值，而书上的w_l_i_j的l记录的是l-1层与l层连接的权值
        self.par = [np.random.rand(neurons[i], neurons[i + 1]) for i in range(self.L - 1)]  # 记录权值参数
        # self.x_i_l[l]记录的是第l层神经的输出
        self.x_i_l = [np.zeros((neurons[i], 1)) for i in range(self.L)]  # 记录神经元输出（第一个array用于占位）
        self.activate_fun = Utils().sigmoid
        self.loss = []

    # 正向传播计算输出，并记录每个神经元的输出
    def forward(self, x):
        if x.ndim == 1:
            self.x_i_l[0] = x[:, np.newaxis]
        else:
            self.x_i_l[0] = x.T

        for i in range(self.L - 1):
            self.x_i_l[i + 1] = self.activate_fun(self.par[i].T.dot(self.x_i_l[i]))

        return self.x_i_l[self.L - 1]

    # 反向传播计算梯度， 注意label要为独热向量哑结点
    def backward(self, data, label, yita=5, iter_times=500):
        idxs = np.random.randint(0, data.shape[0], iter_times)
        for iter in range(iter_times):
            yita = yita * 0.999 ** iter  # 最简单的学习率随训练次数下降
            idx = idxs[iter]
            x = data[idx]
            y = self.forward(x)
            d = label[idx][:, np.newaxis]
            l = self.L - 1
            old_par = self.par.copy()

            delta_j_l = (-y * (1 - y) * (d - y))
            self.par[l - 1] = self.par[l - 1] + (-1) * yita * self.x_i_l[l - 1].dot(delta_j_l.T)
            for l in range(l - 1, 0, -1):
                delta_j_l = self.x_i_l[l] * (1 - self.x_i_l[l]) * old_par[l].dot(delta_j_l)
                self.par[l - 1] = self.par[l - 1] + (-1) * yita * self.x_i_l[l - 1].dot(delta_j_l.T)

            self.loss.append(mean_absolute_error(label, self.forward(data).T))

    def predict(self, data):
        out = self.forward(data).T
        return np.array([np.where(out[i, :] == max(out[i, :]))[0][0] for i in range(out.shape[0])])


# 原书第11章，非参数学习机器与集成学习，不考虑决策树（画图太费劲）,欧式距离，2维向量，c类
class NonParLearning:
    def __init__(self, method, data, c, label=None):
        """
        :param data: 特征向量
        :param c: 类别数
        :param label: 分类标签（注意这里是非参数学习，不是无监督）,为0,1,……
        :param method: 分类方法
        """
        assert method in ['nearest_neighbor', 'k_neighbor', 'clip_neighbor', 'branch_bound']
        self.method = method
        self.data = data
        self.label = label
        self.c = c
        self.N = self.data.shape[0]
        self.dim = self.data.shape[1]

    class BbaTree:
        # 其实如果只是添加几个属性的话，感觉甚至可以在基础Node类上用一下eval函数
        class BbaNode(tc.Node):
            def __init__(self, val, Np, Mp, rp, idxs, L, reason=None, sub_i=None):
                # reason实际上是个占位的参数，为了和基类Node中的reason保持一致
                # 注意这里的val，实际上是数据data所对应的索引（array类型）
                # 一个父节点生成l个子节点，sub_i是这里面第i个
                super(NonParLearning.BbaTree.BbaNode, self).__init__(val=val)
                self.idxs = idxs[sub_i]
                self.Np = Np[sub_i]
                self.Mp = Mp[sub_i]
                self.rp = rp[sub_i]
                self.L = L  # 记录当前水平数

    def _k_neighbor_transform(self, x, k):
        y = -1 * np.ones(self.N, dtype=np.int8)
        label = -1 * np.ones_like(x[:, 0], dtype=np.int8)
        for idx in range(self.N):
            dist = Utils.euc_dist(x[idx, :], self.data)
            label_count = [sum(self.label[dist.argsort()[:k]] == i) for i in range(self.c)]
            label[idx] = np.where(label_count == max(label_count))[0][0]
        return label

    def _branch_bound_alg_transform(self, data, l=3, L=3):
        self.test_data = data
        label = -1 * np.ones_like(data[:, 0], dtype=np.int8)
        self.l, self.L = l, L
        bba_tree = self._bba_tree(l, L)
        for i, x in enumerate(data):
            label[i] = self._bba_single_label(x, bba_tree)
        return label

    # P179的树搜索算法比较复杂，且针对的是单点的，为了明晰，把这个单独提出来
    def _bba_single_label(self, x, bba_tree):
        # do step 1
        B, L = np.inf, 0
        p = bba_tree.root
        not_terminate = True
        # NN_array = -1 * np.ones(k, dtype=np.int32)
        # d_x_xi_array = np.array([np.inf for i in range(k)])

        cal_node_list = []
        succeed_nodes = []

        while not_terminate:
            # do step 2
            cur_node = p
            for child in cur_node.child:
                succeed_nodes.append(child)  # list of str
            d_x_Mp = np.array([Utils.euc_dist(x[np.newaxis, :], node.Mp[np.newaxis, :]) for node in succeed_nodes])
            a=1
            while 1:
                # do step 3
                for i in range(len(succeed_nodes) - 1, -1, -1):
                    if d_x_Mp[i] > B + succeed_nodes[i].rp:
                        del succeed_nodes[i]
                        d_x_Mp = np.delete(d_x_Mp, i)
                if len(succeed_nodes) == 0:
                    # do step4 （回退水平）
                    L -= 1
                    if L == 0:  # 若不满足终止条件，则再一次重新开始内部循环（do step3）
                        not_terminate = False  # break退出内部while，改变布尔值退出外部while
                        break
                else:  # 不满足do step 4条件，跳转step 5
                    # do step 5
                    p_exe_d = min(d_x_Mp)
                    node_idx = np.where(d_x_Mp == p_exe_d)[0][0]
                    p_exe = succeed_nodes[node_idx]
                    del succeed_nodes[node_idx]
                    d_x_Mp = np.delete(d_x_Mp, node_idx)
                    if (p_exe.depth - 1) == self.L:  # L == self.L
                        # do step 6
                        cal_node_list.append(p_exe.val)
                        ''' 下面的语句更符合书上的描述，但是实在太慢了（且是k=1），所以改成下面的利用np广播机制的代码
                        for i, idx in enumerate(p_exe.idxs):
                            if p_exe_d <= Utils.euc_dist(self.data[idx], p_exe.Mp) + B:
                                d_x_xi = Utils.euc_dist(x, self.data[idx])
                                if d_x_xi < B:  # 其实就是找满足d_x_xi<B中，d_x_xi的最小值
                                    NN = idx
                                    B = d_x_xi
                        ################################################失败版任意k      
                        temp_idx = p_exe.idxs[
                            np.where((p_exe_d <= Utils.euc_dist(self.data[p_exe.idxs], p_exe.Mp) + B))[0]]
                        d_x_xi = Utils.euc_dist(x, self.data[temp_idx])
                        if (d_x_xi < B).any():
                            # 求出满足d_x_xi < B的样本
                            possible_idx = temp_idx[np.where(d_x_xi < B)[0]]
                            possible_d = d_x_xi[np.where(d_x_xi < B)[0]]
                            # 如果有第k个最近邻，B就是那个，否则就是最远的那个
                            possible_B = np.sort(possible_d)[k-1] if possible_d.shape[0] >= k else np.sort(possible_d)[-1]
                            if possible_B < max(d_x_xi_array):
                                B = possible_B
                                replace_idx = np.where(d_x_xi_array == max(d_x_xi_array))[0][0]
                                d_x_xi_array[replace_idx] = possible_B
                                NN_array[replace_idx] = possible_idx[possible_d==possible_B]
                        '''
                        temp_idx = p_exe.idxs[
                            np.where((p_exe_d <= Utils.euc_dist(self.data[p_exe.idxs], p_exe.Mp) + B))[0]]
                        d_x_xi = Utils.euc_dist(x, self.data[temp_idx])
                        if (d_x_xi < B).any():
                            B = min(np.where(d_x_xi < B, d_x_xi, B))
                            NN = temp_idx[np.where(d_x_xi == B)[0]]
                        # 运行完step 6之后，再次跳转内部循环(再次step 3)
                    else:  # 不满足step6条件，跳出内部循环(再次step 2)
                        L += 1
                        p = p_exe
                        break

        # 检查该样本点是否正确
        # temp = Utils.euc_dist(x, self.data)
        # if not min(temp) == temp[NN][0]:
        #     print('计算错误'.center(100, '*'))
        #     cal_idx = NN[0]
        #     real_idx = np.where(temp == min(temp))[0]
        #
        #     traverse_instance = tree_class.Find(bba_tree.root)
        #     cal_node = traverse_instance.preorder_idx(cal_idx)
        #     real_node = traverse_instance.preorder_idx(real_idx)
        #
        #     print(f'当前x计算出的的最近邻为样本为：{cal_idx}；计算节点为{cal_node.val}')
        #     print(f'当前x真实的最近邻为样本为：{real_idx}；真实节点为{real_node.val}')
        #
        # else:
        #     print('计算正确'.center(100, '*'))
        return self.label[NN]

    def transform(self, x, k=1):
        # x为未知数据集，k为近邻法
        if self.method == 'nearest_neighbor':
            label = self._k_neighbor_transform(x, 1)
        elif self.method == 'k_neighbor':
            label = self._k_neighbor_transform(x, k)
        elif self.method == 'branch_bound':
            label = self._branch_bound_alg_transform(x)

        return label

    def _bba_tree(self, l, L):
        # 加上根节点，一共L+1层
        # 每个节点再分为l个子集
        start_node_nums = np.cumsum([0] + [l ** i for i in range(L + 1)])  # 从L=1开始，每一层的起始节点标号
        parent_start_node_num = 0
        idxs = [np.arange(self.N, dtype=np.int32)]  # 是为了_cal_Np_Mp_rp函数中代码的同一
        Np, Mp, rp = self._cal_Np_Mp_rp(idxs)
        bba_tree = tc.TreeGenerate(val='0', node_type=NonParLearning.BbaTree.BbaNode, Np=Np, Mp=Mp,
                                   rp=rp, L=0, idxs=idxs, sub_i=0)
        for i in range(L):
            parent_start_node_num += l ** i
            parent_strs = [str(i) for i in np.arange(start_node_nums[i], start_node_nums[i + 1])]
            child_strs = [str(i) for i in np.arange(start_node_nums[i + 1], start_node_nums[i + 2])]
            start_idx = 0
            for j in range(len(parent_strs)):
                parent_idxs = tc.Find(bba_tree.root).preorder(parent_strs[j]).idxs
                idxs_cal = self._cal_idxs(parent_idxs)
                Np, Mp, rp = self._cal_Np_Mp_rp(idxs_cal)
                bba_tree.add_child(parent_str=parent_strs[j], childs_str=child_strs[start_idx:start_idx + l],
                                   node_type=NonParLearning.BbaTree.BbaNode, Np=Np, Mp=Mp,
                                   rp=rp, L=0, idxs=idxs_cal)
                start_idx += l
        return bba_tree

    # 将当前集合的索引，再均匀分为l份
    def _cal_idxs(self, idxs):
        # 不知道当初为啥这样写
        # instance = Cluster(self.test_data[idxs, :], 'c_mean')
        # label = instance.transform(self.l)
        # ans_list = [idxs[np.where(label == i)] for i in range(self.l)]

        temp = idxs.copy()
        np.random.shuffle(temp)
        step = int(idxs.shape[0] / self.l)
        ans_list = [temp[i*step:(i+1)*step] for i in range(self.l - 1)]
        ans_list.append(temp[(self.l-1)*step:])

        return ans_list

    def _cal_Np_Mp_rp(self, idxs):
        # idxs: list of array
        cal_Np = lambda idx: idx.shape[0]
        cal_Mp = lambda idx: np.mean(self.data[idx, :], axis=0)
        cal_rp = lambda idx: max(Utils.euc_dist(cal_Mp(idx)[np.newaxis, :], self.data[idx]))
        Np, Mp, rp = [], [], []
        for i in range(len(idxs)):
            Np.append(cal_Np(idxs[i]))
            Mp.append(cal_Mp(idxs[i]))
            rp.append(cal_rp(idxs[i]))

        return Np, Mp, rp


# 第11章聚类，都是二维聚类
class Cluster:
    def __init__(self, x, method, ini_centers=None):
        self.x = x
        self.N = x.shape[0]
        self.dim = x.shape[1]
        assert method in ['c_mean']
        self.method = method
        self.ini_centers = ini_centers


    def transform(self, c=None, N=10):
        if self.method == 'c_mean':
            label = self._c_mean_transorm(c, N=N)

        return label

    def _c_mean_transorm(self, c, N):
        self.c = c
        if self.ini_centers is None:
            m = self._first_center()
        else:
            m = self.ini_centers

        label = self._first_cluster(m)
        cluster_sample_nums = self._cal_cluster_sample_nums(label)

        unchange_nums = 0
        cal_rho_j = lambda N_j, y, m_j: N_j / (N_j + 1) * Utils.euc_dist(y, m_j) ** 2
        cal_rho_i = lambda N_i, y, m_j: N_i / (N_i - 1) * Utils.euc_dist(y, m_j) ** 2
        while 1:
            # 随机抽取一个样本
            while 1:
                idx = np.random.randint(low=0, high=self.N, size=1)
                i = label[idx]  # 该样本类别
                y = self.x[idx, :]  # 该样本样本坐标
                if cluster_sample_nums[i] > 1:  # 如果N_i=1，则重新取样本点
                    break

            rho_js = np.empty((self.c - 1, 1))
            rho_js_label = np.empty((self.c - 1, 1))
            iter = 0
            for j in range(self.c):
                if j == i:
                    continue
                rho_js[iter] = cal_rho_j(cluster_sample_nums[j], y, m[j][np.newaxis, :])
                rho_js_label[iter] = j
                iter += 1
            rho_i = cal_rho_i(cluster_sample_nums[i], y, m[i])

            if min(rho_js) < rho_i:
                k = rho_js_label[np.where(rho_js == min(rho_js))[0][0]]
                label[idx] = k
                m = self._cal_center(label)

                cluster_sample_nums[i] -= 1
                cluster_sample_nums[int(k)] += 1
                unchange_nums = 0
            else:
                unchange_nums += 1

            if unchange_nums == N:
                break
        return label

    def _first_center(self):
        # P249页，方法二
        label = np.random.randint(low=0, high=self.c, size=self.N)
        return self._cal_center(label)

    # 给定label，计算各类的中点
    def _cal_center(self, label):
        centers = np.empty((self.c, self.dim))
        for i in range(self.c):
            idxs = np.where(label == i)
            centers[i, :] = np.sum(self.x[idxs, :][0], axis=0) / idxs[0].size
        return centers

    def _first_cluster(self, centers):
        # P2491，且是欧式距离
        label = -1 * np.ones(self.N, dtype=np.int8)
        for i in range(self.N):
            dist = Utils.euc_dist(self.x[i, :][np.newaxis, :], centers)
            label[i] = np.where(dist == min(dist))[0]
        return label

    # 计算各类样本的数目
    def _cal_cluster_sample_nums(self, label):
        return np.array([sum(label == i) for i in range(self.c)])


# 原书8.5 Boosting集成学习
class Bossting:
    # 均是使用Fisher判别分类器
    def __init__(self, data, label, M, eps=0.45, ini_rho=0.1, max_iter=1000):
        self.data = data
        self.label = label
        self.N = data.shape[0]
        self.y_original = np.hstack((np.array(self.N * [1])[:, np.newaxis], data))
        self.y = np.hstack((np.array(self.N * [1])[:, np.newaxis], data))
        self.y[np.where(self.label == -1)[0]] = -1 * self.y[np.where(self.label == -1)[0]]  # w2=-1
        self.M = M
        self.N = data.shape[0]
        self.omega = np.array(self.N*[1/self.N])[:, np.newaxis]
        self.eps = eps
        self.rho = lambda iter: ini_rho + iter * (-0.9 * ini_rho) / max_iter  # 线性减少的学习率，最终学习率为原来的1/10
        self.max_iter = max_iter
        self.classifiers = dict(alpha=[], cm=[], acc=[])
        self._mse = lambda alpha: np.sum(self.omega * (self.y.dot(alpha) - 1)**2)

    def lms(self):
        alpha = np.random.rand(self.data.shape[1] + 1, 1)
        iter_times = 0
        run_while = True
        while run_while:
            for i, y in enumerate(self.y):
                y = y[:, np.newaxis]
                if np.abs(1 - (y.T).dot(alpha)) > 1e-6:  # 但其实我就没见<=的
                    alpha = alpha + self.rho(iter_times) * (1 - (y.T).dot(alpha)) * y * self.omega[i]
                    iter_times += 1

                if (self._mse(alpha) < self.eps) or (iter_times >= self.max_iter):
                    run_while = False
                    break

        return alpha

    def predict(self, alpha):
        predict_label = np.sign(self.y_original.dot(alpha))
        return predict_label, accuracy_score(self.label, predict_label)

    def start_boost(self):
        for i in range(self.M):
            alpha = self.lms()
            self.classifiers['alpha'].append(alpha)

            predict_label, acc = self.predict(alpha)
            self._plot_with_alpha(alpha, acc, predict_label)

            em = 1 - acc
            cm = np.log((1 - em) / em)
            self.classifiers['acc'].append(acc)
            self.classifiers['cm'].append(cm)

            indictor_idx = (predict_label != self.label[:, np.newaxis])
            self.omega = self.omega * np.exp(cm * indictor_idx)
            self.omega = self.omega / sum(self.omega)

            a=1

    def boost_predict(self):
        f_m_x = np.empty([self.N, self.M])
        for i in range(self.M):
            f_m_x[:, i][:, np.newaxis] = self.predict(self.classifiers['alpha'][i])[0]

        cm_array = np.array(self.classifiers['cm'])[np.newaxis, :]
        acc_array = np.array(self.classifiers['acc'])[np.newaxis, :]

        predict_label = np.sign(np.sum(cm_array * f_m_x, axis=1))

        temp = cm_array * f_m_x
        cm_array = cm_array.T
        acc_array = acc_array.T
        real_label = self.label[:, np.newaxis]
        predict_label = predict_label[: ,np.newaxis]


        return predict_label, accuracy_score(self.label, predict_label)


    def _plot_with_alpha(self, alpha, acc, predictlabel):
        k = - alpha[1] / alpha[2]
        b = - alpha[0] / alpha[2]

        data = self.data
        label = self.label
        left_right = np.array([1.1 * min(data[:, 0]), 1.1 * max(data[:, 0])])
        data0 = data[np.where(label == -1)[0]]
        data1 = data[np.where(label == 1)[0]]
        f = lambda x: k * x + b
        label_use_decision = np.sign(f(data[:, 0])-data[:, 1])
        decision_acc = sum(label_use_decision == label) / self.N

        plt.figure()
        plt.plot(data1[:, 0], data1[:, 1], '^', label='class 1 (w_1)')
        plt.plot(data0[:, 0], data0[:, 1], 'o', label='class -1 (w_2)')
        plt.plot(left_right, f(left_right), label='decision curve')

        plt.title(f'acc:{acc}; decision_acc: {decision_acc}')
        plt.legend()

        plt.show()


# 第九章，特征选择
class FeatureSelection:
    def __init__(self, data, label, d, prior):
        self.data = data
        self.label = label
        self.d = d
        self.prior = prior

        self.N = self.data.shape[0]
        self.D = self.data.shape[1]
        self.c = len(np.unique(label))


    def J1(self, x):
        x_0 = x[np.where(self.label == -1)[0]]
        x_1 = x[np.where(self.label == 1)[0]]

        mu = np.mean(x, axis=0)[:, np.newaxis]
        mu_0 = np.mean(x_0, axis=0)[:, np.newaxis]
        mu_1 = np.mean(x_1, axis=0)[:, np.newaxis]

        S_b = self.prior[0] * (mu_0 - mu).dot((mu_0 - mu).T)
        S_b += self.prior[1] * (mu_1 - mu).dot((mu_1 - mu).T)

        temp = 0
        for i, x_k in enumerate(x_0):
            x_k = x_k[:, np.newaxis]
            temp += (x_k - mu_0).dot((x_k - mu_0).T)
        temp = temp / x_0.shape[0]
        S_w = self.prior[0] * temp

        temp = 0
        for i, x_k in enumerate(x_1):
            x_k = x_k[:, np.newaxis]
            temp += (x_k - mu_1).dot((x_k - mu_1).T)
        temp = temp / x_1.shape[0]
        S_w += self.prior[1] * temp

        J1 = np.trace(S_b + S_w)

        return J1

# 第11章 非监督学习与聚类
# 单峰子集分离
class UnimodalSeperate:
    def __init__(self, data0, data1, data2, label):
        self.data0 = data0
        self.data1 = data1
        self.data2 = data2
        self.data = np.vstack((data0, data1, data2))
        self.N = self.data.shape[0]
        self.label = label
        self.mean = None
        self.df = None

        left, bottom = np.min(self.data, axis=0)[0], np.min(self.data, axis=0)[1]
        right, top = np.max(self.data, axis=0)[0], np.max(self.data, axis=0)[1]
        horizontal_step = (right - left) / 10
        vertical_step = (top - bottom) / 10
        left -= horizontal_step
        right += horizontal_step
        bottom -= vertical_step
        top += vertical_step

        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top


    def my_plot(self, fun0, fun1, fun_label):
        plt.figure()
        plt.plot(self.data0[:, 0], self.data0[:, 1], 'r^', label='class 0', alpha=0.1)
        plt.plot(self.data1[:, 0], self.data1[:, 1], 'gv', label='class 1', alpha=0.1)
        plt.plot(self.data2[:, 0], self.data2[:, 1], 'bx', label='class 2', alpha=0.1)

        plt.plot([self.left, self.right], [fun0(self.left), fun0(self.right)],  'k', label=fun_label + '0')
        plt.plot([self.left, self.right], [fun1(self.left), fun1(self.right)], 'c--', label=fun_label + '1')

        plt.legend(loc='upper left')
        plt.xlim(self.left, self.right)
        plt.ylim(self.bottom, self.top)
        plt.show()

    def pca(self):
        mean = np.mean(self.data, axis=0)[np.newaxis, :]
        self.mean = mean

        data_shift = self.data - mean
        cov = data_shift.T.dot(data_shift) / (self.N - 1)
        eig, vec = np.linalg.eig(cov)

        dic = {'eig': [eig[0], eig[1]],
               'vec': [vec[:, 0], vec[:, 1]]}
        df = pd.DataFrame(dic)
        df_sort = df.sort_values(by='eig', ascending=False)
        self.df = df_sort

    def hist_estimation(self, direction):
        v = self.data.dot(self.df['vec'].iloc[direction][:, np.newaxis])
        cabin_num = 100
        v0_sort = np.sort(v, axis=0)

        plt.show()
        n, bins, patches = plt.hist(v0_sort, cabin_num, density=True, facecolor='g', alpha=0.75)
        plt.xlabel('Projected data')
        plt.ylabel('Probability')
        plt.title(f'Direction {direction}')
        plt.grid(True)
        plt.show()

        return n, bins

    def decision_line(self, direction, p_v, section):
        vec0 = self.df['vec'].iloc[direction]
        v = self.data.dot(vec0[:, np.newaxis])
        possible_section = list(map(int,input('请输入密度函数波谷的所在区间（空格分割）：').split(" ")))
        assert possible_section[0] < possible_section[1], '区间作值应小于右值'
        dic = {'p_v': [item for item in p_v],
               'left': [item for item in section[:-1]],
               'right': [item for item in section[1:]]}
        df = pd.DataFrame(dic)
        idx_start = np.where((df['left']>possible_section[0])==True)[0][0]
        idx_end = np.where((df['right']<possible_section[1]) == True)[0][-1]

        min_p_v = min(df['p_v'][idx_start:(idx_end + 1)])
        min_idx = np.where(df['p_v']==min_p_v)[0][0]
        min_projected_val = (df['left'][min_idx] + df['right'][min_idx])/2

        fun = lambda x: vec0[0] / vec0[1] * (x - self.mean[0][0]) + self.mean[0][1]
        return fun

    def run(self):
        self.pca()

        vec0 = self.df['vec'].iloc[0]
        fun0 = lambda x: vec0[0]*(self.mean[0][0] - x) / vec0[1] + self.mean[0][1]
        vec1 = self.df['vec'].iloc[1]
        fun1 = lambda x: vec1[0]*(self.mean[0][0] - x) / vec1[1] + self.mean[0][1]
        self.my_plot(fun0, fun1, 'project direction')

        direction = 0
        p_v, section = self.hist_estimation(direction)
        fun0 = self.decision_line(direction, p_v, section)

        direction = 1
        p_v, section = self.hist_estimation(direction)
        fun1 = self.decision_line(direction, p_v, section)

        self.my_plot(fun0, fun1, 'decision_line')







