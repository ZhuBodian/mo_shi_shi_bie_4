import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, log
from scipy.optimize import minimize
from PrivateUtils import util
from PrivateUtils import plot_util
import pandas as pd
import math
from numpy import sqrt, exp, log, max
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint, Bounds, LinearConstraint
from sklearn import datasets
from sklearn.metrics import confusion_matrix

IMAGEFOLDER = '../images/05_linear_model_and_liner_classifier/'


# 原书第五章，线性分类器，均只考虑二维两类
class LinearClassifier:
    """
    感知机与lms太简单了，就不写了
    """
    def __init__(self, data, label, prior=np.array([0.5, 0.5])):
        self.data = data
        self.N = data.shape[0]
        self.dim = data.shape[1]
        if label.ndim == 1:
            self.label = label[:, np.newaxis]
        else:
            self.label = label
        self.prior = prior
        self.decision_function = None  # 二类二维为一条直线的函数（因为形式简单，就写为了显式y=f(x)）
        self._predict_rule = None  # 返回x是否满足式(5-48)大于号的bool数组,要给出包含横纵坐标的向量(注意为列向量)
        self.linear_equation = None

    def fisher_fit_predict(self):
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
        mean0 = mean0.T
        mean1 = mean1.T

        self.decision_function = lambda x: (np.log(self.prior[1] / self.prior[0]) +
                                            (omega[1] * (mean0[1] + mean1[1])) / 2 -
                                            omega[0] * (x - (mean0[0] + mean1[0]) / 2)) / omega[1]

        # 其实这个谁是第一类谁是第二类整的很乱，但试一试就知道了
        is_class_1 = (self.decision_function(self.data[:, 0]) > self.data[:, 1])
        predicted_label = is_class_1 * 2 - 1

        return predicted_label

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

    # 线性支持向量机
    def linear_svm(self, c, G=None):
        self.kernal = util.linear_kernal
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
        self.decision_function = lambda x0: (-b - x0 * omega[0]) / omega[1]
        self.boundary_functions = [lambda x0: (-b - 1 - x0 * omega[0]) / omega[1],
                                   lambda x0: (-b + 1 - x0 * omega[0]) / omega[1]]

        z = 1 - self.label * (self.data.dot(omega[:, np.newaxis]) + b)
        epsilon = np.max(np.hstack((np.zeros_like(z), z)), axis=1)[:, np.newaxis]

        is_class_1 = (self.decision_function(self.data[:, 0]) < self.data[:, 1])
        predicted_label = is_class_1 * 2 - 1

        return predicted_label

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
    def kernal_svm(self, c, kernal=util.rbf_kernal):
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


def generate_linear_classified_data():
    n_samples = 500
    centers = [[1, 1], [-1, -1]]
    std = 0.4
    data, label = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=centers, cluster_std=std)
    label = 2 * label - 1  # 0标签转为-1标签

    return data, label


def generate_linear_classified_data2():
    data = np.array([[0, 0], [1, 1], [1, 2], [2, 1], [4, 1], [4, 2], [2, 3], [3, 2]])
    label = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    return data, label


def generate_linear_unclassified_data():
    n_samples = 500
    centers = [[1, 1], [-1, -1]]
    std = 1.2
    data, label = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=centers, cluster_std=std)
    label = 2 * label - 1

    return data, label


def generate_linear_unclassified_data2():
    data = np.array([[3, 4], [0, 0], [1, 1], [1, 2], [3, 1], [4, 1], [4, 2], [2, 3], [3.5, 2], [1.5, 0], [0, 3],
                     [4, 0]])
    label = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1])

    return data, label


def run_fisher():
    print('Fisher线性变换'.center(100, '*'))

    print('线性可分数据集'.center(50, '*'))
    data, label = generate_linear_classified_data()
    prior = np.array([0.5, 0.5])
    instance = LinearClassifier(data, label, prior)
    predicted_label = instance.fisher_fit_predict()

    plot_sample_and_decision_curve(data, label, instance.decision_function, 'fisher linear classified')

    print('混淆矩阵：')
    print(confusion_matrix(label, predicted_label).T)

    print('线性不可分数据集'.center(50, '*'))
    data, label = generate_linear_unclassified_data()
    prior = np.array([0.5, 0.5])
    instance = LinearClassifier(data, label, prior)
    predicted_label = instance.fisher_fit_predict()

    plot_sample_and_decision_curve(data, label, instance.decision_function, 'fisher linear unclassified')

    print('混淆矩阵：')
    print(confusion_matrix(label, predicted_label).T)


def plot_sample_and_decision_curve(data, label, decision_fun, title, plot_boundary_lines=False, boundary_fun=None):
    left, right, _, _ = plot_util.plot_bound(data)
    left_right = np.array([left, right])
    data0 = data[np.where(label == -1)[0]]
    data1 = data[np.where(label == 1)[0]]

    plt.figure()
    plt.plot(data0[:, 0], data0[:, 1], 'ro', label='class -1')
    plt.plot(data1[:, 0], data1[:, 1], 'b^', label='class 1')
    plt.plot(left_right, decision_fun(left_right), label='decision curve')
    if plot_boundary_lines:
        plt.plot(left_right, boundary_fun[0](left_right), 'g--', label='boundaries line')
        plt.plot(left_right, boundary_fun[1](left_right), 'g--')

    plt.xlabel('x_0')
    plt.ylabel('x_1')
    plt.title(title)
    plt.legend(frameon=True)

    plot_util.save_fig(IMAGEFOLDER, title)
    plt.show()


def run_svm(c):
    print('支持向量机'.center(100, '*'))

    print('线性可分数据集'.center(50, '*'))
    data, label = generate_linear_classified_data2()
    instance = LinearClassifier(data, label)
    predicted_label = instance.linear_svm(c=c)

    plot_sample_and_decision_curve(data, label, instance.decision_function, f'svm linear classified, c={c}', True,
                                   instance.boundary_functions)

    print('混淆矩阵：')
    print(confusion_matrix(label, predicted_label).T)

    print('线性不可分数据集'.center(50, '*'))
    data, label = generate_linear_unclassified_data2()
    instance = LinearClassifier(data, label)
    predicted_label = instance.linear_svm(c=c)

    plot_sample_and_decision_curve(data, label, instance.decision_function, f'svm linear unclassified, c={c}', True,
                                   instance.boundary_functions)

    print('混淆矩阵：')
    print(confusion_matrix(label, predicted_label).T)


if __name__ == '__main__':
    np.random.seed(42)

    # run_fisher()

    # for c in [0.5, 1, 10, 100]:
    #     run_svm(c)

