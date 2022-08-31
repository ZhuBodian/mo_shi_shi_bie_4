import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, log
from scipy.optimize import minimize
from PrivateUtils import util
from PrivateUtils import plot_util
import pandas as pd
import math
from numpy import sqrt, exp, log, max


IMAGEFOLDER = '../images/03_PDF_Estimation/'
Pi = math.pi
epsilon = 1e-100


class BE:
    def __init__(self, p_x_theta, p_theta, x, theta):
        """
        :param p_x_theta:
        :param p_theta:
        :param theta: theta空间的采样点
        """
        self.p_x_theta = p_x_theta
        self.p_theta = p_theta

        self.x = x
        self.theta = theta

        self.p_theta_H = None

    # 取对数计算并添加epsilon防止下溢，且由于是离散计算，式（3-34）分母改为求和
    def cal_theta_posterior(self):
        a_j = np.zeros_like(self.theta)  # 术语参考同文件夹ipynb
        for i, theta in enumerate(self.theta):
            a_j[i] = sum(log(self.p_x_theta(self.x, theta) * self.p_theta(theta) + epsilon))

        p_theta_H = exp(a_j - max(a_j) - log(sum(exp(a_j-max(a_j))) + epsilon))  # array和为1，间接证明变换正确
        self.p_theta_H = p_theta_H

        return p_theta_H

    def cal_theta_hat(self):
        return sum(self.theta * self.p_theta_H)


def run_BE(x_nums=3000, theta_nums=1000):
    print(f'x_nums={x_nums}, theta_nums={theta_nums}'.center(100, '*'))

    # 以3.3.3中正态分布时的贝叶斯估计为例
    sig2 = 10  # 模型的方差
    mu_0 = 0  # 未知参数mean的分布均值
    sig2_0 = 0.1  # 未知参数mean的分布方差，越小越确定，先验越强(式3-51中mu_0系数越大)

    x_sample, theta_sample = generate_uncertain_norm_points(x_nums, theta_nums, sig2, mu_0, sig2_0)
    plot_x_theta_sample(x_sample, theta_sample)

    p_x_mu = lambda x, mu: 1 / sqrt(2 * Pi * sqrt(sig2)) * exp(-1 / (2 * sig2) * (x - mu) ** 2)
    p_mu = lambda mu: 1 / sqrt(2 * Pi * sqrt(sig2_0)) * exp(-1 / (2 * sig2_0) * (mu - mu_0) ** 2)
    # 式(3-51)
    mu_N = (x_nums * sig2_0) / (x_nums * sig2_0 + sig2) * np.mean(x_sample) + sig2 / (x_nums * sig2_0 + sig2) * mu_0

    instance = BE(p_x_mu, p_mu, x_sample, theta_sample)
    instance.cal_theta_posterior()
    theta_hat = instance.cal_theta_hat()

    print(f'式3-51中，采样均值加权：{abs((x_nums * sig2_0) / (x_nums * sig2_0 + sig2)) * 100:.2f}%；'
          f'先验加权：{abs(sig2 / (x_nums * sig2_0 + sig2)) * 100:.2f}%')
    print(f'观测均值m_N：{np.mean(x_sample)}；先验mu_0：{mu_0}')
    print(f'数值计算mu：{theta_hat}；理论计算mu_N：{mu_N}')
    print(f'数值计算与式3-51误差：{abs((mu_N - theta_hat) / (mu_N)) * 100:.2f}%')


# 3.3.3中的模型方差确定，均值不定，这里模拟这种均值不确定的采样
def generate_uncertain_norm_points(x_nums, theta_nums, sig2, mu_0, sig2_0):
    mu_array = np.random.normal(mu_0, sqrt(sig2_0), x_nums)

    # 注意这里并没有返回生成x样本的均值样本，因为这样可能会引入相关性（我瞎猜的）
    return np.sort(np.random.normal(mu_array, sqrt(sig2), x_nums)), \
           np.sort(np.random.normal(mu_0, sqrt(sig2_0), theta_nums))


def plot_x_theta_sample(x_sample, theta_sample):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(x_sample, bins=100, density=True, label='Histogram', alpha=0.6, color='orange')
    x_series = pd.Series(x_sample)
    x_series.plot(kind='kde', label='Probability Density')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(theta_sample, bins=200, density=True, label='Histogram', alpha=0.6, color='orange')
    theta_series = pd.Series(theta_sample)
    theta_series.plot(kind='kde', label='Probability Density')
    plt.legend()
    plt.xlabel('theta')
    plt.ylabel('Probability')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    plot_util.save_fig(IMAGEFOLDER, f'x与mu采样图, x_sample={len(x_sample)}')
    plt.show()


class NonParmEstimation:
    def __init__(self, x):
        self.x = x
        self.n = len(self.x)

        self.k = None
        self.k_n = None
        self.h =None

    def hist(self, k):
        self.k = k

        plt.figure()
        plt.hist(self.x, bins=k, density=True, label='Histogram', alpha=0.6, color='orange')
        x_series = pd.Series(self.x)
        x_series.plot(kind='kde', label='Probability Density')
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.legend()
        plt.title(f'n={len(self.x)}, k={k}')

        plot_util.save_fig(IMAGEFOLDER, f'直方图方法，n={self.n}，k={k}')
        plt.show()

    def k_n_neighbor(self, k_n):
        self.k_n = k_n

        iter_times = int(self.n / self.k_n)
        pdf = np.zeros((iter_times, 2))  # 第一维记录了小舱中点坐标，第二维记录了该点对应的概率值
        # bins_width = np.zeros(iter_times)
        for i in range(iter_times):  # 就算是小数，也是向下取整，for循环之后再考虑余数的问题,循环完后，sum(bins_width * pdf[:, 1])=1
            start = i * self.k_n
            end = (i + 1) * self.k_n - 1

            pdf[i, 0] = (self.x[start] + self.x[end]) / 2
            pdf[i, 1] = self.k_n / self.n / (self.x[end] - self.x[start])

            # bins_width[i] = self.x[end] - self.x[start]

        if np.mod(self.n, self.k_n) != 0:
            if np.mod(self.n, self.k_n) == 1:
                self.n -= 1  # 余数为1没有区间的概念，抛弃该点
            else:
                # 虽然的样本数不足以构成一个小舱，还是将它们放到一个小舱内
                temp_x = (self.x[end + 1] + self.x[-1]) / 2
                temp_y = (self.n - iter_times * self.k_n) / self.n / (self.x[-1] - self.x[end + 1])
                temp = np.array([temp_x, temp_y])
                pdf = np.vstack((pdf, temp))

        # 保证定义域不变（因为pdf是小舱中点，所以抛弃了样本最两端的位置信息）
        pdf = np.vstack((np.array([self.x[0], 0]), pdf))
        pdf = np.vstack((pdf, np.array([self.x[-1], 0])))

        plt.figure()

        plt.plot(pdf[:, 0], pdf[:, 1], label='PDF')
        plt.xlabel('x')
        plt.ylabel('probability')
        plt.title(f'k_N neighbor, n={self.n}, k_n={self.k_n}')
        plt.legend()

        plot_util.save_fig(IMAGEFOLDER, f'k_n近邻法, n={self.n}, k_n={self.k_n}')
        plt.show()

    def parzen(self, h, line_sample_n=10000):
        """
        仅考虑方窗函数，d=1，式子3-70中的x_i是总体样本（例子中是正态模型采样），x不必和x_i一致，可以单独采样（均匀采样）
        """
        self.h = h

        k_x_x_i = lambda x, x_i: 1 / self.h * (abs(x - x_i) <= self.h / 2)  # 因为方窗函数实际上就相当于计数
        line_sample = np.linspace(self.x[0], self.x[-1], line_sample_n)
        p_x_hat = np.zeros_like(line_sample)


        for i, x in enumerate(line_sample):
            p_x_hat[i] = 1 / self.n * sum(k_x_x_i(x, self.x))

        plt.figure()

        plt.plot(line_sample, p_x_hat, label='pdf')
        plt.xlabel('x')
        plt.ylabel('probability')
        plt.title(f'rectangular window parzen, n={self.n}, h={self.h}')
        plt.legend()

        plot_util.save_fig(IMAGEFOLDER, f'rectangular window parzen, n={self.n}, h={self.h}')
        plt.show()


def run_histogram(n=10000, k=100):
    """
    定小舱体积，变落入样本个数
    """
    print(f'直方图方法'.center(100, '*'))

    mu = 0
    sig2 = 1
    x = np.sort(np.random.normal(mu, sqrt(sig2), n))
    
    instance = NonParmEstimation(x)
    instance.hist(k)


def run_k_n_neighbor(n=10000, k_n=100):
    """
    定小舱样本个数，变小舱体积
    """
    print(f'K_N近邻法估计'.center(100, '*'))

    mu = 0
    sig2 = 1
    x = np.sort(np.random.normal(mu, sqrt(sig2), n))

    instance = NonParmEstimation(x)
    instance.k_n_neighbor(k_n)


def run_parzen(n=10000, h=0.01, line_sample_n=10000):
    """
    以单个点为研究对象
    """
    print(f'Parzen窗法'.center(100, '*'))

    mu = 0
    sig2 = 1
    x = np.sort(np.random.normal(mu, sqrt(sig2), n))

    instance = NonParmEstimation(x)
    instance.parzen(h, line_sample_n)


if __name__ == '__main__':
    np.random.seed(42)

    # for x_sample in [10, 30, 300, 3000]:
    #     run_BE(x_sample)

    # for (n, k) in [(1000, 10), (1000,500), (10000, 100)]:
    #     run_histogram(n, k)

    # for (n, k_n) in [(16, 4), (256, 16), (100000000, 100000)]:
    #     run_k_n_neighbor(n, k_n)

    # for (n, h) in [(16, 0.25), (256, 0.25), (10000, 0.25), (100000, 0.25)]:
    #     run_parzen(n, h)

