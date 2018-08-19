#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/14 18:49
# @Author  : runnerxin

import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename):
    """
        Desc:
            对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
        Args:
            filename    文件名
        Returns:
            data_mat    数据特征集
            label_mat   数据分类集
    """
    num_feature = len(open(filename).readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_array = []
        line_now = line.strip().split('\t')
        for i in range(num_feature):
            line_array.append(float(line_now[i]))
        data_mat.append(line_array)
        label_mat.append(float(line_now[-1]))

    return data_mat, label_mat


def stand_regression(x, y):
    """
        Desc:
            线性回归
        Args:
            x       数据特征集
            y       数据分类集
        Returns:
            ws      回归系数
    """
    x_mat = np.mat(x)
    y_mat = np.mat(y).T
    x_x = x_mat.T * x_mat

    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    if np.linalg.det(x_x) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    # 最小二乘法, 求得w的最优解
    ws = x_x.I * (x_mat.T * y_mat)
    return ws


def regression_1():
    """
        Desc:
            线性回归
    """
    x, y = load_dataset('input/data.txt')
    x_mat = np.mat(x)
    y_mat = np.mat(y)

    # 求解最优参数ws
    ws = stand_regression(x, y)

    # 作图
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 原数据
    # ax.scatter([x_mat[:, 1].flatten()], [y_mat.T[:, 0].flatten().A[0]])
    ax.scatter([x_mat[:, 1]], [y_mat.T[:, 0]])

    # 预测的数据
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_predict = x_copy * ws
    ax.plot(x_copy[:, 1], y_predict)

    plt.show()


def local_weight(test_point, x, y, k):
    """
        Desc:
            局部加权线性回归，在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。
        Args:
            test_point      样本点
            x               数据特征集, 即feature
            y               数据分类集
            k               关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
        Returns:
            test_point * ws：数据点与具有权重的系数相乘得到的预测点
        Note:
            算法思路：这其中会用到计算权重的公式，w = e^((x^i - x) / -2k^2)。假设预测点取样本点中的第i个样本点（共m个），
                遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，也就可以计算出每个样本贡献误差的权值，
                可以看出w是一个有m个元素的向量（写成对角阵形式）。其中k是带宽参数，控制w（钟形函数）的宽窄程度，
                类似于高斯函数的标准差。
    """
    x_mat = np.mat(x)
    y_mat = np.mat(y).T

    m = np.shape(x_mat)[0]      # 获得x_mat矩阵的行数
    # eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵weights，该矩阵为每个样本点初始化了一个权重
    weights = np.mat(np.eye(m))

    for j in range(m):
        # 计算 testPoint 与输入样本点之间的距离，然后下面计算出每个样本贡献误差的权值
        diff_mat = test_point - x_mat[j, :]

        # k控制衰减的速度
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))

    # 根据矩阵乘法计算 x_x ，其中的 weights 矩阵是样本点对应的权重矩阵
    x_x = x_mat.T * (weights * x_mat)
    if np.linalg.det(x_x) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    # 计算出回归系数的一个估计
    ws = x_x.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def local_weight_test(test_array, x, y, k=1.0):
    """
        Desc:
            测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
        Args:
            test_array      测试所用的所有样本点
            x               数据特征集, 即feature
            y               数据分类集
            k               关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
        Returns:
            y_predict       预测点的估计值
    """

    # 得到样本点的总数
    m = np.shape(test_array)[0]

    # 构建一个全部都是 0 的 1 * m 的矩阵
    y_predict = np.zeros(m)

    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        y_predict[i] = local_weight(test_array[i], x, y, k)

    # 返回估计值
    return y_predict


def local_weight_plot(x, y, k=1.0):
    """
        Desc:
            首先将 X 排序，其余的都与local_weight_test相同，这样更容易绘图
        Args:
            x               数据特征集, 即feature
            y               数据分类集
            k               关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
        Returns:
            y_predict       样本点的估计值
            x_copy          x的复制
    """

    # 生成一个与目标变量数目相同的 0 向量
    y_predict = np.zeros(np.shape(y))

    x_copy = np.mat(x)
    x_copy.sort(0)
    # 循环，为每个样本点进行局部加权线性回归，得到最终的目标变量估计值
    for i in range(np.shape(x)[0]):
        y_predict[i] = local_weight(x_copy[i], x, y, k)

    return y_predict, x_copy


def regression_2():
    """
    Desc:
        局部加权线性回归(lwlr)
    """
    x, y = load_dataset('input/data.txt')
    y_predict = local_weight_test(x, x, y, k=0.003)

    x_mat = np.mat(x)
    sort_index = x_mat[:, 1].argsort(0)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    x_sort = x_mat[sort_index][:, 0, :]

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_predict[sort_index])
    ax.scatter([x_mat[:, 1]], [np.mat(y).T], s=2, c='red')
    plt.show()

    # x, y = load_dataset('input/data.txt')
    # y_predict, x_copy = local_weight_plot(x, y, k=0.003)
    #
    # # 画图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(x_copy[:, 1], y_predict[:])
    # ax.scatter([np.mat(x)[:, 1]], [np.mat(y).T], s=2, c='red')  # 原数据顺序对画出点无影响
    # plt.show()


def rss_error(y_array, y_hat):
    """
        Desc:
            计算分析预测误差的大小
        Args:
            y_array：真实的目标变量
            y_hat：预测得到的估计值
        Returns:
            计算真实值和估计值得到的值的平方和作为最后的返回值
    """
    return ((y_array - y_hat) ** 2).sum()


def abalone_test():
    """
        Desc:
            预测鲍鱼的年龄,使用局部加权回归和标准回归比较
    """

    # 加载数据
    x, y = load_dataset("input/abalone.txt")

    # 使用不同的核进行预测
    y_hat01 = local_weight_test(x[0:99], x[0:99], y[0:99], 0.1)
    y_hat1 = local_weight_test(x[0:99], x[0:99], y[0:99], 1)
    y_hat10 = local_weight_test(x[0:99], x[0:99], y[0:99], 10)

    # 打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    print("old yHat01 error Size is :", rss_error(y[0:99], y_hat01.T))
    print("old yHat1 error Size is :", rss_error(y[0:99], y_hat1.T))
    print("old yHat10 error Size is :", rss_error(y[0:99], y_hat10.T))

    # 打印出 不同的核预测值 与 新数据集（测试数据集）上的真实值之间的误差大小
    new_y_hat01 = local_weight_test(x[100:199], x[0:99], y[0:99], 0.1)
    new_y_hat1 = local_weight_test(x[100:199], x[0:99], y[0:99], 1)
    new_y_hat10 = local_weight_test(x[100:199], x[0:99], y[0:99], 10)

    print("new yHat01 error Size is :", rss_error(y[0:99], new_y_hat01.T))
    print("new yHat1 error Size is :", rss_error(y[0:99], new_y_hat1.T))
    print("new yHat10 error Size is :", rss_error(y[0:99], new_y_hat10.T))

    # 使用简单的 线性回归 进行预测，与上面的计算进行比较
    stand_ws = stand_regression(x[0:99], y[0:99])
    stand_y_hat = np.mat(x[100:199]) * stand_ws
    print("standRegress error Size is:", rss_error(y[100:199], stand_y_hat.T.A))


def ridge_regress(x, y, lam=0.2):
    """
        Desc:
            这个函数实现了给定 lambda 下的岭回归求解。
        Args:
            x       样本的特征数据，即 feature
            y       每个样本对应的类别标签，即目标变量，实际值
            lam     引入的一个λ值，使得矩阵非奇异
        Returns:
            经过岭回归公式计算得到的回归系数
    """

    x_x = x.T * x
    # 岭回归就是在矩阵 x_x 上加一个 λI 从而使得矩阵非奇异，进而能对 x_x + λI 求逆
    new_x = x_x + np.eye(np.shape(x)[1]) * lam

    # 检查行列式是否为零，即矩阵是否可逆，行列式为0的话就不可逆，不为0的话就是可逆。
    if np.linalg.det(new_x) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = new_x.I * (x.T * y)
    return ws


def ridge_test(x, y):
    """
        Desc：
            函数 ridgeTest() 用于在一组 λ 上测试结果
        Args：
            x       样本数据的特征，即 feature
            y       样本数据的类别标签，即真实数据
        Returns：
            ws      将所有的回归系数输出到一个矩阵并返回
    """
    x_mat = np.mat(x)
    y_mat = np.mat(y).T

    # Y的所有的特征减去均值
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean

    # 标准化 x，所有特征都减去各自的均值并除以方差
    x_means = np.mean(x, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var

    # 可以在 30 个不同的 lambda 下调用 ridge_regress() 函数。
    num_test_lam = 30

    # 创建30 * m 的全部数据为0 的矩阵
    w_mat = np.zeros((num_test_lam, np.shape(x_mat)[1]))
    for i in range(num_test_lam):
        # exp() 返回 e^x
        ws = ridge_regress(x_mat, y_mat, np.exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def regression_3():
    x, y = load_dataset("input/abalone.txt")
    ridge_weights = ridge_test(x, y)

    # 作图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()


def stage_wise(x, y, eps=0.01, num_it=100):
    x_mat = np.mat(x)
    y_mat = np.mat(y).T

    # Y的所有的特征减去均值
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean

    # 标准化 x，所有特征都减去各自的均值并除以方差
    x_means = np.mean(x, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var

    m, n = np.shape(x_mat)
    return_mat = np.zeros((num_it, n))
    ws = np.zeros((n, 1))
    # ws_test = ws.copy()
    ws_max = ws.copy()

    for i in range(num_it):
        lowest_error = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                error =  rss_error(y_mat.A, y_test.A)
                if error < lowest_error:
                    lowest_error = error
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


def regression4():
    x, y = load_dataset("input/abalone.txt")
    # stage_wise(x, y, 0.01, 200)
    print(stage_wise(x, y, 0.01, 200))
    # x_mat = np.mat(x)
    # y_mat = np.mat(y).T

    x_mat = np.mat(x)
    y_mat = np.mat(y).T

    # Y的所有的特征减去均值
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean

    # 标准化 x，所有特征都减去各自的均值并除以方差
    x_means = np.mean(x, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var
    weights = stand_regression(x_mat, y_mat.T)
    print (weights.T)


if __name__ == '__main__':
    # regression_1()
    # regression_2()
    # abalone_test()
    # regression_3()
    regression4()
