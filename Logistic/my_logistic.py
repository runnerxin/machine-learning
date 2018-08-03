#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/1 20:59
# @Author  : runnerxin

import numpy as np
import matplotlib.pyplot as plt


def load_data_set():
    """
        Desc:
            读取数据集
        Args:
        Returns:
            返回数据特征集和对应的label标签
    """
    features = []
    lable = []
    f = open('TestSet.txt', 'r')
    for line in f.readlines():
        line_array = line.strip().split()
        # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
        features.append([1.0, np.float(line_array[0]), np.float(line_array[1])])
        lable.append(int(line_array[2]))
    return features, lable


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def grad_ascent(features, lable):
    """
        Desc:
            梯度下降，求出权重值
        Args:
            features -- 特征值
            lable    -- 标签
        Returns:
            返回权重值
    """
    features_mat = np.mat(features)
    lable_mat = np.mat(lable).transpose()           # n*1 -> 1*n
    num_sample, num_feature = np.shape(features)    # 样本数，特征数
    alpha = 0.001                                   # 学习率
    max_epochs = 500                                # 最大迭代次数

    weights = np.ones((num_feature, 1))
    for epoch in range(max_epochs):
        h = sigmoid(features_mat * weights)
        error = lable_mat - h
        weights = weights + alpha * features_mat.transpose() * error

    return weights


def random_grad_ascent0(features, lable):
    """
        Desc:
            随机梯度上升，只使用一个样本点来更新回归系数
        Args:
            features -- 特征值
            lable    -- 标签
        Returns:
            得到的最佳回归系数
    """
    num_sample, num_feature = np.shape(features)
    alpha = 0.01
    weights = np.ones(num_feature)
    for i in range(num_sample):
        h = sigmoid(sum(features[i] * weights))
        error = lable[i] - h
        weights = weights + alpha * error * features[i]

    return weights


def random_grad_ascent1(features, lable):
    """
        Desc:
            改进版的随机梯度上升，使用随机的一个样本来更新回归系数
        Args:
            features -- 特征值
            lable    -- 标签
        Returns:
            得到的最佳回归系数
    """
    num_sample, num_feature = np.shape(features)  # 样本数，特征数
    weights = np.ones(num_feature)
    max_epochs = 150   # 迭代次数

    for j in range(max_epochs):
        data_index = list(range(num_sample))  # 这里必须要用list，不然后面的del没法使用
        for i in range(num_sample):
            alpha = 4 / (1.0 + j + i) + 0.01  # i和j的不断增大，导致alpha的值不断减少，但是不为0
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(features[data_index[rand_index]] * weights))
            error = lable[data_index[rand_index]] - h
            weights = weights + alpha * error * features[data_index[rand_index]]
            del (data_index[rand_index])
    return weights


def plot_best_fit(weights):
    """
        Desc:
            可视化结果
        Args:
            weights -- 权重值
        Returns:
    """
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    """
    y的由来，卧槽，是不是没看懂？
    首先理论上是这个样子的。
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()


def test():
    features, lable = load_data_set()
    # weights = grad_ascent(features, lable).getA()            # 得到的函数是mat,从中取出数getA()
    # weights = random_grad_ascent0(np.array(features), lable)
    weights = random_grad_ascent1(np.array(features), lable)

    plot_best_fit(weights)


if __name__ == "__main__":
    test()
