#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/1 22:43
# @Author  : runnerxin


import numpy as np


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


def classify_vector(in_x, weights):
    """
        Desc:
            最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
        Args:
            in_x        -- 特征向量，features
            weights     -- 计算得到的回归系数
        Returns:
            分类结果
    """

    prob = sigmoid(np.sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    return 0.0


def colic_test():
    """
        Desc:
            病马demo
        Args:
        Returns:
            错误率
    """
    # 读取数据
    f = open('HorseColicTraining.txt', 'r')
    features = []
    lable = []
    for line in f.readlines():
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue  # 这里如果就一个空的元素，则跳过本次循环
        length = len(curr_line)
        line_array = [float(curr_line[i]) for i in range(length - 1)]

        features.append(line_array)
        lable.append(float(curr_line[length - 1]))

    # 训练
    weights = random_grad_ascent1(np.array(features), lable)

    # 测试
    error_count = 0
    num_test = 0
    f = open('HorseColicTest.txt', 'r')
    for line in f.readlines():
        num_test += 1
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue  # 这里如果就一个空的元素，则跳过本次循环
        length = len(curr_line)
        features = [float(curr_line[i]) for i in range(length - 1)]

        if int(classify_vector(np.array(features), weights)) != int(curr_line[length - 1]):
            error_count += 1

    error_rate = error_count / num_test
    print('the error rate is {}'.format(error_rate))
    return error_rate


def multi_test():
    """
        Desc:
            colicTest() 10次并求结果的平均值
        Args:
        Returns:
            平均后的错误率
    """
    num_tests = 10
    error_sum = 0
    for k in range(num_tests):
        error_sum += colic_test()
    print('after {} iteration the average error rate is {}'.format(num_tests, error_sum / num_tests))


if __name__ == "__main__":
    # colic_test()
    multi_test()
