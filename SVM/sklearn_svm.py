#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/8 20:37
# @Author  : runnerxin

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def load_dataset(filename):
    """
        Desc:
            对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
        Args:
            fileName 文件名
        Returns:
            data_mat  特征矩阵
            lable_mat 类标签
    """
    fr = open(filename)
    data_mat = []
    lable_mat = []
    for line in fr.readlines():
        line_array = line.strip().split('\t')
        length = len(line_array)
        data_mat.append([float(line_array[index]) for index in range(length-1)])
        lable_mat.append(float(line_array[length-1]))

    return data_mat, lable_mat


if __name__ == "__main__":
    data_array, lable_array = load_dataset('testSet.txt')
    data_array = np.mat(data_array)

    clf = svm.SVC(kernel='linear')
    clf.fit(data_array, lable_array)

    # 获取分割超平面
    w = clf.coef_[0]
    # 斜率
    a = -w[0] / w[1]

    # 从-5到5，顺序间隔采样50个样本，默认是num=50
    xx = np.linspace(-2, 10)  # , num=50)
    # 二维的直线方程
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # 通过支持向量绘制分割超平面
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
    plt.scatter(data_array[:, 0].flat, data_array[:, 1].flat, c=lable_array, cmap=plt.cm.Paired)

    plt.axis('tight')
    plt.show()




