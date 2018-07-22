#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/7/22 17:22
# @Author  : runnerxin

import numpy as np
import operator
from sklearn.model_selection import train_test_split
from sklearn import datasets


def load_data(filename):
    """
    函数用来输入数据，并转化为array的格式
    param
        filename:    输入的文件地址
    return:
        features：   特征,
        label：      标签
    """

    features = []
    label = []
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split('\t')

        features.append([float(item) for item in list_from_line[0:3]])
        label.append(int(list_from_line[-1]))
    features = np.array(features)
    return features, label


def data_normalization(dataset):
    """
    将数据归一化
    param:
        dataset:        数据集
    return:
        norm_dataset：  归一化后的数据集
        min:            最小值
        ranges:         极差
    归一化公式：        Y = (X-Xmin)/(Xmax-Xmin)
    """
    min_values = dataset.min(axis=0)  # 空是为全部中最小的，0为列最小，1为行最小
    max_values = dataset.max(axis=0)

    ranges = max_values - min_values
    norm_dataset = (dataset - min_values) / ranges
    return norm_dataset, min_values, ranges


def classify(inx, dataset, labels, k):
    """
    将输入的数据分类
    过程：计算inx与每个数据的距离，找出最近的k个，找出k个中最多的类别
    param
        inx:        输入要分类的x
        dataset:    训练数据集
        labels:     训练数据集分类
        k:          选择最近邻的数目
    return:
        classify_label: 分类后的类别
    """

    dist = (np.sum((inx - dataset) ** 2, axis=1)) ** 0.5
    sort_distance = dist.argsort()
    class_count = {}
    # 选择最近邻的k个数据的lable
    for i in range(k):
        vote_lable = labels[sort_distance[i]]
        class_count[vote_lable] = class_count.get(vote_lable, 0) + 1

    # 统计出现次数最多的lable，作为inx的分类
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    classify_label = sort_class_count[0][0]
    return classify_label


def knn_class_test():
    x, y = load_data("datingTestSet2.txt")
    x, min_values, ranges = data_normalization(x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
    error_count = 0
    for i in range(len(test_x)):
        classify_result = classify(test_x[i], train_x, train_y, 3)
        error_count += classify_result != test_y[i]

    print(error_count)


def handle_number():
    digits = datasets.load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1)

    error_count = 0
    for i in range(len(x_test)):
        classify_result = classify(x_test[i], x_train, y_train, 3)
        error_count += classify_result != y_test[i]

    print((len(x_test)-error_count)/len(x_test))
    print(error_count)


if __name__ == '__main__':
    # knn_class_test()
    handle_number()

