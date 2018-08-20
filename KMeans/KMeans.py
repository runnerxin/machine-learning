#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/20 16:16
# @Author  : runnerxin

import numpy as np
from matplotlib import pyplot as plt


def load_data(filename):
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur = line.strip().split('\t')
        cur_float = [float(x) for x in cur]
        data_mat.append(cur_float)

    return data_mat


def euclidean_distance(vec_a, vec_b):

    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


def rand_center(dataset, k):
    n = np.shape(dataset)[1]            # 列的数量（数据维度）
    center = np.mat(np.zeros((k, n)))   # 创建k个质心矩阵
    for dim in range(n):
        min_value = min(dataset[:, dim])
        max_value = max(dataset[:, dim])
        range_value = float(max_value - min_value)
        # 随机生成k维0~1之间的数，再与最小值最大值的运算，使得产生的数在最小值最大值之间
        center[:, dim] = min_value + range_value * np.random.rand(k, 1)
    return center


def k_means(dataset, k, dist_means=euclidean_distance, create_center=rand_center):
    m = np.shape(dataset)[0]                    # 行数
    cluster_save = np.mat(np.zeros((m, 2)))     # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果
    center = create_center(dataset, k)          # 创建质心,随机K个质心

    # print(center)
    cluster_changed = True                      # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    while cluster_changed:
        cluster_changed = False

        # 遍历所有数据找到距离每个点最近的质心,可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for i in range(m):
            min_distance = np.inf
            min_index = -1
            for j in range(k):      # 遍历每个质心
                dist_ij = dist_means(center[j, :], dataset[i, :])       # 计算数据点到质心的距离
                if dist_ij < min_distance:
                    min_distance = dist_ij
                    min_index = j

            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if cluster_save[i, 0] != min_index:
                cluster_changed = True

            # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)的平方
            cluster_save[i, :] = min_index, min_distance ** 2

        for i in range(k):     # 遍历所有质心并更新它们的取值
            # 通过数据过滤来获得给定簇的所有点
            cluster_index = dataset[np.nonzero(cluster_save[:, 0].A == i)[0]]

            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            center[i, :] = np.mean(cluster_index, axis=0)

    # 返回所有的类质心与点分配结果
    return center, cluster_save


def b_k_means(data_mat, k, dist_means=euclidean_distance):

    m, n = np.shape(data_mat)

    # 创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    cluster_save = np.mat(np.zeros((m, 2)))

    # 计算整个数据集的质心,并使用一个列表来保留所有的质心
    centroid0 = np.mean(data_mat, axis=0).tolist()[0]
    center_list = [centroid0]

    # 遍历数据集中所有点来计算每个点到质心的误差值
    for j in range(m):
        cluster_save[j, 1] = dist_means(np.mat(centroid0), data_mat[j, :]) ** 2

    # 对簇不停的进行划分,直到得到想要的簇数目为止
    while len(center_list) < k:

        lowest_sse = np.inf             # 初始化最小SSE为无穷大,用于比较划分前后的SSE
        # 通过考察簇列表中的值来获得当前簇的数目,遍历所有的簇来决定最佳的簇进行划分
        for i in range(len(center_list)):
            # 对每一个簇,将该簇中的所有点堪称一个小的数据集
            index_i_point_set = data_mat[np.nonzero(cluster_save[:, 0].A == i)[0], :]

            # kMeans会生成两个质心(簇), 同时给出每个簇的误差值,
            # center 中心点, split_cluster 每个点的簇分配结果及平方误差
            center, split_cluster = k_means(index_i_point_set, 2, dist_means)

            # 将误差值与剩余数据集的误差之和作为本次划分的误差
            sse_split = np.sum(split_cluster[:, 1])
            sse_not_split = np.sum(cluster_save[np.nonzero(cluster_save[:, 0].A != i)[0], 1])
            if (sse_split + sse_not_split) < lowest_sse:
                best_center_to_split = i
                best_new_center = center
                best_cluster_ass = split_cluster.copy()
                lowest_sse = sse_split + sse_not_split

        # 找出最好的簇分配结果
        # 调用k_means函数并且指定簇数为2时,会得到两个编号分别为0和1的结果簇
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0].A == 1)[0], 0] = len(center_list)
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0].A == 0)[0], 0] = best_center_to_split

        # 更新质心列表  # 更新原质心list中的第i个质心为使用二分kMeans后bestNewCents的第一个质心
        center_list[best_center_to_split] = best_new_center[0, :].tolist()[0]

        # 添加bestNewCents的第二个质心
        center_list.append(best_new_center[1, :].tolist()[0])

        # 重新分配最好簇下的数据(质心)以及SSE
        cluster_save[np.nonzero(cluster_save[:, 0].A == best_center_to_split)[0], :] = best_cluster_ass

    return np.mat(center_list), cluster_save


def show(dataset, k, center, cluster_save):
    m, n = np.shape(dataset)

    # 画点
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # mark = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    for i in range(m):

        mark_index = int(cluster_save[i, 0])
        plt.plot(dataset[i, 0], dataset[i, 1], mark[mark_index], markersize=4)

    # 画质心
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(center[i, 0], center[i, 1], mark[i], markersize=8)

    plt.show()


def test_k_means():
    data = load_data('input/testSet.txt')
    data_mat = np.mat(data)
    center, cluster_save = k_means(data_mat, 4)
    show(data_mat, 4, center, cluster_save)


def test_bi_k_means():
    data = load_data('input/testSet.txt')
    data_mat = np.mat(data)
    center, cluster_save = b_k_means(data_mat, 4)
    show(data_mat, 4, center, cluster_save)


if __name__ == '__main__':
    # test_k_means()
    test_bi_k_means()
