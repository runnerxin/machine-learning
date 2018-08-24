#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/24 22:12
# @Author  : runnerxin

import numpy as np
import matplotlib.pyplot as plt


def load_data(file_name, split_word='\t'):

    fr = open(file_name)
    data = []
    for line in fr.readlines():
        arr = line.strip().split(split_word)
        to_float = [float(x) for x in arr]
        data.append(to_float)
    return np.mat(data)


def pca(data_mat, dim=99999999):
    """
        Desc:
            基础数据集，降维
        Args:
            data_mat            原数据集矩阵
            dim                 应用的N个特征
        Returns:
            low_data_mat        降维后数据集
            re_eig_vector       降维向量
            mean_value          数据集的均值
    """
    # 计算每一列的均值
    mean_value = np.mean(data_mat, axis=0)

    # 每个向量同时都减去均值
    sub_mean = data_mat - mean_value

    # cov协方差=[(x1-x均值)*(y1-y均值) + (x2-x均值)*(y2-y均值) + ... + (xn-x均值)*(yn-y均值)+] / (n-1)
    cov_mat = np.cov(sub_mean, rowvar=False)

    # feature为特征值， feature_vector为特征向量
    eig_value, eig_vector = np.linalg.eig(np.mat(cov_mat))
    # print(eig_value)

    # 对特征值，进行从小到大的排序，返回从小到大的index序号
    # # 特征值的逆序就可以得到dim个最大的特征向量
    eig_value_index = np.argsort(eig_value)
    eig_value_index = list(reversed(eig_value_index))[:dim]
    # eig_value_index = eig_value_index[:-(dim + 1):-1]

    # 重组 eig_vector 最大到最小
    re_eig_vector = eig_vector[:, eig_value_index]

    # 将数据转换到新空间
    low_data_mat = sub_mean * re_eig_vector                             # 降维后的数据
    return low_data_mat, re_eig_vector, mean_value


def recover_mat(low_data_mat, re_eig_vector, mean_value):
    """
        Desc:
            将降维后的数据恢复出原数据
        Args:
            low_data_mat        降维后数据集
            re_eig_vector       降维向量
            mean_value          数据集的均值
        Returns:
            recover_mat_dim     恢复后的数据集
        """
    recover_mat_dim = (low_data_mat * re_eig_vector.T) + mean_value
    return recover_mat_dim


def show_picture(data_mat, re_mat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(re_mat[:, 0].flatten().A[0], re_mat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()


def replace_nan_data_load():
    data_mat = load_data('input/secom.data', ' ')
    num_feature = np.shape(data_mat)[1]
    for i in range(num_feature):
        # 对value不为NaN的求均值。.A 返回矩阵基于的数组
        mean_value = np.mean(data_mat[np.nonzero(~np.isnan(data_mat[:, i].A))[0], i])
        # 将value为NaN的值赋值为均值
        data_mat[np.nonzero(np.isnan(data_mat[:, i].A))[0], i] = mean_value
    return data_mat


def analyse_data(data_mat):
    mean_value = np.mean(data_mat, axis=0)
    sub_mean = data_mat - mean_value
    cov_mat = np.cov(sub_mean, rowvar=False)
    eig_value, eig_vector = np.linalg.eig(np.mat(cov_mat))
    eig_value_index = np.argsort(eig_value)
    dim = 20
    eig_value_index = list(reversed(eig_value_index))[:dim]

    cov_all_score = np.float(np.sum(eig_value))
    sum_cov_score = 0
    for i in range(0, len(eig_value_index)):
        line_cov_score = np.float(eig_value[eig_value_index[i]])
        sum_cov_score += line_cov_score
        '''
        我们发现其中有超过20%的特征值都是0。
        这就意味着这些特征都是其他特征的副本，也就是说，它们可以通过其他特征来表示，而本身并没有提供额外的信息。
        最前面15个值的数量级大于10^5，实际上那以后的值都变得非常小。
        这就相当于告诉我们只有部分重要特征，重要特征的数目也很快就会下降。
        最后，我们可能会注意到有一些小的负值，他们主要源自数值误差应该四舍五入成0.
        '''
        print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (format(i+1, '2.0f'),
                                                  format(line_cov_score/cov_all_score*100, '4.2f'),
                                                  format(sum_cov_score/cov_all_score*100, '4.1f')))



if __name__ == '__main__':

    # # 加载数据，并转化数据类型为float
    # dataMat = load_data('input/testSet.txt')
    #
    # # # 只需要1个特征向量
    # low_data_mat, low_mat, mean_value = pca(dataMat, 1)
    # # print(np.shape(low_data_mat))
    #
    # # # 只需要2个特征向量，和原始数据一致，没任何变化
    # low_data_mat, low_mat, mean_value = pca(dataMat, 2)
    # # print(recover_mat(low_data_mat, low_mat, mean_value))

    # 项目实践
    # 利用PCA对半导体制造数据降维
    dataMat = replace_nan_data_load()
    print(np.shape(dataMat))
    # 分析数据
    analyse_data(dataMat)
    low_data_mat, re_eig_vector, mean_value = pca(dataMat, 20)
    print(np.shape(low_data_mat))
    show_picture(dataMat, recover_mat(low_data_mat, re_eig_vector, mean_value))
