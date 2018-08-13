#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/12 21:42
# @Author  : runnerxin

import numpy as np


def load_dataset(file_name):
    """
        Desc:
            对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
        Args:
            filename 文件名
        Returns:
            data_set  数据集
    """
    n_feature = len(open(file_name).readline().split('\t')) - 1  # 最后一个是lable
    data_array = []
    lable_array = []
    with open(file_name, 'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            line_array = []

            cur_line = line.strip().split('\t')
            for i in range(n_feature):
                line_array.append(np.float(cur_line[i]))
            data_array.append(line_array)
            lable_array.append(float(cur_line[-1]))
    return np.matrix(data_array), lable_array


def stump_classify(data_mat, dim, boundary, inequal):
    """
        Desc:
            构造弱分类器，以boundary为界，分类为  1\-1
        Args:
            data_mat            特征标签集合
            dim                 表示 feature列
            boundary            分界值
            inequal             计算树左右颠倒
        Returns:
            ret_array           分类后的结果
    """

    ret_array = np.ones((np.shape(data_mat)[0], 1))
    # data_mat[:, dim]    表示数据集中第dim列的所有值
    # boundary == 'left'   表示修改左边的值，gt表示修改右边的值

    if inequal == 'left':
        ret_array[data_mat[:, dim] <= boundary] = -1.0
    else:
        ret_array[data_mat[:, dim] > boundary] = -1.0

    return ret_array


def build_stump(data_array, class_labels, d):
    """
        Desc:
            得到决策树的模型，枚举每一个feature列，每一个分界值，寻找最好的决策树模型
        Args:
            data_mat            特征标签集合
            class_labels        分类标签集合
            d                   最初的特征权重值
        Returns:
            best_stump          最优的分类器模型
            min_error           错误率
            best_class_set      训练后的结果集
    """
    data_mat = np.mat(data_array)
    label_mat = np.mat(class_labels).T

    m, n = np.shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_class_set = np.mat(np.zeros((m, 1)))
    min_error = np.inf           # 无穷大

    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()

        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['left', 'right']:

                boundary = (range_min + float(j) * step_size)
                predicted_values = stump_classify(data_mat, i, boundary, inequal)
                error_array = np.mat(np.ones((m, 1)))
                error_array[predicted_values == label_mat] = 0
                weighted_error = d.T * error_array      # 这里是矩阵乘法

                """
                i                   表示 feature列
                j                   步数
                inequal             表示计算树左右颠倒，的错误率的情况
                boundary            表示树的分界值
                weighted_error      表示整体结果的错误率
                best_class_set      预测的最优结果 （与class_labels对应）
                """
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_set = predicted_values
                    best_stump['dim'] = i
                    best_stump['boundary'] = boundary
                    best_stump['inequal'] = inequal
                    # best_stump 表示分类器的结果，在第几个列上，用大于／小于比较，阈值是多少 (单个弱分类器)
            return best_stump, min_error, best_class_set


def ada_boost_train_ds(data_mat, class_labels, epoch=40):
    """
        Desc:
            adaBoost            训练过程
        Args:
            data_mat            特征数据集合
            class_labels        分类标签集合
            epoch               迭代次数
        Returns:
            weak_class_array    弱分类器的集合
            predict_class_set   预测的分类结果值
    """
    week_class_array = []
    m = np.shape(data_mat)[0]

    # 初始化 D，设置每个特征的权重值，平均分为m份
    d = np.mat(np.ones((m, 1)) / m)
    predict_class_set = np.mat(np.zeros((m, 1)))

    for i in range(epoch):
        # 得到决策树的模型
        best_stump, min_error, best_class_set = build_stump(data_mat, class_labels, d)

        # alpha 目的主要是计算每一个分类器实例的权重(加和就是分类结果)
        # 计算每个分类器的 alpha 权重值
        alpha = float(0.5 * np.log((1.0 - min_error) / max(min_error, 1e-16)))
        best_stump['alpha'] = alpha
        week_class_array.append(best_stump)

        # 分类正确：乘积为1，不会影响结果，-1主要是下面求e的-alpha次方
        # 分类错误：乘积为 -1，结果会受影响，所以也乘以 -1
        exp_on = np.multiply(-1 * alpha * np.mat(class_labels).T, best_class_set)

        # 计算e的exp_on次方，然后计算得到一个综合的概率的值
        # 结果发现： 判断错误的样本，D对于的样本权重值会变大。
        d = np.multiply(d, np.exp(exp_on))
        d = d / d.sum()

        # 预测的分类结果值，在上一轮结果的基础上，进行加和操作
        predict_class_set += alpha * best_class_set

        # sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
        # 结果为：错误的样本标签集合，因为是 !=,那么结果就是0 正, 1 负
        agg_errors = np.multiply(np.sign(predict_class_set) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m

        if error_rate == 0.0:
            break
    return week_class_array, predict_class_set


def ada_classify(data_mat, classifier_array):
    """
        Desc:
            通过刚刚上面那个函数得到的弱分类器的集合进行预测
        Args:
            data_mat            特征数据集合
            classifier_array    分类器列表
        Returns:
            正负一，也就是表示分类的结果
    """
    data_mat = np.mat(data_mat)
    m = np.shape(data_mat)[0]
    predict_class_set = np.mat(np.zeros((m, 1)))

    for i in range(len(classifier_array)):
        class_set = stump_classify(data_mat,
                                   classifier_array[i]['dim'],
                                   classifier_array[i]['boundary'],
                                   classifier_array[i]['inequal'])

        predict_class_set += classifier_array[i]['alpha'] * class_set
        # print(predict_class_set)
    return np.sign(predict_class_set)


def test():

    data_mat, class_labels = load_dataset("horseColicTraining2.txt")
    week_class_array, agg_class_set = ada_boost_train_ds(data_mat, class_labels, 40)

    test_data_mat, test_class_labels = load_dataset("horseColicTest2.txt")
    m = np.shape(test_data_mat)[0]
    predicting10 = ada_classify(test_data_mat, week_class_array)
    err_arr = np.mat(np.ones((m, 1)))

    # 测试：计算总样本数，错误样本数，错误率
    print(m,
          err_arr[predicting10 != np.mat(test_class_labels).T].sum(),
          err_arr[predicting10 != np.mat(test_class_labels).T].sum() / m)


if __name__ == '__main__':
    test()
