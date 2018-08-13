#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/8 21:47
# @Author  : runnerxin

from random import randrange, random, seed
import numpy as np


def load_dataset(filename):
    """
        Desc:
            对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
        Args:
            filename 文件名
        Returns:
            data_set  数据集
    """
    dataset = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            line_array = []
            for feature in line.split(','):
                str_ = feature.strip()

                if str_.isalpha():                   # 判断是否是数字, 将数据集的第column列转换成float形式
                    line_array.append(str_)  # 添加分类标签

                else:
                    line_array.append(np.float(str_))
            dataset.append(line_array)
    return dataset


def cross_validation_split(dataset, n_folds):
    """
        Desc:
            将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次list的元素是无重复的
        Args:
            dataset     原始数据集
            n_folds     数据集dataset分成n_folds份
        Returns:
            dataset_split    list集合，存放的是：将数据集进行抽重抽样 n_folds 份
    """
    data_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset) / n_folds
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))

            # fold.append(dataset_copy.pop(index))  # 无放回的方式
            fold.append(dataset_copy[index])  # 有放回的方式
        data_split.append(fold)
    return data_split


def gini_index(groups, class_values):
    """
        Desc:
            计算代价，分类越准确，则 gini 越小
        Args:
            groups              分类的数据集（left,right)
            class_values        分类值
        Returns:
            gini                计算的代价
    """
    gini = 0.0
    for value in class_values:           # 枚举每个分类值
        for group in groups:             # groups = (left, right)
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(value) / float(size)
            gini += (proportion * (1.0 - proportion))    # 个人理解：计算代价，分类越准确，则 gini 越小
    return gini


def test_split(index, value, dataset):
    """
        Desc:
            根据特征和特征值分割数据集，dataset集里按照index个特征，把值大于values的分为一类，其他的为另一类
        Args:
            index       特征索引
            value       分类值
            dataset     数据集
        Returns:
            left        分类后数据左边的数据
            right       分类后数据左边的数据
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def get_split(dataset, n_features):
    # 找出分割数据集的最优特征，得到最优的特征 index，特征值 row[index]，
    # 以及分割完的数据 groups（left, right）

    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        # print(len(dataset[0]))
        index = randrange(len(dataset[0]) - 1)  # 随机选取n_features个特征
        if index not in features:
            features.append(index)

    # 在 n_features 个特征中选出最优的特征索引，并没有遍历所有特征，从而保证了每课决策树的差异性
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)  # groups=(left, right),
            # row[index] 遍历每一行 # index 索引下的特征值作为分类值 value, 找出最优的分类特征和特征值
            gini = gini_index(groups, class_values)
            # 左右两边的数量越一样，说明数据区分度不高，gini系数越大
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
                # 最后得到最优的分类特征 b_index,分类特征值 b_value,分类结果 b_groups。b_value 为分错的代价成本
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value #
def to_terminal(group):
    """
        Desc:
            输出group中出现次数较多的标签
        Args:
            group  group数据集
        Returns:
            输出 group 中出现次数较多的标签
    """
    outcomes = [row[-1] for row in group]           # max() 函数中，当 key 参数不为空时，就以 key 的函数对象为判断的标准
    return max(set(outcomes), key=outcomes.count)   # 输出 group 中出现次数较多的标签


def split(node, max_depth, min_size, n_features, depth):
    """
        Desc:
            创建子分割器，递归分类，直到分类结束
        Args:
            node            决策树
            max_depth       决策树深度
            min_size        叶子节点的大小
            n_features      选取的特征的个数
            depth           当前决策树的深度
        Returns:
            --
    """
    left, right = node['groups']
    del (node['groups'])

    # 没有左边或者右边的数据（分类只有一种情况）
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # 达到了决策树最大深度
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # 数据量少于最小叶子节点数
    if len(left) < min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)  # node是一个多层字典
        split(node['left'], max_depth, min_size, n_features, depth + 1)  # 递归，depth+1计算递归层数

    # 右子树
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


def build_tree(train, max_depth, min_size, n_features):
    """
        Desc:
            创建一个决策树
        Args:
            train           训练数据集
            max_depth       决策树深度不能太深，不然容易导致过拟合
            min_size        叶子节点的大小
            n_features      选取的特征的个数
        Returns:
            root            返回决策树
    """
    # 返回最优列和相关的信息
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


def predict(node, row):

    # 预测模型分类结果
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):       # isinstance 是 Python 中的一个内建函数。是用来判断一个对象是否是一个已知的类型。
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    """
        Desc:
            bagging预测
        Args:
            trees           决策树的集合
            row             测试数据集的每一行数据
        Returns:
            返回随机森林中，决策树结果出现次数做大的
    """

    # 使用多个决策树trees对测试集test的第row行进行预测，再使用简单投票法判断出该行所属分类
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def subsample(dataset, ratio):
    """
        Desc:
            # 创建数据集的随机子样本
        Args:
            dataset         训练数据集
            ratio           训练数据集的样本比例
        Returns:
            sample          随机抽样的训练样本
    """

    sample = []
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    """
        Desc:
            # 创建数据集的随机子样本
        Args:
            train           训练数据集
            test            测试数据集
            max_depth       决策树深度不能太深，不然容易导致过拟合
            min_size        叶子节点的大小
            sample_size     训练数据集的样本比例
            n_trees         决策树的个数
            n_features      选取的特征的个数
        Returns:
            predictions     每一行的预测结果，bagging 预测最后的分类结果
    """
    trees = []
    for i in range(n_trees):
        # 随机抽样的训练样本， 随机采样保证了每棵决策树训练集的差异性
        sample = subsample(train, sample_size)
        # print(len(sample[0]))
        # 创建一个决策树
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)

    # 每一行的预测结果，bagging 预测最后的分类结果
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


def accuracy_metric(actual, predicted):
    """
        Desc:
            # 导入实际值和预测值，计算精确度
        Args:

        Returns:

    """

    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """
    Desc:
        评估算法性能，返回模型得分
    Args:
        dataset     原始数据集
        algorithm   使用的算法
        n_folds     数据的份数
        *args       其他的参数
    Returns:
        scores      模型得分
    """

    # 将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次 list 的元素是无重复的
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    # 每次循环从 folds 从取出一个 fold 作为测试集，其余作为训练集，遍历整个 folds ，实现交叉验证
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, []) # 设置train里的[]为单个[]

        # fold 表示从原始数据集 dataset 提取出来的测试集
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)

        # print(len(train_set))
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        # 计算随机森林的预测结果的正确率
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


if __name__ == "__main__":

    d_set = load_dataset('sonar-all-data.txt')
    # print(len(d_set[0]))
    # print(d_set)

    n_folds = 5             # 分成5份数据，进行交叉验证
    max_depth = 20          # 调参（自己修改） #决策树深度不能太深，不然容易导致过拟合
    min_size = 1            # 决策树的叶子节点最少的元素数量
    sample_size = 1.0       # 做决策树时候的样本的比例
    n_features = 15         # 调参（自己修改） #准确性与多样性之间的权衡  n_features = int((len(dataset[0])-1))

    for n_trees in [1, 10, 20]:  # 理论上树是越多越好
        scores = evaluate_algorithm(d_set, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        # 每一次执行本文件时都能产生同一个随机数
        seed(1)
        print('random=', random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
