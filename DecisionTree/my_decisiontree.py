#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/7/26 20:55
# @Author  : runnerxin

import copy
import operator
import math
import pickle


def create_dataset():

    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    # labels  露出水面   脚蹼，注意：这里的labels是写的 dataSet 中特征的含义，并不是对应的分类标签或者说目标变量
    feature_labels = ['no surfacing', 'flippers']
    return dataset, feature_labels


def calc_shannon(dataset):
    """
        Desc：
            calculate Shannon entropy -- 计算给定数据集的香农熵
        Args:
            dataSet -- 数据集
        Returns:
            shannonEnt -- 返回 每一组 feature 下的某个分类下，香农熵的信息期望
    """
    data_len = len(dataset)
    # 计算分类标签label出现的次数
    label_counts = {}
    for line_data in dataset:
        current_lable = line_data[-1]
        if current_lable not in label_counts.keys():
            label_counts[current_lable] = 0
        label_counts[current_lable] += 1

    shannon_ent = 0.0
    for key in label_counts:

        prob = float(label_counts[key]) / data_len
        shannon_ent -= prob * math.log(prob, 2)

    return shannon_ent


def split_dataset(dataset, feature_index, values):
    """
    Args:
        dataSet         数据集                     待划分的数据集
        feature_index   表示每一行的index列        划分数据集的特征
        value           表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
    """
    sub_dataset = []
    for line in dataset:
        if line[feature_index] == values:
            reduce_line = line[:feature_index]
            reduce_line.extend(line[feature_index+1:])
            sub_dataset.append(reduce_line)
    return sub_dataset


def choose_best_feature(dataset):
    """
    Desc:
        选择切分数据集的最佳特征
    Args:
        dataset -- 需要切分的数据集
    Returns:
        best_feature -- 切分数据集的最优的特征列
    """
    # num_feature 特征值得个数，（去除最后一列的lable）
    num_feature = len(dataset[0]) - 1
    base_entropy = calc_shannon(dataset)
    best_gain, best_feature = 0.0, -1

    for i in range(num_feature):
        feature_data = [example[i] for example in dataset]
        unique_values = set(feature_data)
        temp_entropy = 0.0

        for values in unique_values:
            sub_dataset = split_dataset(dataset, i, values)
            prob = len(sub_dataset) / float(len(dataset))
            # 计算信息熵
            temp_entropy += prob * calc_shannon(sub_dataset)

        info_gain = base_entropy - temp_entropy
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = i
    return best_feature


def most_cnt(class_list):
    """
    Desc:
        选择出现次数最多的一个class
    Args:
        class_list  label列的集合
    Returns:
        most_class 最多的分类
    """
    count_class = {}
    for vote in class_list:
        if vote not in count_class.keys():
            count_class[vote] = 0
        count_class[vote] += 1

    # 倒叙排列count_class得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    class_count_sorted = sorted(count_class.items(), key=operator.itemgetter(1), reverse=True)

    most_class = class_count_sorted[0][0]
    # most_class = class_count_sorted[0].itemgetter(0)
    return most_class


def create_tree(dataset, feature_labels):
    """
    Desc:
        创建决策树
    Args:
        dataSet -- 要创建决策树的训练数据集
        feature_labels -- 训练数据集中特征对应的含义的labels，不是目标变量
    Returns:
        myTree -- 创建完成的决策树
    """

    # 取出每个数据里的lable
    class_list = [index[-1] for index in dataset]
    # print(class_list)
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果数据集只有1列(1个特征值），那么最初出现label次数最多的一类，作为结果
    if len(dataset[0]) == 1:
        return most_cnt(class_list)
    # print(class_list)

    # 选择最优的列，得到最优列对应的label含义
    best_feature = choose_best_feature(dataset)
    best_feature_values = [example[best_feature] for example in dataset]
    best_feature_label = feature_labels[best_feature]
    # 删除feature_labels中用过的特征
    del (feature_labels[best_feature])

    my_tree = {best_feature_label: {}}
    unique_values = set(best_feature_values)

    for values in unique_values:
        sub_lable = feature_labels[:]
        my_tree[best_feature_label][values] = create_tree(split_dataset(dataset, best_feature, values), sub_lable)

    return my_tree


def classify(input_tree, feature_labels, test_vector):
    """
    Args:
        input_tree          决策树模型
        feature_labels      Feature标签对应的名称
        test_vector             测试输入的数据
    Returns:
        class_label 分类的结果值，需要映射label才能知道名称
    """
    # print(list(input_tree.keys()))
    first_node = list(input_tree.keys())[0]
    second_dict = input_tree[first_node]

    feature_index = feature_labels.index(first_node)

    key = test_vector[feature_index]
    value_of_feature = second_dict[key]

    if isinstance(value_of_feature, dict):
        class_label = classify(value_of_feature, feature_labels, test_vector)
    else:
        class_label = value_of_feature
    return class_label


def store_tree(input_tree, filename):
    """
    将之前训练好的决策树模型存储起来，使用 pickle 模块
    """
    with open(filename, 'wb') as fw:
        pickle.dump(input_tree, fw)


def grab_tree(filename):
    """
    将之前存储的决策树模型使用 pickle 模块 还原出来
    """
    fr = open(filename, 'rb')
    return pickle.load(fr)


def tree_test():
    # 1.创建数据和结果标签
    my_data, feature_labels = create_dataset()

    my_tree = create_tree(my_data, copy.deepcopy(feature_labels))
    print(my_tree)
    print(classify(my_tree, feature_labels, [1, 1]))


def lenses_test():
    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_feature = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lenses_tree = create_tree(lenses, lenses_feature)
    print(lenses_tree)
    # pass


if __name__ == "__main__":
    # tree_test()
    lenses_test()
