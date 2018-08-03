#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/7/29 15:45
# @Author  : runnerxin

import numpy as np


def load_data_set():
    """
        Desc：
            创建数据集
        Args:
        Returns:
            单词列表posting_list, 所属类别class_vec
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
        Desc：
            获取所有单词的集合
        Args:
            data_set -- 数据集
        Returns:
            返回所有单词的集合(即不含重复元素的单词列表)
    """
    vocab_set = set()
    for item in data_set:
        # | 求两个集合的并集
        vocab_set = vocab_set | set(item)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    """
        Desc：
            遍历查看该单词是否出现，出现该单词则将该单词置1
        Args:
            vocab_list -- 所有单词集合列表
            input_set  -- 输入数据集
        Returns:
            其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """

    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    result = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
    return result


def bag_words2vec(vocab_list, input_set):
    """
        Desc：
            遍历查看该单词是否出现，出现该单词则将该单词置1
        Args:
            vocab_list -- 所有单词集合列表
            input_set  -- 输入数据集
        Returns:
            其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """

    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    result = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] += 1
    return result


def train_naive_bayes(train_mat, train_category):
    """
        Desc：
            朴素贝叶斯分类
        Args:
            train_mat -- 总的输入文本
            train_category  -- 文件对应的类别分类
        Returns:
            usual_word_probable_vec    --正常doc下词出现的概率
            abusive_word_probable_vec  --谩骂doc下词出现的概率
            doc_abusive_probable       --谩骂doc出现的概率
    """
    train_doc_num = len(train_mat)      # 文章数目
    vocab_num = len(train_mat[0])
    # 侮辱性文件的出现概率
    doc_abusive_probable = np.sum(train_category) / train_doc_num

    # p0num ,p0all_word     正常的统计
    # p1num ,p1all_word     侮辱的统计

    usual_every_word_count = np.ones(vocab_num)
    usual_all_word = 2
    abusive_every_word_count = np.ones(vocab_num)
    abusive_all_word = 2

    for i in range(train_doc_num):
        # 遍历所有的文件，如果是侮辱性文件，就计算此侮辱性文件中出现的侮辱性单词的个数
        if train_category[i] == 1:
            abusive_every_word_count += train_mat[i]    # 每个词上边加
            abusive_all_word += np.sum(train_mat[i])
        else:
            usual_every_word_count += train_mat[i]  # 每个词上边加
            usual_all_word += np.sum(train_mat[i])

    usual_word_probable_vec = np.log(usual_every_word_count / usual_all_word)
    abusive_word_probable_vec = np.log(abusive_every_word_count / abusive_all_word)

    return usual_word_probable_vec, abusive_word_probable_vec, doc_abusive_probable


def classify_naive_bayes(vec2classify, usual_prob, abusive_prob, doc_abusive_prob):
    """
        Desc：
            朴素贝叶斯分类
            将乘法转换为加法
            P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
        Args:
            vec2classify     -- 待分类的doc向量
            usual_prob       -- 正常doc下词出现的概率
            abusive_prob     -- 谩骂doc下词出现的概率
            doc_abusive_prob -- 谩骂doc出现的概率
        Returns:
            文章的分类情况
    """
    p1 = np.sum(vec2classify * abusive_prob) + np.log(doc_abusive_prob)
    p0 = np.sum(vec2classify * usual_prob) + np.log(1 - doc_abusive_prob)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_naive_bayes():
    # 1. 加载数据集
    list_post, list_classes = load_data_set()

    # 2. 创建单词集合
    vocab_list = create_vocab_list(list_post)

    # 3. 计算单词是否出现并创建数据矩阵
    train_mat = []
    for post_in in list_post:
        train_mat.append(bag_words2vec(vocab_list, post_in))

    # 4. 训练数据
    usual_prob, abusive_prob, doc_abusive_prob = train_naive_bayes(np.array(train_mat), np.array(list_classes))

    # 5. 测试数据
    test_one = ['love', 'my', 'dalmation']
    test_one_doc = np.array(bag_words2vec(vocab_list, test_one))
    print('the result is: {}'.format(classify_naive_bayes(test_one_doc, usual_prob, abusive_prob, doc_abusive_prob)))

    test_two = ['stupid', 'garbage']
    test_two_doc = np.array(bag_words2vec(vocab_list, test_two))
    print('the result is: {}'.format(classify_naive_bayes(test_two_doc, usual_prob, abusive_prob, doc_abusive_prob)))


if __name__ == "__main__":
    testing_naive_bayes()
