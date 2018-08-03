#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/7/29 18:22
# @Author  : runnerxin

import numpy as np
import re
import random


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
    negative_doc = np.sum(train_category) / train_doc_num

    # p0num ,p0all_word     正常的统计
    # p1num ,p1all_word     负例的统计

    p0num = np.ones(vocab_num)
    p0all_word = 2
    p1num = np.ones(vocab_num)
    p1all_word = 2

    for i in range(train_doc_num):
        # 遍历所有的文件，如果是侮辱性文件，就计算此侮辱性文件中出现的侮辱性单词的个数
        if train_category[i] == 1:
            p1num += train_mat[i]    # 每个词上边加
            p1all_word += np.sum(train_mat[i])
        else:
            p0num += train_mat[i]  # 每个词上边加
            p0all_word += np.sum(train_mat[i])

    p0vec = np.log(p0num / p0all_word)
    p1vec = np.log(p1num / p1all_word)

    return p0vec, p1vec, negative_doc


def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class1):
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
    p1 = np.sum(vec2classify * p1vec) + np.log(p_class1)
    p0 = np.sum(vec2classify * p0vec) + np.log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


# def text_parse(big_str):
#     """
#     这里就是做词划分
#     :param big_str: 某个被拼接后的字符串
#     :return: 全部是小写的word列表，去掉少于 2 个字符的字符串
#     """
#     import re
#     # 其实这里比较推荐用　\W+ 代替 \W*，
#     # 因为 \W*会match empty patten，在py3.5+之后就会出现什么问题，推荐自己修改尝试一下，可能就会re.split理解更深了
#     token_list = re.split(r'\W+', big_str)
#     if len(token_list) == 0:
#         print(token_list)
#     return [tok.lower() for tok in token_list if len(tok) > 2]


def text_parse(big_str):

    token_list = re.split(r'\W+', big_str)
    return [tok.lower() for tok in token_list if len(tok) > 2]


def spam_test():
    # 1. 加载数据集
    doc_list = []
    class_list = []
    for i in range(1, 26):
        # 读取垃圾邮件信息
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        class_list.append(1)
        # 读取正常邮件信息
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        class_list.append(0)

    # 2. 创建单词集合

    vocab_list = create_vocab_list(doc_list)
    test_set_index = [int(num) for num in random.sample(range(50), 10)]
    train_set_index = list(set(range(50)) - set(test_set_index))

    # 3. 计算单词是否出现并创建数据矩阵
    train_mat = []
    train_class = []
    for doc_index in train_set_index:
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])

    # 4. 训练数据
    p0v, p1v, p_spam = train_naive_bayes(np.array(train_mat), np.array(train_class))

    # 5. 测试数据
    error_count = 0
    for doc_index in test_set_index:
        test_mat = set_of_words2vec(vocab_list, doc_list[doc_index])
        test_class = class_list[doc_index]
        error_count += classify_naive_bayes(test_mat, p0v, p1v, p_spam) != test_class

    print("the number of wrong classify: ", error_count)
    print('the error rate is {}'.format(error_count / len(test_set_index)))


if __name__ == "__main__":
    spam_test()
