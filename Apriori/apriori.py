#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/22 20:25
# @Author  : runnerxin


# 加载数据集
def load_dataset():
    """
        Desc:
            加载数据
    """
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(dataset):
    """
        Desc:
            创建候选集，即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
        Args:
            dataset     原始数据集
        Returns:
            frozenset   格式的 list
    """
    c1 = []
    for line in dataset:
        for item in line:
            if [item] not in c1:        # 去重
                c1.append([item])
    c1.sort()                           # 对数组进行 `从小到大` 的排序
    return list(map(frozenset, c1))    # frozenset 表示冻结的 set 集合，元素无改变；可以把它当字典的 key 来使用


def scan_d(dataset_list, candidate_key, min_support):
    """
        Desc:
            计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据
        Args:
            dataset_list        原始数据集
            candidate_key       候选项集列表
            min_support         最小支持度
        Returns:
            ret_list            支持度大于 minSupport 的集合
            support_data        候选项集支持度数据
    """
    word_frequency = {}
    for line in dataset_list:
        for key in candidate_key:       # 候选集
            if key.issubset(line):
                if key not in word_frequency.keys():
                    word_frequency[key] = 1
                else:
                    word_frequency[key] += 1

    num_items = float(len(dataset_list))
    ret_list = []
    support_data = {}
    for key in word_frequency:
        support = word_frequency[key] / num_items       # 出现频率 / 总的数据量
        if support >= min_support:          # 将满足的数据保存到 ret_list中
            ret_list.append(key)
        support_data[key] = support

    return ret_list, support_data


def apriori_gen(frequency_set, k):
    """
        Desc:
            输入频繁项集列表 frequency_set 与返回的元素个数 k，然后输出所有可能的候选项集 possible_set
        Args:
            frequency_set   频繁项集列表
            k               返回的项集元素个数（若元素的前 k-2 相同，就进行合并）
        Returns:
            possible_set    元素两两合并的数据集
    """
    possible_set = []
    len_fs = len(frequency_set)
    for i in range(len_fs):
        for j in range(i+1, len_fs):
            l1 = list(frequency_set[i])[:k-2]
            l2 = list(frequency_set[j])[:k-2]
            l1.sort()
            l2.sort()
            if l1 == l2:
                possible_set.append(frequency_set[i] | frequency_set[j])
    return possible_set


def apriori(dataset, min_support=0.5):
    """
        Desc:
            # 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
        Args:
            dataset             原始数据集
            min_support         支持度的阈值（最小支持度）
        Returns:
            satisfy_set         频繁项集的全集
            support_data        所有元素和支持度的全集
    """

    candidate_key = create_c1(dataset)
    dataset_list = list(map(set, dataset))

    satisfy_set, support_data = scan_d(dataset_list, candidate_key, min_support)
    satisfy_set = [satisfy_set]
    k = 2

    while len(satisfy_set[k-2]) > 0:
        ck = apriori_gen(satisfy_set[k-2], k)               # 找出可能的候选集（两两合并）
        lk, sup_k = scan_d(dataset_list, ck, min_support)   # 数据集在候选集上的满足条件的集合，支持度
        support_data.update(sup_k)

        if len(lk) == 0:            # 如果没有满足的集合
            break

        satisfy_set.append(lk)
        k += 1

    return satisfy_set, support_data


def test_apriori():
    # 加载测试数据集
    dataset = load_dataset()
    print('dataSet: ', dataset)

    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    # Apriori 算法生成频繁项集以及它们的支持度
    satisfy_set1, support_data1 = apriori(dataset, min_support=0.7)
    print('L(0.7): ', satisfy_set1)
    print('supportData(0.7): ', support_data1)

    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')

    # Apriori 算法生成频繁项集以及它们的支持度
    satisfy_set2, support_data2 = apriori(dataset, min_support=0.5)
    print('L(0.5): ', satisfy_set2)
    print('supportData(0.5): ', support_data2)


def calc_confidence(frequency_set, fre_item_set, support_data, brl, min_confidence=0.7):
    """
        Desc:
            对两个元素的频繁项，计算可信度。例如： {1,2}/{1} 或者 {1,2}/{2} 看是否满足条件）
        Args:
            frequency_set               频繁项集中的元素，例如: frozenset([1, 3])
            fre_item_set                频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
            support_data                所有元素的支持度的字典
            brl                         关联规则列表
            min_confidence              最小可信度
        Returns:
            confidence_satisfy_set      记录 可信度大于阈值的集合
    """

    confidence_satisfy_set = []        # 记录可信度大于最小可信度（minConf）的集合
    for item in fre_item_set:
        # 可信度定义: a -> b = support(a | b) / support(a)
        confidence = support_data[frequency_set] / support_data[frequency_set - item]
        if confidence >= min_confidence:
            brl.append((frequency_set - item, item, confidence))
            confidence_satisfy_set.append(item)

    return confidence_satisfy_set


def rules_from_confidence(frequency_set, fre_item_set, support_data, brl, min_confidence=0.7):
    """
        Desc:
            # 递归计算频繁项集的规则
        Args:
            Args:
            frequency_set               频繁项集中的元素，例如: frozenset([1, 3])
            fre_item_set                频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
            support_data                所有元素的支持度的字典
            brl                         关联规则列表
            min_confidence              最小可信度
        Returns:
            --
    """
    num_item = len(fre_item_set[0])     # 元素的长度 frozenset([2])，frozenset([2， 3])
    if len(frequency_set) > (num_item + 1):
        possible_set = apriori_gen(fre_item_set, num_item + 1)  # 候选项集长度为 num_item + 1

        # 返回可信度大于最小可信度的集合
        confidence_set = calc_confidence(frequency_set, possible_set, support_data, brl, min_confidence)

        # 计算可信度后，还有数据大于最小可信度的话，那么继续递归调用，否则跳出递归
        if len(confidence_set) > 1:
            rules_from_confidence(frequency_set, confidence_set, support_data, brl, min_confidence)


def generate_rules(frequency_set, support_data, min_confidence=0.7):
    """
        Desc:
            生成关联规则
        Args:
            frequency_set               频繁项集列表
            support_data                频繁项集的支持度
            min_confidence              最小置信度
        Returns:
            confidence_rules_list       可信度规则列表（关于 (A->B+置信度) 3个字段的组合）
    """

    confidence_rules_list = []
    for i in range(1, len(frequency_set)):          #
        for fre_set in frequency_set[i]:
            # 假设：fre_set= frozenset([1, 3]), fre_item_set=[frozenset([1]), frozenset([3])]
            fre_item_set = [frozenset([item]) for item in fre_set]
            # print(fre_item_set)
            # print('#')
            # 2 个的组合，走 else, 2 个以上的组合，走 if
            if i > 1:
                rules_from_confidence(fre_set, fre_item_set, support_data, confidence_rules_list, min_confidence)
            else:
                calc_confidence(fre_set, fre_item_set, support_data, confidence_rules_list, min_confidence)
    return confidence_rules_list


def test_generate_rules():
    # 加载测试数据集
    dataset = load_dataset()
    print('dataset: \n', dataset)
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')

    # Apriori 算法生成频繁项集以及它们的支持度
    satisfy_set1, support_data1 = apriori(dataset, min_support=0.5)
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')

    print('L(0.7): ', satisfy_set1)
    print('supportData(0.7): ', support_data1)
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')

    # 生成关联规则
    rules = generate_rules(satisfy_set1, support_data1, min_confidence=0.5)
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('rules: ', rules)


def test_vote():
    """
        Desc:
            项目案例,发现毒蘑菇的相似特性
        Args:
        Returns:
            得到全集的数据
    """
    dataset = []
    fr = open('input/mushroom.dat')
    for line in fr.readlines():
        cur = line.strip().split()
        dataset.append(cur)

    # Apriori 算法生成频繁项集以及它们的支持度
    satisfy_set1, support_data1 = apriori(dataset, min_support=0.3)

    # 2表示毒蘑菇，1表示可食用的蘑菇
    # 找出关于2的频繁子项出来，就知道如果是毒蘑菇，那么出现频繁的也可能是毒蘑菇
    for item in satisfy_set1[1]:
        if item.intersection('2'):
            print(item)
    for item in satisfy_set1[2]:
        if item.intersection('2'):
            print(item)


if __name__ == '__main__':
    # 测试 Apriori 算法
    # test_apriori()

    # 生成关联规则
    # test_generate_rules()

    test_vote()


"""
    Desc:

    Args:

    Returns:

"""
