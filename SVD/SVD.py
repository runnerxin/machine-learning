#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/25 21:26
# @Author  : runnerxin

import numpy as np
import numpy.linalg as la


def load_data():

    # # 推荐引擎示例矩阵
    # return[[4, 4, 0, 2, 2],
    #        [4, 0, 0, 3, 3],
    #        [4, 0, 0, 1, 1],
    #        [1, 1, 1, 2, 0],
    #        [2, 2, 2, 0, 0],
    #        [1, 1, 1, 0, 0],
    #        [5, 5, 5, 0, 0]]

    # # 原矩阵
    # return[[1, 1, 1, 0, 0],
    #        [2, 2, 2, 0, 0],
    #        [1, 1, 1, 0, 0],
    #        [5, 5, 5, 0, 0],
    #        [1, 1, 0, 2, 2],
    #        [0, 0, 0, 3, 3],
    #        [0, 0, 0, 1, 1]]

    # 原矩阵
    return[[0, -1.6, 0.6],
           [0, 1.2,  0.8],
           [0, 0,    0],
           [0, 0,    0]]


def load_data2():
    # 书上代码给的示例矩阵
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def load_data3():
    # 利用SVD提高推荐效果，菜肴矩阵
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


def euclidean_sim(ina, inb):
    """
        Desc:
            基于欧氏距离相似度计算，假定inA和inB 都是列向量。
        Args:
            ina            向量A
            inb            向量B
        Returns:
            计算结果
    """
    return 1.0/(1.0 + la.norm(ina - inb))


def pears_sim(ina, inb):
    """
        Desc:
            函数会检查是否存在3个或更多的点。corrcoef直接计算皮尔逊相关系数，范围[-1, 1]，归一化后[0, 1]
        Args:
            ina            向量A
            inb            向量B
        Returns:
            计算结果
    """
    # 如果不存在，该函数返回1.0，此时两个向量完全相关。
    if len(ina) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(ina, inb, rowvar=False)[0][1]


def cos_sim(ina, inb):
    """
        Desc:
            计算余弦相似度，如果夹角为90度，相似度为0；如果两个向量的方向相同，相似度为1.0
        Args:
            ina            向量A
            inb            向量B
        Returns:
            计算结果
    """
    mul = float(ina.T*inb)
    norm_mul = la.norm(ina)*la.norm(inb)
    return 0.5 + 0.5*(mul/norm_mul)


def analyse_data(sigma, loop_num=20):
    """
        Desc:
            分析取前多少值时，保留的能量。根据自己的业务情况，就行处理，设置对应的 sigma 次数
            通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。
        Args:
            sigma                   sigma的值
            loop_num                循环次数
    """
    # 总方差的集合（总能量值）
    sig2 = sigma**2
    sigma_sum = sum(sig2)
    for i in range(loop_num):
        sigma_i = sum(sig2[:i+1])
        print('主成分：%s, 方差占比：%s%%' % (format(i+1, '2.0f'), format(sigma_i/sigma_sum*100, '4.2f')))


# 基于物品相似度的推荐引擎
def item_sim_recommend(data_mat, user, sim_meas, item):
    """
        Desc:
            计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分
        Args:
            data_mat                训练数据集
            user                    用户编号
            sim_meas                相似度计算方法
            item                    未评分的物品编号
        Returns:
            评分结果（0～5之间的值）
    """

    # 得到数据集中的物品数目
    n = np.shape(data_mat)[1]

    # 初始化两个评分值
    sim_total = 0.0
    rate_sim_total = 0.0

    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0:        # 未评价直接略过
            continue

        # 寻找两个用户都评级的物品。变量 over_rated_id 给出的是两个物品当中已经被评分的那个元素的索引ID
        # logical_and 计算x1和x2元素的真值。
        over_rated_id = np.nonzero(np.logical_and(data_mat[:, item].A > 0, data_mat[:, j].A > 0))[0]

        # 如果相似度为0，则两着没有任何重合元素，终止本次循环
        if len(over_rated_id) == 0:
            similarity = 0

        # 如果存在重合的物品，则基于这些重合物重新计算相似度。
        else:
            similarity = sim_meas(data_mat[over_rated_id, item], data_mat[over_rated_id, j])

        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity  用户相似度，   userRating 用户评分
        sim_total += similarity
        rate_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    # 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return rate_sim_total / sim_total


def svd_base_recommend(data_mat, user, sim_meas, item):
    """
        Desc:
            计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分
        Args:
            data_mat                训练数据集
            user                    用户编号
            sim_meas                相似度计算方法
            item                    未评分的物品编号
        Returns:
            评分结果（0～5之间的值）
    """

    # 物品数目
    n = np.shape(data_mat)[1]
    sim_total = 0.0
    rate_sim_total = 0.0

    # 奇异值分解。在SVD分解之后，我们只利用包含了90%能量值的奇异值，这些奇异值会以NumPy数组的形式得以保存
    u, sigma, vt = la.svd(data_mat)

    # # # 分析 Sigma 的长度取值
    # analyse_data(sigma, 20)

    # 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵
    sig4 = np.mat(np.eye(4) * sigma[: 4])

    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征)
    x_formed_items = data_mat.T * u[:, :4] * sig4.I

    # 对于给定的用户，for循环在用户对应行的元素上进行遍历。相似度计算时在低维空间下进行的
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating ==0 or j == item:
            continue

        # 相似度的计算方法也会作为一个参数传递给该函数
        similarity = sim_meas(x_formed_items[item, :].T, x_formed_items[j, :].T)
        # print(similarity)
        # 对相似度不断累加求和
        sim_total += similarity
        # 对相似度及对应评分值的乘积求和
        rate_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        # 计算估计评分
        return (rate_sim_total / sim_total)


def recommend(data_mat, user, recommend_num=3, sim_meas=cos_sim, recommend_method=item_sim_recommend):
    """
        Desc:
            推荐引擎，它默认调用standEst()函数，产生了最高的N个推荐结果。该函数另外的参数还包括相似度计算方法和估计方法
        Args:
            data_mat                训练数据集
            user                    用户编号
            recommend_num           推荐结果的数量
            sim_meas                相似度计算方法
            recommend_method        使用的推荐算法
        Returns:
            返回最终 recommend_num 个推荐结果
    """

    # 对给定的用户建立一个未评分的物品列表
    un_rated_items = np.nonzero(data_mat[user, :].A == 0)[1]

    # 如果不存在未评分物品，那么就退出函数
    if len(un_rated_items) == 0:
        return 'you rated everything'

    # 物品的编号和评分值
    item_score = []

    for item in un_rated_items:         # 在未评分物品上进行循环
        estimated_score = recommend_method(data_mat, user, sim_meas, item)     # 获取 item 该物品的评分
        item_score.append((item, estimated_score))

    # 按照评分得分 进行逆排序，获取前N个未评级物品进行推荐
    return sorted(item_score, key=lambda x: x[1], reverse=True)[:recommend_num]


def img_load_data(filename):    # 加载图像数据并转换数据
    img = []
    # 打开文本文件，并从文件以数组方式读入字符
    for line in open(filename).readlines():
        new_row = []
        for i in range(32):
            new_row.append(int(line[i]))
        img.append(new_row)
    my_mat = np.mat(img)
    return my_mat


def print_mat(in_mat, thresh=0.8):      # 打印矩阵
    # 由于矩阵保护了浮点数，因此定义浅色和深色，遍历所有矩阵元素，当元素大于阀值时打印1，否则打印0
    for i in range(32):
        for k in range(32):
            if float(in_mat[i, k]) > thresh:
                print(1, end="")
            else:
                print(0, end="")
        print('')


def img_compress(num_svd=3, thresh=0.8):
    """
        Desc:
            实现图像压缩，允许基于任意给定的奇异值数目来重构图像
        Args:
            num_svd            sigma长度
            thresh             判断的阈值
        Returns:
            --
    """

    # 加载数据
    myMat = img_load_data('input/0_5.txt')
    print("****original matrix****")
    print_mat(myMat)

    # 通过Sigma 重新构成SigRecom来实现.
    # Sigma是一个对角矩阵，因此需要建立一个全0矩阵，然后将前面的那些奇异值填充到对角线上。

    u, sigma, vt = la.svd(myMat)
    sig_recover = np.mat(np.eye(num_svd) * sigma[: num_svd])
    recover_mat = u[:, :num_svd] * sig_recover * vt[:num_svd, :]

    print("****reconstructed matrix using %d singular values *****" % num_svd)
    print_mat(recover_mat, thresh)


if __name__ == '__main__':
    # # 对矩阵进行SVD分解(用python实现SVD)
    # Data = load_data()
    # print('Data:', Data)
    #
    # U, Sigma, VT = la.svd(Data)
    #
    # print('U:', U)
    # print('Sigma', Sigma)
    # print('VT:', VT)
    #
    # # 重构一个3x3的矩阵Sig3
    # Sig3 = np.mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # print(U[:, :3] * Sig3 * VT[:3, :])
    #
    # # 重构一个3x3的矩阵Sig3
    # Sig3 = np.mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]], [0, 0, 0]])
    # print(U * Sig3 * VT)

    # # 计算相似度的方法
    # myMat = np.mat(load_data3())
    #
    # # item_sim_recommend 都是默认基于物品相似度的推荐引擎
    # # 计算相似度的第一种方式
    # print(recommend(myMat, 1, sim_meas=euclidean_sim))
    # # 计算相似度的第二种方式
    # print(recommend(myMat, 1, sim_meas=pears_sim))
    # # 计算相似度的第三种方式
    # print(recommend(myMat, 1, sim_meas=cos_sim))
    # # 默认推荐（菜馆菜肴推荐示例）
    # print(recommend(myMat, 2))
    #
    # # 计算相似度的方法
    # myMat = np.mat(load_data3())
    # # svd_base_recommend 基于奇异值分解的推荐引擎
    # # 计算相似度的第一种方式
    # print(recommend(myMat, 1, sim_meas=euclidean_sim, recommend_method=svd_base_recommend))
    # # 计算相似度的第二种方式
    # print(recommend(myMat, 1, sim_meas=pears_sim, recommend_method=svd_base_recommend))
    # # 计算相似度的第三种方式
    # print(recommend(myMat, 1, sim_meas=cos_sim, recommend_method=svd_base_recommend))
    # # 默认推荐（菜馆菜肴推荐示例）
    # print(recommend(myMat, 2, recommend_method=svd_base_recommend))

    # 压缩图片
    img_compress(2)
    # imgCompress(2)