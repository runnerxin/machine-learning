#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/26 22:43
# @Author  : runnerxin


import numpy as np


# 计算欧氏距离相似度(距离为0时相似度为1,距离非常大时相似度趋于0)
def ecludSim(inA, inB):
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


# 计算余弦相似度
def cosSim(inA, inB):
    num = np.inner(inA, inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)  # 归一化到0到1之间


# 计算在给定相似度计算方法的条件下,用户对物品的估计评分值
# 参数dataMat表示数据矩阵,user表示用户编号,simMeas表示相似度计算方法,item表示物品编号
def standEst(dataMat, user, simMeas, item):
    n = dataMat.shape[1]  # 获取物品数目
    U, Sigma, VT = np.linalg.svd(dataMat)  # 进行奇异值分解
    transform = np.dot(U[:, :4].T, dataMat)  # 对行进行压缩

    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):  # 遍历每个物品
        userRating = dataMat[user, j]  # 用户对第j个用品的评分
        if userRating == 0:
            continue
        similarity = simMeas(transform[:, item], transform[:, j])  # 比较item列与第j列物品的相似度

        simTotal += similarity
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal  # 用户评分归一化到0-5


# 产生最高的N个推荐结果,不过不指定N,默认值为3
def recommend(dataMat, user, N=3, simMeas=cosSim):
    unratedItems = np.nonzero(dataMat[user, :] == 0)[0]  # 寻找未评级的物品
    if len(unratedItems) == 0:  # 如果不存在未评分物品,则退出函数
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:  # 对所有未评分物品进行预测得分
        estimatedScore = standEst(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    # 对itemScores进行从大到小排序，返回前N个未评分物品
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


def reduce_row(datamat,num_svd=3):
    # 按行降维
    U, Sigma, VT = np.linalg.svd(datamat)
    # print(U)
    # print(Sigma)
    # print(VT)

    # 按行降维。这样就会使得本来有n个feature的矩阵，变成了有r个feature了（r < n)，这其实就是对矩阵信息的一种提炼。
    new_data = datamat * VT[:num_svd, :].T

    # sig_recover = np.mat(np.eye(num_svd) * Sigma[: num_svd])
    # new_data2 = U[:, :num_svd] * sig_recover
    # print(new_data, new_data2)
    # 两种写法都可以
    return new_data



def reduce_column(datamat,num_svd=3):
    # 按列降维
    U, Sigma, VT = np.linalg.svd(datamat)
    # print(U)
    # print(Sigma)
    # print(VT)

    # 按列降维。可以理解为，将一些相似的sample合并在一起，或者将一些没有太大价值的sample去掉
    new_data = U[:, :num_svd].T * datamat


    sig_recover = np.mat(np.eye(num_svd) * Sigma[: num_svd])
    new_data2 = sig_recover * VT[: num_svd, :]
    print(new_data, new_data2)
    # 两种写法都可以
    return new_data



if __name__ == "__main__":
    mat = np.array([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
                    [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
                    [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
                    [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
                    [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
                    [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
                    [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
                    [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
                    [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
                    [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

    # ret = recommend(mat, 1, N=3, simMeas=cosSim)
    # print(ret)

    # reduce_row(np.mat(mat), 3)

    reduce_column(np.mat(mat), 3)


