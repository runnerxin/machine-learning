#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/3 21:40
# @Author  : runnerxin

import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename):
    """
        Desc:
            对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
        Args:
            fileName 文件名
        Returns:
            data_mat  特征矩阵
            lable_mat 类标签
    """
    fr = open(filename)
    data_mat = []
    lable_mat = []
    for line in fr.readlines():
        line_array = line.strip().split('\t')
        length = len(line_array)
        data_mat.append([float(line_array[index]) for index in range(length-1)])
        lable_mat.append(float(line_array[length-1]))

    return data_mat, lable_mat


def select_j_rand(i, m):
    """
        Desc:
            随机选择一个整数j 且j不等于i
        Args:
            i  第一个alpha的下标
            m  所有alpha的数目
        Returns:
            j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, hh, ll):
    """
        Desc:
            clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
        Args:
            aj  目标值
            hh   最大值
            ll   最小值
        Returns:
            aj  目标值
    """
    if aj > hh:
        aj = hh
    if ll > aj:
        aj = ll
    return aj


def smo_svm(data_in, lable_in, c, tolerance, max_iter):
    """
        Desc:

        Args:
            data_in     -- 数据集
            lable_in    -- 类别标签
            c           -- 松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            tolerance   -- 容错率
            max_iter    -- 最大的循环次数
        Returns:

        """

    data_mat = np.mat(data_in)
    lable_mat = np.mat(lable_in).transpose()
    m, n = np.shape(data_mat)

    b = 0
    alphas = np.mat(np.zeros((m, 1)))
    epoch = 0
    while epoch < max_iter:

        # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
        alpha_changed = 0
        for i in range(m):
            # 我们预测的类别 y = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*lable[n]*x[n]
            # print(np.shape(data_mat))
            f_xi = float(np.multiply(alphas, lable_mat).T * (data_mat * data_mat[i, :].T)) + b
            # 预测结果与真实结果比对，计算误差Ei
            ei = f_xi - float(lable_mat[i])
            '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
            '''
            if ((lable_mat[i] * ei < -tolerance) and (alphas[i] < c))\
                    or ((lable_mat[i] * ei > tolerance) and (alphas[i] > 0)):
                # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
                j = select_j_rand(i, m)

                # 预测j的结果
                f_xj = float(np.multiply(alphas, lable_mat).T * (data_mat * data_mat[j, :].T)) + b
                ej = f_xj - float(lable_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
                if lable_mat[i] != lable_mat[j]:
                    ll = max(0, alphas[j] - alphas[i])
                    hh = min(c, c + alphas[j] - alphas[i])
                else:
                    ll = max(0, alphas[j] + alphas[i] - c)
                    hh = min(c, alphas[j] + alphas[i])
                # 如果相同，就没发优化了
                if ll == hh:
                    continue

                # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
                eta = 2.0 * data_mat[i, :] * data_mat[j, :].T - data_mat[i, :] * data_mat[i, :].T \
                      - data_mat[j, :] * data_mat[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 计算出一个新的alphas[j]值
                alphas[j] -= lable_mat[j] * (ei - ej) / eta
                # 并使用辅助函数，以及L和H对其进行调整
                alphas[j] = clip_alpha(alphas[j], hh, ll)
                # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    # print("j not moving enough")
                    continue
                # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
                alphas[i] += lable_mat[j] * lable_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - ei - lable_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[i, :].T - \
                     lable_mat[j] * (alphas[j] - alpha_j_old) * data_mat[i, :] * data_mat[j, :].T
                b2 = b - ej - lable_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T - \
                     lable_mat[j] * (alphas[j] - alpha_j_old) * data_mat[j, :] * data_mat[j, :].T

                if (0 < alphas[i]) and (c > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_changed += 1

        # 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
        # 知道更新完毕后，iter次循环无变化，才推出循环。
        if alpha_changed == 0:
            epoch += 1
        else:
            epoch = 0
    return b, alphas


def calc_ws(alphas, data_in, lable_in):
    """
        Desc:
            基于alpha计算w值
        Args:
            alphas      -- 拉格朗日乘子
            data_in     -- feature数据集
            lable_in    -- 目标变量数据集
        Returns:
            wc  回归系数
    """
    x = np.mat(data_in)
    label_mat = np.mat(lable_in).transpose()
    m, n = np.shape(x)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * label_mat[i], x[i, :].T)
    return w


def plot_svm(x_mat, y_mat, ws, b, alphas):
    x_mat = np.mat(x_mat)
    y_mat = np.mat(y_mat)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = np.array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 注意flatten的用法
    ax.scatter(x_mat[:, 0].flatten().A[0], x_mat[:, 1].flatten().A[0])

    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = np.arange(-1.0, 10.0, 0.1)

    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y = (-b-ws[0, 0]*x)/ws[1, 0]
    ax.plot(x, y)

    for i in range(np.shape(y_mat[0, :])[1]):
        if y_mat[0, i] > 0:
            ax.plot(x_mat[i, 0], x_mat[i, 1], 'cx')
        else:
            ax.plot(x_mat[i, 0], x_mat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(x_mat[i, 0], x_mat[i, 1], 'ro')
    plt.show()


def test():
    pass


if __name__ == '__main__':
    data_array, lable_array = load_dataset('testSet.txt')

    # b是常量值， alphas是拉格朗日乘子
    b, alphas = smo_svm(data_array, lable_array, 0.6, 0.001, 40)

    # 画图
    ws = calc_ws(alphas, data_array, lable_array)
    plot_svm(data_array, lable_array, ws, b, alphas)
