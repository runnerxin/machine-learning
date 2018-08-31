#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/31 23:19
# @Author  : runnerxin

import numpy as np
import matplotlib.pyplot as plt
from os import listdir


class Svm:
    def __init__(self, data_in, class_labels, c, tolerance, kernel):  # Initialize the structure with the parameters

        self.X = data_in                                # 数据集
        self.Y = class_labels                           # 类别标签
        self.C = c                                      # 松弛变量(常量值)
        self.tolerance = tolerance                      # 容错率

        self.m = np.shape(data_in)[0]                   # 变量数量
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0

        # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值。
        self.eCache = np.mat(np.zeros((self.m, 2)))

        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_transform(self.X, self.X[i], kernel)


def kernel_transform(data_in, data_in_i, kernel):
    """
        Desc:
            核转换函数,转换数据到高维空间
        Args:
            data_in         dataMatIn数据集
            data_in_i       dataMatIn数据集的第i行的数据
            kernel          核函数的信息
        Returns:

    """

    m, n = np.shape(data_in)
    k = np.mat(np.zeros((m, 1)))

    if kernel[0] == 'lin':
        # linear kernel:   m*n * n*1 = m*1
        k = data_in * data_in_i.T
    elif kernel[0] == 'rbf':
        for j in range(m):
            delta_row = data_in[j, :] - data_in_i
            k[j] = delta_row * delta_row.T
        # 径向基函数的高斯版本
        k = np.exp(k / (-1 * kernel[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return k


def load_dataset(filename):
    """
        Desc:
            对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
        Args:
            fileName        -- 文件名
        Returns:
            data_mat        -- 特征矩阵
            lable_mat       -- 类标签
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
            i               -- 第一个alpha的下标
            m               -- 所有alpha的数目
        Returns:
            j               -- 返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, max_value, min_value):
    """
        Desc:
            clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
        Args:
            aj              -- 目标值
            max_value       -- 最大值
            min_value       -- 最小值
        Returns:
            aj  目标值
    """

    aj = min(aj, max_value)
    aj = max(min_value, aj)
    return aj


def calc_ek(svm, i):
    """
        Desc:
            ek误差      预测值-真实值的差
            我们预测的类别 y = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*lable[n]*x[n]
        Args:
            svm         -- svm对象
            i           -- 具体的某一行
        Returns:
            ek          -- 预测结果与真实结果比对，计算误差ek
    """

    f_k = np.multiply(svm.alphas, svm.Y).T * svm.K[:, i] + svm.b
    ek = f_k - float(svm.Y[i])
    return ek


def select_j(svm, i, ei):
    """
         Desc:
            内循环的启发式方法。选择第二个(内循环)alpha的alpha值
            这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大步长。该函数的误差与第一个alpha值Ei和下标i有关。
        Args:
            svm         -- svm对象
            i           -- 具体的某一行
            ei          -- 预测结果与真实结果比对，计算误差ei
        Returns:
            返回最优的j和ej
            j           -- 随机选出的第j一行
            ej          -- 预测结果与真实结果比对，计算误差ej
    """

    max_k = -1
    max_delta_e = 0
    ej = 0

    # 首先将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了。
    svm.eCache[i] = [1, ei]

    # 非零E值的行的list列表，所对应的alpha值
    valid_e_cache_list = np.nonzero(svm.eCache[:, 0].A)[0]
    if len(valid_e_cache_list) > 1:
        for k in valid_e_cache_list:
            if k == i:
                continue

            # 求 ek误差：预测值-真实值的差
            ek = calc_ek(svm, k)
            delta_e = abs(ei - ek)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                ej = ek
        return max_k, ej

    else:  # 如果是第一次循环，则随机选择一个alpha值
        j = select_j_rand(i, svm.m)

        # 求 Ek误差：预测值-真实值的差
        ej = calc_ek(svm, j)
    return j, ej


def update_ek(svm, i):
    """
        Desc:
            after any alpha has changed update the new value in the cache
            计算误差值并存入缓存中。在对alpha值进行优化之后会用到这个值。
        Args:
            svm         -- svm对象
            i           -- 具体的某一行
        Returns:
            --
    """
    # 求 误差：预测值-真实值的差
    ek = calc_ek(svm, i)
    svm.eCache[i] = [1, ek]


def inner_loop(svm, i):
    """
        Desc:
            内循环代码
        Args:
            svm         -- svm对象
            i           -- 具体的某一行
        Returns:
            0           -- 找不到最优的值
            1           -- 找到了最优的值，并且svm.Cache到缓存中
    """
    # 求 Ek误差：预测值-真实值的差
    ei = calc_ek(svm, i)
    # 约束条件  0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
    # 表示发生错误的概率：Y[i]*ei 如果超出了 tolerance， 才需要优化。至于正负号，我们考虑绝对值就对了。

    '''
    # 检验训练样本(xi, yi)是否满足KKT条件
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha< C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''

    if ((svm.Y[i] * ei < -svm.tolerance) and (svm.alphas[i] < svm.C) or
            (svm.Y[i] * ei > svm.tolerance) and (svm.alphas[i] > 0)):
        # 选择最大的误差对应的j进行优化。效果更明显
        j, ej = select_j(svm, i, ei)
        alpha_i_old = svm.alphas[i].copy()
        alpha_j_old = svm.alphas[j].copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if svm.Y[i] != svm.Y[j]:
            min_values = max(0, svm.alphas[j] - svm.alphas[i])
            max_values = min(svm.C, svm.C + svm.alphas[j] - svm.alphas[i])
        else:
            min_values = max(0, svm.alphas[j] + svm.alphas[i] - svm.C)
            max_values = min(svm.C, svm.alphas[j] + svm.alphas[i])
        if min_values == max_values:
            return 0

        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
        eta = 2.0 * svm.K[i, j] - svm.K[i, i] - svm.K[j, j]  # changed for kernel
        if eta >= 0:
            return 0

        # 计算出一个新的alphas[j]值
        svm.alphas[j] -= svm.Y[j] * (ei - ej) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        svm.alphas[j] = clip_alpha(svm.alphas[j], max_values, min_values)
        # 更新误差缓存
        update_ek(svm, j)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if abs(svm.alphas[j] - alpha_j_old) < 0.00001:
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        svm.alphas[i] += svm.Y[j] * svm.Y[i] * (alpha_j_old - svm.alphas[j])
        # 更新误差缓存
        update_ek(svm, i)

        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        # w= Σ[1~n] ai*yi*xi => b = yj Σ[1~n] ai*yi(xi*xj)
        # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
        b1 = svm.b - ei - svm.Y[i] * (svm.alphas[i] - alpha_i_old) * svm.K[i, i] - svm.Y[j] * (
                svm.alphas[j] - alpha_j_old) * svm.K[i, j]
        b2 = svm.b - ej - svm.Y[i] * (svm.alphas[i] - alpha_i_old) * svm.K[i, j] - svm.Y[j] * (
                svm.alphas[j] - alpha_j_old) * svm.K[j, j]
        if (0 < svm.alphas[i]) and (svm.C > svm.alphas[i]):
            svm.b = b1
        elif (0 < svm.alphas[j]) and (svm.C > svm.alphas[j]):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2
        return 1
    else:
        return 0


def smo_svm(data_in, lable_in, c, tolerance, max_iter, kernel=('lin', 0)):
    """
        Desc:
            完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
        Args:
            data_in     -- 数据集
            lable_in    -- 类别标签
            c           -- 松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            tolerance   -- 容错率
            max_iter    -- 最大的循环次数
        Returns:
            b           -- 模型的常量值
            alphas      -- 拉格朗日乘子
    """

    # 创建一个 svm 对象
    svm = Svm(np.mat(data_in), np.mat(lable_in).transpose(), c, tolerance, kernel)
    epoch = 0
    entire_set = True
    alpha_changed = 0

    # 循环遍历：循环maxIter次 并且 （alpha_changed存在可以改变 or 所有行遍历一遍）,alphaPairs还是没变化,循环结束
    while (epoch < max_iter) and (alpha_changed > 0 or entire_set):
        alpha_changed = 0

        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entire_set:
            # 在数据集上遍历所有可能的alpha, # 是否存在alpha对，存在就+1
            for i in range(svm.m):
                alpha_changed += inner_loop(svm, i)
            epoch += 1

        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            no_bounds = np.nonzero((svm.alphas.A > 0) * (svm.alphas.A < c))[0]
            for i in no_bounds:
                alpha_changed += inner_loop(svm, i)
            epoch += 1

        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entire_set:
            entire_set = False  # toggle entire set loop
        elif alpha_changed == 0:
            entire_set = True
    return svm.b, svm.alphas


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
    data_array, lable_array = load_dataset('testSet.txt')

    # b是常量值， alphas是拉格朗日乘子
    b, alphas = smo_svm(data_array, lable_array, 0.6, 0.001, 40, ['lin', 0])

    # 画图
    ws = calc_ws(alphas, data_array, lable_array)
    plot_svm(data_array, lable_array, ws, b, alphas)


def test_rbf_kernel(k1=1.3):
    data_array, lable_array = load_dataset('testSetRBF.txt')
    b, alphas = smo_svm(data_array, lable_array, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important

    data_mat = np.mat(data_array)
    label_mat = np.mat(lable_array).transpose()

    support_vector_index = np.nonzero(alphas.A > 0)[0]
    support_vector = data_mat[support_vector_index]               # get matrix of only support vectors
    label_support_vector = label_mat[support_vector_index]
    print("there are %d Support Vectors" % np.shape(support_vector)[0])
    m, n = np.shape(data_mat)

    # 训练数据集
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_transform(support_vector, data_mat[i, :], ('rbf', k1))

        # 和这个svm-simple类似： fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
        predict = kernel_eval.T * np.multiply(label_support_vector, alphas[support_vector_index]) + b
        if np.sign(predict) != np.sign(lable_array[i]):
            error_count += 1
    print("the training error rate is: %f" % (float(error_count) / m))

    # 测试数据集
    data_array, lable_array = load_dataset('testSetRBF2.txt')
    error_count = 0
    data_mat = np.mat(data_array)
    # label_mat = np.mat(lable_array).transpose()
    m, n = np.shape(data_mat)
    for i in range(m):
        kernel_eval = kernel_transform(support_vector, data_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * np.multiply(label_support_vector, alphas[support_vector_index]) + b
        if np.sign(predict) != np.sign(lable_array[i]):
            error_count += 1
    print("the test error rate is: %f" % (float(error_count) / m))


def img2vector(filename):
    vector = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            vector[0, 32 * i + j] = int(line_str[j])
    return vector


def load_images(dir_name):
    hw_labels = []
    # print(dirName)
    training_file_list = listdir(dir_name)  # load the training set
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]  # take off .txt
        # str_class = file_str.split('_')[0]
        class_num_str = int(file_str.split('_')[0])
        if class_num_str == 9:
            hw_labels.append(-1)
        else:
            hw_labels.append(1)
        training_mat[i, :] = img2vector('%s/%s' % (dir_name, file_name_str))
    return training_mat, hw_labels


def test_digits(kernel=('rbf', 10)):
    data_array, lable_array = load_images('trainingDigits')
    b, alphas = smo_svm(data_array, lable_array, 200, 0.0001, 10000, kernel)  # C=200 important

    data_mat = np.mat(data_array)
    label_mat = np.mat(lable_array).transpose()

    support_vector_index = np.nonzero(alphas.A > 0)[0]
    support_vector = data_mat[support_vector_index]               # get matrix of only support vectors
    label_support_vector = label_mat[support_vector_index]
    print("there are %d Support Vectors" % np.shape(support_vector)[0])
    m, n = np.shape(data_mat)

    # 训练数据集
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_transform(support_vector, data_mat[i, :], kernel)

        # 和这个svm-simple类似： fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
        predict = kernel_eval.T * np.multiply(label_support_vector, alphas[support_vector_index]) + b
        if np.sign(predict) != np.sign(lable_array[i]):
            error_count += 1
    print("the training error rate is: %f" % (float(error_count) / m))

    # 测试数据集
    data_array, lable_array = load_images('testDigits')
    error_count = 0
    data_mat = np.mat(data_array)
    # label_mat = np.mat(lable_array).transpose()
    m, n = np.shape(data_mat)
    for i in range(m):
        kernel_eval = kernel_transform(support_vector, data_mat[i, :], kernel)
        predict = kernel_eval.T * np.multiply(label_support_vector, alphas[support_vector_index]) + b
        if np.sign(predict) != np.sign(lable_array[i]):
            error_count += 1
    print("the test error rate is: %f" % (float(error_count) / m))


if __name__ == '__main__':
    # test()

    # # 有核函数的测试
    # test_rbf_kernel(0.8)
    # 项目实战
    # 示例：手写识别问题回顾
    # testDigits(('rbf', 0.1))
    test_digits(('rbf', 5))
    # test_digits(('rbf', 10))
    # testDigits(('rbf', 50))
    # testDigits(('rbf', 100))
    # testDigits(('lin', 10))
