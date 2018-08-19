#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/19 15:39
# @Author  : runnerxin

import numpy as np


def load_dataset(filename):
    """
        Desc:
            该函数读取一个以 tab 键为分隔符的文件，然后将每行的内容保存成一组浮点数,假定最后一列是结果值
        Args:
            filename        文件名
        Returns:
            data_mat        每一行的数据集array类型
    """
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        to_float = [float(x) for x in cur_line]
        data_mat.append(to_float)

    return data_mat


def reg_leaf(dataset):
    """
        Desc:
            regLeaf 是产生叶节点的函数，就是求均值，即用聚类中心点来代表这类数据
        Args:
            dataset        数据集
        Returns:
            每一个叶子结点的均值
    """
    return np.mean(dataset[:, -1])


def reg_error(dataset):
    """
        Desc:
            求这组数据的方差，即通过决策树划分，可以让靠近的数据分到同一类中去。计算总方差=方差*样本数
        Args:
            dataset        数据集
        Returns:
            这组数据的方差
    """
    # shape(dataSet)[0] 表示行数
    return np.var(dataset[:, -1]) * np.shape(dataset)[0]


def b_split_dataset(dataset, feature_index, split_value):
    """
        Desc:
            在给定特征和特征值的情况下，该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回。
            将数据集，按照feature列的value进行 二元切分
        Args:
            dataset             数据集
            feature_index       待切分的特征列
            split_value         特征列要比较的值
        Returns:
            mat0                小于等于 value 的数据集在左边
            mat1                大于 value 的数据集在右边
    """

    # nonzero(dataSet[:, feature] > value)  返回结果为true行的index下标
    mat0 = dataset[np.nonzero(dataset[:, feature_index] <= split_value)[0], :]
    mat1 = dataset[np.nonzero(dataset[:, feature_index] > split_value)[0], :]
    return mat0, mat1


def choose_best_split(dataset, leaf_function=reg_leaf, err_function=reg_error, ops=(1, 4)):
    """
        Desc:
            用最佳方式切分数据集 和 生成相应的叶节点
        Args:
            dataset             加载的原始数据集
            leaf_function       建立叶子点的函数
            err_function        误差计算函数
            ops=(1, 4)          [容许误差下降值，切分的最少样本数]
        Returns:
            best_index          feature的index坐标
            best_value          切分的最优值
    """

    # ops=(1,4)，它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
    # 防止决策树的过拟合，当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。

    tol_s = ops[0]          # 最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分
    tol_n = ops[1]          # 划分最小 size 小于，就不继续划分了

    # 如果集合size为1，也就是说全部的数据都是同一个类别，不用继续划分。
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leaf_function(dataset)

    s = err_function(dataset)  # 无分类误差的总方差和
    m, n = np.shape(dataset)
    best_s, best_index, best_value = np.inf, 0, 0

    # 循环处理每一列对应的feature值
    for feature_index in range(n-1):
        # 下面的一行表示的是将某一列全部的数据转换为行，然后设置为list形式
        for split_value in set(dataset[:, feature_index].T.tolist()[0]):
            # 对该列进行分组，然后组内的成员的val值进行 二元切分
            mat0, mat1 = b_split_dataset(dataset, feature_index, split_value)

            # 判断二元切分的方式的元素数量是否符合预期
            if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):
                continue
            new_s = err_function(mat0) + err_function(mat1)

            # 如果二元切分，算出来的误差在可接受范围内，那么就记录切分点，并记录最小误差
            # 如果划分后误差小于 bestS，则说明找到了新的bestS
            if new_s < best_s:
                best_s = new_s
                best_index = feature_index
                best_value = split_value

    # 判断二元切分的方式的元素误差是否符合预期
    if (s - best_s) < tol_s:
        return None, leaf_function(dataset)

    mat0, mat1 = b_split_dataset(dataset, best_index, best_value)
    # 对整体的成员进行判断，是否符合预期
    # 如果集合的 size 小于 tolN
    if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):  # 当最佳划分后，集合过小，也不划分，产生叶节点
        return None, leaf_function(dataset)
    return best_index, best_value


def create_tree(dataset, leaf_function=reg_leaf, err_function=reg_error, ops=(1, 4)):
    """
        Desc:
            递归函数：如果构建的是回归树，该模型是一个常数，如果是模型树，其模型是一个线性方程。
        Args:
            dataset             加载的原始数据集
            leaf_function       建立叶子点的函数
            err_function        误差计算函数
            ops=(1, 4)          [容许误差下降值，切分的最少样本数]
        Returns:
            reg_tree        决策树最后的结果
    """

    # 选择最好的切分方式： feature索引值，最优切分值

    feature, value = choose_best_split(dataset, leaf_function, err_function, ops)

    # 如果 splitting 达到一个停止条件，那么返回 val
    if feature is None:
        return value

    reg_tree = dict()
    reg_tree['split_index'] = feature
    reg_tree['split_value'] = value

    # # 大于在右边，小于在左边，分为2个数据集
    left_set, right_set = b_split_dataset(dataset, feature, value)
    # # 递归的进行调用，在左右子树中继续递归生成树
    reg_tree['left'] = create_tree(left_set, leaf_function, err_function, ops)
    reg_tree['right'] = create_tree(right_set, leaf_function, err_function, ops)

    return reg_tree


def is_tree(obj):
    """
        Desc:
            测试输入变量是否是一棵树,即是否是一个字典,
        Args:
            obj -- 输入变量
        Returns:
            返回布尔类型的结果。如果 obj 是一个字典，返回true，否则返回 false
    """
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    """
       Desc:
           从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的平均值。
           对 tree 进行塌陷处理，即返回树平均值。
       Args:
           tree -- 输入的树
       Returns:
           返回 tree 节点的平均值
    """
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):
    """
       Desc:
           从上而下找到叶节点，用测试数据集来判断将这些叶节点合并是否能降低测试误差
       Args:
           tree         待剪枝的树
           testData     剪枝所需要的测试数据 testData
       Returns:
           prune_tree   剪枝完成的树
    """

    # 判断是否测试数据集没有数据，如果没有，就直接返回tree本身的均值
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)

    # 判断分枝是否是dict字典，如果是就将测试数据集进行切分
    if is_tree(tree['right']) or is_tree(tree['left']):
        left_set, right_set = b_split_dataset(test_data, tree['split_index'], tree['split_value'])

        # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
        if is_tree(tree['left']):
            tree['left'] = prune(tree['left'], left_set)

        # 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
        if is_tree(tree['right']):
            tree['right'] = prune(tree['right'], right_set)

    if not is_tree(tree['left']) and not is_tree(tree['right']):
        left_set, right_set = b_split_dataset(test_data, tree['split_index'], tree['split_value'])

        # power(x, y)表示x的y次方
        error_before_merge = \
            sum(np.power(left_set[:, -1] - tree['left'], 2)) + sum(np.power(right_set[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = sum(np.power(test_data[:, -1] - tree_mean, 2))

        # 如果 合并的总方差 < 不合并的总方差，那么就进行合并
        if error_merge < error_before_merge:
            print("merging")
            return tree_mean
        else:
            return tree
    else:
        return tree


def linear_solve(dataset):
    """
        Desc:
            将数据集格式化成目标变量y和自变量x，执行简单的线性回归，得到ws
        Args:
            dataset     输入数据
        Returns:
            ws          执行线性回归的回归系数
            x           格式化自变量X
            y           格式化目标变量Y
    """
    m, n = np.shape(dataset)
    # print(m, n)

    x = np.mat(np.ones((m, n)))
    # y = np.mat(np.ones((m, 1)))
    x[:, 1:n] = dataset[:, 0:n-1]   # X的0列为1，常数项，用于计算平衡误差
    y = dataset[:, -1]

    # 转置矩阵*矩阵
    x_x = x.T * x
    # 如果矩阵的逆不存在，会造成程序异常
    if np.linalg.det(x_x) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
    # 最小二乘法求最优解:  w0*1+w1*x1=y
    ws = x_x.I * (x.T * y)
    return ws, x, y


def model_leaf(dataset):
    """
        Desc:
            当数据不再需要切分的时候，生成叶节点的模型。
            #得到模型的ws系数：f(x) = x0 + x1*featrue1+ x3*featrue2 ...
        Args:
            dataset -- 输入数据集
        Returns:
            调用 linear_solve 函数，返回得到的 回归系数ws
    """
    ws, x, y = linear_solve(dataset)
    return ws


def model_error(dataset):
    """
        Desc:
            在给定数据集上计算误差。
        Args:
            dataset -- 输入数据集
        Returns:
            调用 linear_solve 函数，返回 y_hat 和 Y 之间的平方误差。
    """
    ws, x, y = linear_solve(dataset)
    y_hat = x * ws
    return sum(np.power(y - y_hat, 2))


def reg_tree_eval(model, in_data):   # 为了和 modelTreeEval() 保持一致，保留两个输入参数
    """
        Desc:
            对 回归树 进行预测
        Args:
            model               指定模型，可选值为 回归树模型 或者 模型树模型，这里为回归树
            in_data              输入的测试数据
        Returns:
            float(model)        将输入的模型数据转换为 浮点数 返回
    """
    return float(model)


def model_tree_eval(model, in_data):
    """
        Desc:
            对 模型树 进行预测
        Args:
            model               输入模型，可选值为 回归树模型 或者 模型树模型，这里为模型树模型
            in_data             输入的测试数据
        Returns:
            float(X * model)    将测试数据乘以 回归系数 得到一个预测值 ，转化为 浮点数 返回
    """

    n = np.shape(in_data)[1]
    x = np.mat(np.ones((1, n + 1)))
    x[:, 1: n + 1] = in_data            # 线性预测
    return float(x * model)


def tree_forecast(tree, in_data, model_eval=reg_tree_eval):
    """
        Desc:
            对特定模型的树进行预测，可以是 回归树 也可以是 模型树
        Args:
            tree         已经训练好的树的模型
            in_data      输入的测试数据
            model_eval   预测的树的模型类型，可选值为 reg_tree_eval（回归树） 或 model_tree_eval（模型树），默认为回归树
        Returns:
            返回预测值
    """
    if not is_tree(tree):
        return model_eval(tree, in_data)

    if in_data[tree['split_index']] <= tree['split_value']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'], in_data, model_eval)
        else:
            return model_eval(tree['left'], in_data)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], in_data, model_eval)
        else:
            return model_eval(tree['right'], in_data)


def create_forecast(tree, test_data, model_eval=reg_tree_eval):
    """
        Desc:
            调用 create_forecast ，对特定模型的树进行预测，可以是 回归树 也可以是 模型树
        Args:
            tree         已经训练好的树的模型
            test_data    输入的测试数据
            model_eval   预测的树的模型类型，可选值为 reg_tree_eval（回归树） 或 model_tree_eval（模型树），默认为回归树
        Returns:
            返回预测值矩阵
    """
    m = len(test_data)
    y_hat = np.mat(np.zeros((m, 1)))
    for index in range(m):
        y_hat[index, 0] = tree_forecast(tree, np.mat(test_data[index]), model_eval)
    return y_hat


if __name__ == '__main__':
    # test_1  ----------------------------------------------------------------
    # # 1、回归树
    # myDat = load_dataset('input/data1.txt')
    # myMat = np.mat(myDat)
    # myTree = create_tree(myMat)
    # print(myTree)

    # test_2 ----------------------------------------------------------------
    # # 1、预剪枝就是：提起设置最大误差数和最少元素数
    # myDat = load_dataset('input/data3.txt')
    # myMat = np.mat(myDat)
    # myTree = create_tree(myMat, ops=(0, 1))
    # print(myTree)
    #
    # # 2、后剪枝就是：通过测试数据，对预测模型进行合并判断
    # myDatTest = load_dataset('input/data3test.txt')
    # myMat2Test = np.mat(myDatTest)
    # myFinalTree = prune(myTree, myMat2Test)
    # print('\n\n\n-------------------')
    # print(myFinalTree)

    # test_3 ----------------------------------------------------------------
    # # 模型树求解
    # myDat = load_dataset('input/data4.txt')
    # myMat = np.mat(myDat)
    # myTree = create_tree(myMat, model_leaf, model_error)
    # print(myTree)

    # test_4 ----------------------------------------------------------------
    # # 回归树 VS 模型树 VS 线性回归
    trainMat = np.mat(load_dataset('input/bikeSpeedVsIq_train.txt'))
    testMat = np.mat(load_dataset('input/bikeSpeedVsIq_test.txt'))

    # # 回归树
    myTree1 = create_tree(trainMat, reg_leaf, reg_error, ops=(1, 20))
    y_Hat1 = create_forecast(myTree1, testMat[:, 0], reg_tree_eval)
    print("回归树:", np.corrcoef(y_Hat1, testMat[:, 1], rowvar=False)[0, 1])

    # 模型树
    myTree2 = create_tree(trainMat, model_leaf, model_error, ops=(1, 20))
    y_Hat2 = create_forecast(myTree2, testMat[:, 0], model_tree_eval)
    print("模型树:", np.corrcoef(y_Hat2, testMat[:, 1], rowvar=False)[0, 1])

    # 线性回归
    ws_, x_, y_ = linear_solve(trainMat)
    m_ = len(testMat[:, 0])
    yHat3 = np.mat(np.zeros((m_, 1)))
    for i in range(np.shape(testMat)[0]):
        yHat3[i] = [1, testMat[i, 0]] * ws_
        # yHat3[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print("线性回归:", np.corrcoef(yHat3, testMat[:, 1], rowvar=False)[0, 1])
