#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/7/29 14:10
# @Author  : runnerxin


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 参数
n_classes = 3
plot_colors = "bry"
plot_step = 0.02

# 加载数据
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)


clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
ans = clf.predict(x_test)

accuracy_knn = (ans == y_test).astype(int).mean()
print(accuracy_knn)


