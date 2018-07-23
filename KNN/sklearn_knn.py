#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/7/23 20:25
# @Author  : runnerxin


# from sklearn import datasets
# from sklearn import neighbors
#
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# print(len(X))
#
# knn_clf = neighbors.KNeighborsClassifier()
# knn_clf.fit(X,y)
# knn_y = knn_clf.predict(X)
# print(knn_y)
# print(y)
#
# accuracy_knn = (knn_y == y).astype(int).mean()
# print(accuracy_knn)


from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1)

knn_model = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
knn_y = knn_model.predict(x_test)

accuracy_knn = (knn_y == y_test).astype(int).mean()
print(accuracy_knn)

