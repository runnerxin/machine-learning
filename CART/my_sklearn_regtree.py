# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# # @Time    : 2018/8/19 23:49
# # @Author  : runnerxin
#
# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# import matplotlib.pyplot as plt
#
#
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)  # rng.rand(80, 1)即矩阵的形状是 80行，1列
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - rng.rand(16))
#
#
# # 拟合回归模型
# # regr_1 = DecisionTreeRegressor(max_depth=2)
# # regr_2 = DecisionTreeRegressor(max_depth=5)
# # regr_3 = DecisionTreeRegressor(max_depth=4)
# # regr_1.fit(X, y)
# # regr_2.fit(X, y)
# # regr_3.fit(X, y)
# #
# # # 预测
# # X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# # y_1 = regr_1.predict(X_test)
# # y_2 = regr_2.predict(X_test)
# # y_3 = regr_3.predict(X_test)
# #
# # # 绘制结果
# # plt.figure()
# # plt.scatter(X, y, c="darkorange", label="data")
# # plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# # plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# # plt.plot(X_test, y_3, color="red", label="max_depth=3", linewidth=2)
# # plt.xlabel("data")
# # plt.ylabel("target")
# # plt.title("Decision Tree Regression")
# # plt.legend()
# # plt.show()
#
# # 保持 max_depth=5 不变，增加 min_samples_leaf=6 的参数，效果进一步提升了
# regr_1 = DecisionTreeRegressor(max_depth=5)
# regr_2 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=6)
#
# regr_1.fit(X, y)
# regr_2.fit(X, y)
# # 预测
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)
#
# # 绘制结果
# plt.figure()
# plt.scatter(X, y, c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=5", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Create the dataset
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng)

regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()