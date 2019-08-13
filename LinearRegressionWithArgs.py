import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#最小二乘线性回归
x_train = np.array([1.7, 1.5, 1.3, 5, 1.3, 2.2])

y_train = np.array([368, 340, 376, 954, 332, 556])
'''
fit_intercept:模型中是否包含截距项，如果数据已经中心化(训练样本集的y的均值为0，则设置为False)
normalize:是否对输入特征X做归一化（减去均值，除以L2模），使得每个样本的模长为1。
          对数据进行归一化会使得超参数学习更鲁棒，且几乎和样本数目没有关系，回归中用的不多，更多的时候用标准化
copy_X:是否拷贝数据X，设置为False,X会被重写覆盖
n_jobs:并行计算时使用的CPU的数据 -1表示使用所有的CPU资源(与设置为CPU核的数目相同)
'''
lr = LinearRegression(True, True, False, 4)
#在scikit-learn中，训练数据x是二维数组，例子中是单维特征，需要变成二维数组
x_train = x_train.reshape(-1, 1)
#训练模型参数
lr.fit(x_train, y_train)
y_predict = lr.predict(x_train)
print("y_predict:{}".format(y_predict))
score = lr.score(x_train, y_train)
print("train score:{}".format(score))
plt.scatter(x_train, y_train, label="Train Simples")

plt.xlabel("online advertising dollars")
plt.ylabel("monthly e-commenerce sales")
plt.show()
