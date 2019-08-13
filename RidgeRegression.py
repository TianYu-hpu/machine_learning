import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

x_train = np.array([1.7, 1.5, 1.3, 5, 1.3, 2.2])

y_train = np.array([368, 340, 376, 954, 332, 556])
'''
alpha:目标函数J（W;lanbuta）中的lanbuta
fit_intercept/normalize/copy_x意义同LinearRegression
'''
ridge = Ridge(1.0, True, True, False)
#在scikit-learn中，训练数据x是二维数组，例子中是单维特征，需要变成二维数组
x_train = x_train.reshape(-1, 1)
#训练模型参数
ridge.fit(x_train, y_train)

plt.scatter(x_train, y_train, label="Train Simples")
y_predict = ridge.predict(x_train)
print("y_predict:{}".format(y_predict))
score = ridge.score(x_train, y_train)
print("train score:{}".format(score))
plt.xlabel("online advertising dollars")
plt.ylabel("monthly e-commenerce sales")
plt.show()
