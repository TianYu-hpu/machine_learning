import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
#Huber损失函数
#向量用大写形式表示
X_train = np.array([1.7, 1.5, 1.3, 5, 1.3, 2.2])
#标量或者预测值用小写表示
y_train = np.array([368, 340, 376, 954, 332, 556])
#使用默认的值进行初始化
huber = HuberRegressor()
#在scikit-learn中，训练数据x是二维数组，例子中是单维特征，需要变成二维数组
X_train = X_train.reshape(-1, 1)
#训练模型参数
huber.fit(X_train, y_train)
y_train_pred_huber = huber.predict(X_train)
plt.scatter(X_train, y_train_pred_huber, label="Train Simples")

plt.xlabel("online advertising dollars")
plt.ylabel("monthly e-commenerce sales")
plt.show()
