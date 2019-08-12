#散点图

import numpy as np
import matplotlib.pyplot as plt
#X大写表示这个变量是向量或者是矩阵的形式
X_train = np.array([1.7, 1.5, 1.3, 5, 1.3, 2.2])
#y小写表示这个矩阵是标量
y_train = np.array([368, 340, 376, 954, 332, 556])
#画散点图
plt.scatter(X_train, y_train, label="Train Simples")
#x轴解释
plt.xlabel("Online Advertising Dollars")
#y轴解释
plt.ylabel("Monthly E-commerce Sales")
#显示散点图
plt.show()
