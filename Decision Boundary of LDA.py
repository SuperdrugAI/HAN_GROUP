# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 16:22:17 2023

@author: hp
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties

# 读取数据
data=pd.read_csv(r'C:\Users\hp\Desktop\Yy LDAD.csv',encoding='gb2312')

# 清除数据集中的NaN以及无穷大的值
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# 分割特征和标签
X=data.iloc[:,1:13].values  # 确保我们有10个特征
y=data.iloc[:, 0].values

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# 标准化特征
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 应用LDA
lda = LDA(n_components=2) # 在这种情况下，降维到2D以便于绘制决策边界
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# 用LDA训练分类器
classifier = LDA()
classifier.fit(X_train_lda, y_train)

# 绘制决策边界
x_min, x_max = X_train_lda[:, 0].min() - 1, X_train_lda[:, 0].max() + 1
y_min, y_max = X_train_lda[:, 1].min() - 1, X_train_lda[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(dpi=600)  # 设置DPI
plt.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')
scatter = plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, s=50, edgecolor='k', cmap='rainbow')

# 设置标题字体为Arial
title_font = FontProperties(family='Arial')
plt.title('Decision boundary of LDA', fontproperties=title_font)
plt.show()
