from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 加载数据
data = pd.read_csv(r'C:\Users\hp\Desktop\ROC.csv', encoding='gb2312')
# 清除数据集中的NaN以及无穷大的值
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# 分割特征和标签
X = data.iloc[:, 1:13].values  # 确保我们有10个特征
y = data.iloc[:, 0].values

# 确保y只包含两类
assert len(np.unique(y)) == 2, 'y should have exactly two classes.'

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# 创建RF模型并进行训练
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 其余部分与原先相同


# 获取测试集的预测概率
y_scores = model.predict_proba(X_test)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])

# 计算AUC值
roc_auc = auc(fpr, tpr)

# Calculate Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
sensitivity = tpr
specificity = 1 - fpr

# Calculate Accuracy
accuracy = (tpr + specificity) / 2

# Output the metrics
print('Sensitivity (True Positive Rate):', sensitivity)
print('Specificity (True Negative Rate):', specificity)
print('Accuracy:', accuracy)

# Calculate the mean values
mean_sensitivity = np.mean(sensitivity)
mean_specificity = np.mean(specificity)
mean_accuracy = np.mean(accuracy)

# Plot ROC curve
title_font = FontProperties(family='Arial')
plt.figure(dpi=300, figsize=(8, 6))
roc_curve, = plt.plot(fpr, tpr, color=(0.58, 0.7, 0.9), lw=4, label='ROC curve (area = %0.2f)' % roc_auc)
line, = plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

# Create dummy lines for the specificity, sensitivity, and accuracy
spec_line = plt.Line2D([0], [0], linestyle='none', color='white', label='Mean Specificity = %0.2f' % mean_specificity)
sens_line = plt.Line2D([0], [0], linestyle='none', color='white', label='Mean Sensitivity = %0.2f' % mean_sensitivity)
accu_line = plt.Line2D([0], [0], linestyle='none', color='white', label='Mean Accuracy = %0.2f' % mean_accuracy)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic example', fontproperties=title_font)
plt.legend(handles=[roc_curve, line, spec_line, sens_line, accu_line], loc="lower right")
plt.savefig('roc_curve.png', dpi=300)
plt.show()

