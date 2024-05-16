import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt



# 读取训练集
train_X = np.load('./data/train/train_minist.npy')  # 数字矩阵
train_X_flat = train_X.reshape(train_X.shape[0], -1)
train_label = pd.read_csv('./data/train/train_label.csv')
train_number = train_label['number']  # 数字标签
train_size = train_label['size']  # 粗细标签
# 读取测试集
test_X = np.load('./data/test/test_minist.npy')
test_X_flat = test_X.reshape(test_X.shape[0], -1)
test_label = pd.read_csv('./data/test/test_label.csv')
test_number = test_label['number']
test_size = test_label['size']
# 查看数据集规模
print(f"训练集的尺度是：{train_X.shape}, 测试集的尺度是：{test_X.shape}")
# ----------------------------->第一题（必做）
# Initialize the Logistic Regression model
logistic_regression = LogisticRegression(solver='sag')

# Fit the model on the training data and labels
logistic_regression.fit(train_X_flat, train_size)

# Make predictions on the test set
test_size_predictions = logistic_regression.predict(test_X_flat)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 计算准确率
accuracy = accuracy_score(test_size, test_size_predictions)

# 计算精确率
precision = precision_score(test_size, test_size_predictions)

# 计算召回率
recall = recall_score(test_size, test_size_predictions)

# 计算F1-score
f1 = f1_score(test_size, test_size_predictions)

# 计算auROC
au_roc = roc_auc_score(test_size, test_size_predictions)

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(test_size, test_size_predictions)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % au_roc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('1-Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

print("准确率 (Accuracy): {:.2f}".format(accuracy))
print("精确率 (Precision): {:.2f}".format(precision))
print("召回率 (Recall): {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))
print("auROC: {:.2f}".format(au_roc))

print("--------------------------------------------------------")
#
#
# ---------------------------->第二题（必做）

# Initialize the Logistic Regression model with softmax activation
softmax_regression = LogisticRegression(multi_class='multinomial', solver='sag')

# Fit the model on the training data and labels
softmax_regression.fit(train_X_flat, train_number)
print("aaaaaaaaaaaaaaaaa")

# Make predictions on the test set
test_number_predictions = softmax_regression.predict(test_X_flat)
test_number_predictions_proba = softmax_regression.predict_proba(test_X_flat)
print(test_number_predictions)

# 计算准确率
accuracy = accuracy_score(test_number, test_number_predictions)

# 计算精确率
precision = precision_score(test_number, test_number_predictions, average='macro')

# 计算召回率
recall = recall_score(test_number, test_number_predictions, average='macro')

# 计算F1-score
f1 = f1_score(test_number, test_number_predictions, average='macro')
print("准确率 (Accuracy): {:.2f}".format(accuracy))
print("精确率 (Precision): {:.2f}".format(precision))
print("召回率 (Recall): {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))

# 计算auROC
au_roc = roc_auc_score(test_number, test_number_predictions_proba, multi_class='ovr')

cm = confusion_matrix(test_number, test_number_predictions)

print("准确率 (Accuracy): {:.2f}".format(accuracy))
print("精确率 (Precision): {:.2f}".format(precision))
print("召回率 (Recall): {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))
print("auROC: {:.2f}".format(au_roc))
print("confusion_matrix: ", cm)


