import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt



# 读取训练集
train_X = np.load('./data/train/train_minist.npy')  # 数字矩阵
train_label = pd.read_csv('./data/train/train_label.csv')
train_number = train_label['number']  # 数字标签
train_size = train_label['size']  # 粗细标签
# 读取测试集
test_X = np.load('./data/test/test_minist.npy')
test_label = pd.read_csv('./data/test/test_label.csv')
test_number = test_label['number']
test_size = test_label['size']
# 查看数据集规模
print(f"训练集的尺度是：{train_X.shape}, 测试集的尺度是：{test_X.shape}")

# 第一题：Logistic回归拟合size标签
log_reg_size = LogisticRegression()
log_reg_size.fit(train_X, train_size)

# 预测测试集的size标签
pred_size = log_reg_size.predict(test_X)

# 计算准确率和F1分数
acc_size = accuracy_score(test_size, pred_size)
f1_size = f1_score(test_size, pred_size)

print("Logistic回归对size标签的预测结果：")
print("准确率：", acc_size)
print("F1分数：", f1_size)

# 第二题：Softmax回归拟合number标签
softmax_reg_number = LogisticRegression(multi_class="multinomial", solver="lbfgs")
softmax_reg_number.fit(train_X, train_number)

# 预测测试集的number标签
pred_number = softmax_reg_number.predict(test_X)

# 计算准确率和F1分数
acc_number = accuracy_score(test_number, pred_number)
f1_number = f1_score(test_number, pred_number)

print("\nSoftmax回归对number标签的预测结果：")
print("准确率：", acc_number)
print("F1分数：", f1_number)
