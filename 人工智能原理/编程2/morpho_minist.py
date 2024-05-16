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
# ----------------------------->第一题（必做）
# TODO 1:使用Logistic回归拟合训练集的X数据和size标签,并对测试集进行预测
#
#
#
#
#
#
#
#
# ---------------------------->第二题（必做）
# TODO 2:使用Softmax回归拟合训练集的X数据和number标签,并对测试集进行预测
#
#
#
#
#
#
#
#

