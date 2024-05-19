import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_curve, roc_auc_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 预测函数
def predict(X, w, b):
    return sigmoid(np.dot(X, w) + b)


# 训练Logistic回归模型
def train_logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for i in range(num_iterations):
        idx = np.random.randint(m)
        x_i = X[idx]
        y_i = y[idx]

        # 计算预测值
        z = np.dot(x_i, w) + b
        h = sigmoid(z)

        # 计算梯度
        gradient_w = (h - y_i) * x_i
        gradient_b = h - y_i

        # 更新参数
        w -= learning_rate * gradient_w
        b -= learning_rate * gradient_b

    return w, b


if __name__ == '__main__':
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
    w, b = train_logistic_regression(X, y, learning_rate=0.01, num_iterations=10000)

    # Make predictions on the test set
    test_size_predictions = logistic_regression.predict(test_X_flat)

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
    plt.title('Receiver Operating Characteristic (ROC)')
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

    # Make predictions on the test set
    test_number_predictions = softmax_regression.predict(test_X_flat)
    test_number_predictions_proba = softmax_regression.predict_proba(test_X_flat)

    # 计算准确率
    accuracy = accuracy_score(test_number, test_number_predictions)

    # 计算精确率
    precision = precision_score(test_number, test_number_predictions, average='macro')

    # 计算召回率
    recall = recall_score(test_number, test_number_predictions, average='macro')

    # 计算F1-score
    f1 = f1_score(test_number, test_number_predictions, average='macro')

    # 计算auROC
    au_roc = roc_auc_score(test_number, test_number_predictions_proba, multi_class='ovr')

    cm = confusion_matrix(test_number, test_number_predictions)

    print("准确率 (Accuracy): {:.2f}".format(accuracy))
    print("精确率 (Precision): {:.2f}".format(precision))
    print("召回率 (Recall): {:.2f}".format(recall))
    print("F1-score: {:.2f}".format(f1))
    print("auROC: {:.2f}".format(au_roc))
    print("confusion_matrix: \n", cm)


