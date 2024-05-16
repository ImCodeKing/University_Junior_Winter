import matplotlib.pyplot as plt

A = [0.47, 0.64, 1., 1.47, 1.60, 2.86, 3.21, 4.05, 4.71, 5.12]
B = [20.23, 27.9, 23.77, 28.85, 27.62, 36.08, 34.52, 37.45, 40.71, 46.58]

x_bar = sum(A) / 10
y_bar = sum(B) / 10
Sxx = 0
Sxy = 0
Syy = 0
MAE = 0
MSE = 0
a = 0
b = 0
c = 0

for i in range(len(A)):
    Sxx += (A[i] - x_bar) ** 2
    Sxy += (A[i] - x_bar) * (B[i] - y_bar)
    Syy += (B[i] - y_bar) ** 2

w = Sxy / Sxx

b = y_bar - w * x_bar

for i in range(len(A)):
    MAE += abs(B[i] - (w * A[i] + b)) / len(A)
    # MSE += (B[i] - (w * A[i] + b)) ** 2 / len(A)
    a += (B[i] - y_bar) ** 2
    b += (B[i] - (w*A[i]+b)) ** 2
    # c += ((w*A[i]+b) - y_bar) ** 2
r = (Sxy ** 2) / (Sxx * Syy)
print(w, b, r)
print(MAE)
print(MSE)
print(a, b, c)
# 绘制散点图
plt.scatter(A, B)

# 显示图形
plt.show()