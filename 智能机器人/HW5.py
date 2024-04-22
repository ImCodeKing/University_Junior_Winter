import numpy as np

# x_k = 0
# P_k =
# Z_k = [1, -2, 3, 2, -1, 1]
x_k = 1
P_k = 1
Z_k = [-2, 3, 2, -1, 1]

print(f'--------------P_k = {P_k}--------------')

for zk in Z_k:
    x_k1_k = x_k
    P_k1_k = P_k + 2
    K_k1 = P_k1_k / (P_k1_k + 1)
    P_k = (1 - K_k1) * P_k1_k
    x_k = x_k1_k + K_k1 * (zk - x_k1_k)

    print("x_k1_k = ", x_k1_k)
    print("P_k1_k = ", P_k1_k)
    print("x_k = ", x_k)
    print("P_k = ", P_k)
    print('-----------------------------------')
