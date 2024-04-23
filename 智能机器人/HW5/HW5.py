import numpy as np
import matplotlib.pyplot as plt

x_k = 0
P_k = 1
Z_k = [1, -2, 3, 2, -1, 1]
# x_k = 1
# P_k = 1
# Z_k = [-2, 3, 2, -1, 1]

P_k1_k_list = [10]
iteration_list = [0]

print(f'--------------P_k = {P_k}--------------')

for i, zk in enumerate(Z_k, start=0):
    x_k1_k = x_k
    P_k1_k = P_k + 2
    K_k1 = P_k1_k / (P_k1_k + 1)
    P_k = (1 - K_k1) * P_k1_k
    x_k = x_k1_k + K_k1 * (zk - x_k1_k)

    print("x_k1_k = ", x_k1_k)
    print("P_k1_k = ", P_k1_k)
    print("x_k = ", x_k)
    print("P_k = ", P_k)
    P_k1_k_list.append(P_k1_k)
    iteration_list.append(i)
    print('-----------------------------------')

plt.plot(iteration_list, P_k1_k_list)
plt.xlabel('i')
plt.ylabel('P_k1_k')
plt.title('P_k1_k - i')
plt.grid(True)
plt.show()