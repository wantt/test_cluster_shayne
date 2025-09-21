# import numpy as np
# import matplotlib.pyplot as plt

# # 参数设置
# lambda0 = 0.5
# max_k = 50
# growth = 9.0
# power = 2.0

# # 生成k值
# k = np.linspace(0, 100, 500)

# # 计算y值
# y = lambda0 * (1.0 + growth * ((k / max(max_k, 1)) ** power))

# # 绘制图形
# plt.figure(figsize=(10, 6))
# plt.plot(k, y, 'b-', linewidth=2)
# plt.title('Relationship between y and k', fontsize=14)
# plt.xlabel('k', fontsize=12)
# plt.ylabel('y', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)

# # 标记关键点
# plt.axvline(x=max_k, color='r', linestyle='--', label=f'max_k = {max_k}')
# plt.legend()

# plt.show()

#-----------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
max_k = 50
growth = 9.0
power = 2.0
lambda_values = [0.4, 0.5, 0.6, 0.7]  # 不同的lambda0值

# 生成k值
k = np.linspace(0, 100, 500)

# 创建图形
plt.figure(figsize=(10, 6))

# 计算并绘制每条曲线
for lambda0 in lambda_values:
    y = lambda0 * (1.0 + growth * ((k / max(max_k, 1)) ** power))
    plt.plot(k, y, label=f'λ₀ = {lambda0}', linewidth=2)

# 图表装饰
plt.title('Comparison of y-k relationships with different λ₀ values', fontsize=14)
plt.xlabel('k', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.axvline(x=max_k, color='gray', linestyle='--', label=f'max_k = {max_k}')
plt.legend()

plt.show()
'''
#___________________________

'''
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
lambda0 = 0.5
max_k = 50
power = 2.0
growth_values = [9.0, 7.0, 5.0, 2.0]  # 不同的growth值

# 生成k值
k = np.linspace(0, 100, 500)

# 创建图形
plt.figure(figsize=(10, 6))

# 计算并绘制每条曲线
for growth in growth_values:
    y = lambda0 * (1.0 + growth * ((k / max(max_k, 1)) ** power))
    plt.plot(k, y, label=f'growth = {growth}', linewidth=2)

# 图表装饰
plt.title('Comparison of y-k relationships with different growth values', fontsize=14)
plt.xlabel('k', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.axvline(x=max_k, color='gray', linestyle='--', label=f'max_k = {max_k}')
plt.legend()

plt.show()
'''

#___________________________

import numpy as np
import matplotlib.pyplot as plt

# 固定参数
lambda0 = 0.5
growth = 9.0
power = 2.0
max_k_values = [50, 30, 10, 5]  # 不同的max_k值

# 生成k值（0到100）
k = np.linspace(0, 100, 500)

# 创建图形
plt.figure(figsize=(10, 6))

# 计算并绘制每条曲线
for max_k in max_k_values:
    y = lambda0 * (1.0 + growth * ((k / max(max_k, 1)) ** power))
    plt.plot(k, y, label=f'max_k = {max_k}', linewidth=2)

# 图表装饰
plt.title('Comparison of y-k relationships with different max_k values', fontsize=14)
plt.xlabel('k', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# 标记所有max_k值
for max_k in max_k_values:
    plt.axvline(x=max_k, color='gray', linestyle='--', alpha=0.5)

plt.show()