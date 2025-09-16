import numpy as np
import matplotlib.pyplot as plt

# 定义 x 的范围（x ∈ (1, ∞)）
x = np.linspace(1, 509, 4000)  # 从1到10，400个点

# 计算四种函数的 y 值
y1 = 1 / (x ** 0.7)           # 幂函数 y = 1/x^0.7
y2 = 1 - np.log(x)/np.log(x+1) # 对数变换 y = 1 - ln(x)/ln(x+1)
y3 = np.exp(1 - x)             # 指数衰减 y = e^(1-x)
y4 = 0.6 / (np.sqrt(x - 0.5) )     # 调整幂函数 y = 1/sqrt(x-0.5)
y5  =  (1.0 / (1+x))**0.4

# 创建画布
plt.figure(figsize=(12, 7))

# 绘制四条曲线
plt.plot(x, y1, label=r'$y = \frac{1}{x^{0.7}}$', color='blue', linewidth=2)
plt.plot(x, y2, label=r'$y = 1 - \frac{\ln(x)}{\ln(x+1)}$', color='green', linewidth=2)
plt.plot(x, y3, label=r'$y = e^{1 - x}$', color='red', linewidth=2)
plt.plot(x, y4, label=r'$y = \frac{1}{\sqrt{x - 0.5}}$', color='purple', linewidth=2)
plt.plot(x, y5, label=r'$y5', color='purple', linewidth=2)
# 添加标题和标签
plt.title('Comparison of Slowly Decreasing Concave Curves\n(Domain: $x \in (1, \infty)$, Range: $y \in (1, 0)$)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 设置坐标轴范围
# plt.xlim(1, 10)
# plt.ylim(0, 1.1)

# 添加图例并调整位置
plt.legend(fontsize=12, loc='upper right')

# 显示图像
plt.tight_layout()
plt.show()