import numpy as np
import matplotlib.pyplot as plt

# 定义 x 的范围（注意：x > 0，因为分母不能为零）
x = np.linspace(1, 100, 800)  # 从 0.1 到 10 生成 400 个点（避免 x=0）
y1 = (1/x)**0.1 #(1 / x) ** 0.5  # 计算 y = (1/x)^0.1
y2 = np.exp(0.1-0.1*x)      # 计算 y = 1/x

# 绘制图像
plt.figure()
plt.plot(x, y1, label='$y = (1/x)^{0.1}$', color='blue', linewidth=2)
plt.plot(x, y2, label='$y = 1/x$', color='red', linewidth=2)

# 添加标题和标签
plt.title('Comparison of $y = (1/x)^{0.1}$ and $y = 1/x$', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# 设置坐标轴范围（可选）
# plt.xlim(0, 10)
# plt.ylim(0, 10)

# 显示图像
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# x = np.linspace(0, 5, 400)
# y = np.exp(-x)  # y = e^{-x}

# plt.figure(figsize=(8, 5))
# plt.plot(x, y, label=r'$y = e^{-x}$', color='blue', linewidth=2)
# plt.title("Slowly Decreasing Concave Curve ($y = e^{-x}$)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.show()