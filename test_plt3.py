import matplotlib.pyplot as plt
import numpy as np
'''
# 示例数据：直角坐标系下的点 (x, y)
points_cartesian = np.array([
    [1, 0],    # x=1, y=0
    [0, 1],    # x=0, y=1
    [-1, 0],   # x=-1, y=0
    [0, -1],   # x=0, y=-1
    [0.5, 0.5] # x=0.5, y=0.5
])

# 转换为极坐标 (r, θ)
x = points_cartesian[:, 0]
y = points_cartesian[:, 1]
r = np.sqrt(x**2 + y**2)          # 半径
theta = np.arctan2(y, x)          # 角度（弧度）

# 转换为极坐标的数组
points_polar = np.column_stack((r, theta))
print("极坐标 (r, θ):\n", points_polar)

# 创建极坐标子图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='polar')

# 绘制离散点
ax.scatter(theta, r, color='red', s=100, label='Points')

# 设置极坐标图的标题和网格
ax.set_title('Polar Coordinate Plot', pad=20)
ax.grid(True)

# 显示图例
ax.legend()

plt.show()


'''


import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据（直角坐标系）
np.random.seed(42)  # 固定随机种子
points_cartesian = np.random.randn(20, 2)  # 20个二维正态分布点

# 转换为极坐标
x, y = points_cartesian[:, 0], points_cartesian[:, 1]
r = np.sqrt(x**2 + y**2)          # 半径
theta = np.arctan2(y, x)          # 角度（弧度）

# 创建画布和子图
fig = plt.figure(figsize=(12, 6))

# 子图1：直角坐标系
ax1 = fig.add_subplot(121)
ax1.scatter(x, y, color='red', s=50, label='Cartesian Points')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)  # x轴
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)  # y轴
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Cartesian Coordinate System')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal')  # 保证x/y轴比例一致

# 子图2：极坐标系
ax2 = fig.add_subplot(122, projection='polar')
ax2.scatter(theta, r, color='red', s=50, label='Polar Points')
ax2.set_title('Polar Coordinate System', pad=20)
ax2.grid(True)
ax2.legend()

plt.tight_layout()  # 自动调整子图间距
plt.show()