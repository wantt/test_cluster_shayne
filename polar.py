import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# https://yuanbao.tencent.com/chat/naQivTmsDa/7a3c6d5a-c249-4f03-ac1b-622a16226b6f
def cartesian_to_polar(points):
    """
    将直角坐标系二维点转换为极坐标 (r, θ)
    
    参数:
        points : numpy.ndarray, 形状为 (N, 2) 的二维数组，表示N个点的直角坐标 (x, y)
    
    返回:
        polar_points : numpy.ndarray, 形状为 (N, 2) 的极坐标数组 (r, θ)，θ单位为弧度
    """
    x = points[:, 0]
    y = points[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.column_stack((r, theta))

def plot_coordinate_comparison(cartesian_points, polar_points=None, labels: np.ndarray = None,
                          centroids: np.ndarray = None,
                          out_path: str = None,
                          title: str = None):
    """
    并排绘制直角坐标和极坐标对比图，支持聚类标签着色和中心点标记
    
    参数:
        cartesian_points : numpy.ndarray, 形状为 (N, 2) 的直角坐标点 (x, y)
        polar_points     : numpy.ndarray (可选), 形状为 (N, 2) 的极坐标点 (r, θ)。若未提供则自动计算
        labels           : numpy.ndarray (可选), 形状为 (N,) 的标签数组，用于点着色
        centroids        : numpy.ndarray (可选), 形状为 (K, 2) 的中心点坐标（直角坐标系）
        out_path         : str, 输出图片路径
        title            : str (可选), 整个图的标题
    """
    if polar_points is None:
        polar_points = cartesian_to_polar(cartesian_points)
    
    r, theta = polar_points[:, 0], polar_points[:, 1]
    x, y = cartesian_points[:, 0], cartesian_points[:, 1]
    print(x[0],y[0],r[0],theta[0])
    # 创建颜色映射（如果提供了labels）
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
        color_map = {label: colors(i) for i, label in enumerate(unique_labels)}
        point_colors = [color_map[label] for label in labels]
    else:
        point_colors = 'red'
    
    fig = plt.figure(figsize=(12, 5))
    if title:
        fig.suptitle(title, fontsize=14)
    
    # ----------------- 直角坐标系 -----------------
    ax1 = fig.add_subplot(121)
    
    # 绘制数据点（按label着色）
    ax1.scatter(x, y, c=point_colors, s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
    
    # 绘制中心点（如果提供）
    if centroids is not None:
        ax1.scatter(centroids[:, 0], centroids[:, 1], 
                   c='black', marker='X', s=200, 
                   linewidths=1.5, label='Centroids')
    
    # 坐标轴和网格
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_title('Cartesian Coordinates')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # 添加图例（如果有中心点）
    if centroids is not None:
        ax1.legend()
    
    # ----------------- 极坐标系 -----------------
    ax2 = fig.add_subplot(122)
    # ax2 = fig.add_subplot(122, projection='polar')
    
    # 绘制数据点（保持与直角坐标相同的颜色）
    ax2.scatter(theta, r, c=point_colors, s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
    
    # 绘制中心点（转换为极坐标）
    if centroids is not None:
        centroids_polar = cartesian_to_polar(centroids)
        ax2.scatter(centroids_polar[:, 1], centroids_polar[:, 0],
                   c='black', marker='X', s=200,
                   linewidths=1.5, label='Centroids')
    
    # 极坐标设置
    ax2.set_title('Polar Coordinates', pad=20)
    ax2.grid(True)
    
    # 添加图例（如果有中心点）
    if centroids is not None:
        ax2.legend()
    
    plt.tight_layout()
    
    # 保存或显示
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # 测试示例
    np.random.seed(42)
    points = np.random.randn(100, 2) * 10
    labels = np.random.randint(0, 3, size=100)
    centroids = np.array([[5, 5], [-5, -5], [5, -5]])
    
    plot_coordinate_comparison(points, labels=labels, centroids=centroids, out_path=None, title="Coordinate System Comparison")