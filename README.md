# Stream Clustering Framework (Visualization-Ready)

本框架用于**流式输入的聚类算法**对比与可视化。你可以在 `model.py` 中实现/注册自己的在线聚类算法，
使用 `run.py` 读取统一的 2D 随机点流（范围 [-100, 100]×[-100, 100]，共 10,000 点），并在固定步长生成快照图像、输出评估指标。

> 目前 `model.py` 里只提供了 **DEMO** 级别的占位算法（非真实聚类，便于跑通管线）：
> - `demo_random`: 随机分配到固定 K 个标签（只用于演示管线/可视化，不代表任何有效聚类）。
> - `demo_grid`: 将二维平面按网格划分，以网格索引作为“簇”。

你可以删除 DEMO 算法、添加自己的算法类，并通过注册表 `ALGORITHM_REGISTRY` 暴露给 `run.py`。

## 用法

```bash
# 基本运行（默认 10000 点，间隔 1000 步保存一次快照到 outputs/）
python run.py --algos demo_random,demo_grid

# 指定点数、步长、种子等
python run.py --algos demo_random --n_points 10000 --snapshot_every 500 --seed 42

# 只看某个算法
python run.py --algos demo_grid
```

运行结束后：
- 可视化快照：`outputs/<algo_name>_step_<t>.png`
- 最终指标：`outputs/metrics.json`
- 最终散点：`outputs/<algo_name>_final.png`

## 扩展自己的算法

在 `model.py` 里：
1. 继承 `StreamClusterer`，实现 `partial_fit(self, x)`（返回该点的簇标签）和可选的 `get_state()`。
2. 在文件底部把你的类加入 `ALGORITHM_REGISTRY`，例如：

```python
from model import StreamClusterer

class MyOnlineKMeans(StreamClusterer):
    def __init__(self, dim=2, name="my_okmeans", **kwargs):
        super().__init__(dim=dim, name=name)
        # TODO: 初始化参数/质心

    def partial_fit(self, x):
        # TODO: 更新质心并返回该点的标签（int）
        return 0

    def get_state(self):
        return { "k": 0, "centroids": None }

# 注册
ALGORITHM_REGISTRY["my_okmeans"] = MyOnlineKMeans
```

`run.py` 会自动从注册表中构建算法实例（传入 `dim=2`），并在流式数据上测试。

## 指标说明（可自行扩展）
- `k`: 最终簇数（由算法状态返回，若未知会回退为标签最大值+1）
- `sse`: 若算法提供质心，则计算 SSE（组内平方和）；否则为 `null`
- 你可以在 `metrics.py` 中添加 Silhouette（注意 10k 点 O(n^2)），
  或对 10k 点做下采样计算更昂贵的指标。

## 依赖
- Python 3.8+
- numpy
- matplotlib

> 不依赖额外库。若要保存动画/GIF，可自行在 `viz.py` 中扩展（需安装 pillow/ffmpeg）。
