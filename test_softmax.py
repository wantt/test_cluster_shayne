import numpy as np

def softmax(x, axis=1):
    # 减去最大值提高数值稳定性（防止指数爆炸）
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax_scores(scores, tau=1.0):
        scores = np.array(scores, dtype=float)
        exps =np.exp(-scores / tau)
        return exps / exps.sum()

# 示例输入（2个样本，3个类别）
x = np.array([[1.0, 2.0, 3.0]
              ])

# 沿axis=1计算（对每一行操作）
output = softmax(x, axis=1)
print(output)

# output = softmax_scores(x, tau=1)
# print(output)

# import torch
# print(torch.nn.Softmax(dim=0)(torch.tensor(x, dtype=torch.float32)))