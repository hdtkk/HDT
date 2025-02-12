# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# t = torch.rand(2, 2, 16, 16)
# # tt = torch.rand(2, 32, 32, 8)
# gate_pool = nn.AvgPool2d(2, 2)
#
# q = gate_pool(t)
# q = q.permute(0, 2, 3, 1)
# gate = F.gumbel_softmax(q, dim=-1, hard=True)
# gate = gate.permute(0, 3, 1, 2)
# indices = gate.argmax(dim=1)
# indices_repeat = indices.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2).unsqueeze(1)
# print(indices, indices.shape)
# print(indices_repeat, indices_repeat.shape)
import torch
import torch.nn as nn
# 输入数据大小为 137
import torch
import torch.nn as nn
import numpy as np
import random
import math
def gamma():
    return lambda r: 1 - r ** 2
x = gamma()
t = np.random.uniform()
print(t)
print(x(t))
r = math.floor(x(t) * 12)
print(r)

import torch
import torch.nn.functional as F

# q = [[] for _ in range(3)]
# print(q)
# data = torch.tensor([0.42, 0.86, 0.73, 0.53, 0.20, 0.15, 0.83, 0.77]).view(1, 1, -1)
# padded_data = F.pad(data, (1, 1), mode='constant', value=0.0)
# padded_data_2 = F.pad(data_2, (1, 1), mode='constant', value=0.0)
# d = nn.Conv1d(
#     in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=0
# )
#
# q = nn.Conv1d(
#     in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
# )
# r = d(padded_data)
# t = d(padded_data_2)
# result = torch.cat((r, t), dim=-1)
# print(result)
#
# result2 = d(torch.cat((data, data_2), dim=-1))
# print(result2)

# # 创建一个长度为64的列表，其中元素为 [1, 2, 3, 4] 循环出现
# sequence = [1, 2, 3, 4] * (64 // len([1, 2, 3, 4]))
# print(sequence) # 显示列表的前20个元素进行检查



