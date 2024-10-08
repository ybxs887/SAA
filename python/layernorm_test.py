import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义矩阵的大小
num_features = 4  # 假设我们有一个10个特征的矩阵
batch_size = 4     # 假设我们的矩阵有5个样本（即批量大小为5）

# 创建一个LayerNorm层，它将在最后一个维度上进行归一化
# 即对每个样本的特征进行归一化
layernorm = nn.LayerNorm(num_features)

# 创建一个随机初始化的矩阵，这里假设每个样本是一行，特征是列
# input_matrix = torch.randn(batch_size, num_features)
input_matrix = torch.tensor([[-9.959 ,8.467  ,-3.666     ,16.5 ],
                             [9.169  ,5.724  ,1.478      ,19.358 ],
                             [16.962 ,14.464 ,-4.295     ,18.145 ],
                             [13.281 ,6.827  ,-0.0389996 ,-9.509 ]])

# 打印原始矩阵
print("Original Matrix:")
print(input_matrix)

# 应用LayerNorm
normalized_matrix = layernorm(input_matrix)

# 打印归一化后的矩阵
print("\nNormalized Matrix:")
print(normalized_matrix)


# 应用Softmax
# dim参数指定了在哪个维度上进行softmax计算，通常是特征维度
softmax_matrix = F.softmax(input_matrix, dim=-1)

# 打印softmax后的矩阵
print("\nSoftmax Matrix:")
print(softmax_matrix)



# 验证归一化效果，计算归一化后矩阵的均值和标准差
mean = normalized_matrix.mean(dim=-1)
std = normalized_matrix.std(dim=-1)
print("\nMean of each sample in the normalized matrix:")
print(mean)  # 应接近0
print("\nStandard deviation of each sample in the normalized matrix:")
print(std)   # 应接近1



# 打印LayerNorm层的可学习参数
print("LayerNorm gamma (weight):", layernorm.weight.data)
print("LayerNorm beta (bias):", layernorm.bias.data)

# 导出参数
torch.save(layernorm.state_dict(), 'layernorm_params.pth')

# 如果你想加载这些参数到另一个相同配置的LayerNorm层
layernorm_new = nn.LayerNorm(num_features)
layernorm_new.load_state_dict(torch.load('layernorm_params.pth'))

# 验证导入的参数
print("\nLoaded LayerNorm gamma (weight):", layernorm_new.weight.data)
print("Loaded LayerNorm beta (bias):", layernorm_new.bias.data)
