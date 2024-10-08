import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 假设我们有一个28x28的单通道图像作为输入
input_data = torch.rand(1, 28, 28)


def quantize_tensor(x, scale, zero_point, dtype):
    x = x / scale + zero_point
    x = torch.clamp(x, 0, 255)
    return x.to(dtype)

# 量化参数示例
scale = 0.1
zero_point = 0
input_dtype = torch.uint8

# 量化输入
quantized_input = quantize_tensor(input_data, scale, zero_point, input_dtype)


def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x.float() - zero_point)

# 假设模型输出也是量化的，我们将其反量化
# 注意：这里只是为了示例。实际上，模型输出通常不是量化的，除非你明确进行了量化操作。
quantized_output = torch.tensor([120], dtype=torch.uint8)  # 假设的量化输出
output_scale = 0.1
output_zero_point = 0
dequantized_output = dequantize_tensor(quantized_output, output_scale, output_zero_point)

print(dequantized_output)

# import torch
# import numpy as np

# # 假设模型是一个简单的全连接层
# input_features = 28 * 28  # 输入特征数量，例如一个28x28的图像
# output_features = 10  # 输出特征数量，例如10类分类问题

# # 随机生成权重和偏置作为示例
# # 在实际应用中，这些将是训练好的模型参数
# weight = torch.randn((output_features, input_features), dtype=torch.float32)
# bias = torch.randn(output_features, dtype=torch.float32)


# def quantize_tensor(x, scale, zero_point, dtype):
#     x = x / scale + zero_point
#     x = torch.clamp(x, 0, 255)
#     return x.to(dtype)

# # 为简单起见，这里我们使用相同的scale和zero_point来量化所有数据
# # 在实际应用中，你可能需要为输入、权重和偏置分别计算这些值
# scale = 0.1
# zero_point = 0
# input_dtype = torch.int8

# # 准备输入数据
# input_data = torch.rand((1, input_features), dtype=torch.float32)  # 假设输入数据
# quantized_input = quantize_tensor(input_data, scale, zero_point, input_dtype)

# # 量化权重和偏置
# quantized_weight = quantize_tensor(weight, scale, zero_point, input_dtype)
# quantized_bias = quantize_tensor(bias, scale, zero_point, torch.int32)  # 偏置通常用更宽的类型


# # 执行int8矩阵乘法
# # 注意：PyTorch不直接支持int8矩阵乘法，这里我们使用float32进行演示，并假设存在一个可以执行int8乘法的函数
# # 在实际应用中，你可能需要使用专门的硬件或库来执行int8运算
# output = torch.matmul(quantized_input.float(), quantized_weight.float().t()) + quantized_bias.float()

# # 假设的int8乘法函数
# # def int8_matmul(a, b):
# #     # 实现int8矩阵乘法
# #     pass
# # output = int8_matmul(quantized_input, quantized_weight.t()) + quantized_bias

# def dequantize_tensor(q_x, scale, zero_point):
#     return scale * (q_x.float() - zero_point)

# # 反量化输出
# dequantized_output = dequantize_tensor(output, scale, zero_point)
