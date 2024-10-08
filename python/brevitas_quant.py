import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int32Bias
import numpy as np
import os

# 自己对float进行量化的函数

def compute_scale(tensor, qmin=-128, qmax=127):
    """
    计算对称量化的scale因子。
    """
    max_val = np.max(np.abs(tensor))
    scale = max_val / qmax
    return scale

def symmetric_quantize(tensor, scale):
    """
    对称量化给定的张量到INT8。
    """
    q_tensor = np.round(tensor / scale).astype(np.int8)
    return q_tensor

def symmetric_quantize_int32(tensor, scale):
    """
    对称量化给定的张量到INT32。 // bias
    """
    q_tensor = np.round(tensor / scale).astype(np.int32)
    return q_tensor

def symmetric_dequantize(q_tensor, scale):
    """
    反量化给定的INT8张量到浮点数。
    """
    return q_tensor * scale

#---------------------------------------------------#
#   brevitas量化
#---------------------------------------------------#

################### Linear层量化 ###################
# torch.manual_seed(0)
# from brevitas.nn import QuantIdentity

# ROWS =  4 # 矩阵行数 I
# COL1S = 4 # 矩阵列数 J
# COLS =  4 # 公共维度 K
# float_input = torch.randn(ROWS, COLS)
# quant_identity = QuantIdentity(return_quant_tensor=True) # 量化输入，开启return_quant_tensor传递输入量化因子给Linear层
# quant_linear = qnn.QuantLinear(COLS, COL1S, bias=False, return_quant_tensor=True) # 执行量化

# # 进行量化推理
# dequant_input = quant_identity(float_input)
# dequant_weight = quant_linear.weight.T # 获取linear层的float权重，实际上这就是反量化权重,注意转置权重
# dequant_output = quant_linear(dequant_input)

# # 获得量化层的参数
# quant_input_scale = dequant_input.scale # 获得input的input_scale
# quant_weight_scale = quant_linear.quant_weight_scale() # 获得Linear层的weight_scale
# print(f"quant_input_scale:\n {quant_input_scale} \n")
# print(f"quant_weight_scale:{quant_weight_scale} \n")
# print(f"quant_input_scale * quant_weight_scale:{quant_weight_scale*quant_input_scale} \n")
# print(f"quant_output_scale:{dequant_output.scale} \n") # 输出scale是权重scale和输入scale的乘积

# # float推理
# float_output = torch.matmul(float_input, dequant_weight)  # 使用浮点权重和浮点输入进行矩阵乘法

# # 模拟量化推理
# float_input = float_input.detach().cpu().numpy()
# dequant_weight = dequant_weight.detach().cpu().numpy()
# quant_input = symmetric_quantize(float_input, quant_input_scale.detach().numpy()) #得到量化后的int8输入矩阵
# quant_weight = symmetric_quantize(dequant_weight, quant_weight_scale.detach().numpy()) #得到量化后的int8权重矩阵,需要转置为4,2
# quant_output = np.matmul(quant_input.astype(np.int32), quant_weight.astype(np.int32)) # 执行量化后的矩阵乘法
# mydequant_output = symmetric_dequantize(quant_output, dequant_output.scale.detach().numpy()) # 反量化结果
# # 保存为二进制文件
# save_dir ="./data/brevitas_quant"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# quant_input.tofile(os.path.join(save_dir, 'quant_input.bin'))
# quant_weight.tofile(os.path.join(save_dir, 'quant_weight.bin'))

# # 打印输出
# print(f"Float input:\n {float_input} \n")
# print(f"Float weight:\n {dequant_weight} \n")
# print(f"Float output:\n {float_output} \n")
# print(f"brevitas deQuant input:\n {dequant_input} \n")
# print(f"brevitas deQuant weight:\n {dequant_weight} \n")
# print(f"brevitas deQuant output:\n {dequant_output}")
# print(f"Quant input:\n {quant_input} \n")
# print(f"Quant weight:\n {quant_weight} \n")
# print(f"Quant output:\n {quant_output}")
# print(f"deQuant output:\n {mydequant_output}")


################### Linear层接layernorm ###################

# torch.manual_seed(0)
# from brevitas.nn import QuantIdentity

# ROWS =  4 # 矩阵行数 I
# COL1S = 4 # 矩阵列数 J
# COLS =  4 # 公共维度 K
# float_input = torch.randn(ROWS, COLS)
# quant_identity = QuantIdentity(return_quant_tensor=True) # 量化输入，开启return_quant_tensor传递输入量化因子给Linear层
# quant_linear = qnn.QuantLinear(COLS, COL1S, bias=False, return_quant_tensor=True) # 执行量化
# layernorm = nn.LayerNorm(COL1S) # 沿着列进行layernorm

# # 进行量化推理
# dequant_input = quant_identity(float_input)
# dequant_weight = quant_linear.weight.T # 获取linear层的float权重，实际上这就是反量化权重,注意转置权重
# dequant_output = quant_linear(dequant_input)
# dequant_layernorm = layernorm(dequant_output.value) # 对上面的float矩阵乘法求Layernorm

# # 获得量化层的参数
# quant_input_scale = dequant_input.scale # 获得input的input_scale
# quant_weight_scale = quant_linear.quant_weight_scale() # 获得Linear层的weight_scale
# print(f"quant_input_scale:\n {quant_input_scale} \n")
# print(f"quant_weight_scale:{quant_weight_scale} \n")
# print(f"quant_input_scale * quant_weight_scale:{quant_weight_scale*quant_input_scale} \n")
# print(f"quant_output_scale:{dequant_output.scale} \n") # 输出scale是权重scale和输入scale的乘积

# # float推理
# float_output = torch.matmul(float_input, dequant_weight)  # 使用浮点权重和浮点输入进行矩阵乘法

# # 模拟量化推理
# float_input = float_input.detach().cpu().numpy()
# dequant_weight = dequant_weight.detach().cpu().numpy()
# quant_input = symmetric_quantize(float_input, quant_input_scale.detach().numpy()) #得到量化后的int8输入矩阵
# quant_weight = symmetric_quantize(dequant_weight, quant_weight_scale.detach().numpy()) #得到量化后的int8权重矩阵,需要转置为4,2
# quant_output = np.matmul(quant_input.astype(np.int32), quant_weight.astype(np.int32)) # 执行量化后的矩阵乘法
# mydequant_output = symmetric_dequantize(quant_output, dequant_output.scale.detach().numpy()) # 反量化结果
# # 保存为二进制文件
# save_dir ="./data/brevitas_quant"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# quant_input.tofile(os.path.join(save_dir, 'quant_input.bin'))
# quant_weight.tofile(os.path.join(save_dir, 'quant_weight.bin'))

# # 打印输出
# print(f"Float input:\n {float_input} \n")
# print(f"Float weight:\n {dequant_weight} \n")
# print(f"Float output:\n {float_output} \n")
# print(f"layernorm output:\n {dequant_layernorm} \n")
# print(f"brevitas deQuant input:\n {dequant_input} \n")
# print(f"brevitas deQuant weight:\n {dequant_weight} \n")
# print(f"brevitas deQuant output:\n {dequant_output}")
# print(f"Quant input:\n {quant_input} \n")
# print(f"Quant weight:\n {quant_weight} \n")
# print(f"Quant output:\n {quant_output}")
# print(f"deQuant output:\n {mydequant_output}")


################### Linear层融合bias###################

# torch.manual_seed(0)
# from brevitas.nn import QuantIdentity
# import torch.nn.functional as F

# ROWS =  100 # 矩阵行数 I
# COL1S = 100 # 矩阵列数 J
# COLS =  100 # 公共维度 K
# float_input = torch.randn(ROWS, COLS)
# quant_identity = QuantIdentity(return_quant_tensor=True) # 量化输入，开启return_quant_tensor传递输入量化因子给Linear层
# quant_linear = qnn.QuantLinear(COLS, COL1S, bias=True, return_quant_tensor=True) # 执行量化，使用Bias
# layernorm = nn.LayerNorm(COL1S) # 沿着列进行layernorm

# # 进行量化推理
# dequant_input = quant_identity(float_input)
# dequant_weight = quant_linear.weight.T # 获取linear层的float权重，实际上这就是反量化权重,注意转置权重
# dequant_bias = quant_linear.bias # 获取linear层的bias权重，实际上是float偏置
# dequant_output = quant_linear(dequant_input)
# dequant_layernorm = layernorm(dequant_output.value) # 对上面的float矩阵乘法求Layernorm
# dequant_softmax = F.softmax(dequant_output.value, dim=-1) # 对上面float矩阵乘法求softmax

# # 获得量化层的参数
# quant_input_scale = dequant_input.scale # 获得input的input_scale
# quant_weight_scale = quant_linear.quant_weight_scale() # 获得Linear层的weight_scale
# print(f"quant_input_scale:\n {quant_input_scale} \n")
# print(f"quant_weight_scale:{quant_weight_scale} \n")
# print(f"quant_input_scale * quant_weight_scale:{quant_weight_scale*quant_input_scale} \n")
# print(f"quant_output_scale:{dequant_output.scale} \n") # 输出scale是权重scale和输入scale的乘积

# # float推理
# float_output = torch.matmul(float_input, dequant_weight) + dequant_bias  # 使用浮点权重和浮点输入进行矩阵乘法

# # 模拟量化推理
# # 获取float原矩阵
# float_input = float_input.detach().cpu().numpy()
# dequant_weight = dequant_weight.detach().cpu().numpy()
# dequant_bias = np.tile(dequant_bias.detach().cpu().numpy(),(ROWS, 1))# 将偏置向量复制到每一行，以形成一个新的偏置矩阵
# # 将float矩阵进行量化得到定点矩阵
# quant_input = symmetric_quantize(float_input, quant_input_scale.detach().numpy()) #得到量化后的int8输入矩阵
# quant_weight = symmetric_quantize(dequant_weight, quant_weight_scale.detach().numpy()) #得到量化后的int8权重矩阵,需要转置为4,2
# quant_bias = symmetric_quantize_int32(dequant_bias, dequant_output.scale.detach().numpy()) # 使用输出scale得到量化后的int32偏置矩阵
# quant_output = np.matmul(quant_input.astype(np.int32), quant_weight.astype(np.int32)) + quant_bias # 执行量化后的矩阵乘法
# mydequant_output = symmetric_dequantize(quant_output, dequant_output.scale.detach().numpy()) # 反量化结果
# # 保存为二进制文件
# save_dir ="./data/brevitas_quant"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# quant_input.tofile(os.path.join(save_dir, 'quant_input.bin'))
# quant_weight.tofile(os.path.join(save_dir, 'quant_weight.bin'))
# quant_bias.tofile(os.path.join(save_dir, 'quant_bias.bin'))
# np.save(os.path.join(save_dir, 'quant_input.npy'),quant_input)
# np.save(os.path.join(save_dir, 'quant_weight.npy'),quant_weight)
# np.save(os.path.join(save_dir, 'quant_bias.npy'),quant_bias)

# # 打印输出
# print(f"Float input:\n {float_input} \n")
# print(f"Float weight:\n {dequant_weight} \n")
# print(f"Float bias:\n {dequant_bias} \n")
# print(f"Float output:\n {float_output} \n")
# print(f"layernorm output:\n {dequant_layernorm} \n")
# print(f"softmax output:\n {dequant_softmax} \n")
# print(f"brevitas deQuant input:\n {dequant_input} \n")
# print(f"brevitas deQuant weight:\n {dequant_weight} \n")
# print(f"brevitas deQuant output:\n {dequant_output}")
# print(f"Quant input:\n {quant_input} \n")
# print(f"Quant weight:\n {quant_weight} \n")
# print(f"Quant bias:\n {quant_bias} \n")
# print(f"Quant output:\n {quant_output}")
# print(f"deQuant output:\n {mydequant_output}")



################### Linear层分类minist###################

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from brevitas.nn import QuantIdentity, QuantLinear
import os
import torch.nn.functional as F
from brevitas.nn import QuantReLU
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int16Bias

# 设置随机种子以确保结果可复现
torch.manual_seed(0)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# # 定义模型
# class QuantizedMLP(nn.Module):
#     def __init__(self):
#         super(QuantizedMLP, self).__init__()
#         self.quant_input = QuantIdentity(return_quant_tensor=True)
#         self.fc1 = QuantLinear(28*28, 512,output_quant=Int8ActPerTensorFloat, weight_bit_width=8, bias=True, return_quant_tensor=True)
#         self.quant_relu = QuantReLU(return_quant_tensor=True)
#         # self.fc2 = QuantLinear(512, 10, bias=True, return_quant_tensor=True)
#         self.fc2 = QuantLinear(512, 10, weight_bit_width=8, bias=True) # 输出不返回return_quant_tensor

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the image
#         x = self.quant_input(x)
#         x = self.fc1(x)
#         x = self.quant_relu(x)
#         x = self.fc2(x)
#         return x

# # 实例化模型并移动到设备
# model = QuantizedMLP().to(device)

# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 训练模型
# def train_model(model, train_loader, criterion, optimizer, epochs):
#     model.train()
#     for epoch in range(epochs):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             if batch_idx % 100 == 0:
#                 print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

# # 评估模型
# def evaluate_model(model, test_loader):
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             _, predicted = torch.max(output.data, 1)
#             correct += (predicted == target).sum().item()
#     accuracy = 100 * correct / len(test_loader.dataset)
#     print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# # 训练和评估
# train_model(model, train_loader, criterion, optimizer, epochs=5)
# evaluate_model(model, test_loader)

# # 保存模型权重
# save_dir ="./data/brevitas_quant/brevitas_minist"
# torch.save(model.state_dict(), save_dir)


########
# 加载模型字典并进行一次推理，获取层的权重scale和推理时的input scale
########

# 定义模型
class QuantizedMLP(nn.Module):
    def __init__(self):
        super(QuantizedMLP, self).__init__()
        self.quant_input = QuantIdentity(return_quant_tensor=True)
        self.fc1 = QuantLinear(28*28, 512,output_quant=Int8ActPerTensorFloat, weight_bit_width=8, bias=True, return_quant_tensor=True)
        self.quant_relu = QuantReLU(return_quant_tensor=True)
        # self.fc2 = QuantLinear(512, 10, bias=True, return_quant_tensor=True)
        self.fc2 = QuantLinear(512, 10, weight_bit_width=8, bias=True) # 输出不返回return_quant_tensor

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = self.quant_input(x)
        # print("quant_input:",x)
        quant_input_scale = x.scale
        x = self.fc1(x)
        # print("fc1_output:",x)
        fc1_output_scale = x.scale
        x = self.quant_relu(x)
        # print("relu_output:",x)
        relu_output_scale = x.scale
        x = self.fc2(x)
        # print("fc2_output:",x)
        scales = {
            'quant_input_scale': quant_input_scale,
            'fc1_output_scale': fc1_output_scale,
            'relu_output_scale': relu_output_scale,
        }
        return x, scales
    
# 创建模型的实例（不需要定义参数，只需要架构）
model = QuantizedMLP()
# 加载模型权重
save_dir ="./data/brevitas_quant/brevitas_minist"
model.load_state_dict(torch.load(save_dir))
# 将模型设置为评估模式
model.eval()

def floating_point_matrix_multiplication_with_bias(float_input, float_weight, float_bias, apply_relu = False):
    """
    执行浮点数矩阵乘法，包括偏置计算。
    
    参数:
    - float_input: 输入矩阵（numpy array）
    - float_weight: 权重矩阵（numpy array）
    - float_bias: 偏置向量（numpy array）
    
    返回:
    - float_output: 浮点数矩阵乘法的输出（numpy array）
    """
    # 执行浮点数矩阵乘法
    float_output = np.matmul(float_input, float_weight)
    # print(f"float_output.shape:\n {float_output.shape} \n")
    # print(f"float_bias.shape:\n {float_bias.shape} \n")
    # 添加偏置
    float_output += float_bias
    
    # 如果需要，应用 ReLU 激活函数
    if apply_relu:
        float_output = np.maximum(float_output, 0)

    return float_output

def quantized_matrix_multiplication_with_relu(float_input, float_weight, float_bias, input_scale, weight_scale, apply_relu=False):
    """
    自动量化输入和权重矩阵，并执行定点矩阵乘法，包括偏置计算。
    根据 apply_relu 参数决定是否应用 ReLU 激活函数。
    
    参数:
    - float_input: 输入矩阵（numpy array）
    - float_weight: 权重矩阵（numpy array）
    - float_bias: 偏置向量（numpy array）
    - input_scale: 输入量化尺度（float）
    - weight_scale: 权重量化尺度（float）
    - apply_relu: 是否应用 ReLU 激活函数（bool）
    
    返回:
    - quantized_output: 量化和反量化后的输出矩阵（numpy array）
    """
    # 对输入和权重进行量化
    quant_input = symmetric_quantize(float_input, input_scale)
    quant_weight = symmetric_quantize(float_weight, weight_scale)
    
    # 对偏置进行量化
    output_scale = input_scale * weight_scale
    quant_bias = symmetric_quantize_int32(float_bias, output_scale)
    # print(f"float_input:\n {float_input} \n")
    # print(f"float_weight:\n {float_weight} \n")
    # print(f"float_bias:\n {float_bias} \n")
    # print(f"quant_input:\n {quant_input} \n")
    # print(f"quant_weight:\n {quant_weight} \n")
    # print(f"quant_bias:\n {quant_bias} \n")
    # 执行定点矩阵乘法
    quant_output = np.matmul(quant_input.astype(np.int32), quant_weight.astype(np.int32))
    
    # 添加量化后的偏置
    quant_output += quant_bias
    
    # # 如果需要，应用 ReLU 激活函数
    # if apply_relu:
    #     dequant_output = np.maximum(quant_output, 0)

    # 反量化输出结果
    dequant_output = symmetric_dequantize(quant_output, output_scale)
    
    # 如果需要，应用 ReLU 激活函数
    if apply_relu:
        dequant_output = np.maximum(dequant_output, 0)
    
    return dequant_output


# 获取推理时的tensor的scale
# 生成一个输入tensor
test_images = torch.randn(1, 28*28)
output, scales = model(test_images)

# # 2 浮点计算forward
# fc1_float_output = floating_point_matrix_multiplication_with_bias(test_images.detach().cpu().numpy(), 
#                                                 model.fc1.weight.T.detach().cpu().numpy(), 
#                                                 model.fc1.bias.detach().cpu().numpy(),
#                                                 apply_relu=True )
# print(f"fc1_float_output:\n {fc1_float_output} \n")

# fc2_float_output = floating_point_matrix_multiplication_with_bias(fc1_float_output, 
#                                                 model.fc2.weight.T.detach().cpu().numpy(), 
#                                                 model.fc2.bias.detach().cpu().numpy(),
#                                                 apply_relu=False )
# print(f"fc2_float_output:\n {fc2_float_output} \n")


# # 2 定点计算forward
# # fc1 层的计算 对输入和权重进行量化同时进行定点矩阵乘,融合relu算子
# fc1_quant_output = quantized_matrix_multiplication_with_relu(test_images.detach().cpu().numpy(), 
#                                                    model.fc1.weight.T.detach().cpu().numpy(), 
#                                                    model.fc1.bias.detach().cpu().numpy(),
#                                                    scales['quant_input_scale'].detach().cpu().numpy(), 
#                                                    model.fc1.quant_weight_scale().detach().cpu().numpy(),
#                                                    apply_relu=True); 
# print(f"fc1_quant_output:\n {fc1_quant_output} \n")

# # fc2 层的计算 对输入和权重进行量化同时输出,不融合relu算子
# fc2_quant_output = quantized_matrix_multiplication_with_relu(fc1_quant_output, 
#                                                    model.fc2.weight.T.detach().cpu().numpy(), 
#                                                    model.fc2.bias.detach().cpu().numpy(),
#                                                    scales['fc1_output_scale'].detach().cpu().numpy(), 
#                                                    model.fc2.quant_weight_scale().detach().cpu().numpy(),
#                                                    apply_relu=False); 
# print(f"fc2_quant_output:\n {fc2_quant_output} \n")


# 将forwar中的Linear层替换为定点矩阵乘法，利用linear+relu算子搭建计算图，完成定点推理

# 原推理函数
def floating_point_forward(model, test_images):
    float_output = model(test_images)
    return float_output

# 浮点推理函数
def floating_point_inference(model, test_images):
    # 获取模型中的权重和偏置
    fc1_weight = model.fc1.weight.T.detach().cpu().numpy()
    fc1_bias = model.fc1.bias.detach().cpu().numpy()
    fc2_weight = model.fc2.weight.T.detach().cpu().numpy()
    fc2_bias = model.fc2.bias.detach().cpu().numpy()

    # 执行 fc1 层的浮点数矩阵乘法和 ReLU 激活函数
    fc1_float_output = floating_point_matrix_multiplication_with_bias(
        test_images.detach().cpu().numpy(),
        fc1_weight,
        fc1_bias,
        apply_relu=True
    )

    # 执行 fc2 层的浮点数矩阵乘法
    fc2_float_output = floating_point_matrix_multiplication_with_bias(
        fc1_float_output,
        fc2_weight,
        fc2_bias,
        apply_relu=False
    )

    return fc2_float_output


# 定点推理函数
def quantized_inference(model, test_images):
    # 获取模型参数
    fc1_weight = model.fc1.weight.T.detach().cpu().numpy()
    fc1_bias = model.fc1.bias.detach().cpu().numpy()
    fc2_weight = model.fc2.weight.T.detach().cpu().numpy()
    fc2_bias = model.fc2.bias.detach().cpu().numpy()

    # 前向推理量化尺度参数
    output, scales = model(test_images)
    # print(f"quant_input_scale:\n {scales['quant_input_scale']} \n")
    # print(f"fc1_output_scale:\n {scales['fc1_output_scale']} \n")
    # print(f"relu_output_scale:\n {scales['relu_output_scale']} \n")

    # 获取量化尺度参数
    fc1_input_scale = scales['quant_input_scale'].detach().cpu().numpy()  # 假设输入有scale
    fc2_input_scale = scales['fc1_output_scale'].detach().cpu().numpy()  # fc1和relu融合了，因此输出tensor的scale是relu的
    # fc2_input_scale = scales['relu_output_scale'].detach().cpu().numpy()  # fc1和relu融合了，因此输出tensor的scale是relu的
    fc1_weight_scale = model.fc1.quant_weight_scale().detach().cpu().numpy()
    fc2_weight_scale = model.fc2.quant_weight_scale().detach().cpu().numpy()

    # 将输入张量转换为numpy数组
    input_images = test_images.detach().cpu().numpy()

    # 执行第一层的浮点矩阵乘法和ReLU激活函数
    fc1_output = quantized_matrix_multiplication_with_relu(
        input_images, fc1_weight, fc1_bias,fc1_input_scale,fc1_weight_scale, apply_relu=True
    )

    # 执行第二层的浮点矩阵乘法
    fc2_output = quantized_matrix_multiplication_with_relu(
        fc1_output, fc2_weight, fc2_bias,fc2_input_scale,fc2_weight_scale, apply_relu=False
    )

    return fc2_output


# 执行
forward_output = floating_point_forward(model, test_images)
float_output = floating_point_inference(model, test_images)
quant_output = quantized_inference(model, test_images)
# print(f"forward_output:\n {forward_output}")
# print(f"float_output:\n {float_output}")
# print(f"quant_output:\n {quant_output}")

print(f'device: {device}')
# 评估准确率
# 评估forward推理

import time

def evaluate_forward_point_inference(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            start_time = time.time()
            float_output,scale = model(data)  # 确保使用浮点数进行推理
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"The function took {execution_time} seconds to run.")
            _, predicted = torch.max(float_output.data, 1)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Accuracy of the forward : {accuracy:.2f}%')

# 评估浮点推理
def evaluate_floating_point_inference(model, test_loader):
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten the image
            float_output = floating_point_inference(model, data)  # 确保使用浮点数进行推理
            _, predicted = torch.max(torch.tensor(float_output, dtype=torch.float32), 1)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Accuracy of the floating : {accuracy:.2f}%')


# 评估定点推理
def evaluate_quantized_inference(model, test_loader):
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # Flatten the image
        quant_output = quantized_inference(model, data)  # 使用定点推理函数
        _, predicted = torch.max(torch.tensor(quant_output, dtype=torch.float32), 1)
        correct += (predicted == target).sum().item()
        # print(f"quant_output:\n {quant_output}   ")
        # print(f"target:\n {target}   ")
        # break
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Accuracy of the quantized : {accuracy:.2f}%')



# 评估定点推理
evaluate_forward_point_inference(model, test_loader)
evaluate_floating_point_inference(model, test_loader)
evaluate_quantized_inference(model, test_loader)


# 将权重、偏置、scale、输入测试数据导出保存为bin文件用于测试前向推理

# 加载模型
model = QuantizedMLP() # 创建模型的实例（不需要定义参数，只需要架构）
save_dir ="./data/brevitas_quant/brevitas_minist" # 加载模型权重
model.load_state_dict(torch.load(save_dir))
model.eval() # 将模型设置为评估模式

# 保存路径创建
save_dir ="./data/brevitas_quant/minist"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存scale
test_images = torch.randn(1, 28*28)
output, scales = model(test_images)
fc1_input_scale = scales['quant_input_scale'].detach().cpu().numpy()  # 假设输入有scale
fc2_input_scale = scales['fc1_output_scale'].detach().cpu().numpy()  # fc1和relu融合了，因此输出tensor的scale是relu的
fc1_weight_scale = model.fc1.quant_weight_scale().detach().cpu().numpy()
fc2_weight_scale = model.fc2.quant_weight_scale().detach().cpu().numpy()
fc1_output_scale = fc1_input_scale * fc1_weight_scale
fc2_output_scale = fc2_input_scale * fc2_weight_scale
print(f"fc1_input_scale:\n {fc1_input_scale}   ")
print(f"fc2_input_scale:\n {fc2_input_scale}   ")
print(f"fc1_weight_scale:\n {fc1_weight_scale} ")
print(f"fc2_weight_scale:\n {fc2_weight_scale} ")
print(f"fc1_output_scale:\n {fc1_output_scale} ")
print(f"fc2_output_scale:\n {fc2_output_scale} ")

# 保存量化的权重和偏置
batch_size=1000
fc1_weight = model.fc1.weight.T.detach().cpu().numpy()
fc1_bias = model.fc1.bias.detach().cpu().numpy()
fc1_bias = np.tile(fc1_bias,(batch_size, 1))# 将偏置向量复制到每一行，以形成一个新的偏置矩阵
fc2_weight = model.fc2.weight.T.detach().cpu().numpy()
fc2_bias = model.fc2.bias.detach().cpu().numpy()
fc2_bias = np.tile(fc2_bias,(batch_size, 1))# 将偏置向量复制到每一行，以形成一个新的偏置矩阵
# 对权重和偏置进行量化
quant_fc1_weight = symmetric_quantize(fc1_weight, fc1_weight_scale)
quant_fc1_bias = symmetric_quantize_int32(fc1_bias, fc1_output_scale)
quant_fc2_weight = symmetric_quantize(fc2_weight, fc2_weight_scale)
quant_fc2_bias = symmetric_quantize_int32(fc2_bias, fc2_output_scale)
quant_fc1_weight.tofile(os.path.join(save_dir, 'quant_fc1_weight.bin'))
quant_fc1_bias.tofile(os.path.join(save_dir, 'quant_fc1_bias.bin'))
quant_fc2_weight.tofile(os.path.join(save_dir, 'quant_fc2_weight.bin'))
quant_fc2_bias.tofile(os.path.join(save_dir, 'quant_fc2_bias.bin'))
print(f"quant_fc1_bias shape:\n {quant_fc1_bias.shape} ")
print(f"quant_fc2_bias shape:\n {quant_fc2_bias.shape} ")

# 遍历测试数据集，保存量化后的测试矩阵
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print(f"Batch size: {test_loader.batch_size}")
print(f"Number of batches: {len(test_loader)}")
batch_index = 0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    data = data.view(data.size(0), -1)  # 将输入图像降维成一维
    # 为当前遍历生成唯一的文件名
    data_file_name = f'data_{batch_index}.bin'
    target_file_name = f'target_{batch_index}.bin'
    # 保存到bin文件
    data.cpu().numpy().tofile(os.path.join(save_dir, data_file_name))
    target.cpu().numpy().astype(np.int32).tofile(os.path.join(save_dir, target_file_name))
    # 更新计数器
    batch_index += 1
    # print(f"data.shape {data.shape}")
    # print(f"target.shape {target.shape}")
    # print(f"target {target}")