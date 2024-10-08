import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int32Bias
import numpy as np
import os
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

import brevitas.nn as qnn


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

# 卷积核的im2col
def kernels2row(kernels):
    """
    将卷积核转换为行形式，以便进行矩阵乘法
    
    参数:
    - kernels: 卷积核，其形状为 (num_kernels, channels, kernel_height, kernel_width)
    
    返回:
    - row: 转换后的行形式，其形状为 (num_kernels, channels * kernel_height * kernel_width)
    """
    num_kernels, channels, kernel_height, kernel_width = kernels.shape
    row = kernels.reshape(num_kernels, -1)

    # 转置矩阵，因为我们矩阵乘法时，wgt是weight在右边
    row = row.T
    return row


# 图像的im2col
def im2col_batch_nchw(input_data, block_size, stride, pad):
    """
    将一个批次的图像数据（NCHW格式）转换为列-major形式的矩阵。
    
    参数:
    - input_data: 输入批次的四维张量，形状为 [batch_size, channels, height, width]
    - block_size: 卷积核的大小，假设为正方形卷积核，因此是一个单一的整数
    - stride: 卷积核移动的步长
    - pad: 边缘填充的量，假设填充后尺寸为 (height + 2 * pad - block_size) / stride + 1
    
    返回:
    - cols: 列-major形式的矩阵，形状为 (channels * block_size * block_size, output_height * output_width * batch_size)
    """
    batch_size, channels, height, width = input_data.shape
    padded_height = height + 2 * pad
    padded_width = width + 2 * pad
    output_height = (padded_height - block_size) // stride + 1
    output_width = (padded_width - block_size) // stride + 1

    # 初始化一个空列表来存储每个批次的列-major矩阵
    batch_cols = []

    # 填充输入数据
    input_data = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant', constant_values=0)
    # print(f"input_data:\n {input_data} \n")
    # 执行im2col转换
    for b in range(batch_size):
        cols = np.zeros((channels * block_size * block_size, output_height * output_width), dtype=input_data.dtype)
        for c in range(channels):
            for h in range(output_height):
                for w in range(output_width):
                    rows = h * stride
                    cols_start = w * stride
                    cols[c * (block_size * block_size) : (c + 1) * (block_size * block_size), h * output_width + w] = input_data[b, c, rows:rows + block_size, cols_start:cols_start + block_size].flatten()
        batch_cols.append(cols)

    # 将所有批次的列-major矩阵水平栈起来
    cols = np.hstack(batch_cols)

    # 转置矩阵，因为我们矩阵乘法时，img是input在左边
    cols = cols.T

    return cols

def reshape_output_feature_maps(cols, batch_size, channels_out, output_height, output_width):
    """
    将列形式的矩阵重塑回输出特征图的格式。
    
    参数:
    - cols: 列形式的矩阵，形状为 (output_height * output_width * batch_size, channels_out)
    - batch_size: 批次数
    - output_height: 输出特征图的高度
    - output_width: 输出特征图的宽度
    - channels_out: 输出通道数
    
    返回:
    - output_data: 重塑后的输出特征图，形状为 (batch_size, channels_out, output_height, output_width)
    """
    # 首先将矩阵重塑为中间形状 (batch_size, output_height * output_width, channels_out)
    intermediate_shape = (batch_size, output_height * output_width, channels_out)
    intermediate = cols.reshape(intermediate_shape)
    
    # 然后将中间形状重塑为 (batch_size, output_height, output_width, channels_out)
    reshaped = intermediate.reshape(batch_size, output_height, output_width, channels_out)
    
    # 最后，交换最后两个维度以获得 (batch_size, channels_out, output_height, output_width)
    output_data = reshaped.transpose(0, 3, 1, 2)
    
    return output_data


def floating_point_conv(conv_nchw_input, conv_nchw_weight, conv_c_bias,batch_size , kernel_size , stride ,padding ,apply_relu = False):
    """
    利用im2col和我们的浮点矩阵乘法，完成batch的的conv的矩阵乘法实现

    """
    # print(f"conv_nchw_input:\n {conv_nchw_input} \n")
    # print(f"conv_nchw_weight:\n {conv_nchw_weight} \n")
    # print(f"conv_c_bias:\n {conv_c_bias} \n")

    batch_size, channels_in, input_height, input_width = conv_nchw_input.shape
    channels_out, _, kernel_height, kernel_width = conv_nchw_weight.shape
    
    # 计算输出特征图的高度和宽度
    output_height = ((input_height + 2 * padding - kernel_height) // stride) + 1
    output_width = ((input_width + 2 * padding - kernel_width) // stride) + 1

    conv_input = im2col_batch_nchw(conv_nchw_input, kernel_size, stride, padding)# im2col
    conv_weight = kernels2row(conv_nchw_weight)# kernels2row
    conv_bias = conv_c_bias# bias需要按照列添加，因为列代表输出通道数量
    
    # print(f"conv_input:\n {conv_input} \n")
    # print(f"conv_weight:\n {conv_weight} \n")
    # print(f"conv_bias:\n {conv_bias} \n")
    # print(f"conv_input.shape:\n {conv_input.shape} \n")
    # print(f"conv_weight.shape:\n {conv_weight.shape} \n")
    # print(f"conv_bias.shape:\n {conv_bias.shape} \n")
    conv_output = floating_point_matrix_multiplication_with_bias(
        conv_input,
        conv_weight,
        conv_bias,
        apply_relu=apply_relu
    )
    # 重塑输出为特征图，也就是col2im
    conv_output = reshape_output_feature_maps(conv_output, batch_size, channels_out, output_height, output_width)
    # print(f"conv_output.shape:\n {conv_output.shape} \n")

    # 重塑输出为batch行，如果下接mlp
    conv_output = conv_output.reshape(batch_size, -1)
    # print(f"conv_output.shape:\n {conv_output.shape} \n")

    return conv_output

################### CNN分类minist###################

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from brevitas.nn import QuantIdentity, QuantLinear,QuantMaxPool2d
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
# class QuantWeightActBiasLeNet(Module):
#     def __init__(self):
#         super(QuantWeightActBiasLeNet, self).__init__()
#         self.quant_inp = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)

#         # 第一个卷积层，in_channels=1对应MNIST的灰度图像
#         self.conv1 = qnn.QuantConv2d(
#             1,  # in_channels
#             6,  # out_channels
#             kernel_size=5,
#             stride=1,
#             padding=2,  # 保持尺寸
#             bias=True,
#             weight_bit_width=8,
#             bias_quant=Int32Bias,
#             output_quant=Int8ActPerTensorFloat,
#             return_quant_tensor=True
#         )
#         self.relu1 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
#         self.fc1   = qnn.QuantLinear(28*28*6, 10, bias=True, weight_bit_width=8, bias_quant=Int32Bias)

#     def forward(self, x):
#         out = self.quant_inp(x)
#         out = self.relu1(self.conv1(out))
#         out = out.reshape(out.shape[0], -1)
#         out = self.fc1(out)
#         return out

# # 实例化模型并移动到设备
# model = QuantWeightActBiasLeNet().to(device)

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
# save_dir ="./data/brevitas_quant/brevitas_minist_cnn"
# torch.save(model.state_dict(), save_dir)


########
# 加载模型字典并进行一次推理，获取层的权重scale和推理时的input scale
########

# # 评估模型
# def evaluate_model(model, test_loader):
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             print("data.shape:",data.shape)
#             output = model(data)
#             _, predicted = torch.max(output.data, 1)
#             correct += (predicted == target).sum().item()
#     accuracy = 100 * correct / len(test_loader.dataset)
#     print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# evaluate_model(model, test_loader)

# 定义模型
class QuantWeightActBiasLeNet(Module):
    def __init__(self):
        super(QuantWeightActBiasLeNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)

        # 第一个卷积层，in_channels=1对应MNIST的灰度图像
        self.conv1 = qnn.QuantConv2d(
            1,  # in_channels
            6,  # out_channels
            kernel_size=5,
            stride=1,
            padding=2,  # 保持尺寸
            bias=True,
            weight_bit_width=8,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.relu1 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.fc1   = qnn.QuantLinear(28*28*6, 10, bias=True, weight_bit_width=8, bias_quant=Int32Bias)

    def forward(self, x):
        # print("input:",x)
        out = self.quant_inp(x)
        quant_input_scale = out.scale
        # print("quant_input:",out)
        out = self.conv1(out)
        quant_conv1_scale = out.scale
        # print("conv1_output:",out)
        # print("conv1_output.shape:",out.shape)
        out = self.relu1(out)
        quant_relu1_scale = out.scale
        # print("relu1_output:",out)
        out = out.reshape(out.shape[0], -1)
        # print("relu1_reshape_output:",out)
        # print("relu1_reshape_output.shape:",out.shape)
        out = self.fc1(out)
        # print("fc1_output:",out)
        scales = {
            'quant_input_scale': quant_input_scale,
            'quant_conv1_scale': quant_conv1_scale,
            'quant_relu1_scale': quant_relu1_scale,
        }
        return out,scales

# 创建模型的实例（不需要定义参数，只需要架构）
model = QuantWeightActBiasLeNet()
# 加载模型权重
save_dir ="./data/brevitas_quant/brevitas_minist_cnn"
model.load_state_dict(torch.load(save_dir))
# 将模型设置为评估模式
model.eval()

# # 获取推理时的tensor的scale
# # 生成一个输入tensor
# test_images = torch.randn(1, 1 ,28 ,28) # 四维输入
# output,scales= model(test_images)


# # 打印查看权重
# conv1_weight = model.conv1.weight.detach().cpu().numpy()
# conv1_bias = model.conv1.bias.detach().cpu().numpy()
# # print(f"conv1_weight:\n {conv1_weight}")
# # print(f"conv1_bias:\n {conv1_bias}")
# # print(f"conv1_weight.shape:\n {conv1_weight.shape}")
# # print(f"conv1_bias.shape:\n {conv1_bias.shape}")


# # 测试batch的卷积函数

# batch_size = 2
# test_images = torch.randn(batch_size, 1 ,28 ,28) # 四维输入
# # test_images = np.arange(batch_size * 1 * 28 * 28)
# # test_images = np.reshape(test_images, (batch_size, 1, 28, 28))
# # test_images = torch.tensor(test_images)
# # print(test_images)
# # print(test_images.shape)

# output,scales= model(test_images)
# print(f"output:\n {output} \n")
# print(f"output.shape:\n {output.shape} \n")

# # 获取模型中的权重和偏置
# conv1_weight = model.conv1.weight.detach().cpu().numpy()
# conv1_bias = model.conv1.bias.detach().cpu().numpy()
# fc1_weight = model.fc1.weight.T.detach().cpu().numpy()
# fc1_bias = model.fc1.bias.detach().cpu().numpy()


# # 计算第一层conv+relu，输出被reshape为batch_size行
# conv1_relu_float_output = floating_point_conv(test_images, 
#                                         conv1_weight, 
#                                         conv1_bias,
#                                         batch_size = batch_size,
#                                         kernel_size = 5,
#                                         stride = 1,
#                                         padding = 2,
#                                         apply_relu=True )

# # 执行 fc1 层的浮点数矩阵乘法
# fc1_float_output = floating_point_matrix_multiplication_with_bias(
#     conv1_relu_float_output,
#     fc1_weight,
#     fc1_bias,
#     apply_relu=False
# )
# print(f"fc1_float_output:\n {fc1_float_output} \n")
# print(f"fc1_float_output.shape:\n {fc1_float_output.shape} \n")



# 浮点推理函数
def floating_point_inference(model, test_images):
    # 获取模型中的权重和偏置
    conv1_weight = model.conv1.weight.detach().cpu().numpy()
    conv1_bias = model.conv1.bias.detach().cpu().numpy()
    fc1_weight = model.fc1.weight.T.detach().cpu().numpy()
    fc1_bias = model.fc1.bias.detach().cpu().numpy()

    # 计算第一层conv+relu，输出被reshape为batch_size行
    conv1_relu_float_output = floating_point_conv(test_images, 
                                            conv1_weight, 
                                            conv1_bias,
                                            batch_size = test_images[0],
                                            kernel_size = 5,
                                            stride = 1,
                                            padding = 2,
                                            apply_relu=True )

    # 执行 fc1 层的浮点数矩阵乘法
    fc1_float_output = floating_point_matrix_multiplication_with_bias(
        conv1_relu_float_output,
        fc1_weight,
        fc1_bias,
        apply_relu=False
    )
    return fc1_float_output



def quant_point_conv(conv_nchw_input, conv_nchw_weight, conv_c_bias, batch_size , kernel_size , stride ,padding , input_scale, weight_scale, apply_relu = False):
    """
    利用im2col和我们的浮点矩阵乘法，完成batch的的conv的矩阵乘法实现

    """
    # print(f"conv_nchw_input:\n {conv_nchw_input} \n")
    # print(f"conv_nchw_weight:\n {conv_nchw_weight} \n")
    # print(f"conv_c_bias:\n {conv_c_bias} \n")

    batch_size, channels_in, input_height, input_width = conv_nchw_input.shape
    channels_out, _, kernel_height, kernel_width = conv_nchw_weight.shape
    
    # 计算输出特征图的高度和宽度
    output_height = ((input_height + 2 * padding - kernel_height) // stride) + 1
    output_width = ((input_width + 2 * padding - kernel_width) // stride) + 1

    # 进行im2col和kernel2row
    conv_input = im2col_batch_nchw(conv_nchw_input, kernel_size, stride, padding)# im2col
    conv_weight = kernels2row(conv_nchw_weight)# kernels2row
    conv_bias = conv_c_bias# bias需要按照列添加，因为列代表输出通道数量

    # 使用定点矩阵乘法函数，可以自行进行scale
    conv_output = quantized_matrix_multiplication_with_relu(
        conv_input, conv_weight, conv_bias,input_scale,weight_scale, apply_relu=apply_relu
    )

    # 重塑输出为特征图，也就是col2im
    conv_output = reshape_output_feature_maps(conv_output, batch_size, channels_out, output_height, output_width)

    # 重塑输出为batch行，如果下接mlp
    conv_output = conv_output.reshape(batch_size, -1)

    return conv_output


# 定点推理函数
def quantized_inference(model, test_images):
    # 获取模型中的权重和偏置
    conv1_weight = model.conv1.weight.detach().cpu().numpy()
    conv1_bias = model.conv1.bias.detach().cpu().numpy()
    fc1_weight = model.fc1.weight.T.detach().cpu().numpy()
    fc1_bias = model.fc1.bias.detach().cpu().numpy()

    # 前向推理量化尺度参数
    output, scales = model(test_images)
    # print(f"quant_input_scale:\n {scales['quant_input_scale']} \n")
    # print(f"quant_conv1_scale:\n {scales['quant_conv1_scale']} \n")
    # print(f"quant_relu1_scale:\n {scales['quant_relu1_scale']} \n")

    # 获取量化尺度参数
    conv1_input_scale = scales['quant_input_scale'].detach().cpu().numpy()  # 假设输入有scale
    fc1_input_scale = scales['quant_conv1_scale'].detach().cpu().numpy()  # fc1和relu融合了，因此输出tensor的scale是relu的
    # fc2_input_scale = scales['relu_output_scale'].detach().cpu().numpy()  # fc1和relu融合了，因此输出tensor的scale是relu的
    conv1_weight_scale = model.conv1.quant_weight_scale().detach().cpu().numpy()
    fc1_weight_scale = model.fc1.quant_weight_scale().detach().cpu().numpy()

    # 将输入张量转换为numpy数组
    input_images = test_images.detach().cpu().numpy()

    # 执行第一层的浮点矩阵乘法和ReLU激活函数
    conv1_output = quant_point_conv(input_images, 
                                    conv1_weight, 
                                    conv1_bias,
                                    batch_size = input_images[0],
                                    kernel_size = 5,
                                    stride = 1,
                                    padding = 2,
                                    input_scale =conv1_input_scale,
                                    weight_scale =conv1_weight_scale,
                                    apply_relu=True )
    
    # 执行第二层的浮点矩阵乘法
    fc2_output = quantized_matrix_multiplication_with_relu(
        conv1_output, fc1_weight, fc1_bias,fc1_input_scale,fc1_weight_scale, apply_relu=False
    )

    return fc2_output



# # 评估准确率
# # 评估forward推理
# def evaluate_forward_point_inference(model, test_loader):
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             float_output,scale = model(data)  # 确保使用浮点数进行推理
#             _, predicted = torch.max(float_output.data, 1)
#             correct += (predicted == target).sum().item()
#     accuracy = 100 * correct / len(test_loader.dataset)
#     print(f'Accuracy of the forward : {accuracy:.2f}%')

# # 评估浮点推理
# def evaluate_floating_point_inference(model, test_loader):
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             float_output = floating_point_inference(model, data)  # 确保使用浮点数进行推理
#             _, predicted = torch.max(torch.tensor(float_output, dtype=torch.float32), 1)
#             correct += (predicted == target).sum().item()
#     accuracy = 100 * correct / len(test_loader.dataset)
#     print(f'Accuracy of the floating : {accuracy:.2f}%')


# # 评估定点推理
# def evaluate_quantized_inference(model, test_loader):
#     correct = 0
#     for data, target in test_loader:
#         data, target = data.to(device), target.to(device)
#         quant_output = quantized_inference(model, data)  # 使用定点推理函数
#         _, predicted = torch.max(torch.tensor(quant_output, dtype=torch.float32), 1)
#         correct += (predicted == target).sum().item()
#         # print(f"quant_output:\n {quant_output}   ")
#         # print(f"target:\n {target}   ")
#         # break
#     accuracy = 100 * correct / len(test_loader.dataset)
#     print(f'Accuracy of the quantized : {accuracy:.2f}%')

import time



# 评估准确率
# 评估forward推理
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
            start_time = time.time()
            float_output = floating_point_inference(model, data)  # 确保使用浮点数进行推理
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"The function took {execution_time} seconds to run.")
            _, predicted = torch.max(torch.tensor(float_output, dtype=torch.float32), 1)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Accuracy of the floating : {accuracy:.2f}%')


# 评估定点推理
def evaluate_quantized_inference(model, test_loader):
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
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















# # 软件卷积
# import numpy as np

# def conv2d(input, weight, bias=None, stride=1, padding=0):
#     """
#     使用循环实现的二维卷积。

#     参数:
#     - input: 输入特征图，形状为(batch_size, channels_in, height_in, width_in)
#     - weight: 卷积核权重，形状为(channels_out, channels_in, kernel_height, kernel_width)
#     - bias: 卷积核偏置，形状为(channels_out,)
#     - stride: 卷积步长
#     - padding: 在输入特征图周围填充的0的层数

#     返回:
#     - output: 输出特征图，形状为(batch_size, channels_out, height_out, width_out)
#     """
#     batch_size, channels_in, height_in, width_in = input.shape
#     channels_out, _, kernel_height, kernel_width = weight.shape
    
#     # 计算输出特征图的尺寸
#     height_out = (height_in - kernel_height + 2 * padding) // stride + 1
#     width_out = (width_in - kernel_width + 2 * padding) // stride + 1
    
#     # 初始化输出特征图
#     output = np.zeros((batch_size, channels_out, height_out, width_out))
    
#     # 填充输入特征图
#     input_padded = np.pad(input, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    
#     # 进行卷积操作
#     for b in range(batch_size):
#         for c_out in range(channels_out):
#             for h in range(height_out):
#                 for w in range(width_out):
#                     # 计算卷积核在输入特征图上的位置
#                     h_start = h * stride
#                     h_end = h_start + kernel_height
#                     w_start = w * stride
#                     w_end = w_start + kernel_width
                    
#                     # 提取卷积核覆盖区域的输入特征图
#                     region = input_padded[b, :, h_start:h_end, w_start:w_end]
                    
#                     # 执行卷积操作
#                     output[b, c_out, h, w] = np.sum(region * weight[c_out, :, :, :])
                    
#                     # 添加偏置
#                     if bias is not None:
#                         output[b, c_out, h, w] += bias[c_out]
    
#     return output

# # im2col函数测试
# import numpy as np

# # 图像的im2col
# def im2col_batch_nchw(input_data, block_size, stride, pad):
#     """
#     将一个批次的图像数据（NCHW格式）转换为列-major形式的矩阵。
    
#     参数:
#     - input_data: 输入批次的四维张量，形状为 [batch_size, channels, height, width]
#     - block_size: 卷积核的大小，假设为正方形卷积核，因此是一个单一的整数
#     - stride: 卷积核移动的步长
#     - pad: 边缘填充的量，假设填充后尺寸为 (height + 2 * pad - block_size) / stride + 1
    
#     返回:
#     - cols: 列-major形式的矩阵，形状为 (channels * block_size * block_size, output_height * output_width * batch_size)
#     """
#     batch_size, channels, height, width = input_data.shape
#     padded_height = height + 2 * pad
#     padded_width = width + 2 * pad
#     output_height = (padded_height - block_size) // stride + 1
#     output_width = (padded_width - block_size) // stride + 1

#     # 初始化一个空列表来存储每个批次的列-major矩阵
#     batch_cols = []

#     # 填充输入数据
#     input_data = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant', constant_values=0)

#     # 执行im2col转换
#     for b in range(batch_size):
#         cols = np.zeros((channels * block_size * block_size, output_height * output_width), dtype=input_data.dtype)
#         for c in range(channels):
#             for h in range(output_height):
#                 for w in range(output_width):
#                     rows = h * stride
#                     cols_start = w * stride
#                     cols[c * (block_size * block_size) : (c + 1) * (block_size * block_size), h * output_width + w] = input_data[b, c, rows:rows + block_size, cols_start:cols_start + block_size].flatten()
#         batch_cols.append(cols)

#     # 将所有批次的列-major矩阵水平栈起来
#     cols = np.hstack(batch_cols)

#     # 转置矩阵，因为我们矩阵乘法时，img是input在左边
#     cols = cols.T

#     return cols

# # 示例使用im2col_batch_nchw
# input_data_nchw = np.random.rand(2, 3, 3, 3)  # 一个随机生成的批次，包含2个3x3的图像，3个通道
# block_size = 2
# stride = 1
# pad = 0
# cols = im2col_batch_nchw(input_data_nchw, block_size, stride, pad)
# print(f"input_data_nchw:\n {input_data_nchw} \n")
# print(f"cols.shape:\n {cols.shape} \n")
# print(f"cols:\n {cols} \n")

# # 卷积核的im2col
# def kernels2row(kernels):
#     """
#     将卷积核转换为行形式，以便进行矩阵乘法
    
#     参数:
#     - kernels: 卷积核，其形状为 (num_kernels, channels, kernel_height, kernel_width)
    
#     返回:
#     - row: 转换后的行形式，其形状为 (num_kernels, channels * kernel_height * kernel_width)
#     """
#     num_kernels, channels, kernel_height, kernel_width = kernels.shape
#     row = kernels.reshape(num_kernels, -1)

#     # 转置矩阵，因为我们矩阵乘法时，wgt是weight在右边
#     row = row.T
#     return row

# weight_data_nchw = np.random.rand(2, 3, 2, 2)  # 卷积核，2个输出通道，3个输入通道，卷积核大小2*2
# row = kernels2row(weight_data_nchw)
# print(f"weight_data_nchw:\n {weight_data_nchw} \n")
# print(f"row.shape:\n {row.shape} \n")
# print(f"row:\n {row} \n")


# # col2im函数
# def reshape_output_feature_maps(cols, batch_size, channels_out, output_height, output_width):
#     """
#     将列形式的矩阵重塑回输出特征图的格式。
    
#     参数:
#     - cols: 列形式的矩阵，形状为 (output_height * output_width * batch_size, channels_out)
#     - batch_size: 批次数
#     - output_height: 输出特征图的高度
#     - output_width: 输出特征图的宽度
#     - channels_out: 输出通道数
    
#     返回:
#     - output_data: 重塑后的输出特征图，形状为 (batch_size, channels_out, output_height, output_width)
#     """
#     # 首先将矩阵重塑为中间形状 (batch_size, output_height * output_width, channels_out)
#     intermediate_shape = (batch_size, output_height * output_width, channels_out)
#     intermediate = cols.reshape(intermediate_shape)
    
#     # 然后将中间形状重塑为 (batch_size, output_height, output_width, channels_out)
#     reshaped = intermediate.reshape(batch_size, output_height, output_width, channels_out)
    
#     # 最后，交换最后两个维度以获得 (batch_size, channels_out, output_height, output_width)
#     output_data = reshaped.transpose(0, 3, 1, 2)
    
#     return output_data

# # 示例用法：
# batch_size, channels_in, height_in, width_in = 2, 3, 3, 3
# channels_out, kernel_height, kernel_width = 2, 2, 2
# stride = 1
# padding = 0
# input = np.random.rand(batch_size, channels_in, height_in, width_in)
# weight = np.random.rand(channels_out, channels_in, kernel_height, kernel_width)
# bias = np.random.rand(channels_out)
# output = conv2d(input, weight, bias, stride=stride, padding=padding)
# print(f"input:\n {input} \n")
# print(f"weight:\n {weight} \n")
# print(f"bias:\n {bias} \n")
# print(f"output:\n {output} \n")
# print(f"output.shape:\n {output.shape} \n")

# output = output.reshape(batch_size, -1)
# print(f"output:\n {output} \n")
# print(f"output.shape:\n {output.shape} \n")

# im2col_input = im2col_batch_nchw(input, kernel_height, stride = stride, pad = padding) # 转换输入张量为col
# im2col_weight = kernels2row(weight) # 转换卷积核为row
# im2col_output = np.dot(im2col_input, im2col_weight) + bias
# print(f"im2col_input:\n {im2col_input} \n")
# print(f"im2col_weight:\n {im2col_weight} \n")
# print(f"im2col_bias:\n {bias} \n")
# print(f"im2col_output:\n {im2col_output} \n")
# print(f"im2col_output.shape:\n {im2col_output.shape} \n")

# im2col_output = reshape_output_feature_maps(im2col_output, batch_size, channels_out, 2, 2)
# im2col_output = im2col_output.reshape(batch_size, -1)
# print(f"im2col_output:\n {im2col_output} \n")
# print(f"im2col_output.shape:\n {im2col_output.shape} \n")


# im2col_output = reshape_output_feature_maps(im2col_output, batch_size, channels_out, 2, 2)
# print(f"im2col_output:\n {im2col_output} \n")
# print(f"im2col_output.shape:\n {im2col_output.shape} \n")
# # # 将im2col的输入和kernels2row的权重进行矩阵乘法
# # result = np.dot(cols, row)
# # print(result.shape)
# # print(result)
# # # 重塑 result
# # reshaped_result = result.reshape(batch_size, -1)
# # print(reshaped_result.shape)
# # print(reshaped_result)# 现在 reshaped_result 的形状应该是 (batch_size, num_kernels * output_height * output_width)



# # # col2im函数
# # import numpy as np

# # def reshape_output_feature_maps(cols, batch_size, channels_out, output_height, output_width):
# #     """
# #     将列形式的矩阵重塑回输出特征图的格式。
    
# #     参数:
# #     - cols: 列形式的矩阵，形状为 (output_height * output_width * batch_size, channels_out)
# #     - batch_size: 批次数
# #     - output_height: 输出特征图的高度
# #     - output_width: 输出特征图的宽度
# #     - channels_out: 输出通道数
    
# #     返回:
# #     - output_data: 重塑后的输出特征图，形状为 (batch_size, channels_out, output_height, output_width)
# #     """
# #     # 首先将矩阵重塑为中间形状 (batch_size, output_height * output_width, channels_out)
# #     intermediate_shape = (batch_size, output_height * output_width, channels_out)
# #     intermediate = cols.reshape(intermediate_shape)
    
# #     # 然后将中间形状重塑为 (batch_size, output_height, output_width, channels_out)
# #     reshaped = intermediate.reshape(batch_size, output_height, output_width, channels_out)
    
# #     # 最后，交换最后两个维度以获得 (batch_size, channels_out, output_height, output_width)
# #     output_data = reshaped.transpose(0, 3, 1, 2)
    
# #     return output_data

# # # 示例使用reshape_output_feature_maps
# # # 假设我们有一个列形式的矩阵 cols，形状为 (output_height * output_width * batch_size, channels_out)
# # # 并且我们知道 batch_size, output_height, output_width, channels_out
# # cols = np.random.rand(2 * 2 * 2, 2)  # 假设 batch_size=2, output_height=2, output_width=2, channels_out=2
# # batch_size = 2
# # output_height = 2
# # output_width = 2
# # channels_out = 2

# # output_data = reshape_output_feature_maps(cols, batch_size, channels_out, output_height, output_width)
# # print("原始列形式的矩阵:")
# # print(cols)
# # print("重塑后的输出特征图形状:", output_data.shape)
# # print("重塑后的输出特征图:")
# # print(output_data)


