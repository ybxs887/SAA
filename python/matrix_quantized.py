# import numpy as np

# # 创建一个fp32全连接层权重矩阵和输入向量
# W_fp32 = np.random.randn(4, 4).astype(np.float32)  # 假设这是训练好的权重
# x_fp32 = np.random.randn(4, 1).astype(np.float32)  # 假设这是输入向量

# # fp32矩阵乘法
# y_fp32 = np.dot(W_fp32, x_fp32)

# # 计算量化因子（以最大绝对值为基准）
# scale_factor = np.max(np.abs(W_fp32)) / 127

# # 量化权重矩阵和输入向量
# W_int8 = np.clip((W_fp32 / scale_factor).round(), -128, 127).astype(np.int8)
# x_int8 = np.clip((x_fp32 / scale_factor).round(), -128, 127).astype(np.int8)

# # int8矩阵乘法
# y_int8 = np.dot(W_int8, x_int8)

# # 反量化结果
# y_fp32_from_int8 = y_int8 * scale_factor  # 正确的反量化方法

# # 输出结果
# print("fp32 result:\n", y_fp32)
# print("int8 result after dequantization:\n", y_fp32_from_int8)

# # 对比误差
# error = np.abs(y_fp32 - y_fp32_from_int8)
# print("Absolute error:\n", error)




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.quantization
# import numpy  as np
# # 设置随机种子以确保可重复性
# torch.manual_seed(0)

# # 定义一个简单的全连接层
# class FullyConnected(nn.Module):
#     def __init__(self, input_features, output_features):
#         super(FullyConnected, self).__init__()
#         self.fc = nn.Linear(input_features, output_features,bias=False)

#     def forward(self, x):
#         x = self.fc(x)
#         return x

# # 创建模型实例
# input_features = 10
# output_features = 5
# model = FullyConnected(input_features, output_features)

# # 创建输入矩阵
# input_matrix = torch.randn(5, input_features)

# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# # 训练模型
# target = torch.randn(5, output_features)  # 创建目标输出矩阵
# for epoch in range(100):  # 运行100个训练周期
#     optimizer.zero_grad()
#     output = model(input_matrix)
#     loss = criterion(output, target)
#     loss.backward()
#     optimizer.step()

# # 保存未量化的模型输出
# output_before_quantization = model(input_matrix)

# # 量化模型
# model.eval()
# quantized_model = torch.quantization.quantize_dynamic(
#     model, {nn.Linear}, dtype=torch.qint8
# )


# # 使用量化模型进行前向传播
# output_after_quantization = quantized_model(input_matrix)
# print("output_after_quantization:\n", output_after_quantization)
# # 反量化输出以比较
# output_after_dequantization = output_after_quantization.dequantize()

# # 比较未量化和反量化后的结果
# print("quantization model:\n", model)
# print("linear matrix :\n", model.fc.weight)
# print("input matrix :\n", input_matrix)
# print(" np.dot result :\n", np.dot(input_matrix,(model.fc.weight.detach().numpy()).transpose()))
# print("Output before quantization:\n", output_before_quantization)
# print("Output after dequantization:\n", output_after_dequantization)










# import numpy as np

# def compute_quantization_parameters(tensor, qmin=-128, qmax=127):
#     """
#     计算量化的scale和zero_point。
#     """
#     min_val, max_val = tensor.min(), tensor.max()
#     scale = (max_val - min_val) / (qmax - qmin)
#     zero_point = qmax - max_val / scale
#     zero_point = np.clip(zero_point, qmin, qmax).round().astype(np.int8)
#     return scale, zero_point

# def quantize(tensor, scale, zero_point, qmin=-128, qmax=127):
#     """
#     量化给定的张量。
#     """
#     q_tensor = (zero_point + tensor / scale)
#     q_tensor = np.clip(q_tensor, qmin, qmax).round().astype(np.int8)
#     return q_tensor

# def dequantize(q_tensor, scale, zero_point):
#     """
#     反量化给定的张量。
#     """
#     return scale * (q_tensor - zero_point)

# # 创建两个随机矩阵
# matrix1 = np.random.randn(3, 5).astype(np.float32)
# matrix2 = np.random.randn(5, 3).astype(np.float32)
# result = np.matmul(matrix1, matrix2)
# print("matrix1:\n", matrix1)
# print("matrix2:\n", matrix2)


# # 量化矩阵
# scale1, zero_point1 = compute_quantization_parameters(matrix1)
# scale2, zero_point2 = compute_quantization_parameters(matrix2)
# q_matrix1 = quantize(matrix1, scale1, zero_point1)
# q_matrix2 = quantize(matrix2, scale2, zero_point2)
# print("scale1:",scale1,",zero_point1:",zero_point1,",q_matrix1:\n", q_matrix1)
# print("scale2:",scale2,",zero_point2:",zero_point2,",q_matrix2:\n", q_matrix2)

# # 执行量化矩阵乘法
# q_result = np.matmul(q_matrix1, q_matrix2)

# # 反量化结果
# scale_out = scale1 * scale2
# zero_point_out = 0  # 对于简化起见，我们可以假设输出的zero_point为0
# result_after_dequantization = dequantize(q_result, scale_out, zero_point_out)

# # 结果
# print("results:", result)
# print("quantized results:", result_after_dequantization)


# # 对比不量化的结果
# result_without_quantization = np.matmul(matrix1, matrix2)

# # 计算误差
# error = np.abs(result_without_quantization - result_after_dequantization).max()
# print("Maximum error between non-quantized and dequantized results:", error)




# import numpy as np

# def compute_quantization_parameters(tensor, qmin=-128, qmax=127):
#     """
#     计算量化的scale和zero_point。
#     """
#     min_val, max_val = tensor.min(), tensor.max()
#     scale = (max_val - min_val) / (qmax - qmin)
#     zero_point = qmin - min_val / scale
#     zero_point = np.clip(zero_point, qmin, qmax).round().astype(np.int8)
#     return scale, zero_point

# def quantize(tensor, scale, zero_point, qmin=-128, qmax=127):
#     """
#     量化给定的张量。
#     """
#     q_tensor = zero_point + tensor / scale
#     q_tensor = np.clip(q_tensor, qmin, qmax).round().astype(np.int8)
#     return q_tensor

# def dequantize(q_tensor, scale, zero_point):
#     """
#     反量化给定的张量。
#     """
#     return scale * (q_tensor - zero_point)

# # 创建两个随机矩阵
# matrix1 = 100*np.random.randn(3, 5).astype(np.float32)
# matrix2 = np.random.randn(5, 3).astype(np.float32)
# result = np.matmul(matrix1, matrix2)
# print("matrix1:\n", matrix1)
# print("matrix2:\n", matrix2)

# # 量化矩阵
# scale1, zero_point1 = compute_quantization_parameters(matrix1)
# scale2, zero_point2 = compute_quantization_parameters(matrix2)
# q_matrix1 = quantize(matrix1, scale1, 0)
# q_matrix2 = quantize(matrix2, scale2, 0)
# q_matrix1_int32 = q_matrix1.astype(np.int32)
# q_matrix2_int32 = q_matrix2.astype(np.int32)
# print("scale1:",scale1,",zero_point1:",zero_point1,",q_matrix1:\n", q_matrix1)
# print("scale2:",scale2,",zero_point2:",zero_point2,",q_matrix2:\n", q_matrix2)

# # 执行量化矩阵乘法
# q_result = np.matmul(q_matrix1_int32, q_matrix2_int32)

# # 反量化结果
# scale_out = scale1 * scale2
# zero_point_out = 0  # 对于简化起见，我们可以假设输出的zero_point为0
# result_after_dequantization = dequantize(q_result, scale_out, zero_point_out)

# # 结果
# print("results:", result)
# print("quantized results:", result_after_dequantization)

# # 对比不量化的结果
# result_without_quantization = np.matmul(matrix1, matrix2)

# # 计算误差
# error = np.abs(result_without_quantization - result_after_dequantization).max()
# print("Maximum error between non-quantized and dequantized results:", error)



# import numpy as np

# def compute_scale(tensor, qmin=-128, qmax=127):
#     """
#     计算对称量化的scale因子。
#     """
#     max_val = np.max(np.abs(tensor))
#     scale = max_val / qmax
#     return scale

# def symmetric_quantize(tensor, scale):
#     """
#     对称量化给定的张量到INT8。
#     """
#     q_tensor = np.round(tensor / scale).astype(np.int8)
#     return q_tensor

# def symmetric_dequantize(q_tensor, scale):
#     """
#     反量化给定的INT8张量到浮点数。
#     """
#     return q_tensor * scale

# # 示例：量化两个矩阵并进行矩阵乘法
# matrix1 = np.random.randn(3, 5).astype(np.float32)
# matrix2 = np.random.randn(5, 3).astype(np.float32)

# # 计算scale因子
# scale1 = compute_scale(matrix1)
# scale2 = compute_scale(matrix2)

# # 量化
# q_matrix1 = symmetric_quantize(matrix1, scale1)
# q_matrix2 = symmetric_quantize(matrix2, scale2)
# q_matrix1_int32 = q_matrix1.astype(np.int32)
# q_matrix2_int32 = q_matrix2.astype(np.int32)

# # 执行量化后的矩阵乘法
# q_result = np.matmul(q_matrix1_int32, q_matrix2_int32)

# # 反量化结果
# scale_out = scale1 * scale2
# result_after_dequantization = symmetric_dequantize(q_result, scale_out)

# # 打印结果
# print("Original matrix multiplication result:\n", np.matmul(matrix1, matrix2))
# print("Dequantized matrix multiplication result:\n", result_after_dequantization)

# # 计算并打印误差
# error = np.abs(np.matmul(matrix1, matrix2) - result_after_dequantization).max()
# print("Maximum error between original and dequantized results:", error)





############################连续两个矩阵全量化###############################

# import numpy as np

# def compute_scale(tensor, qmin=-128, qmax=127):
#     """
#     计算对称量化的scale因子。
#     """
#     max_val = np.max(np.abs(tensor))
#     scale = max_val / qmax
#     return scale

# def symmetric_quantize(tensor, scale):
#     """
#     对称量化给定的张量到INT8。
#     """
#     q_tensor = np.round(tensor / scale).astype(np.int8)
#     return q_tensor

# def symmetric_dequantize(q_tensor, scale):
#     """
#     反量化给定的INT8张量到浮点数。
#     """
#     return q_tensor * scale

# # 示例：量化两个矩阵并进行两次矩阵乘法
# matrix1 = np.random.randn(3, 5).astype(np.float32)
# matrix2 = np.random.randn(5, 4).astype(np.float32)
# matrix3 = np.random.randn(4, 3).astype(np.float32)

# # 计算scale因子
# scale1 = compute_scale(matrix1)
# scale2 = compute_scale(matrix2)
# scale3 = compute_scale(matrix3)

# # 量化
# q_matrix1 = symmetric_quantize(matrix1, scale1)
# q_matrix2 = symmetric_quantize(matrix2, scale2)
# q_matrix3 = symmetric_quantize(matrix3, scale3)

# # 执行第一次量化后的矩阵乘法
# q_result1 = np.matmul(q_matrix1.astype(np.int32), q_matrix2.astype(np.int32))

# # 计算中间结果的scale因子并量化中间结果
# scale_intermediate = compute_scale(q_result1, qmin=0, qmax=2147483647)  # 使用int32的范围
# q_result1_int8 = symmetric_quantize(q_result1, scale_intermediate)

# # 执行第二次量化后的矩阵乘法
# q_result2 = np.matmul(q_result1_int8.astype(np.int32), q_matrix3.astype(np.int32))

# # 反量化最终结果
# scale_out = scale1 * scale2 * scale3 * scale_intermediate
# result_after_dequantization = symmetric_dequantize(q_result2, scale_out)

# # 打印结果
# print("Original matrix multiplication result:\n", np.matmul(np.matmul(matrix1, matrix2), matrix3))
# print("Dequantized matrix multiplication result:\n", result_after_dequantization)

# # 计算并打印误差
# error = np.abs(np.matmul(np.matmul(matrix1, matrix2), matrix3) - result_after_dequantization).max()
# print("Maximum error between original and dequantized results:", error)







import numpy as np

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
    对称量化给定的张量到INT8。
    """
    q_tensor = np.round(tensor / scale).astype(np.int32)
    return q_tensor

def symmetric_dequantize(q_tensor, scale):
    """
    反量化给定的INT8张量到浮点数。
    """
    return q_tensor * scale

# 示例：量化两个矩阵并进行两次矩阵乘法
ROWS =  4 # I
COL1S = 1 # J
COLS =  5 # K
matrix1 = np.random.randn(ROWS, COLS).astype(np.float32)
matrix2 = np.random.randn(COLS, COL1S).astype(np.float32)
matrix3 = np.random.randn(COL1S, ROWS).astype(np.float32)
bias1 = np.random.randn(ROWS, COL1S).astype(np.float32)
bias2 = np.random.randn(ROWS, ROWS).astype(np.float32)


# 进行float32的矩阵乘法,并打印第一次矩阵乘法结果和第二次矩阵乘法结果
result1 = np.matmul(matrix1, matrix2) + bias1
result2 = np.matmul(result1, matrix3) + bias2
print("result1:\n", result1)
print("result2:\n", result2)


#########################测试计算完第一次进行反量化,第二次进行量化############################

########################第一次矩阵乘法###########################
# 计算scale因子
scale1 = compute_scale(matrix1) # 第一次矩阵乘法的输入
scale2 = compute_scale(matrix2) # 第一次矩阵乘法的权重
# 量化
q_matrix1 = symmetric_quantize(matrix1, scale1)
q_matrix2 = symmetric_quantize(matrix2, scale2)
q_bias1 = symmetric_quantize_int32(bias1, scale1*scale2) # 使用Int32量化
# 查看量化后矩阵反量化和原来矩阵的精度损失
q_matrix1_dequantize = symmetric_dequantize(q_matrix1, scale1)
q_matrix2_dequantize = symmetric_dequantize(q_matrix2, scale2)
q_bias1_dequantize = symmetric_dequantize(q_bias1, scale1*scale2)
print("matrix1:\n", matrix1)
print("q_matrix1_dequantize:\n", q_matrix1_dequantize)
print("bias1:\n", bias1)
print("q_bias1_dequantize:\n", q_bias1_dequantize)
print("q_bias1:\n", q_bias1)
# 执行量化后的矩阵乘法
q_result1 = np.matmul(q_matrix1.astype(np.int32), q_matrix2.astype(np.int32)) + q_bias1
# 反量化结果
scale_out = scale1 * scale2
q_result1 = symmetric_dequantize(q_result1, scale_out)
print("qresult1:\n", q_result1)


########################第二次矩阵乘法###########################
# 再次量化q_result1得到第二次矩阵乘法的输入参数
scale3 = compute_scale(q_result1) # 第一次矩阵乘法的结果矩阵计算缩放
scale4 = compute_scale(matrix3) # 第二次矩阵乘法的权重
# 量化
q_result1 = symmetric_quantize(q_result1, scale3)
q_matrix3 = symmetric_quantize(matrix3, scale4)
q_bias2 = symmetric_quantize_int32(bias2, scale3*scale4) # 使用Int32量化
# 执行量化后的矩阵乘法
q_result2 = np.matmul(q_result1.astype(np.int32), q_matrix3.astype(np.int32)) + q_bias2
# 反量化结果
scale_out = scale3 * scale4
q_result2 = symmetric_dequantize(q_result2, scale_out)
print("q_result2:\n", q_result2)



# import math

# value = 8.0
# mantissa, exponent = math.frexp(value)
# print(mantissa, exponent)  # 输出尾数和指数
# 输出结果可能是：0.5 4





# # 量化
# q_matrix1 = symmetric_quantize(matrix1, scale1)
# q_matrix2 = symmetric_quantize(matrix2, scale2)
# q_matrix3 = symmetric_quantize(matrix3, scale3)

# # 第一次矩阵乘法
# q_matrix1_int32 = q_matrix1.astype(np.int32)
# q_matrix2_int32 = q_matrix2.astype(np.int32)
# q_result1 = np.matmul(q_matrix1_int32, q_matrix2_int32)

# # 中间结果量化为INT8
# scale_out1 = scale1 * scale2
# q_result1_int8 = symmetric_quantize(q_result1, scale_out1)

# # 第二次矩阵乘法
# q_result1_int32 = q_result1_int8.astype(np.int32)
# q_matrix3_int32 = q_matrix3.astype(np.int32)
# q_result2 = np.matmul(q_result1_int32, q_matrix3_int32)

# # 反量化最终结果
# scale_out2 = scale_out1 * scale3
# result_after_dequantization = symmetric_dequantize(q_result2, scale_out2)

# # 打印结果
# print("Original matrix multiplication result:\n", np.matmul(np.matmul(matrix1, matrix2), matrix3))
# print("Dequantized matrix multiplication result:\n", result_after_dequantization)

# # 计算并打印误差
# error = np.abs(np.matmul(np.matmul(matrix1, matrix2), matrix3) - result_after_dequantization).max()
# print("Maximum error between original and dequantized results:", error)
