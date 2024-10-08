import torch
import torch.nn as nn
import numpy as np
import os
# 定义一个简单的Transformer模块
class SimpleTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads=1):
        super(SimpleTransformer, self).__init__()
        self.embed = nn.Embedding(num_tokens, dim_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Linear(dim_model * 4, dim_model)
        )
        self.layernorm2 = nn.LayerNorm(dim_model)

    def forward(self, x):
        x_embed = self.embed(x)
        attn_output, _ = self.self_attn(x_embed, x_embed, x_embed)
        x = self.layernorm1(x_embed + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x

# 创建模型实例
num_tokens = 10  # 假设词汇表大小为10
dim_model = 128  # 假设模型维度为512
model = SimpleTransformer(num_tokens=num_tokens, dim_model=dim_model)

# 创建一个随机的输入序列
sequence_length = 5  # 假设序列长度为5
input_sequence = torch.randint(0, num_tokens, (sequence_length,)).unsqueeze(0)  # 添加批次维度

# 通过模型传递输入
output = model(input_sequence)

# 打印输出
print(output)

# 导出模型权重
torch.save(model.state_dict(), './data/simple_transformer_weights.pth')

# 如果要加载权重到另一个模型实例
model_loaded = SimpleTransformer(num_tokens=num_tokens, dim_model=dim_model)
model_loaded.load_state_dict(torch.load('./data/simple_transformer_weights.pth'))

# 验证加载的权重
output_loaded = model_loaded(input_sequence)
print(torch.equal(output, output_loaded))  # 应该输出True，表示输出相同

################################训练################################

import torch.optim as optim

# 创建一个随机的目标embedding序列与模型输出尺寸相同
target_embedding = torch.randn(sequence_length, dim_model).unsqueeze(0)  # 添加批次维度

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 对于分类任务，可以使用nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义训练循环
num_epochs = 100  # 训练轮数
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清除梯度
    output = model(input_sequence)  # 前向传播
    loss = criterion(output, target_embedding)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 保存训练后的模型权重
torch.save(model.state_dict(), './data/simple_transformer_trained_weights.pth')


###############################量化#################################
import quanto
from quanto import quantize,freeze

quantize(model, weights=quanto.qint8, activations=quanto.qint8)
print(model)
# print(model.self_attn.query_key_value.weight)
# print(model.ffn.weight)
# print(model.layernorm1.weight)

freeze(model)

# # 遍历模型中的所有层和参数
# for name, param in model.named_parameters():
#     print(f"Layer: {name}, Shape: {param.shape}, Value: {param}")




#---------------------------------------------------#
#   递归导出量化权重和scale ,并保存bias
#---------------------------------------------------#

# # 定义一个函数来递归保存层的权重和偏置为.npy格式
# def save_layer_params_as_npy(layer, layer_path):
#     # 保存权重参数
#     if hasattr(layer, 'weight') and hasattr(layer.weight, 'data'):
#         weight = layer.weight.data.cpu().numpy()
#         if hasattr(layer.weight, '_data'):  # 检查是否为QTensor,如果是的话weight就被替换为qtensor的量化整数版本,同时存储scale
#             weight = layer.weight._data.numpy()
#             scale = layer.weight._scale.numpy()
#             np.save(os.path.join(layer_path, 'scale.npy'), scale)
#             print(f"-- saved scale{scale.shape}")
#         np.save(os.path.join(layer_path, 'weight.npy'), weight)
#         print(f"-- saved weight{weight.shape}")
    
#     # 保存偏置参数
#     if hasattr(layer, 'bias') and hasattr(layer.bias, 'data'):
#         bias = layer.bias.data.cpu().numpy()
#         if hasattr(layer.bias, '_data'):  # 检查是否为QTensor
#             bias = layer.bias._data.numpy()
#         np.save(os.path.join(layer_path, 'bias.npy'), bias)
#         print(f"-- saved bias{bias.shape}")

# # 定义一个函数来遍历模型的每一层
# def save_model_layers_as_npy(model, save_dir='./data/mnist_linear_model'):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     for name, layer in model.named_children():
#         layer_name = name.replace('.', '_')  # 将层的名称中的点替换为下划线
#         layer_dir = os.path.join(save_dir, layer_name)
#         if not os.path.exists(layer_dir):
#             os.makedirs(layer_dir)
        
#         save_layer_params_as_npy(layer, layer_dir)
#         print(f"Saved parameters for layer {layer_name}")
        
#         # 如果子模块有进一步的子模块，则递归调用
#         if len(list(layer.children())) > 0:
#             save_model_layers_as_npy(layer, layer_dir)

# # 假设我们有一个名为model的PyTorch模型
# # 保存模型的每一层参数为.npy格式
# save_model_layers_as_npy(model,'./data/quanted_transformer_model')




# import os
# import numpy as np
# import torch.nn as nn

# # 定义一个函数来保存层的所有参数为.npy格式
# def save_layer_params_as_npy(layer, layer_path):
#     for param_name, param in layer.named_parameters(recurse=False):
#         param_data = param.data.cpu().numpy()
#         file_name = param_name.replace('.', '_')  # 替换点以创建有效的文件名
#         if hasattr(param, '_data'):  # 检查是否为QTensor
#             param_data = param._data.numpy()
#             scale = param._scale.numpy()
#             np.save(os.path.join(layer_path, f'{file_name}_scale.npy'), scale)
#             print(f"-- saved {file_name}_scale{scale.shape}")
#         np.save(os.path.join(layer_path, f'{file_name}.npy'), param_data)
#         print(f"-- saved {file_name}{param_data.shape}")

# # 定义一个函数来遍历模型的每一层
# def save_model_layers_as_npy(model, save_dir='./data/mnist_linear_model'):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     for name, layer in model.named_modules():  # 使用named_modules遍历所有子模块
#         layer_name = name.replace('.', '_')  # 将层的名称中的点替换为下划线
#         layer_dir = os.path.join(save_dir, layer_name)
#         if not os.path.exists(layer_dir):
#             os.makedirs(layer_dir)
        
#         save_layer_params_as_npy(layer, layer_dir)
#         print(f"Saved parameters for layer {layer_name}")

# # 假设我们有一个名为model的PyTorch模型
# # 保存模型的每一层参数为.npy格式
# save_model_layers_as_npy(model,'./data/quanted_transformer_model')


import os
import numpy as np
import torch

# 定义一个函数来保存层的所有参数为.npy格式
def save_layer_params_as_npy(layer, layer_path):
    for param_name, param in layer.named_parameters(recurse=False):
        param_data = param.data.cpu().numpy()

        if hasattr(param, '_data'):  # 检查是否为QTensor
            param_data = param._data.numpy()
            scale = param._scale.numpy()
            np.save(os.path.join(layer_path, f'{param_name}_scale.npy'), scale)
            print(f"-- saved {param_name}_scale{scale.shape}")
        
        np.save(os.path.join(layer_path, f'{param_name}.npy'), param_data)
        print(f"-- saved {param_name} with shape {param_data.shape}")


# 定义一个函数来递归遍历模型的每一层
def save_model_layers_as_npy(model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for name, layer in model.named_modules():  # 使用named_modules来递归遍历所有层
        # 跳过根模块，因为它已经被包含在子模块中
        if name == "":
            continue
        layer_name = name.replace('.', '_')  # 将层的名称中的点替换为下划线
        layer_dir = os.path.join(save_dir, layer_name)
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
        
        save_layer_params_as_npy(layer, layer_dir)
        print(f"Saved parameters for layer {layer_name}")

# 假设我们有一个名为model的PyTorch模型
# 保存模型的每一层参数为.npy格式
save_model_layers_as_npy(model, './data/quanted_transformer_model')




# # 如果您只想打印特定层的参数，例如self_attn
# for name, param in model.self_attn.named_parameters():
#     print(f"Layer: {name}, Shape: {param.shape}")

# # 假设SimpleTransformer是之前定义的模型类
# model = SimpleTransformer(num_tokens=num_tokens, dim_model=dim_model)

# # 假设模型已经训练完成，这里是加载模型权重的代码
# model.load_state_dict(torch.load('./data/simple_transformer_trained_weights.pth'))

# # 使用动态量化量化模型的权重
# quantized_model = torch.quantization.quantize_dynamic(
#     model,  # 要量化的模型
#     {nn.Linear,  nn.MultiheadAttention},  # 指定要量化的层类型
#     dtype=torch.qint8  # 量化到8位整数
# )

# # 保存量化后的模型权重
# quantized_model_path = './data/quantized_simple_transformer_weights.pth'
# torch.save(quantized_model.state_dict(), quantized_model_path)

# # 验证是否成功保存
# if os.path.exists(quantized_model_path):
#     print(f"Quantized model weights have been saved to {quantized_model_path}")
# else:
#     print("Failed to save quantized model weights.")

# import torch
# from torch import nn
# from torch.quantization import get_default_qconfig, quantize_dynamic

# # 假设SimpleTransformer是待量化的模型
# model = SimpleTransformer(num_tokens=num_tokens, dim_model=dim_model)

# # 指定量化配置并应用动态量化
# qconfig = get_default_qconfig('fbgemm')  # 'fbgemm'是针对x86 CPU优化的后端
# model.qconfig = qconfig
# quantized_model = quantize_dynamic(model, {nn.Linear,  nn.MultiheadAttention}, dtype=torch.qint8)

# for name, param in quantized_model.named_parameters():
#         layer_name = name.replace('.', '_')  # 将层的名称中的点替换为下划线
#         file_path = os.path.join("./data/npy_layer_weights", f'{layer_name}.npy')
#         np.save(file_path, param.data.cpu().numpy())  # 将参数转换为NumPy数组并保存

###############################导出#################################

# import numpy as np

# # 定义一个函数来保存每一层的参数为.npy格式
# def save_layer_weights_as_npy(model, save_dir='./data/npy_layer_weights'):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     for name, param in model.named_parameters():
#         layer_name = name.replace('.', '_')  # 将层的名称中的点替换为下划线
#         file_path = os.path.join(save_dir, f'{layer_name}.npy')
#         np.save(file_path, param.data.cpu().numpy())  # 将参数转换为NumPy数组并保存
#         print(f"Saved {layer_name} ")
        
# # 保存模型的每一层参数为.npy格式
# save_layer_weights_as_npy(model)

# # 如果需要加载.npy格式的权重到模型
# # 例如加载嵌入层的权重
# embed_weights_path = os.path.join('./data/npy_layer_weights', 'embed_weight.npy')
# embed_weights = np.load(embed_weights_path)
# model_loaded.embed.weight.data.copy_(torch.from_numpy(embed_weights))

# # 打印加载的权重，检查是否与保存的权重相同
# print("Loaded Embedding weights from npy:", model_loaded.embed.weight.data)


###############################量化导出#################################


# # 假设quantized_model是你的量化模型
# # 例如：quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# def save_quantized_weights_as_npy(model, save_dir='./data/quantized_weights_npy'):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # 遍历模型的状态字典
#     for name, param in model.state_dict().items():
#         # 构建保存路径
#         layer_name = name.replace('.', '_')  # 替换掉名字中的点，避免路径问题
#         file_path = os.path.join(save_dir, f'{layer_name}.npy')
        
#         # 检查参数是否为量化权重
#         if hasattr(param, 'dequantize'):
#             # 对于量化的权重，首先进行反量化
#             param = param.dequantize()
        
#         # 保存为.npy文件
#         np.save(file_path, param.numpy())

# # 调用函数
# save_quantized_weights_as_npy(quantized_model)



# def save_quantized_model(model, save_dir='./data/quantized_model'):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # 保存模型的参数
#     for name, param in model.named_parameters():
#         # 量化权重
#         if hasattr(param, 'q_per_channel_scales'):
#             # 通道级量化参数
#             scale = param.q_per_channel_scales().numpy()
#             zero_point = param.q_per_channel_zero_points().numpy()
#             np.save(os.path.join(save_dir, f"{name}_scale.npy"), scale)
#             np.save(os.path.join(save_dir, f"{name}_zero_point.npy"), zero_point)
#             weight = param.int_repr().numpy()
#             np.save(os.path.join(save_dir, f"{name}_weight.npy"), weight)
#         elif hasattr(param, 'q_scale'):
#             # 整体量化参数
#             scale = np.array([param.q_scale()])
#             zero_point = np.array([param.q_zero_point()])
#             np.save(os.path.join(save_dir, f"{name}_scale.npy"), scale)
#             np.save(os.path.join(save_dir, f"{name}_zero_point.npy"), zero_point)
#             weight = param.int_repr().numpy()
#             np.save(os.path.join(save_dir, f"{name}_weight.npy"), weight)
#         else:
#             # 未量化的权重
#             weight = param.data.numpy()
#             np.save(os.path.join(save_dir, f"{name}.npy"), weight)

#     # 保存模型的缓冲区（例如，BatchNorm的running_mean和running_var）
#     for name, buf in model.named_buffers():
#         np.save(os.path.join(save_dir, f"{name}.npy"), buf.numpy())

# # 假设quantized_model是你已经量化的模型
# save_quantized_model(quantized_model)




# # 导出模型权重到npy文件
# def save_model_weights(model, base_path):
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             np.save(f"{base_path}/{name}.npy", param.data.numpy())

# # 指定基础路径
# base_path = './data/simple_transformer_weights'
# # 确保保存路径存在
# os.makedirs(base_path, exist_ok=True)

# # 导出权重
# save_model_weights(model, base_path)





# # 定义一个函数来保存每一层的权重
# def save_layer_weights(model, save_dir='layer_weights'):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     for name, param in model.named_parameters():
#         layer_name = name.replace('.', '_')  # 将层的名称中的点替换为下划线
#         file_path = os.path.join(save_dir, f'{layer_name}.pth')
#         torch.save(param.data, file_path)

# # 保存模型的每一层权重
# save_layer_weights(model)

# # 你也可以选择加载特定层的权重
# # 例如加载嵌入层的权重
# embed_weights = torch.load('./data/layer_weights/embed_weight.pth')
# model_loaded.embed.weight.data.copy_(embed_weights)

# # 打印加载的权重，检查是否与保存的权重相同
# print("Loaded Embedding weights:", model_loaded.embed.weight.data)

