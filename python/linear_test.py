import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# 定义一个简单的 Linear 线性层模型
class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 64,bias =False)
        self.linear2 = nn.Linear(64, output_size,bias =False)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)

# 实例化模型
model = SimpleLinearModel(input_size=28*28, output_size=10)


#---------------------------------------------------#
#   训练
#---------------------------------------------------#

# 下载并准备数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # 训练模型
# for epoch in range(5):  # 只训练5个epoch作为示例
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs = inputs.view(-1, 28*28)  # 将输入数据展平成一维向量
#         # print("inputs")
#         # print(inputs)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         # print("outputs")
#         # print(outputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 100 == 99:    # 每100个batch打印一次损失
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0

# print('Finished Training')

# # 保存训练好的模型
# torch.save(model.state_dict(), './data/mnist_linear_model.pth')





#---------------------------------------------------#
#   推理
#---------------------------------------------------#
# 下载并准备测试集数据
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# # 加载预训练的模型权重
# model.load_state_dict(torch.load('./data/mnist_linear_model.pth'))

# # 计算模型在测试集上的预测精度
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images = images.view(-1, 28*28)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the model on the 10000 test images: {:.2f}%'.format(100 * correct / total))


#---------------------------------------------------#
#   quanto量化
#---------------------------------------------------#

# import quanto
# from quanto import quantize,freeze


# # 加载训练好的模型权重
# model.load_state_dict(torch.load('./data/mnist_linear_model.pth'))

# # 量化模型,同时量化激活,不对偏差bias进行量化
# quantize(model, weights=quanto.qint8, activations=quanto.qint8)
# print(model)
# # print(model.self_attn.query_key_value.weight)
# # print(model.ffn.weight)
# # print(model.layernorm1.weight)

# freeze(model)

# # 遍历模型中的所有层和参数
# for name, param in model.named_parameters():
#     print(f"Layer: {name}, Shape: {param.shape}, Value: {param}")


#---------------------------------------------------#
#   pytorch量化
#---------------------------------------------------#

model = model.eval()
model_int8 = torch.quantization.quantize_dynamic(
    model=model,  # 原始模型
    qconfig_spec={nn.Linear},  # 要动态量化的NN算子
    dtype=torch.qint8, inplace=True)  # 将权重量化为：float16 \ qint8+
torch.save(model_int8.state_dict(), "./data/mnist_linear_model_int8.pth")
model_int8.load_state_dict(torch.load("./data/mnist_linear_model_int8.pth"))
print(model_int8)
print(model_int8.linear1.weight().int_repr())
print(model_int8.linear1.weight().q_scale())

#---------------------------------------------------#
#   递归导出量化权重和scale ,并保存bias
#---------------------------------------------------#


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
save_model_layers_as_npy(model, './data/mnist_linear_model')

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
# save_model_layers_as_npy(model,'./data/mnist_linear_model')


# import os
# import numpy as np
# import torch

# # 定义一个函数来保存层的所有参数为.npy格式
# def save_layer_params_as_npy(layer, layer_path):
#     for param_name, param in layer.named_parameters(recurse=False):
#         param_data = param.data.cpu().numpy()

#         if hasattr(param, '_data'):  # 检查是否为QTensor
#             param_data = param._data.numpy()
#             scale = param._scale.numpy()
#             np.save(os.path.join(layer_path, f'{param_name}_scale.npy'), scale)
#             print(f"-- saved {param_name}_scale{scale.shape}")
        
#         np.save(os.path.join(layer_path, f'{param_name}.npy'), param_data)
#         print(f"-- saved {param_name} with shape {param_data.shape}")


# # 定义一个函数来递归遍历模型的每一层
# def save_model_layers_as_npy(model, save_dir):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     for name, layer in model.named_modules():  # 使用named_modules来递归遍历所有层
#         # 跳过根模块，因为它已经被包含在子模块中
#         if name == "":
#             continue
#         layer_name = name.replace('.', '_')  # 将层的名称中的点替换为下划线
#         layer_dir = os.path.join(save_dir, layer_name)
#         if not os.path.exists(layer_dir):
#             os.makedirs(layer_dir)
        
#         save_layer_params_as_npy(layer, layer_dir)
#         print(f"Saved parameters for layer {layer_name}")

# # 假设我们有一个名为model的PyTorch模型
# # 保存模型的每一层参数为.npy格式
# save_model_layers_as_npy(model, './data/mnist_linear_model')


