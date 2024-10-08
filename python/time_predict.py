
# # 第一步：下载和预处理数据

# import pandas as pd
# import numpy as np
# import os
    
# # 读取数据
# df = pd.read_csv('./python/AirQualityUCI.csv', delimiter=';')

# # 选取CO浓度列，并去除无效数据
# df['CO(GT)'] = pd.to_numeric(df['CO(GT)'], errors='coerce')
# df = df.dropna(subset=['CO(GT)'])

# # 仅用于演示，选取前1000个有效数据点
# data = df['CO(GT)'].values
# print("data_shape",data.shape)

# # 第二步：创建数据集
# num = 10

# def create_dataset(data, n=5):
#     X, y = [], []
#     for i in range(len(data)-n):
#         X.append(data[i:i+n])
#         y.append(data[i+n])
#     return np.array(X), np.array(y)

# X, y = create_dataset(data, num)


# # 第三步：定义全连接网络模型


# import torch
# import torch.nn as nn
# import torch.optim as optim

# class FCN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FCN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, output_dim)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# model = FCN(num,4)  # 假设我们使用过去5个时间点的数据进行预测


# # 第四步：训练模型
# # 转换为PyTorch张量
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# # 分割数据集
# split_ratio = int(len(X_tensor) * 0.5)
# X_train, X_test = X_tensor[:split_ratio], X_tensor[split_ratio:]
# y_train, y_test = y_tensor[:split_ratio], y_tensor[split_ratio:]

# # 训练模型
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 训练循环
# epochs = 1000
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     output = model(X_train)
#     loss = criterion(output, y_train)
#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item()}')


# # 第五步：模型预测和评估

# with torch.no_grad():
#     predictions = model(X_test)
#     test_loss = criterion(predictions, y_test)
#     print(f'Test Loss: {test_loss.item()}')



# # 假设 model 是我们的 PyTorch 模型，X_test 是我们的测试特征
# predictions = model(X_test)

# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from math import sqrt

# # 将预测和真实标签转换为 NumPy 数组
# y_true = y_test.detach().numpy()
# y_pred = predictions.detach().numpy()

# # 计算 MSE
# mse = mean_squared_error(y_true, y_pred)
# # 计算 RMSE
# rmse = sqrt(mse)
# # 计算 MAE
# mae = mean_absolute_error(y_true, y_pred)
# # 计算 R²
# r2 = r2_score(y_true, y_pred)

# # 打印精度指标
# print(f"MSE: {mse:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"R²: {r2:.4f}")


# import matplotlib.pyplot as plt

# # 假设 y_true 是测试集的真实标签，y_pred 是模型的预测结果
# # 这些数据已经转换为 NumPy 数组

# # 绘制真实数据
# plt.figure(figsize=(14, 7))
# plt.plot(y_true, label='Actual Data', color='blue', marker='o')

# # 绘制预测数据
# plt.plot(y_pred, label='Predicted Data', color='red', marker='x')

# # 添加标题和标签
# plt.title('Comparison of Actual and Predicted Data')
# plt.xlabel('Time Steps')
# plt.ylabel('CO Concentration')
# plt.legend()

# # 显示图表
# plt.show()












# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# # 假设我们有以下数据格式：
# # sequences 是一个形状为 (num_samples, sequence_length, num_features) 的 NumPy 数组
# # targets 是一个形状为 (num_samples,) 的 NumPy 数组，表示每个序列的最终目标值

# # 示例数据
# sequences = np.random.rand(100, 5, 3)  # 100个样本，每个样本5个时间步长，每个时间步3个特征
# targets = np.random.rand(100)  # 100个目标值

# # 将 NumPy 数组转换为 PyTorch 张量
# sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
# targets_tensor = torch.tensor(targets, dtype=torch.float32)

# # 定义一个简单的 PyTorch 模型
# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 假设每个时间步的输入特征数量为3，序列长度为5
# input_size = sequences.shape[2] * sequences.shape[1]
# hidden_size = 50  # 隐藏层大小
# output_size = 1   # 输出大小（例如，预测一个连续值）

# # 实例化模型
# model = SimpleMLP(input_size, hidden_size, output_size)

# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 训练模型
# for epoch in range(100):  # 简单的训练循环，实际应用中需要更复杂的逻辑
#     # 将序列展平为单个特征向量
#     x = sequences_tensor.view(-1, input_size)
#     y = targets_tensor

#     # 前向传播
#     outputs = model(x)

#     # 计算损失
#     loss = criterion(outputs, y)

#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # 输出损失信息
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# # 预测
# # 假设我们有一个新的序列进行预测
# new_sequence = torch.tensor(np.random.rand(5, 3), dtype=torch.float32)
# # 展平序列
# new_sequence_flattened = new_sequence.view(1, -1)
# # 预测
# prediction = model(new_sequence_flattened)
# print(f'Prediction: {prediction.item()}')









# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader

# # 假设我们生成一些模拟的股票价格数据
# np.random.seed(42)  # 为了可重复性设置随机种子
# dates = np.arange(100)  # 100天的数据
# daily_returns = np.random.normal(loc=0.001, scale=0.02, size=100)  # 模拟每日收益率
# stock_prices = np.exp(np.cumsum(daily_returns))  # 计算股票价格

# # 将股票价格转换为 PyTorch 张量
# stock_prices_tensor = torch.tensor(stock_prices, dtype=torch.float32)



# class StockPriceDataset(Dataset):
#     def __init__(self, prices, sequence_length=5):
#         self.prices = prices
#         self.sequence_length = sequence_length

#     def __len__(self):
#         return len(self.prices) - self.sequence_length

#     def __getitem__(self, idx):
#         return self.prices[idx:idx + self.sequence_length], self.prices[idx + self.sequence_length]

# # 创建数据集实例
# sequence_length = 5
# dataset = StockPriceDataset(stock_prices_tensor, sequence_length)



# # 创建 DataLoader 实例
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# class StockPricePredictor(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(StockPricePredictor, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 实例化模型
# input_size = sequence_length  # 每个样本的时间步长数量
# hidden_size = 20
# output_size = 1
# model = StockPricePredictor(input_size, hidden_size, output_size)


# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # 训练模型
# for epoch in range(100):
#     for inputs, targets in dataloader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets.unsqueeze(1))  # 确保 targets 是正确的形状
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')


#     # 进行预测
# last_sequence = stock_prices_tensor[-sequence_length:].unsqueeze(0)  # 使用最后的价格序列作为输入
# prediction = model(last_sequence).item()
# print(f'Predicted next day stock price: {prediction}')



# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import numpy as np

# # 分割单变量序列为样本
# def split_sequence(sequence, n_steps_in, n_steps_out):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out
#         if out_end_ix > len(sequence):
#             break
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return np.array(X), np.array(y)

# # 定义输入序列
# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# n_steps_in, n_steps_out = 3, 2
# X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# # 定义 PyTorch Dataset 类
# class TimeSeriesDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# # 创建数据集实例
# dataset = TimeSeriesDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# # 定义模型
# class MultiStepMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MultiStepMLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 实例化模型
# input_size = n_steps_in
# hidden_size = 100
# output_size = n_steps_out
# model = MultiStepMLP(input_size, hidden_size, output_size)

# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # 训练模型
# epochs = 2000
# for epoch in range(epochs):
#     for inputs, targets in dataloader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

# # 保存模型权重
# torch.save(model.state_dict(), 'model_weights.pth')


# # 创建模型的实例（不需要定义参数，只需要架构）
# model = MultiStepMLP(input_size, hidden_size, output_size)
# # 加载模型权重
# model.load_state_dict(torch.load('model_weights.pth'))
# # 将模型设置为评估模式
# model.eval()

# # 进行预测
# x_input = np.array([[70, 80, 90],[70, 80, 90]]).reshape((2, n_steps_in))
# # x_input = np.array([90, 100, 110]).reshape((1, n_steps_in))
# x_input = torch.tensor(x_input, dtype=torch.float32)
# yhat = model(x_input)
# print(yhat.detach().numpy())



#####################使用MLP######################

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # 分割单变量序列为样本
# def split_sequence(sequence, n_steps_in, n_steps_out):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out
#         if out_end_ix > len(sequence):
#             break
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return np.array(X), np.array(y)

# # 读取Air Passengers数据集
# data = pd.read_csv('./python/AirQualityUCI.csv', delimiter=';')

# # 处理数据，选取CO浓度列，并去除无效数据
# # 用 ',-200' 替换非数值的条目，然后转换为数值类型
# data['CO(GT)'] = data['CO(GT)'].replace(',', '.', regex=True).replace('-200', np.nan).astype(float)
# data['CO(GT)'] = pd.to_numeric(data['CO(GT)'], errors='coerce')

# # 删除缺失值
# data = data.dropna(subset=['CO(GT)'])

# # 准备数据
# raw_seq = data['CO(GT)'].values
# print("raw_seq.shape:",raw_seq.shape)

# n_steps_in, n_steps_out = 3, 2
# X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("X_train.shape:",X_train.shape)
# print("X_test.shape:",X_test.shape)
# print("y_train.shape:",y_train.shape)
# print("y_test.shape:",y_test.shape)

# # 定义 PyTorch Dataset 类
# class TimeSeriesDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# # 创建训练集和测试集的数据集实例
# train_dataset = TimeSeriesDataset(X_train, y_train)
# test_dataset = TimeSeriesDataset(X_test, y_test)

# # 创建训练集和测试集的数据加载器
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=500, shuffle=False)

# # 定义模型
# class MultiStepMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MultiStepMLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, 2 * output_size)  # 增加输出层的大小
#         self.fc3 = nn.Linear(2 * output_size, output_size)  # 添加另一个线性层

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)  # 通过另一个线性层
#         return x

# # 实例化模型
# input_size = n_steps_in
# hidden_size = 1000
# output_size = n_steps_out
# model = MultiStepMLP(input_size, hidden_size, output_size)

# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # 训练模型
# epochs = 200
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0.0
#     for inputs, targets in train_dataloader:
#         optimizer.zero_grad()
#         # print("inputs.shape:",inputs.shape)
#         outputs = model(inputs)
#         # print("outputs.shape:",outputs.shape)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/len(train_dataloader)}')

# # 测试模型
# model.eval()
# test_losses = []
# with torch.no_grad():
#     for inputs, targets in test_dataloader:
#         outputs = model(inputs)
#         test_loss = criterion(outputs, targets)
#         test_losses.append(test_loss.item())

# # 打印测试集上的损失
# print("Test set mean squared error: ", np.mean(test_losses))

# # 保存模型权重
# import os
# # 检查目录是否存在，不存在则创建
# save_dir = './data/time_predict'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # 保存模型权重
# torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))



# # 加载模型
# model = MultiStepMLP(input_size, hidden_size, output_size) # 创建模型的实例（不需要定义参数，只需要架构）
# save_dir ="./data/time_predict/model_weights.pth" # 加载模型权重
# model.load_state_dict(torch.load(save_dir))

# # 测试模型
# model.eval() # 将模型设置为评估模式
# test_inputs, test_outputs, test_targets = [], [], []
# with torch.no_grad():
#     for inputs, targets in test_dataloader:
#         outputs = model(inputs)
#         test_inputs.append(inputs.numpy())
#         test_outputs.append(outputs.numpy())
#         test_targets.append(targets.numpy())

# # 将列表转换为NumPy数组以便于处理
# test_inputs = np.concatenate(test_inputs)
# test_outputs = np.concatenate(test_outputs)
# test_targets = np.concatenate(test_targets)

# # 打印输入测试数据
# print("Test Input Data:")
# print(test_inputs)

# # 打印模型的输出测试数据
# print("Model Output Data:")
# print(test_outputs)

# # 打印原来正确的测试数据
# print("Original Target Data:")
# print(test_targets)




#####################使用LSTM######################


#####################使用CNN######################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 分割单变量序列为样本
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# 读取数据集
data = pd.read_csv('./python/AirQualityUCI.csv', delimiter=';')

# 处理数据，选取CO浓度列，并去除无效数据
data['CO(GT)'] = data['CO(GT)'].replace(',', '.', regex=True).replace('-200', np.nan).astype(float)
data['CO(GT)'] = pd.to_numeric(data['CO(GT)'], errors='coerce')
data = data.dropna(subset=['CO(GT)'])

# 准备数据
raw_seq = data['CO(GT)'].values
n_steps_in, n_steps_out = 30, 10
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义 PyTorch Dataset 类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # 确保X和y是按照正确的形状tensor化的，即(batch_size, channels, length)
        self.X = torch.tensor(X.reshape(-1, 1, X.shape[1]), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 确保返回的输入数据是按照(batch_size, channels, length)的格式
        return self.X[idx], self.y[idx]
    
# 创建训练集和测试集的数据集实例
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# 创建训练集和测试集的数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义CNN模型
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels, input_size, hidden_size, output_size):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(hidden_size * (input_size // 2), output_size)  # Pooling reduces the length by half

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the FC layer
        x = self.fc1(x)
        return x

# 实例化模型
input_channels = 1  # Input channels, for single time series data it's 1
input_size = n_steps_in
hidden_size = 64  # Number of filters in the convolutional layer
output_size = n_steps_out
model = TimeSeriesCNN(input_channels, input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/len(train_dataloader)}')

# 测试模型
model.eval()
test_losses = []
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        test_losses.append(test_loss.item())

# 打印测试集上的损失
print("Test set mean squared error: ", np.mean(test_losses))

# 保存模型权重
import os
# 检查目录是否存在，不存在则创建
save_dir = './data/time_predict'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_dir = './data/time_predict'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))


# 加载模型
model = TimeSeriesCNN(input_channels, input_size, hidden_size, output_size) # 创建模型的实例（不需要定义参数，只需要架构）
save_dir ="./data/time_predict/model_weights.pth" # 加载模型权重
model.load_state_dict(torch.load(save_dir))

# 测试模型
model.eval() # 将模型设置为评估模式
test_inputs, test_outputs, test_targets = [], [], []
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        test_inputs.append(inputs.numpy())
        test_outputs.append(outputs.numpy())
        test_targets.append(targets.numpy())

# 将列表转换为NumPy数组以便于处理
test_inputs = np.concatenate(test_inputs)
test_outputs = np.concatenate(test_outputs)
test_targets = np.concatenate(test_targets)

# 打印输入测试数据
print("Test Input Data:")
print(test_inputs)

# 打印模型的输出测试数据
print("Model Output Data:")
print(test_outputs)

# 打印原来正确的测试数据
print("Original Target Data:")
print(test_targets)