import torch
import torch.nn as nn
import numpy as np
import os
import brevitas.nn as qnn  # 导入brevitas量化神经网络模块

# 定义一个简单的Transformer模块，使用Brevitas进行量化
class QuantizedSimpleTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads=1):
        super(QuantizedSimpleTransformer, self).__init__()
        self.embed = qnn.QuantEmbedding(num_embeddings=num_tokens, embedding_dim=dim_model, weight_bit_width=8)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads)  # 注意力机制目前不量化
        self.layernorm1 = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            qnn.QuantLinear(dim_model, dim_model * 4, weight_bit_width=8, bias=True),
            nn.ReLU(),
            qnn.QuantLinear(dim_model * 4, dim_model, weight_bit_width=8, bias=True)
        )
        self.layernorm2 = nn.LayerNorm(dim_model)

    def forward(self, x):
        x_embed = self.embed(x)
        attn_output, _ = self.self_attn(x_embed, x_embed, x_embed)
        x = self.layernorm1(x_embed + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x

# 创建量化模型实例
model = QuantizedSimpleTransformer(num_tokens=10, dim_model=5)

# 创建一个随机的输入序列并通过模型传递输入
input_sequence = torch.randint(0, 10, (5,)).unsqueeze(0)  # 添加批次维度
output = model(input_sequence)

# 打印输出
print(output)

# 导出量化模型的权重
if not os.path.exists('./data/brevitas'):
    os.makedirs('./data/brevitas')

for name, param in model.named_parameters():
    np.save(f'./data/brevitas/{name}.npy', param.detach().numpy())

# 请注意，量化注意力层可能需要额外的工作，因为Brevitas主要针对量化线性和卷积层。
# 上述代码中的注意力机制没有被量化，这是因为Brevitas还不直接支持MultiheadAttention层的量化。
