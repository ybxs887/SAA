import numpy as np

def generate_matrix(rows, cols, filename):
    # 生成随机整数矩阵
    matrix = np.random.randint(low=-5, high=5, size=(rows, cols))
    
    # 保存矩阵到文本文件
    np.savetxt(filename, matrix, fmt='%d')

# 定义权重矩阵的尺寸
row = 6  # 你可以根据需要修改这些尺寸
col = 6
col1 = 6

# 定义文件名
inputfile = './data/inputs.txt'
weightfile = './data/weights.txt'
resultfile = './data/results.txt'

# 生成和保存矩阵
generate_matrix(row, col, inputfile)
generate_matrix(col, col1, weightfile)


########################结果##########################
import numpy as np

# 读取输入矩阵和权重矩阵
inputs_matrix = np.loadtxt(inputfile, dtype=int)
weights_matrix = np.loadtxt(weightfile, dtype=int)

# 执行矩阵相乘操作
result_matrix = np.dot(inputs_matrix, weights_matrix)

print("inputs_matrix:\n",inputs_matrix)
print("weights_matrix:\n",weights_matrix)
print("result_matrix:\n",result_matrix)

# 保存结果矩阵到文件
np.savetxt(resultfile, result_matrix, fmt='%d')
