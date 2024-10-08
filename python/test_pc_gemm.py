# import numpy as np
# import time



# # # 生成两个256x256的随机矩阵
# # A = np.random.rand(1000, 1000)
# # B = np.random.rand(1000, 1000)

# # # 记录开始时间
# # start_time = time.time()

# # # 执行矩阵乘法
# # C = np.dot(A, B)

# # # 记录结束时间
# # end_time = time.time()

# # # 计算并打印执行时间
# # print(f"Execution time: {end_time - start_time} seconds")



# import numpy as np

# def pad_matrix(mat, block_size):
#     """填充矩阵使其行数和列数是block_size的倍数"""
#     rows, cols = mat.shape
#     pad_rows = (block_size - rows % block_size) % block_size
#     pad_cols = (block_size - cols % block_size) % block_size
#     return np.pad(mat, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

# def block_multiply(A, B, block_size):
#     """将矩阵分块并执行块乘法"""
#     n, m = A.shape
#     p, q = B.shape
#     C = np.zeros((n, q))

#     for i in range(0, n, block_size):
#         for j in range(0, q, block_size):
#             for k in range(0, m, block_size):
#                 A_block = A[i:i+block_size, k:k+block_size]
#                 B_block = B[k:k+block_size, j:j+block_size]
#                 C[i:i+block_size, j:j+block_size] += np.dot(A_block, B_block)
#     return C


# block_size = 4

# # 定义矩阵A和B
# A = np.random.randint(10, size=(8, 9))
# B = np.random.randint(10, size=(9, 10))

# # 将矩阵填充到合适的大小
# A_padded = pad_matrix(A, block_size)
# B_padded = pad_matrix(B, block_size)

# # 执行分块矩阵乘法
# C = block_multiply(A_padded, B_padded, block_size)


# res=np.dot(A, B)


# # 打印结果
# print(C)
# print(res)



# import numpy as np

# # 定义矩阵
# input_matrix = np.array([
#     [-19, 7, -6],
#     [0, -11, -16],
#     [18, 18, -18],
# ])

# weight_matrix = np.array([
#     [4, 5, 5],
#     [-19, 7, -19],
#     [-9, 15, 2],
# ])

# # 矩阵相乘
# result_matrix = np.dot(input_matrix, weight_matrix)

# print(result_matrix)



import numpy as np

matrix = np.array([
    [0, 1],
    [2, 3]
])

for j in range(0,2):
    for i in range(0,2):
        t=matrix[i][j]
        print(t)