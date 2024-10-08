import numpy as np
import time
def pack_matrix_to_buffer_np(matrix, block_size):
    rows, cols = matrix.shape
    assert rows % block_size == 0 and cols % block_size == 0, "Matrix dimensions must be multiples of block_size"
    # 将矩阵分割成 block_size x block_size 的块
    temp = matrix.reshape(rows // block_size, block_size, cols // block_size, block_size)
    # 交换块内行列的索引，使得块按行列排列
    temp = temp.transpose(0, 2, 1, 3)
    # 展平数组
    buffer = temp.ravel()
    return buffer

# 示例使用
row = 500
col = 500
block_size = 10
matrix = np.random.randint(0, 100, size=(row, col), dtype=np.int8)
print(matrix)


pt0 = time.perf_counter()
buffer = pack_matrix_to_buffer_np(matrix, block_size)
pt1 = time.perf_counter()
time_sw = pt1 - pt0
print("pure software: %fs" % time_sw)

print(buffer)
