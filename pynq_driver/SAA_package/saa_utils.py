
# 数据重排
# 矩阵按块大小分割并打包到缓冲区
def pack_matrix_to_buffer(matrix, block_size, buffer):
    rows, cols = matrix.shape
    block_row_count, block_col_count = rows // block_size, cols // block_size
    
    buffer_index = 0
    for block_row in range(block_row_count):
        for block_col in range(block_col_count):
            block = matrix[block_row*block_size:(block_row+1)*block_size, block_col*block_size:(block_col+1)*block_size]
            buffer[buffer_index:buffer_index+block.size] = block.flatten()
            buffer_index += block.size


# 解数据重排
def unpack_buffer_to_matrix(target_matrix, block_size ,buffer):
    rows, cols = target_matrix.shape
    block_row_count, block_col_count = rows // block_size, cols // block_size
    
    buffer_index = 0
    for block_row in range(block_row_count):
        for block_col in range(block_col_count):
            # 计算当前块在矩阵中的起始和结束位置
            start_row = block_row * block_size
            start_col = block_col * block_size
            end_row = (block_row + 1) * block_size
            end_col = (block_col + 1) * block_size
            
            # 从缓冲区中取出当前块的数据
            block_data = buffer[buffer_index:buffer_index + block_size*block_size]
            
            # 将块数据放回目标矩阵的对应位置
            target_matrix[start_row:end_row, start_col:end_col] = block_data.reshape(block_size, block_size)
            
            # 更新缓冲区索引
            buffer_index += block_size * block_size



# import numpy as np
# from pynq import allocate

# # 假设矩阵和块的大小
# matrix_rows, matrix_cols = 11, 11  # 矩阵大小，这次尝试一个不能整除的情况
# block_size = 4  # 块大小为 2x2

# # 创建一个示例矩阵
# matrix = np.arange(matrix_rows * matrix_cols).reshape(matrix_rows, matrix_cols)
# print("Original matrix:")
# print(matrix)

# # 计算填充后的新大小
# new_rows = matrix_rows + (block_size - matrix_rows % block_size) % block_size
# new_cols = matrix_cols + (block_size - matrix_cols % block_size) % block_size

# # 创建填充后的矩阵
# padded_matrix = np.zeros((new_rows, new_cols), dtype=matrix.dtype)
# padded_matrix[:matrix_rows, :matrix_cols] = matrix  # 将原矩阵复制到填充矩阵中
# print("Padded matrix:")
# print(padded_matrix)

# # 分配连续缓冲区，这次根据填充后的尺寸来分配
# buffer_size = new_rows * new_cols  # 缓冲区大小等于填充后矩阵的元素个数
# buffer = allocate(shape=(buffer_size,), dtype=np.int32)
            
# # 执行打包操作
# pack_matrix_to_buffer(padded_matrix, block_size, buffer)

# # 将连续缓冲区的内容打印出来，以验证结果
# print("Buffer content (flattened):")
# print(buffer)

# # 解包
# unpack_matrix = np.zeros((new_rows, new_cols), dtype=matrix.dtype)
# unpack_buffer_to_matrix(unpack_matrix, block_size, buffer)
# print("Matrix unpack:")
# print(unpack_matrix)



