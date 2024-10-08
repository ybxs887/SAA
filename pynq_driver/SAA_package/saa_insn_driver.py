import struct
import numpy as np

# 操作码和位宽定义
OP_CODE_WIDTH = 3  # 假设操作码位宽为8位
BUFFER_ID_WIDTH = 3  # 假设缓冲区ID位宽为8位
DRAM_ADDR_WIDTH = 32  # 假设DRAM地址位宽为32位
BUFFER_ADDR_WIDTH = 16 # 假设BUFFER行寻址位宽为16位
TRANSFER_SIZE_WIDTH = 16  # 假设矩阵大小位宽为16位
TRANSFER_STRIDE_WIDTH = 16  # 假设步进位宽为16位
COMPUTE_TYPE_WIDTH = 3  # 假设计算类型位宽为3位
WEIGHT_SWITCH_WIDTH = 1  # 假设权重切换位宽为1位
COMPUTE_SWITCH_WIDTH = 1  # 假设计算切换位宽为1位
COMPUTE_ACCUMULATE_WIDTH = 1  # 假设累加位宽为1位
ITER_WIDTH = 14 # 定义了分块计算的循环最大值

# 操作码定义
OPCODE_LOAD = 0 # 定义加载指令的操作码
OPCODE_COMPUTE = 1 # 定义计算指令的操作码
OPCODE_STORE = 2 # 定义存储指令的操作码
OPCODE_DONE = 3 # 定义计算完成指令的操作码
OPCODE_ANU = 4 # 定义ANU指令
OPCODE_GEMM = 5 # 定义GEMM指令

# compute_type 计算类型定义
WEIGHT_PRELOAD = 0 # 权重预加载
COMPUTE = 1 # 使用当前脉动阵列权重计算
COMPUTE_WEIGHT_PRELOAD = 2 # 加载权重同时进行计算，用于双缓冲操作

# buffer id 定义
WEIGHT_BUFFER_ID = 0
INPUT_BUFFER_ID = 1
ACCUMULATOR_BUFFER_ID = 2
OUTPUT_BUFFER_ID = 3
UOP_BUFFER_ID = 4


#ANU操作码
UNUSE_ANU    =0
ANU_SOFTMAX   =1
ANU_LAYERNORM = 2
ANU_SIGMOID   =3

#脉动阵列大小
MATRIX_WIDTH = 4

# 定义指令的数据类型
Instruct_DataType = np.dtype([('low', np.uint64), ('high', np.uint64)]) # 定义一个由两个64位整数组成的128位数据类型\

# 定义微操作序列的数据类型
Uop_DataType = np.dtype([('low', np.uint16), ('high', np.uint16)]) # 定义一个由两个16位数组成

# 计算矩阵的数据类型
Input_DataType =  np.int8
Weight_DataType = np.int8
Output_DataType = np.int32

# 使用struct模块生成128位指令
def create_load_instruction(opcode, buffer_id, dram_addr, buffer_addr, y_size, x_size, x_stride):
    # 我们需要将所有的字段合并成一个128位的整数
    # 第一个64位
    instruction = (opcode & ((1 << OP_CODE_WIDTH) - 1))
    instruction |= (buffer_id & ((1 << BUFFER_ID_WIDTH) - 1)) << OP_CODE_WIDTH
    instruction |= (dram_addr & ((1 << DRAM_ADDR_WIDTH) - 1)) << (OP_CODE_WIDTH + BUFFER_ID_WIDTH)
    instruction |= (buffer_addr & ((1 << BUFFER_ADDR_WIDTH) - 1)) << (OP_CODE_WIDTH + BUFFER_ID_WIDTH + DRAM_ADDR_WIDTH)
    # 第二个64位
    instruction |= (y_size & ((1 << TRANSFER_SIZE_WIDTH) - 1)) << (64)
    instruction |= (x_size & ((1 << TRANSFER_SIZE_WIDTH) - 1)) << (64 + TRANSFER_SIZE_WIDTH)
    instruction |= (x_stride & ((1 << TRANSFER_STRIDE_WIDTH) - 1)) << (64 + 2 * TRANSFER_SIZE_WIDTH)

    # 将128位整数转换成16字节的二进制数据
    # 使用'Q'格式符代表无符号的长长整型（64位），注意这里需要两个'Q'来表示128位
    # '<'代表小端字节序
    packed_instruction = struct.pack('<QQ',instruction & ((1 << 64) - 1) , (instruction >> 64) & ((1 << 64) - 1))
    return packed_instruction

# 创建计算指令的函数
def create_compute_instruction(opcode, compute_type, weight_addr, input_addr, output_addr, weight_switch, compute_switch, accumulate):
    # 第一个64位
    instruction = (opcode & ((1 << OP_CODE_WIDTH) - 1))
    instruction |= (compute_type & ((1 << COMPUTE_TYPE_WIDTH) - 1)) << OP_CODE_WIDTH
    instruction |= (weight_addr & ((1 << BUFFER_ADDR_WIDTH) - 1)) << (OP_CODE_WIDTH + COMPUTE_TYPE_WIDTH)
    instruction |= (input_addr & ((1 << BUFFER_ADDR_WIDTH) - 1)) << (OP_CODE_WIDTH + COMPUTE_TYPE_WIDTH + BUFFER_ADDR_WIDTH)
    instruction |= (output_addr & ((1 << BUFFER_ADDR_WIDTH) - 1)) << (OP_CODE_WIDTH + COMPUTE_TYPE_WIDTH + 2 * BUFFER_ADDR_WIDTH)
    # 目前没有第二个64位
    instruction |= (weight_switch & ((1 << WEIGHT_SWITCH_WIDTH) - 1)) << (OP_CODE_WIDTH + COMPUTE_TYPE_WIDTH + 3 * BUFFER_ADDR_WIDTH)
    instruction |= (compute_switch & ((1 << COMPUTE_SWITCH_WIDTH) - 1)) << (OP_CODE_WIDTH + COMPUTE_TYPE_WIDTH + 3 * BUFFER_ADDR_WIDTH + WEIGHT_SWITCH_WIDTH)
    instruction |= (accumulate & ((1 << COMPUTE_ACCUMULATE_WIDTH) - 1)) << (OP_CODE_WIDTH + COMPUTE_TYPE_WIDTH + 3 * BUFFER_ADDR_WIDTH + WEIGHT_SWITCH_WIDTH + COMPUTE_ACCUMULATE_WIDTH)

    # 将128位整数转换成16字节的二进制数据
    # 使用'Q'格式符代表无符号的长长整型（64位），注意这里需要两个'Q'来表示128位
    # '<'代表小端字节序
    packed_instruction = struct.pack('<QQ', instruction & ((1 << 64) - 1), (instruction >> 64) & ((1 << 64) - 1))
    return packed_instruction

# 创建GEMM函数 矩阵乘法的指令
def create_gemm_instruction(dim_I_block, 
                   dim_J_block, 
                   dim_K_block, 
                   bias_use):
    # 第一个64位
    instruction = (OPCODE_GEMM & ((1 << OP_CODE_WIDTH) - 1))
    instruction |= (bias_use & ((1 << 1) - 1)) << (OP_CODE_WIDTH + COMPUTE_TYPE_WIDTH + 3 * BUFFER_ADDR_WIDTH + WEIGHT_SWITCH_WIDTH + COMPUTE_ACCUMULATE_WIDTH+1)
    # 第二个64位
    instruction |= (0 & ((1 << BUFFER_ADDR_WIDTH) - 1)) << (64)
    instruction |= (dim_I_block & ((1 << BUFFER_ADDR_WIDTH) - 1)) << (64 +BUFFER_ADDR_WIDTH)
    instruction |= (dim_J_block & ((1 << ITER_WIDTH) - 1)) << (64 +BUFFER_ADDR_WIDTH +BUFFER_ADDR_WIDTH)
    instruction |= (dim_K_block & ((1 << ITER_WIDTH) - 1)) << (64 +BUFFER_ADDR_WIDTH +BUFFER_ADDR_WIDTH+ITER_WIDTH)
    
    # 将128位整数转换成16字节的二进制数据
    # 使用'Q'格式符代表无符号的长长整型（64位），注意这里需要两个'Q'来表示128位
    # '<'代表小端字节序
    packed_instruction = struct.pack('<QQ', instruction & ((1 << 64) - 1), (instruction >> 64) & ((1 << 64) - 1))
    return packed_instruction

    
# 创建微操作指令
def create_uop_instruction(input_idx, output_idx):
    instruction = (input_idx & ((1 << BUFFER_ADDR_WIDTH) - 1))
    instruction |= (output_idx & ((1 << BUFFER_ADDR_WIDTH) - 1)) << BUFFER_ADDR_WIDTH
    
    # 打包到16位数里，H代表16位，两个H代表32位
    packed_instruction = struct.pack('<HH', instruction & ((1 << 16) - 1), (instruction >> 16) & ((1 << 16) - 1))
    return packed_instruction
    
def print_binary(instruction_bytes):
    # 将字节字符串转换回两个64位整数
    high, low = struct.unpack('<QQ', instruction_bytes)
    # 将两个64位整数转换为二进制字符串，并去掉前缀'0b'
    binary_low = bin(low)[2:].zfill(64)    # 填充低位以确保长度为64位
    binary_high = bin(high)[2:].zfill(64)  # 填充高位以确保长度为64位
    # 合并二进制字符串并打印，注意这里先打印低地址的low，再打印高地址的high
    binary_instruction = binary_low + binary_high
    print(binary_instruction)

# 生成微操作指令
def getUOPInsn(input_idx, 
           output_idx):
    instruction=np.frombuffer(create_uop_instruction(input_idx, 
                                     output_idx),dtype=Uop_DataType)
    return instruction


# 生成gemm指令
def getGEMMInsn(dim_I_block, 
           dim_J_block,
           dim_K_block,
           bias_use):
    instruction=np.frombuffer( create_gemm_instruction(dim_I_block, 
                                       dim_J_block, 
                                       dim_K_block, 
                                       bias_use),dtype=Instruct_DataType)
    return instruction

    
# 生成2D加载、存储指令，加载存储矩阵块，需要指定opcode是加载还是存储
def get2DLoadStoreInsn(opcode, 
               buffer_id, 
               buffer_offset, 
               dram_offset, 
               y_size, 
               x_size, 
               x_stride):
    instruction=np.frombuffer(create_load_instruction(opcode, 
                                      buffer_id, 
                                      dram_offset, 
                                      buffer_offset, 
                                      y_size, 
                                      x_size, 
                                      x_stride),dtype=Instruct_DataType)
    return instruction
    
    
# 生成GEMM指令，包括多种计算类型
# 权重预加载指令，只需要权重相关参数
def getWeightPreloadInsn(weigth_offset, # 权重块加载偏移
                 weight_switch): # 加载到哪个权重位置
    instruction=np.frombuffer(create_compute_instruction(OPCODE_COMPUTE, 
                                       WEIGHT_PRELOAD, 
                                       weigth_offset, 
                                       0, 
                                       0, 
                                       weight_switch, 
                                       0, 
                                       0),dtype=Instruct_DataType)
    return instruction
    

# 计算指令，只需要读取输入和存储输出，以及选择进行矩阵乘法的寄存器和累加  
def getComputeInsn(input_offset, # 权重块加载偏移
             output_offset, # 输出存储偏移
             compute_switch, # 使用哪个寄存器进行计算
             accumulate): # 当前计算是否进行累加   
    instruction=np.frombuffer(create_compute_instruction(OPCODE_COMPUTE, 
                                       COMPUTE, 
                                       0, 
                                       input_offset, 
                                       output_offset, 
                                       0, 
                                       compute_switch, 
                                       accumulate),dtype=Instruct_DataType)
    return instruction
    
# 计算预加载指令，同时进行计算和预加载，只不过预加载和计算的寄存器在内部就做了乒乓缓冲
# 计算指令计算的就是参数的寄存器，预加载加载的是另一个寄存器
# 因此预加载寄存器后，调用该函数继续计算上面那个寄存器，而这个最后执行完，调用计算计算相反寄存器
def getWeightPreloadComputeInsn(input_offset,    # 权重块加载偏移
                      weigth_offset,   # 权重块加载偏移
                      output_offset,   # 输出存储偏移
                      weight_switch,   # 加载到哪个权重位置
                      compute_switch,  # 使用哪个寄存器进行计算
                      accumulate):    # 当前计算是否进行累加 
    instruction=np.frombuffer(create_compute_instruction(OPCODE_COMPUTE, 
                                       COMPUTE_WEIGHT_PRELOAD, 
                                       weigth_offset, 
                                       input_offset, 
                                       output_offset, 
                                       weight_switch, 
                                       compute_switch, 
                                       accumulate),dtype=Instruct_DataType)
    return instruction

# 生成运算完成指令
def getFinishInsn():
    instruction=np.frombuffer(create_compute_instruction(OPCODE_DONE, 
                                           0, 
                                           0, 
                                           0, 
                                           0, 
                                           0, 
                                           0, 
                                           0),dtype=Instruct_DataType)
    return instruction
    

# 加载库
from pynq import allocate
from saa_utils import *
import time
import numpy as np

wait_cycles = 100000 # 定义一次最多等待周期为1000万周期

    
def blocked_gemm_test(saa_driver,
              dim_I, 
              dim_J, 
              dim_K, 
              input, 
              weight,
              bias, 
              output, 
              block, 
              bias_use):
    
    print("=====================================================================================")
    print(f"INFO - Blocked GEMM test: dim_I={dim_I}, dim_J={dim_J}, dim_K={dim_K}, block={block}, bias_use={bias_use}")
    # 加载分块
    dim_I_load_block = dim_I // block
    dim_J_load_block = dim_J // block
    dim_K_load_block = dim_K // block 
    
    # 加载分块和计算分块的比值   
    load_compute_block_ratio = block // MATRIX_WIDTH
    
    # 计算分块
    dim_I_block = dim_I // MATRIX_WIDTH
    dim_J_block = dim_J // MATRIX_WIDTH
    dim_K_block = dim_K // MATRIX_WIDTH
    
#     # 声明连续缓冲区
#     input_buffer = allocate(shape = (dim_I*dim_K), cacheable = 0, dtype = Input_DataType)
#     weight_buffer = allocate(shape = (dim_K*dim_J), cacheable = 0, dtype = Weight_DataType)

#     # 进行打包操作，输入矩阵要和缓冲区的大小一致
#     pack_matrix_to_buffer(input, block, input_buffer)
#     pack_matrix_to_buffer(weight, block, weight_buffer)

    # 计算指令数量
    insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block) +(dim_I_block * dim_J_block)
#     insn_compute_size = 2 * dim_I_block * dim_K_block * dim_J_block  # 不使用权重复用
#     insn_compute_size = (dim_I_block + 1) * dim_K_block * dim_J_block  # 使用权重复用
    insn_compute_size = dim_I_block * dim_K_block * dim_J_block + 1  # 使用权重复用和双缓冲
    insn_store_size = dim_I_block * dim_J_block
    insn_size = insn_load_size + insn_store_size + insn_compute_size + 1

    # 初始化指令队列
    insn_buf = allocate(shape = (insn_size), cacheable = 0, dtype = Instruct_DataType)
    insn_idx = 0
    
    # 初始化微操作序列
    uop_buf = allocate(shape = (dim_I_block), cacheable = 0, dtype = Uop_DataType)
    # 生成微操作
    for i in range(dim_I_block):
        input_idx=i * dim_K_block * MATRIX_WIDTH
        output_idx=i * dim_J_block * MATRIX_WIDTH
        uop_buf[i] = getUOPInsn(input_idx,output_idx)
    
    # 加载微操作
    insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
                              UOP_BUFFER_ID, 
                              0, 
                              0, 
                              1, 
                              dim_I_block, 
                              0)# 直接加载一整个块
    insn_idx += 1
    

    # 加载偏置biases(dim_I,dim_J)
    if bias_use == 1 and bias is not None:
        insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
                                  OUTPUT_BUFFER_ID, 
                                  0, 
                                  0, 
                                  dim_I_block, 
                                  dim_J_block, 
                                  dim_J_block)# 直接加载一整个块
        insn_idx += 1

    
    # 加载输入input(dim_I,dim_K)
    insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
                              INPUT_BUFFER_ID, 
                              0, 
                              0, 
                              dim_I_block, 
                              dim_K_block, 
                              dim_K_block)# 直接加载一整个块
    insn_idx += 1


    # 加载权重weight(dim_K,dim_J)
    insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
                              WEIGHT_BUFFER_ID, 
                              0, 
                              0, 
                              dim_K_block, 
                              dim_J_block, 
                              dim_J_block)# 直接加载一整个块
    insn_idx += 1
    
    # 生成计算指令
    insn_buf[insn_idx] = getGEMMInsn(dim_I_block, 
                           dim_J_block,
                           dim_K_block,
                           bias_use)
    insn_idx += 1

    
    # 生成存储指令     
    insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_STORE, 
                              OUTPUT_BUFFER_ID, 
                              0, 
                              0, 
                              dim_I_block, 
                              dim_J_block, 
                              dim_J_block)# 直接加载一整个块
    insn_idx += 1
            
    # 生成结束指令
    insn_buf[insn_idx] = getFinishInsn()
    insn_idx += 1
    print(f"insn_idx: {insn_idx}") 
    
    # 运行SAA硬件
    pt0 = time.perf_counter()
    saa_driver.run_saa(insn_idx,
           insn_buf.physical_address,
           uop_buf.physical_address,
           input.physical_address,
           weight.physical_address,
           bias.physical_address,
           output.physical_address,
           wait_cycles)
    pt1 = time.perf_counter()
    t_fpga = pt1 - pt0
    print("INFO - Saa run time: %fs" % t_fpga)     
    
    # 计算同步时间和吞吐量
    synchronization_time_ms = t_fpga * 1E3  # 将秒转换为毫秒
    throughput_gops_s = (dim_I * dim_J * dim_K * 3) / (t_fpga*1E9) # 计算每秒的浮点运算次数（GOPs）（矩阵乘法算*2，再加偏置算*3）

    # 打印结果，保留 6 位小数
    print(f"INFO - Synchronization time: {synchronization_time_ms:.6f}ms")
    print(f"INFO - Throughput: {throughput_gops_s:.6f}GOPs/s")

    # 清空指令buf
    del insn_buf
    
    return 0 



# def blocked_gemm_test(saa_driver,
#               dim_I, 
#               dim_J, 
#               dim_K, 
#               input, 
#               weight,
#               bias, 
#               output, 
#               block, 
#               bias_use):
    
#     print("=====================================================================================")
#     print(f"INFO - Blocked GEMM test: dim_I={dim_I}, dim_J={dim_J}, dim_K={dim_K}, block={block}, bias_use={bias_use}")
    
#     # 计算分块
#     dim_I_block = dim_I // MATRIX_WIDTH
#     dim_J_block = dim_J // MATRIX_WIDTH
#     dim_K_block = dim_K // MATRIX_WIDTH

#     # 计算指令数量
#     insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block)
# #     insn_compute_size = 2 * dim_I_block * dim_K_block * dim_J_block  # 不使用权重复用
# #     insn_compute_size = (dim_I_block + 1) * dim_K_block * dim_J_block  # 使用权重复用
#     insn_compute_size = dim_I_block * dim_K_block * dim_J_block + 1  # 使用权重复用和双缓冲
#     insn_store_size = dim_I_block * dim_J_block
#     insn_size = insn_load_size + insn_store_size + insn_compute_size + 1

#     # 初始化指令队列
#     insn_buf = allocate(shape = (insn_size), cacheable = 0, dtype = Instruct_DataType)
#     insn_idx = 0
    
#     # 生成加载Input指令
#     for i in range(dim_I_block):
#         for k in range(dim_K_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i*dim_K_block+k
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + i * dim_K_block * MATRIX_WIDTH * MATRIX_WIDTH + k * MATRIX_WIDTH
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       INPUT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       MATRIX_WIDTH, 
#                                       MATRIX_WIDTH, 
#                                       dim_K)# 直接加载一整个块
#             insn_idx += 1

#     # 生成加载weight指令
#     for k in range(dim_K_block):
#         for j in range(dim_J_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = k * dim_J_block + j
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + k * dim_J_block * MATRIX_WIDTH * MATRIX_WIDTH + j * MATRIX_WIDTH 
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       WEIGHT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       MATRIX_WIDTH, 
#                                       MATRIX_WIDTH, 
#                                       dim_J)
#             insn_idx += 1
    
#     # 生成计算指令
#     # 用于切换权重寄存器，最先使用 weight1
#     pingpang = 0
#     wb_start_addr = 0
#     input_start_addr = 0
#     output_start_addr = 0
#     weight_offset = 0
#     output_offset = 0
#     input_offset = 0
#     accumulate = 0
#     accumulate_delay = 0
    
#     # 初始化指令计数
#     compute_count = insn_idx

#     # 迭代公共维度块和输出列块
#     for k in range(dim_K_block):
#         for j in range(dim_J_block):
#             # 计算权重偏移
#             weight_offset = wb_start_addr + (k * dim_J_block + j) * MATRIX_WIDTH
#             accumulate = 0 if k == 0 else 1
#             # 第一次加载权重，使用初始寄存器，无法双缓冲
#             if k == 0 and j == 0:
#                 insn_buf[insn_idx] = getWeightPreloadInsn(weight_offset, pingpang)
#                 insn_idx += 1
#             else:
#                 # 剩下的权重加载可以进行双缓冲
#                 insn_buf[insn_idx] = getWeightPreloadComputeInsn(
#                     input_offset,
#                     weight_offset,
#                     output_offset,
#                     pingpang,
#                     pingpang,
#                     accumulate_delay)
#                 pingpang = not pingpang  # 切换加载寄存器和计算寄存器
#                 insn_idx += 1

#             # 迭代输出行块
#             for i in range(dim_I_block):
#                 output_offset = output_start_addr + (i * dim_J_block + j) * MATRIX_WIDTH
#                 input_offset = input_start_addr + (i * dim_K_block + k) * MATRIX_WIDTH
#                 accumulate_delay = accumulate# 将累加标志位延迟一位，这个值是原来下面执行的getComputeInsn，当getWeightPreloadComputeInsn的功能之一是执行下面的getComputeInsn时，必须用原来的值
#                 # 如果不是最后一个计算，使用 getComputeInsn 计算
                
#                 if i != dim_I_block - 1:
#                     insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                             output_offset, 
#                                             pingpang, 
#                                             accumulate)
#                     insn_idx += 1
#                 # 如果是最后一个权重块，使用当前寄存器进行计算
#                 if i == dim_I_block - 1 and j == dim_J_block - 1 and k == dim_K_block - 1:
#                     insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                             output_offset, 
#                                             pingpang, 
#                                             accumulate)
#                     insn_idx += 1
                    
#     # 更新计算指令的数量
#     compute_count = insn_idx - compute_count
#     print(f"compute_insn_count: {compute_count}")    
#     print(f"insn_size: {insn_size}") 
    
#     # 生成存储指令
#     for i in range(dim_I_block):
#         for j in range(dim_J_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i * dim_J_block + j
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH * MATRIX_WIDTH + j * MATRIX_WIDTH
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_STORE, 
#                                        OUTPUT_BUFFER_ID, 
#                                        buffer_offset, 
#                                        dram_offset, 
#                                        MATRIX_WIDTH, 
#                                        MATRIX_WIDTH, 
#                                        dim_J)
#             insn_idx += 1
            
#     # 生成结束指令
#     insn_buf[insn_idx] = getFinishInsn()
#     insn_idx += 1

# #     # 打印所有计算指令
# #     print("insn_size",insn_size)
# #     print("insn_idx",insn_idx)
# #     print("insn",insn_idx)
# #     for i in range(insn_idx):
# #         if i>=insn_load_size and i<insn_load_size+compute_count:
# #             print_binary(insn_buf[i])
            
#     # 运行SAA硬件
#     pt0 = time.perf_counter()
#     saa_driver.run_saa(insn_idx,
#            insn_buf.physical_address,
#            input.physical_address,
#            weight.physical_address,
#            output.physical_address,
#            wait_cycles)
#     pt1 = time.perf_counter()
#     t_fpga = pt1 - pt0
#     print("INFO - Saa run time: %fs" % t_fpga)     
    
#     # 计算吞吐量
    
#     # 计算同步时间和吞吐量
#     synchronization_time_ms = t_fpga * 1E3  # 将秒转换为毫秒
#     throughput_gops_s = (dim_I * dim_J * dim_K * 2) / (t_fpga*1E9) # 计算每秒的浮点运算次数（GOPs）

#     # 打印结果，保留 6 位小数
#     print(f"INFO - Synchronization time: {synchronization_time_ms:.6f}ms")
#     print(f"INFO - Throughput: {throughput_gops_s:.6f}GOPs/s")

#     # 清空指令buf
#     del insn_buf
    
#     return 0 
            

    
    


# def blocked_gemm_test(saa_driver,
#               dim_I, 
#               dim_J, 
#               dim_K, 
#               input, 
#               weight,
#               bias, 
#               output, 
#               block, 
#               bias_use):
    
#     print("=====================================================================================")
#     print(f"INFO - Blocked GEMM test: dim_I={dim_I}, dim_J={dim_J}, dim_K={dim_K}, block={block}, bias_use={bias_use}")
#     # 加载分块
#     dim_I_load_block = dim_I // block
#     dim_J_load_block = dim_J // block
#     dim_K_load_block = dim_K // block 
    
#     # 加载分块和计算分块的比值   
#     load_compute_block_ratio = block // MATRIX_WIDTH
    
#     # 计算分块
#     dim_I_block = dim_I // MATRIX_WIDTH
#     dim_J_block = dim_J // MATRIX_WIDTH
#     dim_K_block = dim_K // MATRIX_WIDTH
    
# #     # 声明连续缓冲区
# #     input_buffer = allocate(shape = (dim_I*dim_K), cacheable = 0, dtype = Input_DataType)
# #     weight_buffer = allocate(shape = (dim_K*dim_J), cacheable = 0, dtype = Weight_DataType)

# #     # 进行打包操作，输入矩阵要和缓冲区的大小一致
# #     pack_matrix_to_buffer(input, block, input_buffer)
# #     pack_matrix_to_buffer(weight, block, weight_buffer)

#     # 计算指令数量
#     insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block) +(dim_I_block * dim_J_block)
# #     insn_compute_size = 2 * dim_I_block * dim_K_block * dim_J_block  # 不使用权重复用
# #     insn_compute_size = (dim_I_block + 1) * dim_K_block * dim_J_block  # 使用权重复用
#     insn_compute_size = dim_I_block * dim_K_block * dim_J_block + 1  # 使用权重复用和双缓冲
#     insn_store_size = dim_I_block * dim_J_block
#     insn_size = insn_load_size + insn_store_size + insn_compute_size + 1

#     # 初始化指令队列
#     insn_buf = allocate(shape = (insn_size), cacheable = 0, dtype = Instruct_DataType)
#     insn_idx = 0
    
    
#     # 加载偏置biases(dim_I,dim_J)
#     if bias_use == 1 and bias is not None:
#         for i in range(dim_I_load_block):
#             for j in range(dim_J_load_block):
#                 buffer_start = 0
#                 dram_start = 0
#                 A_block = i*dim_J_load_block+j
#                 buffer_offset = buffer_start + (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH
#                 x_size = block*block
#                 dram_offset = dram_start + A_block * x_size # 直接按顺序计算
#                 insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                           OUTPUT_BUFFER_ID, 
#                                           buffer_offset, 
#                                           dram_offset, 
#                                           1, 
#                                           x_size, 
#                                           dim_J)# 直接加载一整个块
#                 insn_idx += 1

    
#     # 加载输入input(dim_I,dim_K)
#     for i in range(dim_I_load_block):
#         for k in range(dim_K_load_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i*dim_K_load_block+k
#             buffer_offset = buffer_start + (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH
#             x_size = block*block
#             dram_offset = dram_start + A_block * x_size # 直接按顺序计算
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       INPUT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       1, 
#                                       x_size, 
#                                       dim_K)# 直接加载一整个块
#             insn_idx += 1

#     # 加载权重weight(dim_K,dim_J)
#     for k in range(dim_K_load_block):
#         for j in range(dim_J_load_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = k * dim_J_load_block + j
#             buffer_offset = buffer_start + (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH
#             x_size = block*block
#             dram_offset = dram_start + A_block * x_size # 直接按顺序计算
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       WEIGHT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       1, 
#                                       x_size, 
#                                       dim_J)
#             insn_idx += 1
    
#     # 生成计算指令
#     # 用于切换权重寄存器，最先使用 weight1
#     pingpang = 0
#     wb_start_addr = 0
#     input_start_addr = 0
#     output_start_addr = 0
#     weight_offset = 0
#     output_offset = 0
#     input_offset = 0
#     accumulate = 0
#     accumulate_delay = 0
    
#     # 初始化指令计数
#     compute_count = insn_idx

#     # 迭代公共维度块和输出列块
#     for k in range(dim_K_block):
#         for j in range(dim_J_block):
#             # 计算权重偏移
#             weight_offset = wb_start_addr + (k * dim_J_block + j) * MATRIX_WIDTH
#             accumulate = 1 if bias_use == 1 else (0 if k == 0 else 1) # 如果是第一个块，刷新累加器中的累加值,其他块进行累加,如果使用了biase，那么全部累加

#             # 第一次加载权重，使用初始寄存器，无法双缓冲
#             if k == 0 and j == 0:
#                 insn_buf[insn_idx] = getWeightPreloadInsn(weight_offset, pingpang)
#                 insn_idx += 1
#             else:
#                 # 剩下的权重加载可以进行双缓冲
#                 insn_buf[insn_idx] = getWeightPreloadComputeInsn(
#                     input_offset,
#                     weight_offset,
#                     output_offset,
#                     pingpang,
#                     pingpang,
#                     accumulate_delay)
#                 pingpang = not pingpang  # 切换加载寄存器和计算寄存器
#                 insn_idx += 1

#             # 迭代输出行块
#             for i in range(dim_I_block):
#                 output_offset = output_start_addr + (i * dim_J_block + j) * MATRIX_WIDTH
#                 input_offset = input_start_addr + (i * dim_K_block + k) * MATRIX_WIDTH
#                 accumulate_delay = accumulate# 将累加标志位延迟一位，这个值是原来下面执行的getComputeInsn，当getWeightPreloadComputeInsn的功能之一是执行下面的getComputeInsn时，必须用原来的值
#                 # 如果不是最后一个计算，使用 getComputeInsn 计算
                
#                 if i != dim_I_block - 1:
#                     insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                             output_offset, 
#                                             pingpang, 
#                                             accumulate)
#                     insn_idx += 1
#                 # 如果是最后一个权重块，使用当前寄存器进行计算
#                 if i == dim_I_block - 1 and j == dim_J_block - 1 and k == dim_K_block - 1:
#                     insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                             output_offset, 
#                                             pingpang, 
#                                             accumulate)
#                     insn_idx += 1
                    
#     # 更新计算指令的数量
#     compute_count = insn_idx - compute_count
#     print(f"compute_insn_count: {compute_count}")    
#     print(f"insn_size: {insn_size}") 
    
#     # 生成存储指令
#     for i in range(dim_I_block):
#         for j in range(dim_J_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i * dim_J_block + j
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH*MATRIX_WIDTH + j * MATRIX_WIDTH

#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_STORE, 
#                                        OUTPUT_BUFFER_ID, 
#                                        buffer_offset, 
#                                        dram_offset, 
#                                        MATRIX_WIDTH, 
#                                        MATRIX_WIDTH, 
#                                        dim_J)
#             insn_idx += 1
            
#     # 生成结束指令
#     insn_buf[insn_idx] = getFinishInsn()
#     insn_idx += 1
#     print(f"insn_idx: {insn_idx}") 
    
#     # 运行SAA硬件
#     pt0 = time.perf_counter()
#     saa_driver.run_saa(insn_idx,
#            insn_buf.physical_address,
#            input.physical_address,
#            weight.physical_address,
#            bias.physical_address,
#            output.physical_address,
#            wait_cycles)
#     pt1 = time.perf_counter()
#     t_fpga = pt1 - pt0
#     print("INFO - Saa run time: %fs" % t_fpga)     
    
#     # 计算同步时间和吞吐量
#     synchronization_time_ms = t_fpga * 1E3  # 将秒转换为毫秒
#     throughput_gops_s = (dim_I * dim_J * dim_K * 3) / (t_fpga*1E9) # 计算每秒的浮点运算次数（GOPs）（矩阵乘法算*2，再加偏置算*3）

#     # 打印结果，保留 6 位小数
#     print(f"INFO - Synchronization time: {synchronization_time_ms:.6f}ms")
#     print(f"INFO - Throughput: {throughput_gops_s:.6f}GOPs/s")

#     # 清空指令buf
#     del insn_buf
    
#     return 0 
    
# def blocked_gemm_test(saa_driver,
#               dim_I, 
#               dim_J, 
#               dim_K, 
#               input, 
#               weight,
#               bias, 
#               output, 
#               block, 
#               bias_use):
    
#     print("=====================================================================================")
#     print(f"INFO - Blocked GEMM test: dim_I={dim_I}, dim_J={dim_J}, dim_K={dim_K}, block={block}, bias_use={bias_use}")
    
#     # 计算分块
#     dim_I_block = dim_I // MATRIX_WIDTH
#     dim_J_block = dim_J // MATRIX_WIDTH
#     dim_K_block = dim_K // MATRIX_WIDTH

#     # 计算指令数量
#     insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block)
# #     insn_compute_size = 2 * dim_I_block * dim_K_block * dim_J_block  # 不使用权重复用
# #     insn_compute_size = (dim_I_block + 1) * dim_K_block * dim_J_block  # 使用权重复用
#     insn_compute_size = dim_I_block * dim_K_block * dim_J_block + 1  # 使用权重复用和双缓冲
#     insn_store_size = dim_I_block * dim_J_block
#     insn_size = insn_load_size + insn_store_size + insn_compute_size + 1

#     # 初始化指令队列
#     insn_buf = allocate(shape = (insn_size), cacheable = 0, dtype = Instruct_DataType)
#     insn_idx = 0
    
#     # 生成加载Input指令
#     for i in range(dim_I_block):
#         for k in range(dim_K_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i*dim_K_block+k
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + i * dim_K_block * MATRIX_WIDTH * MATRIX_WIDTH + k * MATRIX_WIDTH
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       INPUT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       MATRIX_WIDTH, 
#                                       MATRIX_WIDTH, 
#                                       dim_K)
#             insn_idx += 1

#     # 生成加载weight指令
#     for k in range(dim_K_block):
#         for j in range(dim_J_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = k * dim_J_block + j
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + k * dim_J_block * MATRIX_WIDTH * MATRIX_WIDTH + j * MATRIX_WIDTH 
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       WEIGHT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       MATRIX_WIDTH, 
#                                       MATRIX_WIDTH, 
#                                       dim_J)
#             insn_idx += 1
    
#     # 生成计算指令
#     # 用于切换权重寄存器，最先使用 weight1
#     pingpang = 0
#     wb_start_addr = 0
#     input_start_addr = 0
#     output_start_addr = 0
#     weight_offset = 0
#     output_offset = 0
#     input_offset = 0
#     accumulate = 0
#     accumulate_delay = 0
    
#     # 初始化指令计数
#     compute_count = insn_idx

#     # 迭代公共维度块和输出列块
#     for k in range(dim_K_block):
#         for j in range(dim_J_block):
#             # 计算权重偏移
#             weight_offset = wb_start_addr + (k * dim_J_block + j) * MATRIX_WIDTH
#             accumulate = 0 if k == 0 else 1
#             # 第一次加载权重，使用初始寄存器，无法双缓冲
#             if k == 0 and j == 0:
#                 insn_buf[insn_idx] = getWeightPreloadInsn(weight_offset, pingpang)
#                 insn_idx += 1
#             else:
#                 # 剩下的权重加载可以进行双缓冲
#                 insn_buf[insn_idx] = getWeightPreloadComputeInsn(
#                     input_offset,
#                     weight_offset,
#                     output_offset,
#                     pingpang,
#                     pingpang,
#                     accumulate_delay)
#                 pingpang = not pingpang  # 切换加载寄存器和计算寄存器
#                 insn_idx += 1

#             # 迭代输出行块
#             for i in range(dim_I_block):
#                 output_offset = output_start_addr + (i * dim_J_block + j) * MATRIX_WIDTH
#                 input_offset = input_start_addr + (i * dim_K_block + k) * MATRIX_WIDTH
#                 accumulate_delay = accumulate# 将累加标志位延迟一位，这个值是原来下面执行的getComputeInsn，当getWeightPreloadComputeInsn的功能之一是执行下面的getComputeInsn时，必须用原来的值
#                 # 如果不是最后一个计算，使用 getComputeInsn 计算
                
#                 if i != dim_I_block - 1:
#                     insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                             output_offset, 
#                                             pingpang, 
#                                             accumulate)
#                     insn_idx += 1
#                 # 如果是最后一个权重块，使用当前寄存器进行计算
#                 if i == dim_I_block - 1 and j == dim_J_block - 1 and k == dim_K_block - 1:
#                     insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                             output_offset, 
#                                             pingpang, 
#                                             accumulate)
#                     insn_idx += 1
                    
#     # 更新计算指令的数量
#     compute_count = insn_idx - compute_count
#     print(f"compute_insn_count: {compute_count}")    
#     print(f"insn_size: {insn_size}") 
    
#     # 生成存储指令
#     for i in range(dim_I_block):
#         for j in range(dim_J_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i * dim_J_block + j
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH * MATRIX_WIDTH + j * MATRIX_WIDTH
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_STORE, 
#                                        OUTPUT_BUFFER_ID, 
#                                        buffer_offset, 
#                                        dram_offset, 
#                                        MATRIX_WIDTH, 
#                                        MATRIX_WIDTH, 
#                                        dim_J)
#             insn_idx += 1
            
#     # 生成结束指令
#     insn_buf[insn_idx] = getFinishInsn()
#     insn_idx += 1

# #     # 打印所有计算指令
# #     print("insn_size",insn_size)
# #     print("insn_idx",insn_idx)
# #     print("insn",insn_idx)
# #     for i in range(insn_idx):
# #         if i>=insn_load_size and i<insn_load_size+compute_count:
# #             print_binary(insn_buf[i])
            
#     # 运行SAA硬件
#     pt0 = time.perf_counter()
#     saa_driver.run_saa(insn_idx,
#            insn_buf.physical_address,
#            input.physical_address,
#            weight.physical_address,
#            output.physical_address,
#            wait_cycles)
#     pt1 = time.perf_counter()
#     t_fpga = pt1 - pt0
#     print("INFO - Saa run time: %fs" % t_fpga)     
    
#     # 计算吞吐量
    
#     # 计算同步时间和吞吐量
#     synchronization_time_ms = t_fpga * 1E3  # 将秒转换为毫秒
#     throughput_gops_s = (dim_I * dim_J * dim_K * 2) / (t_fpga*1E9) # 计算每秒的浮点运算次数（GOPs）

#     # 打印结果，保留 6 位小数
#     print(f"INFO - Synchronization time: {synchronization_time_ms:.6f}ms")
#     print(f"INFO - Throughput: {throughput_gops_s:.6f}GOPs/s")

#     # 清空指令buf
#     del insn_buf
    
#     return 0 
    
    
# def blocked_gemm_test(saa_driver,
#               dim_I, 
#               dim_J, 
#               dim_K, 
#               input, 
#               weight,
#               bias, 
#               output, 
#               block, 
#               bias_use):
    
#     print("=====================================================================================")
#     print(f"INFO - Blocked GEMM test: dim_I={dim_I}, dim_J={dim_J}, dim_K={dim_K}, block={block}, bias_use={bias_use}")
    
#     # 计算分块
#     dim_I_block = dim_I // MATRIX_WIDTH
#     dim_J_block = dim_J // MATRIX_WIDTH
#     dim_K_block = dim_K // MATRIX_WIDTH

#     # 计算指令数量
#     insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block)
#     insn_compute_size = 2 * dim_I_block * dim_K_block * dim_J_block  # 不使用权重复用
# #     insn_compute_size = (dim_I_block + 1) * dim_K_block * dim_J_block  # 使用权重复用
# #     insn_compute_size = dim_I_block * dim_K_block * dim_J_block + 1  # 使用权重复用和双缓冲
#     insn_store_size = dim_I_block * dim_J_block
#     insn_size = insn_load_size + insn_store_size + insn_compute_size + 1

#     # 初始化指令队列
#     insn_buf = allocate(shape = (insn_size), cacheable = 0, dtype = Instruct_DataType)
#     insn_idx = 0
    
#     # 生成加载Input指令
#     for i in range(dim_I_block):
#         for k in range(dim_K_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i*dim_K_block+k
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + i * dim_K_block * MATRIX_WIDTH * MATRIX_WIDTH + k * MATRIX_WIDTH
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       INPUT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       MATRIX_WIDTH, 
#                                       MATRIX_WIDTH, 
#                                       dim_K)
#             insn_idx += 1

#     # 生成加载weight指令
#     for k in range(dim_K_block):
#         for j in range(dim_J_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = k * dim_J_block + j
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + k * dim_J_block * MATRIX_WIDTH * MATRIX_WIDTH + j * MATRIX_WIDTH 
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       WEIGHT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       MATRIX_WIDTH, 
#                                       MATRIX_WIDTH, 
#                                       dim_J)
#             insn_idx += 1
    
#     # 生成计算指令
#     # 用于切换权重寄存器，最先使用 weight1
#     pingpang = 0
#     wb_start_addr = 0
#     input_start_addr = 0
#     output_start_addr = 0
#     weight_offset = 0
    
#     output_offset = 0
#     input_offset = 0
#     accumulate = 0
#     accumulate_delay = 0
    
#     # 初始化指令计数
#     compute_count = insn_idx

#     # 迭代公共维度块和输出列块
#     for k in range(dim_K_block):
#         for j in range(dim_J_block):
#             # 计算权重偏移
#             weight_offset = wb_start_addr + (k * dim_J_block + j) * MATRIX_WIDTH
#             accumulate = 0 if k == 0 else 1

#             # 第一次加载权重，使用初始寄存器，无法双缓冲
#             if k == 0 and j == 0:
#                 insn_buf[insn_idx] = getWeightPreloadInsn(weight_offset, pingpang)
#                 insn_idx += 1
#                 print("getWeightPreloadInsn")    
#             else:
#                 insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                         output_offset, 
#                                         pingpang, 
#                                         accumulate_delay)
#                 insn_idx += 1
#                 print("getComputeInsn")
#                 insn_buf[insn_idx] = getWeightPreloadInsn(weight_offset, pingpang)
#                 insn_idx += 1
#                 print("getWeightPreloadInsn")
                
# #                 pingpang = not pingpang  # 切换加载寄存器和计算寄存器
#             # 迭代输出行块
#             for i in range(dim_I_block):
#                 output_offset = output_start_addr + (i * dim_J_block + j) * MATRIX_WIDTH
#                 input_offset = input_start_addr + (i * dim_K_block + k) * MATRIX_WIDTH
#                 accumulate_delay = accumulate
#                 # 如果不是最后一个计算，使用 getComputeInsn 计算
#                 if i != dim_I_block - 1:
#                     insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                             output_offset, 
#                                             pingpang, 
#                                             accumulate)
#                     insn_idx += 1
#                     print("getComputeInsn")
#                 # 如果是最后一个权重块，使用当前寄存器进行计算
#                 if i == dim_I_block - 1 and j == dim_J_block - 1 and k == dim_K_block - 1:
#                     insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                             output_offset, 
#                                             pingpang, 
#                                             accumulate)
#                     insn_idx += 1
#                     print("getComputeInsn")

#     # 更新计算指令的数量
#     compute_count = insn_idx - compute_count
#     print(f"compute_count: {compute_count}")    

    
#     # 生成存储指令
#     for i in range(dim_I_block):
#         for j in range(dim_J_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i * dim_J_block + j
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH * MATRIX_WIDTH + j * MATRIX_WIDTH
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_STORE, 
#                                        OUTPUT_BUFFER_ID, 
#                                        buffer_offset, 
#                                        dram_offset, 
#                                        MATRIX_WIDTH, 
#                                        MATRIX_WIDTH, 
#                                        dim_J)
#             insn_idx += 1
            
#     # 生成结束指令
#     insn_buf[insn_idx] = getFinishInsn()
#     insn_idx += 1

#     print("insn_size",insn_size)
#     print("insn_idx",insn_idx)
#     print("insn",insn_idx)
#     for i in range(insn_idx):
#         if i>=insn_load_size and i<insn_load_size+compute_count:
#             print_binary(insn_buf[i])
            
#     # 运行SAA硬件
#     pt0 = time.perf_counter()
#     saa_driver.run_saa(insn_idx,
#            insn_buf.physical_address,
#            input.physical_address,
#            weight.physical_address,
#            output.physical_address,
#            wait_cycles)
#     pt1 = time.perf_counter()
#     time_sw = pt1 - pt0
#     print("saa run time: %fs" % time_sw)     
    
#     # 计算吞吐量
    
#     # 清空指令buf
#     del insn_buf
    
#     return 0       
            
    
    
    

    
# def blocked_gemm_test(saa_driver,
#               dim_I, 
#               dim_J, 
#               dim_K, 
#               input, 
#               weight,
#               bias, 
#               output, 
#               block, 
#               bias_use):
    
#     print("=====================================================================================")
#     print(f"INFO - Blocked GEMM test: dim_I={dim_I}, dim_J={dim_J}, dim_K={dim_K}, block={block}, bias_use={bias_use}")
    
#     # 计算分块
#     dim_I_block = dim_I // MATRIX_WIDTH
#     dim_J_block = dim_J // MATRIX_WIDTH
#     dim_K_block = dim_K // MATRIX_WIDTH

#     # 计算指令数量
#     insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block)
#     insn_compute_size = 2 * dim_I_block * dim_K_block * dim_J_block  # 不使用权重复用
# #     insn_compute_size = (dim_I_block + 1) * dim_K_block * dim_J_block  # 使用权重复用
# #     insn_compute_size = dim_I_block * dim_K_block * dim_J_block + 1  # 使用权重复用和双缓冲
#     insn_store_size = dim_I_block * dim_J_block
#     insn_size = insn_load_size + insn_store_size + insn_compute_size + 1

#     # 初始化指令队列
#     insn_buf = allocate(shape = (insn_size), cacheable = 0, dtype = Instruct_DataType)
#     insn_idx = 0
    
#     # 生成加载Input指令
#     for i in range(dim_I_block):
#         for k in range(dim_K_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i*dim_K_block+k
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + i * dim_K_block * MATRIX_WIDTH * MATRIX_WIDTH + k * MATRIX_WIDTH
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       INPUT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       MATRIX_WIDTH, 
#                                       MATRIX_WIDTH, 
#                                       dim_K)
#             insn_idx += 1

#     # 生成加载weight指令
#     for k in range(dim_K_block):
#         for j in range(dim_J_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = k * dim_J_block + j
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + k * dim_J_block * MATRIX_WIDTH * MATRIX_WIDTH + j * MATRIX_WIDTH 
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_LOAD, 
#                                       WEIGHT_BUFFER_ID, 
#                                       buffer_offset, 
#                                       dram_offset, 
#                                       MATRIX_WIDTH, 
#                                       MATRIX_WIDTH, 
#                                       dim_J)
#             insn_idx += 1
    
#     # 生成计算指令
#     # 用于切换权重寄存器，最先使用 weight1
#     pingpang = 0
#     wb_start_addr = 0
#     input_start_addr = 0
#     output_start_addr = 0
#     weight_offset = 0
#     output_offset = 0
#     input_offset = 0
#     accumulate = 0
    
#     # 初始化指令计数
#     compute_count = insn_idx

#     # 迭代公共维度块和输出列块
#     for k in range(dim_K_block):
#         for j in range(dim_J_block):
#             # 计算权重偏移
#             weight_offset = wb_start_addr + (k * dim_J_block + j) * MATRIX_WIDTH
#             accumulate = 0 if k == 0 else 1

#             insn_buf[insn_idx] = getWeightPreloadInsn(weight_offset, pingpang)
#             insn_idx += 1
#             print("getWeightPreloadInsn")
            
#             # 迭代输出行块
#             for i in range(dim_I_block):
#                 output_offset = output_start_addr + (i * dim_J_block + j) * MATRIX_WIDTH
#                 input_offset = input_start_addr + (i * dim_K_block + k) * MATRIX_WIDTH
#                 insn_buf[insn_idx] = getComputeInsn(input_offset, 
#                                         output_offset, 
#                                         pingpang, 
#                                         accumulate)
#                 insn_idx += 1
#                 print("getComputeInsn")
            
#     # 更新计算指令的数量
#     compute_count = insn_idx - compute_count
#     print(f"compute_count: {compute_count}")    

    
#     # 生成存储指令
#     for i in range(dim_I_block):
#         for j in range(dim_J_block):
#             buffer_start = 0
#             dram_start = 0
#             A_block = i * dim_J_block + j
#             buffer_offset = buffer_start + A_block * MATRIX_WIDTH
#             dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH * MATRIX_WIDTH + j * MATRIX_WIDTH
#             insn_buf[insn_idx] = get2DLoadStoreInsn(OPCODE_STORE, 
#                                        OUTPUT_BUFFER_ID, 
#                                        buffer_offset, 
#                                        dram_offset, 
#                                        MATRIX_WIDTH, 
#                                        MATRIX_WIDTH, 
#                                        dim_J)
#             insn_idx += 1
            
#     # 生成结束指令
#     insn_buf[insn_idx] = getFinishInsn()
#     insn_idx += 1

#     print("insn_size",insn_size)
#     print("insn_idx",insn_idx)
#     print("insn",insn_idx)
#     for i in range(insn_idx):
#         if i>=insn_load_size and i<insn_load_size+compute_count:
#             print_binary(insn_buf[i])
            
#     # 运行SAA硬件
#     pt0 = time.perf_counter()
#     saa_driver.run_saa(insn_idx,
#            insn_buf.physical_address,
#            input.physical_address,
#            weight.physical_address,
#            output.physical_address,
#            wait_cycles)
#     pt1 = time.perf_counter()
#     time_sw = pt1 - pt0
#     print("saa run time: %fs" % time_sw)     
    
#     # 计算吞吐量
    
#     # 清空指令buf
#     del insn_buf
    
#     return 0       
            
    
    
    