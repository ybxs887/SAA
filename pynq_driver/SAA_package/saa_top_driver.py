import time
from pynq import Overlay

class SaaDriver:
    def __init__(self, bitfile_name="saa_top.bit"):
        """
        初始化 SAA 驱动类实例，并加载 Overlay。
        
        参数:
        bitfile_name -- Overlay 位文件的名称
        """
        self.overlay = Overlay(bitfile_name)
        print("saa_top Overlay downloaded successfully!")
        
        # 从 overlay 获取 SAA IP 实例
        self.saa_top_ip = self.overlay.saa_top_0

    def write_ip_register(self, offset, value):
        """
        向指定 IP 核的寄存器写入值。
        
        参数:
        offset -- 寄存器的偏移地址
        value -- 要写入的值
        """
        self.saa_top_ip.write(offset, value)

    def read_ip_register(self, offset):
        """
        从指定 IP 核的寄存器读取值。
        
        参数:
        offset -- 寄存器的偏移地址
        
        返回值:
        寄存器中的值
        """
        return self.saa_top_ip.read(offset)

    def run_saa(self, 
            insn_count, 
            insn_phy_addr, 
            uop_phy_addr, 
            input_phy_addr, 
            weight_phy_addr, 
            bias_phy_addr, 
            output_phy_addr, 
            wait_cycles):
        """
        向 SAA 提交指令并等待一次大批量指令执行完成。
        
        参数:
        insn_count -- 这一次批量执行的指令数量
        insn_phy_addr -- 这一次执行的指令的缓冲区首地址
        uop_phy_addr -- 这一次执行的微操作缓冲区首地址
        input_phy_addr -- 这一次执行的指令的输入缓冲区首地址
        weight_phy_addr -- 这一次执行的指令的权重缓冲区首地址
        output_phy_addr -- 这一次执行的指令的输出缓冲区首地址
        wait_cycles -- 最大等待的时间周期
        
        返回值:
        如果执行成功且没有超时，返回 0；如果超时，返回 1。
        """
        # 配置指令数量寄存器
        self.write_ip_register(0x10, insn_count)
        
        # 配置物理地址寄存器
        self.write_ip_register(0x18, insn_phy_addr)
        self.write_ip_register(0x24, uop_phy_addr)
        self.write_ip_register(0x30, input_phy_addr)
        self.write_ip_register(0x3c, weight_phy_addr)
        self.write_ip_register(0x48, bias_phy_addr)
        self.write_ip_register(0x54, output_phy_addr)
        
        # 启动 IP 进行计算
        self.write_ip_register(0x0, 0x81)
        
        # 延时 1 微秒
        time.sleep(0.0000001)
        
        # 读取完成信号
        for t in range(0, wait_cycles):
            done_flag = self.read_ip_register(0x60)
            if done_flag == 0x1:
                print("done:", t)
                return 0
            time.sleep(0.000001)
        
        # 超时返回 1
        return 1










# class SaaDriver:
#     def __init__(self, bitfile_name="saa_top.bit"):
#         """
#         初始化 SAA 驱动类实例，并加载 Overlay。
        
#         参数:
#         bitfile_name -- Overlay 位文件的名称
#         """
#         self.overlay = Overlay(bitfile_name)
#         print("saa_top Overlay downloaded successfully!")
        
#         # 从 overlay 获取 SAA IP 实例
#         self.saa_top_ip = self.overlay.saa_top_0

#     def write_ip_register(self, offset, value):
#         """
#         向指定 IP 核的寄存器写入值。
        
#         参数:
#         offset -- 寄存器的偏移地址
#         value -- 要写入的值
#         """
#         self.saa_top_ip.write(offset, value)

#     def read_ip_register(self, offset):
#         """
#         从指定 IP 核的寄存器读取值。
        
#         参数:
#         offset -- 寄存器的偏移地址
        
#         返回值:
#         寄存器中的值
#         """
#         return self.saa_top_ip.read(offset)

#     def run_saa(self, insn_count, insn_phy_addr, input_phy_addr, weight_phy_addr, bias_phy_addr, output_phy_addr, wait_cycles):
#         """
#         向 SAA 提交指令并等待一次大批量指令执行完成。
        
#         参数:
#         insn_count -- 这一次批量执行的指令数量
#         insn_phy_addr -- 这一次执行的指令的缓冲区首地址
#         input_phy_addr -- 这一次执行的指令的输入缓冲区首地址
#         weight_phy_addr -- 这一次执行的指令的权重缓冲区首地址
#         output_phy_addr -- 这一次执行的指令的输出缓冲区首地址
#         wait_cycles -- 最大等待的时间周期
        
#         返回值:
#         如果执行成功且没有超时，返回 0；如果超时，返回 1。
#         """
#         # 配置指令数量寄存器
#         self.write_ip_register(0x10, insn_count)
        
#         # 配置物理地址寄存器
#         self.write_ip_register(0x18, insn_phy_addr)
#         self.write_ip_register(0x24, input_phy_addr)
#         self.write_ip_register(0x30, weight_phy_addr)
#         self.write_ip_register(0x3c, bias_phy_addr)
#         self.write_ip_register(0x48, output_phy_addr)
        
#         # 启动 IP 进行计算
#         self.write_ip_register(0x0, 0x81)
        
#         # 延时 1 微秒
#         time.sleep(0.0000001)
        
#         # 读取完成信号
#         for t in range(0, wait_cycles):
#             done_flag = self.read_ip_register(0x54)
#             if done_flag == 0x1:
#                 print("done:", t)
#                 return 0
#             time.sleep(0.000001)
        
#         # 超时返回 1
#         return 1

# class SaaDriver:
#     def __init__(self, bitfile_name="saa_top.bit"):
#         """
#         初始化 SAA 驱动类实例，并加载 Overlay。
        
#         参数:
#         bitfile_name -- Overlay 位文件的名称
#         """
#         self.overlay = Overlay(bitfile_name)
#         print("saa_top Overlay downloaded successfully!")
        
#         # 从 overlay 获取 SAA IP 实例
#         self.saa_top_ip = self.overlay.saa_top_0

#     def write_ip_register(self, offset, value):
#         """
#         向指定 IP 核的寄存器写入值。
        
#         参数:
#         offset -- 寄存器的偏移地址
#         value -- 要写入的值
#         """
#         self.saa_top_ip.write(offset, value)

#     def read_ip_register(self, offset):
#         """
#         从指定 IP 核的寄存器读取值。
        
#         参数:
#         offset -- 寄存器的偏移地址
        
#         返回值:
#         寄存器中的值
#         """
#         return self.saa_top_ip.read(offset)

#     def run_saa(self, insn_count, insn_phy_addr, input_phy_addr, weight_phy_addr, output_phy_addr, wait_cycles):
#         """
#         向 SAA 提交指令并等待一次大批量指令执行完成。
        
#         参数:
#         insn_count -- 这一次批量执行的指令数量
#         insn_phy_addr -- 这一次执行的指令的缓冲区首地址
#         input_phy_addr -- 这一次执行的指令的输入缓冲区首地址
#         weight_phy_addr -- 这一次执行的指令的权重缓冲区首地址
#         output_phy_addr -- 这一次执行的指令的输出缓冲区首地址
#         wait_cycles -- 最大等待的时间周期
        
#         返回值:
#         如果执行成功且没有超时，返回 0；如果超时，返回 1。
#         """
#         # 配置指令数量寄存器
#         self.write_ip_register(0x10, insn_count)
        
#         # 配置物理地址寄存器
#         self.write_ip_register(0x18, insn_phy_addr)
#         self.write_ip_register(0x24, input_phy_addr)
#         self.write_ip_register(0x30, weight_phy_addr)
#         self.write_ip_register(0x3c, output_phy_addr)
        
#         # 启动 IP 进行计算
#         self.write_ip_register(0x0, 0x1)
        
#         # 延时 1 微秒
#         time.sleep(0.0000001)
        
#         # 读取完成信号
#         for t in range(0, wait_cycles):
#             done_flag = self.read_ip_register(0x48)
#             if done_flag == 0x1:
#                 print("done:", t)
#                 return 0
#             time.sleep(0.000001)
        
#         # 超时返回 1
#         return 1

    
# import time
# import random
# from pynq import Overlay
# import numpy as np
# from pynq import allocate
        
# # 加载Overlay
# overlay = Overlay("saa_top.bit")
# print("saa_top Overlay downloaded successfully!")


# # 定义写入IP寄存器的函数，可以对IP的对应位置进行写入
# def write_ip_register(ip, offset, value):
#     """
#     向指定IP核的寄存器写入值。
    
#     参数:
#     ip -- IP核实例
#     offset -- 寄存器的偏移地址
#     value -- 要写入的值
#     """
#     # 假设IP核实例有一个名为'write'的方法来写入寄存器
#     ip.write(offset, value)

# def read_ip_register(ip, offset):
#     """
#     从指定IP核的寄存器读取值。
    
#     参数:
#     ip -- IP核实例
#     offset -- 寄存器的偏移地址
    
#     返回值:
#     寄存器中的值
#     """
#     # 通过寄存器偏移地址直接访问字典属性
#     return ip.read(offset)

# # 从overlay获取IP实例,也就是handle
# saa_top_ip = overlay.saa_top_0

# # 使用写入寄存器函数，对四个IP进行配置
# # 配置和VTA不同，我们的三个缓冲区的物理起始地址是有值的，
# # 这是因为我使用memcpy时，指令中的dram_base代表的是dram的索引而不是首地址
# # 因此传入指令时要传入索引，索引按照dram存储数据大小寻址
# # 因此真正的数组首地址就是这里定义的物理地址
# def RunSaa(insn_count,
#            insn_phy_addr,
#            input_phy_addr,
#            weight_phy_addr,
#            output_phy_addr,
#            wait_cycles):
#     """
#     向saa提交指令等待一次大批量指令执行完成,注意要有done信号表示计算完成以退出RunSaa(暂时没有)
    
#     参数:
#     insn_count -- 这一次批量执行的指令数量
#     insn_phy_addr -- 这一次执行的指令的缓冲区首地址
#     input_phy_addr -- 这一次执行的指令的输入缓冲区首地址
#     weight_phy_addr -- 这一次执行的指令的权重缓冲区首地址
#     output_phy_addr -- 这一次执行的指令的输出缓冲区首地址
#     wait_cycles -- 最大等待的时间周期,可以设置很大很大,查询done信号等待这一批指令执行完成
#     """
#     # 配置insn_count
#     write_ip_register(saa_top_ip,0x10,insn_count) # 配置指令数量寄存器
    
#     # 配置phy_addr
#     write_ip_register(saa_top_ip,0x18,insn_phy_addr) # 配置指令物理地址寄存器，也就是指令缓冲区物理首地址
#     write_ip_register(saa_top_ip,0x24,input_phy_addr) # 配置输入缓冲区物理地址
#     write_ip_register(saa_top_ip,0x30,weight_phy_addr) # 配置权重缓冲区物理地址
#     write_ip_register(saa_top_ip,0x3c,output_phy_addr) # 配置输出缓冲区物理地址

#     #写入IP控制寄存器，启动IP进行计算
#     write_ip_register(saa_top_ip,0x0,0x81) # 指令寄存器写入0x1启动本次模块

#     #延时1微秒使得设备响应
#     time.sleep(0.0000001) # 让出CPU，等待0.000001秒（1u秒）
    
#     # 读取compute的done信号是否完成
#     for t in range(0, wait_cycles):
#         done_flag = read_ip_register(saa_top_ip,0x48) # 从computeIP的done寄存器读取本次指令是否执行完毕
#         if done_flag == 0x1: # 如果done_flag被置为1，代表这次执行的是FINISH指令，本批次指令执行完毕
#             print("done：",t)
#             break
# #         time.sleep(0.000001) # 让出CPU，等待0.000001秒（1u秒）

#     # 根据是否超时返回，如果没超时返回0，超时返回1
#     return 0 if t < wait_cycles else 1


