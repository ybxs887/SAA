

import time
from pynq import Overlay

class SAADevice:
    def __init__(self, overlay):
        self.overlay = overlay
        self.saa_fetch_handle_ = self._map_register(SAA_FETCH_ADDR)
        self.saa_load_handle_ = self._map_register(SAA_LOAD_ADDR)
        self.saa_compute_handle_ = self._map_register(SAA_COMPUTE_ADDR)
        self.saa_store_handle_ = self._map_register(SAA_STORE_ADDR)

    def __del__(self):
        self._unmap_register(self.saa_fetch_handle_)
        self._unmap_register(self.saa_load_handle_)
        self._unmap_register(self.saa_compute_handle_)
        self._unmap_register(self.saa_store_handle_)

    def _map_register(self, address):
        # 假设overlay对象有一个方法来映射寄存器
        return self.overlay.map_register(address)

    def _unmap_register(self, handle):
        # 假设overlay对象有一个方法来取消映射寄存器
        self.overlay.unmap_register(handle)

    def write_ip_register(self, ip, offset, value):
        # 调用写入寄存器的函数
        write_ip_register(ip, offset, value)

    def read_ip_register(self, ip, offset):
        # 调用读取寄存器的函数
        return read_ip_register(ip, offset)

    def run(self, insn_count, insn_phy_addr, input_phy_addr, weight_phy_addr, output_phy_addr, wait_cycles):
        # 配置各IP的寄存器
        self.write_ip_register(self.saa_fetch_handle_, RegisterOffset.FETCH_INSN_COUNT_OFFSET, insn_count)
        self.write_ip_register(self.saa_fetch_handle_, RegisterOffset.FETCH_INSN_ADDR_OFFSET, insn_phy_addr)
        self.write_ip_register(self.saa_load_handle_, RegisterOffset.LOAD_INP_ADDR_OFFSET, input_phy_addr)
        self.write_ip_register(self.saa_load_handle_, RegisterOffset.LOAD_WGT_ADDR_OFFSET, weight_phy_addr)
        self.write_ip_register(self.saa_store_handle_, RegisterOffset.STORE_OUT_ADDR_OFFSET, output_phy_addr)

        # 启动SAA
        self.write_ip_register(self.saa_fetch_handle_, 0x0, 0x1)
        self.write_ip_register(self.saa_load_handle_, 0x0, 0x81)
        self.write_ip_register(self.saa_compute_handle_, 0x0, 0x81)
        self.write_ip_register(self.saa_store_handle_, 0x0, 0x81)

        # 等待SAA完成
        for t in range(wait_cycles):
            done_flag = self.read_ip_register(self.saa_compute_handle_, RegisterOffset.COMPUTE_DONE_OFFSET)
            if done_flag == 0x1:
                break
            else:
                time.sleep(0.000001)

        # 根据是否超时返回结果
        return 0 if t < wait_cycles else 1

# 假设RegisterOffset类和write_ip_register, read_ip_register函数已经定义
# 假设overlay对象已经被正确初始化
# 创建SAADevice实例
saa_device = SAADevice(overlay)

# 使用SAADevice实例运行SAA
result = saa_device.run(insn_count, insn_phy_addr, input_phy_addr, weight_phy_addr, output_phy_addr, wait_cycles)
print("SAA operation result:", "PASS" if result == 0 else "FAIL")

