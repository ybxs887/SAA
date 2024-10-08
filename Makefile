


# 目录定义
ROOTDIR = $(CURDIR)# 当前目录作为根目录
BUILD_DIR = $(CURDIR)/build/xilinx# 构建目录，用于存放编译和构建结果
SCRIPT_DIR = $(CURDIR)/scripts# 脚本目录，存放构建脚本
SRC_DIR = $(CURDIR)/src# 源代码目录，存放源文件

#定义生成的IP和bit流名
CONF = saa# 使用配置名称，这里暂时使用字符串

#路径定义
IP_BUILD_PATH := $(BUILD_DIR)/hls/$(CONF)  # IP 生成路径
HW_BUILD_PATH := $(BUILD_DIR)/vivado/$(CONF)  # 硬件构建路径

# 配置文件
CONFIG_TCL = $(SCRIPT_DIR)/hls_config.tcl  # SAA hls 配置的 Tcl 文件

# 可执行命令
VITIS_HLS_PATH = D:\Xilinx2021.2\Vitis_HLS\2021.2\bin\vitis_hls_cmd.bat
VITIS_HLS = vitis_hls  # vitis HLS 工具，用于高层次综合
VIVADO = vivado  # Vivado 工具，用于 FPGA 设计的实现和生成

# IP 文件路径
IP_PATH := $(BUILD_DIR)/hls/$(CONF)/saa_compute/soln/impl/ip/xilinx_com_hls_compute_1_0.zip  # IP 文件的完整路径,作为生成目标

# 比特流文件路径
BIT_PATH := $(BUILD_DIR)/vivado/$(CONF)/export/$(CONF).bit  # 比特流文件的完整路径

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 创建IP构建路径
$(IP_BUILD_PATH): $(BUILD_DIR)
	mkdir -p $@

# 创建硬件构建路径
$(HW_BUILD_PATH): $(BUILD_DIR)
	mkdir -p $@

# 确保在构建IP文件之前创建IP目录
$(IP_PATH): $(IP_BUILD_PATH)

# 定义构建伪目标，all代表生成全部，必须生成bit流，ip代表生成ip,bit代表生成比特流，必须依赖于IP的文件路径才能生成
.PHONY: all ip bit clean clean_all

# 构建所有目标
all: bit  # 依赖于比特流文件
# 生成 IP 目标
ip: $(IP_PATH)  # 依赖于 IP 文件路径,如果路径不存在会自动创建路径
# 生成比特流目标
bit: $(BIT_PATH)  # 依赖于比特流文件路径,如果路径不存在会自动创建路径

# 生成 IP 文件
$(IP_PATH): $(SRC_DIR)/* $(CONFIG_TCL)# 依赖于源代码文件（使用SRC_DIR下所有文件）和配置的 Tcl 文件
	mkdir -p $(IP_BUILD_PATH)
	cd $(IP_BUILD_PATH) && \
		$(VITIS_HLS) \
		-f $(SCRIPT_DIR)/hls.tcl \
		-tclargs \
			$(ROOTDIR) \
			$(CONFIG_TCL)
			
# # 生成比特流文件
# $(BIT_PATH): $(IP_PATH)  # 依赖于 IP 文件路径
# 	mkdir -p $(HW_BUILD_PATH)  # 创建硬件构建路径
# 	cd $(HW_BUILD_PATH) && \
# 		$(VIVADO) \
# 		-mode tcl \
# 		-source $(SCRIPT_DIR)/vivado.tcl \
# 		-tclargs \
# 			$(BUILD_DIR)/hls/$(CONF) \
# 			$(CONFIG_TCL)

# 清理当前目录下的输出文件
clean:
	rm -rf *.out *.log

# 清理所有构建目录和文件
cleanall: clean  # 先执行清理操作
	rm -rf $(BUILD_DIR)  # 删除构建目录


print-config-tcl:
	echo $(IP_BUILD_PATH)

	