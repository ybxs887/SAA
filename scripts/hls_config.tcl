proc const {name value} {
    uplevel 1 [list set $name $value]
    uplevel 1 [list trace var $name w {error "constant $name read-only"}]
}

# 包含HLS路径
const CFLAGS "-I/home/fpga_pc/my_accelerator_generate/SAA/src"

# 使用的芯片型号device(vivado/hls)
const FPGA_DEVICE xck26-sfvc784-2LV-c

# 属于哪一组(vivado)
const FPGA_FAMILY zynq-7000

# 板子型号(vivado)
const FPGA_BOARD xilinx.com:kv260_som

# 板子型号修订(vivado)
const FPGA_BOARD_REV part0:1.2

# 生成IP的最高时钟周期
const FPGA_PERIOD 5

# 设置pll的时钟频率,也就是总的模块运行频率
const FPGA_FREQ 100

# 每个IP的范围
const IP_REG_MAP_RANGE 0x1000

# fetch模块的基地址
const FETCH_BASE_ADDR 0xA0000000

# load模块的基地址
const LOAD_BASE_ADDR 0xA0001000

# compute模块的基地址
const COMPUTE_BASE_ADDR 0xA0002000

# store模块的基地址
const STORE_BASE_ADDR 0xA0003000

# 脉动阵列大小
const MATRIX_WIDTH 4

# 将缓冲区数组第二维度变成一个元素，使得能够按行寻址
const MEM_BLOCK_FRACTOR 1

# 数据总线宽度
const TRANSFER_WIDTH 128

# 指令总线宽度
const INSTRUCT_WIDTH 128