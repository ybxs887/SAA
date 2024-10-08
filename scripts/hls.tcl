# 命令行参数：
# 参数 1：SAA 根目录的路径
# 参数 2：配置参数脚本的路径

# 检查是否提供了足够的命令行参数
if { [llength $argv] eq 4 } {
    set root_dir        [lindex $argv 2]
    set saa_config      [lindex $argv 3] 
} else {
    puts "提供的参数不足！"
    exit
}

# 推导路径
set src_dir "$root_dir/src"  
set sim_dir "$root_dir/sim"
# set test_dir "$root_dir/tests/hardware/common"  # 测试目录

# 源（读取）SAA 配置变量
source $saa_config

# 获取 C 编译器标志
set cflags $CFLAGS

# 获取 SAA 配置参数
set ::device $FPGA_DEVICE 
set ::period $FPGA_PERIOD  

# 获取 SAA SRAM 重塑/分区因子，以确保所有内存具有相同的 AXI 宽度,
set ::inp_reshape_factor    $MEM_BLOCK_FRACTOR  
set ::wgt_reshape_factor    $MEM_BLOCK_FRACTOR  
set ::out_reshape_factor    $MEM_BLOCK_FRACTOR  


# 初始化 HLS 设计并设置内存分区的 HLS 指令
# 这是必要的，因为 Vivado 的限制不允许总线宽度超过 1024 位
# 可以复用多个工程
proc init_design {} {
    # 设置设备 ID
    set_part $::device

    # 设置时钟频率
    create_clock -period $::period -name default

    # # HLS 指令来重塑/分区输入内存的读写端口
    # set_directive_array_reshape -type complete -dim 2 "load" input_buffer
    # set_directive_array_reshape -type complete -dim 2 "compute" input_buffer

    # # HLS 指令来重塑/分区权重内存的读写端口
    # set_directive_array_reshape -type block -factor $::wgt_reshape_factor -dim 2 "load" weight_buffer
    # set_directive_array_reshape -type block -factor $::wgt_reshape_factor -dim 2 "compute" weight_buffer

    # # HLS 指令来重塑/分区输出内存的读写端口
    # set_directive_array_reshape -type block -factor $::out_reshape_factor -dim 2 "compute" output_buffer
    # set_directive_array_reshape -type block -factor $::out_reshape_factor -dim 2 "store" output_buffer
}


# HLS 行为级仿真
# open_project saa_sim
# set_top saa_top
# add_files $src_dir/SAA.cpp -cflags $cflags
# add_files -tb $sim_dir/saa_test.cpp -cflags $cflags
# open_solution "soln"
# init_design
# csim_design -clean
# close_project

# 生成 fetch 阶段
open_project saa_fetch
set_top fetch
add_files $src_dir/SAA.cpp -cflags $cflags
open_solution "soln"
init_design
csynth_design
export_design -format ip_catalog
close_project


# 生成 load 阶段（重复上述步骤）
open_project saa_load
set_top load
add_files $src_dir/SAA.cpp -cflags $cflags
open_solution "soln"
init_design
csynth_design
export_design -format ip_catalog
close_project

# 生成 compute 阶段（重复上述步骤）
open_project saa_compute
set_top compute
add_files $src_dir/SAA.cpp -cflags $cflags
open_solution "soln"
init_design
csynth_design
export_design -format ip_catalog
close_project

# 生成 store 阶段（重复上述步骤）
open_project saa_store
set_top store
add_files $src_dir/SAA.cpp -cflags $cflags
open_solution "soln"
init_design
csynth_design
export_design -format ip_catalog
close_project

exit 
