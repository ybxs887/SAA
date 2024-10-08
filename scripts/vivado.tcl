# 检查脚本是否与正确的Vivado版本兼容
set scripts_vivado_version 2021.2  # 脚本生成时使用的Vivado版本
set current_vivado_version [version -short]  # 当前系统中的Vivado版本
if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   catch {common::send_msg_id "BD_TCL-109" "ERROR" "此脚本是使用Vivado <$scripts_vivado_version>生成的，\
    当前运行环境为Vivado <$current_vivado_version>。"}
   return 1  # 如果版本不匹配，则退出脚本
}

# 解析命令行参数，获取IP文件路径和VTA配置文件路径
if { [llength $argv] eq 2 } {
  set ip_path     [lindex $argv 0]  # IP文件路径
  set vta_config  [lindex $argv 1]  # VTA配置文件路径
} else {
  puts "参数列表不完整：<path to ip dir> <path to vta_config.py>"
  return 1  # 如果参数不完整，则退出脚本
}

# 导入VTA配置文件中的变量
source $vta_config

# 获取VTA配置参数
set target            $TARGET  # 目标平台
set device            $FPGA_DEVICE  # FPGA型号
set device_family     $FPGA_FAMILY  # FPGA系列
set clock_freq        $FPGA_FREQ  # 时钟频率
set board             $FPGA_BOARD  # 开发板型号
set board_rev         $FPGA_BOARD_REV  # 开发板版本

# SRAM尺寸参数
set inp_part          $INP_MEM_BANKS  # 输入内存分区数
set inp_mem_width     $INP_MEM_WIDTH  # 输入内存宽度
set inp_mem_depth     $INP_MEM_DEPTH  # 输入内存深度
set wgt_part          $WGT_MEM_BANKS  # 权重内存分区数
set wgt_mem_width     $WGT_MEM_WIDTH  # 权重内存宽度
set wgt_mem_depth     $WGT_MEM_DEPTH  # 权重内存深度
set out_part          $OUT_MEM_BANKS  # 输出内存分区数
set out_mem_width     $OUT_MEM_WIDTH  # 输出内存宽度
set out_mem_depth     $OUT_MEM_DEPTH  # 输出内存深度
set num_wgt_mem_uram  $NUM_WGT_MEM_URAM  # 使用URAM的权重内存数量

# AXI总线信号参数
set axi_cache         $AXI_CACHE_BITS  # AXI缓存位
set axi_prot          $AXI_PROT_BITS  # AXI保护位

# 地址映射参数
set ip_reg_map_range  $IP_REG_MAP_RANGE  # IP寄存器映射范围
set fetch_base_addr   $FETCH_BASE_ADDR  # fetch模块基地址
set load_base_addr    $LOAD_BASE_ADDR   # load模块基地址
set compute_base_addr $COMPUTE_BASE_ADDR  # compute模块基地址
set store_base_addr   $STORE_BASE_ADDR  # store模块基地址

# IP库路径
set proj_name vta  # 项目名称
set design_name $proj_name  # 设计名称
set proj_path "."  # 项目路径
set ip_lib "ip_lib"  # IP库目录
set fetch_ip "${ip_path}/vta_fetch/soln/impl/ip/xilinx_com_hls_fetch_1_0.zip"  # fetch模块IP文件路径
set load_ip "${ip_path}/vta_load/soln/impl/ip/xilinx_com_hls_load_1_0.zip"  # load模块IP文件路径
set compute_ip "${ip_path}/vta_compute/soln/impl/ip/xilinx_com_hls_compute_1_0.zip"  # compute模块IP文件路径
set store_ip "${ip_path}/vta_store/soln/impl/ip/xilinx_com_hls_store_1_0.zip"  # store模块IP文件路径

# 创建自定义Vivado项目
create_project -force $proj_name $proj_path -part $device  # 强制创建或覆盖项目

# 应用开发板预设（如果存在）
if {$board != "None" && $board_rev != "None"} {
  set_property board_part $board:$board_rev [current_project]  # 设置开发板型号和版本
}

# 更新IP库，添加生成的IP文件
file mkdir $ip_lib  # 创建IP库目录
set_property ip_repo_paths $ip_lib [current_project]  # 设置IP库路径
update_ip_catalog  # 更新IP目录
update_ip_catalog -add_ip $fetch_ip -repo_path $ip_lib  # 添加fetch模块IP
update_ip_catalog -add_ip $load_ip -repo_path $ip_lib  # 添加load模块IP
update_ip_catalog -add_ip $compute_ip -repo_path $ip_lib  # 添加compute模块IP
update_ip_catalog -add_ip $store_ip -repo_path $ip_lib  # 添加store模块IP

##################################################################
# 配置块图设计
##################################################################

# 创建bd设计
create_bd_design $design_name  # 创建bd设计
current_bd_design $design_name  # 切换到当前bd设计

# 初始化FIFO属性的函数
proc init_fifo_property {fifo width_bytes depth} {
  # 设置FIFO属性
  set_property -dict [ list \
    CONFIG.FIFO_Implementation_rach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wrch {Common_Clock_Distributed_RAM} \
    CONFIG.Full_Flags_Reset_Value {1} \
    CONFIG.INTERFACE_TYPE {AXI_STREAM} \
    CONFIG.Input_Depth_axis $depth \
    CONFIG.Reset_Type {Asynchronous_Reset} \
    CONFIG.TDATA_NUM_BYTES $width_bytes \
  ] $fifo
}

# 初始化BRAM属性的函数
proc init_bram_property {bram width depth} {
  # 设置BRAM属性
  set_property -dict [ list \
    CONFIG.use_bram_block {Stand_Alone} \
    CONFIG.Assume_Synchronous_Clk {true} \
    CONFIG.Byte_Size {8} \
    CONFIG.Enable_32bit_Address {true} \
    CONFIG.Enable_B {Use_ENB_Pin} \
    CONFIG.Memory_Type {True_Dual_Port_RAM} \
    CONFIG.Read_Width_A $width \
    CONFIG.Read_Width_B $width \
    CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
    CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
    CONFIG.Use_Byte_Write_Enable {true} \
    CONFIG.Use_RSTA_Pin {true} \
    CONFIG.Use_RSTB_Pin {true} \
    CONFIG.Write_Depth_A $depth \
    CONFIG.Write_Width_A $width \
    CONFIG.Write_Width_B $width \
  ] $bram
}

# 创建proc_sys_reset实例，并设置属性
set proc_sys_reset [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset ]

# 创建pll_clk实例，并设置属性
set pll_clk [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 pll_clk ]
set_property -dict [ list \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $clock_freq \
  CONFIG.RESET_PORT {resetn} \
  CONFIG.RESET_TYPE {ACTIVE_LOW} \
  CONFIG.USE_LOCKED {false} \
] $pll_clk

# 创建axi_smc0实例，并设置属性
set axi_smc0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc0 ]
set_property -dict [ list \
  CONFIG.NUM_MI {1} \
  CONFIG.NUM_SI {5} \
] $axi_smc0

# 创建axi_xbar实例，并设置属性
set axi_xbar [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_xbar ]
set_property -dict [ list \
  CONFIG.NUM_MI {4} \
  CONFIG.NUM_SI {1} \
] $axi_xbar

# 创建fetch_0实例，并设置属性
set fetch_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:fetch:1.0 fetch_0 ]
set_property -dict [ list \
  CONFIG.C_M_AXI_INS_PORT_CACHE_VALUE $axi_cache \
  CONFIG.C_M_AXI_INS_PORT_PROT_VALUE $axi_prot \
] $fetch_0

# 创建load_0实例，并设置属性
set load_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:load:1.0 load_0 ]
set_property -dict [ list \
  CONFIG.C_M_AXI_DATA_PORT_CACHE_VALUE $axi_cache \
  CONFIG.C_M_AXI_DATA_PORT_PRO
] $load_0

# Create instance: compute_0, and set properties
set compute_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:compute:1.0 compute_0 ]
set_property -dict [ list \
  CONFIG.C_M_AXI_DATA_PORT_CACHE_VALUE $axi_cache \
  CONFIG.C_M_AXI_DATA_PORT_PROT_VALUE $axi_prot \
  CONFIG.C_M_AXI_UOP_PORT_CACHE_VALUE $axi_cache \
  CONFIG.C_M_AXI_UOP_PORT_PROT_VALUE $axi_prot \
] $compute_0

# Create instance: store_0, and set properties
set store_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:store:1.0 store_0 ]
set_property -dict [ list \
  CONFIG.C_M_AXI_DATA_PORT_CACHE_VALUE $axi_cache \
  CONFIG.C_M_AXI_DATA_PORT_PROT_VALUE $axi_prot \
] $store_0

# Create command queues and set properties
set cmd_queue_list {load_queue gemm_queue store_queue}
foreach cmd_queue $cmd_queue_list {
  set tmp_cmd_queue [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 $cmd_queue ]
  # Width is 16B (128b, as set in hw_spec.h), depth is 512 (depth of FIFO on Zynq 7000 and Zynq Ultrascale+)
  # TODO: derive it from vta_config.h
  [ init_fifo_property $tmp_cmd_queue 16 512 ]
}

# Create dependence queues and set properties
set dep_queue_list {l2g_queue g2l_queue g2s_queue s2g_queue}
foreach dep_queue $dep_queue_list {
  set tmp_dep_queue [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 $dep_queue ]
  # Width is 1B (min width), depth is 1024
  # TODO: derive it from vta_config.h
  [ init_fifo_property $tmp_dep_queue 1 1024 ]
}

# Create and connect inp_mem partitions
for {set i 0} {$i < $inp_part} {incr i} {
  # Create instance: inp_mem, and set properties
  set inp_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 inp_mem_${i} ]
  [ init_bram_property $inp_mem $inp_mem_width $inp_mem_depth ]
  # If module has more than 1 mem port, the naming convention changes
  if {$inp_part > 1} {
    set porta [get_bd_intf_pins load_0/inp_mem_${i}_V_PORTA]
    set portb [get_bd_intf_pins compute_0/inp_mem_${i}_V_PORTA]
  } else {
    set porta [get_bd_intf_pins load_0/inp_mem_V_PORTA]
    set portb [get_bd_intf_pins compute_0/inp_mem_V_PORTA]
  }
  # Create interface connections
  connect_bd_intf_net -intf_net load_0_inp_mem_V_PORTA \
    [get_bd_intf_pins $inp_mem/BRAM_PORTA] \
    $porta
  connect_bd_intf_net -intf_net compute_0_inp_mem_V_PORTA \
    [get_bd_intf_pins $inp_mem/BRAM_PORTB] \
    $portb
}

# Create and connect wgt_mem partitions
for {set i 0} {$i < $wgt_part} {incr i} {
  # Create instance: wgt_mem, and set properties
  set wgt_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 wgt_mem_${i} ]
  [ init_bram_property $wgt_mem $wgt_mem_width $wgt_mem_depth ]
  # If module has more than 1 mem port, the naming convention changes
  if {$wgt_part > 1} {
    set porta [get_bd_intf_pins load_0/wgt_mem_${i}_V_PORTA]
    set portb [get_bd_intf_pins compute_0/wgt_mem_${i}_V_PORTA]
  } else {
    set porta [get_bd_intf_pins load_0/wgt_mem_V_PORTA]
    set portb [get_bd_intf_pins compute_0/wgt_mem_V_PORTA]
  }
  # Create interface connections
  connect_bd_intf_net -intf_net load_0_wgt_mem_${i}_V_PORTA \
    [get_bd_intf_pins $wgt_mem/BRAM_PORTA] \
    $porta
  connect_bd_intf_net -intf_net compute_0_wgt_mem_${i}_V_PORTA \
    [get_bd_intf_pins $wgt_mem/BRAM_PORTB] \
    $portb
  if { $device_family eq "zynq-ultrascale+" && $i < $num_wgt_mem_uram } {
    set_property -dict [list CONFIG.PRIM_type_to_Implement {URAM}] $wgt_mem
  }
}

# Create and connect out_mem partitions
for {set i 0} {$i < $out_part} {incr i} {
  # Create instance: out_mem, and set properties
  set out_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 out_mem_${i} ]
  [ init_bram_property $out_mem $out_mem_width $out_mem_depth ]
  # If module has more than 1 mem port, the naming convention changes
  if {$out_part > 1} {
    set porta [get_bd_intf_pins compute_0/out_mem_${i}_V_PORTA]
    set portb [get_bd_intf_pins store_0/out_mem_${i}_V_PORTA]
  } else {
    set porta [get_bd_intf_pins compute_0/out_mem_V_PORTA]
    set portb [get_bd_intf_pins store_0/out_mem_V_PORTA]
  }
  # Create interface connections
  connect_bd_intf_net -intf_net compute_0_out_mem_${i}_V_PORTA \
    [get_bd_intf_pins $out_mem/BRAM_PORTA] \
    $porta
  connect_bd_intf_net -intf_net store_0_out_mem_${i}_V_PORTA \
    [get_bd_intf_pins $out_mem/BRAM_PORTB] \
    $portb
}

# Create instance: processing_system, and set properties
if { $device_family eq "zynq-7000" } {
  set processing_system [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system ]
  set_property -dict [ list \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_DEFAULT_ACP_USER_VAL {1} \
    CONFIG.PCW_USE_S_AXI_ACP {1} \
    CONFIG.preset {ZC702} \
  ] $processing_system
  # Get ports that are specific to the Zynq 7000 processing system
  set ps_clk    [get_bd_pins processing_system/FCLK_CLK0]
  set ps_rstn   [get_bd_pins processing_system/FCLK_RESET0_N]
  set maxi_clk  [get_bd_pins processing_system/M_AXI_GP0_ACLK]
  set saxi_clk  [get_bd_pins processing_system/S_AXI_ACP_ACLK]
  set maxi      [get_bd_intf_pins processing_system/M_AXI_GP0]
  set saxi      [get_bd_intf_pins processing_system/S_AXI_ACP]
} elseif { $device_family eq "zynq-ultrascale+" } {
  set processing_system [ create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.3 processing_system ]
  apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells processing_system]
  set_property -dict [ list \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {1}
  ] $processing_system
  # Get ports that are specific to the Zynq Ultrascale MPSoC processing system
  set ps_clk    [get_bd_pins processing_system/pl_clk0]
  set ps_rstn   [get_bd_pins processing_system/pl_resetn0]
  set maxi_clk  [get_bd_pins processing_system/maxihpm0_fpd_aclk]
  set saxi_clk  [get_bd_pins processing_system/saxihpc0_fpd_aclk]
  set maxi      [get_bd_intf_pins processing_system/M_AXI_HPM0_FPD]
  set saxi      [get_bd_intf_pins processing_system/S_AXI_HPC0_FPD]
}

# Create interface connections
connect_bd_intf_net -intf_net axi_xbar_M00_AXI [get_bd_intf_pins axi_xbar/M00_AXI] [get_bd_intf_pins fetch_0/s_axi_CONTROL_BUS]
connect_bd_intf_net -intf_net axi_xbar_M01_AXI [get_bd_intf_pins axi_xbar/M01_AXI] [get_bd_intf_pins load_0/s_axi_CONTROL_BUS]
connect_bd_intf_net -intf_net axi_xbar_M02_AXI [get_bd_intf_pins axi_xbar/M02_AXI] [get_bd_intf_pins compute_0/s_axi_CONTROL_BUS]
connect_bd_intf_net -intf_net axi_xbar_M03_AXI [get_bd_intf_pins axi_xbar/M03_AXI] [get_bd_intf_pins store_0/s_axi_CONTROL_BUS]
connect_bd_intf_net -intf_net fetch_0_l2g_dep_queue_V [get_bd_intf_pins l2g_queue/S_AXIS] [get_bd_intf_pins load_0/l2g_dep_queue_V]
connect_bd_intf_net -intf_net fetch_0_load_queue_V_V [get_bd_intf_pins fetch_0/load_queue_V_V] [get_bd_intf_pins load_queue/S_AXIS]
connect_bd_intf_net -intf_net fetch_0_gemm_queue_V_V [get_bd_intf_pins fetch_0/gemm_queue_V_V] [get_bd_intf_pins gemm_queue/S_AXIS]
connect_bd_intf_net -intf_net fetch_0_store_queue_V_V [get_bd_intf_pins fetch_0/store_queue_V_V] [get_bd_intf_pins store_queue/S_AXIS]
connect_bd_intf_net -intf_net compute_0_g2l_dep_queue_V [get_bd_intf_pins compute_0/g2l_dep_queue_V] [get_bd_intf_pins g2l_queue/S_AXIS]
connect_bd_intf_net -intf_net compute_0_g2s_dep_queue_V [get_bd_intf_pins compute_0/g2s_dep_queue_V] [get_bd_intf_pins g2s_queue/S_AXIS]
connect_bd_intf_net -intf_net store_0_s2g_dep_queue_V [get_bd_intf_pins s2g_queue/S_AXIS] [get_bd_intf_pins store_0/s2g_dep_queue_V]
connect_bd_intf_net -intf_net load_queue_M_AXIS [get_bd_intf_pins load_0/load_queue_V_V] [get_bd_intf_pins load_queue/M_AXIS]
connect_bd_intf_net -intf_net gemm_queue_M_AXIS [get_bd_intf_pins compute_0/gemm_queue_V_V] [get_bd_intf_pins gemm_queue/M_AXIS]
connect_bd_intf_net -intf_net store_queue_M_AXIS [get_bd_intf_pins store_0/store_queue_V_V] [get_bd_intf_pins store_queue/M_AXIS]
connect_bd_intf_net -intf_net l2g_queue_M_AXIS [get_bd_intf_pins compute_0/l2g_dep_queue_V] [get_bd_intf_pins l2g_queue/M_AXIS]
connect_bd_intf_net -intf_net g2l_queue_M_AXIS [get_bd_intf_pins g2l_queue/M_AXIS] [get_bd_intf_pins load_0/g2l_dep_queue_V]
connect_bd_intf_net -intf_net g2s_queue_M_AXIS [get_bd_intf_pins g2s_queue/M_AXIS] [get_bd_intf_pins store_0/g2s_dep_queue_V]
connect_bd_intf_net -intf_net s2g_queue_M_AXIS [get_bd_intf_pins compute_0/s2g_dep_queue_V] [get_bd_intf_pins s2g_queue/M_AXIS]
connect_bd_intf_net -intf_net fetch_0_m_axi_ins_port [get_bd_intf_pins axi_smc0/S00_AXI] [get_bd_intf_pins fetch_0/m_axi_ins_port]
connect_bd_intf_net -intf_net load_0_m_axi_data_port [get_bd_intf_pins axi_smc0/S01_AXI] [get_bd_intf_pins load_0/m_axi_data_port]
connect_bd_intf_net -intf_net compute_0_m_axi_uop_port [get_bd_intf_pins axi_smc0/S02_AXI] [get_bd_intf_pins compute_0/m_axi_uop_port]
connect_bd_intf_net -intf_net compute_0_m_axi_data_port [get_bd_intf_pins axi_smc0/S03_AXI] [get_bd_intf_pins compute_0/m_axi_data_port]
connect_bd_intf_net -intf_net store_0_m_axi_data_port [get_bd_intf_pins axi_smc0/S04_AXI] [get_bd_intf_pins store_0/m_axi_data_port]
connect_bd_intf_net -intf_net axi_smc0_M00_AXI [get_bd_intf_pins axi_smc0/M00_AXI] $saxi
connect_bd_intf_net -intf_net processing_system_m_axi [get_bd_intf_pins axi_xbar/S00_AXI] $maxi

# Create port connections
connect_bd_net -net processing_system_reset \
  [get_bd_pins pll_clk/resetn] \
  [get_bd_pins proc_sys_reset/ext_reset_in] \
  $ps_rstn
connect_bd_net -net ps_clk_net \
  [get_bd_pins pll_clk/clk_in1] \
  $ps_clk
connect_bd_net -net proc_sys_reset_interconnect_aresetn \
  [get_bd_pins axi_xbar/ARESETN] \
  [get_bd_pins proc_sys_reset/interconnect_aresetn]
connect_bd_net -net proc_sys_reset_peripheral_aresetn \
  [get_bd_pins proc_sys_reset/peripheral_aresetn] \
  [get_bd_pins axi_smc0/aresetn] \
  [get_bd_pins axi_xbar/M00_ARESETN] \
  [get_bd_pins axi_xbar/M01_ARESETN] \
  [get_bd_pins axi_xbar/M02_ARESETN] \
  [get_bd_pins axi_xbar/M03_ARESETN] \
  [get_bd_pins axi_xbar/S00_ARESETN] \
  [get_bd_pins fetch_0/ap_rst_n] \
  [get_bd_pins load_0/ap_rst_n] \
  [get_bd_pins store_0/ap_rst_n] \
  [get_bd_pins compute_0/ap_rst_n] \
  [get_bd_pins load_queue/s_aresetn] \
  [get_bd_pins gemm_queue/s_aresetn] \
  [get_bd_pins store_queue/s_aresetn] \
  [get_bd_pins l2g_queue/s_aresetn] \
  [get_bd_pins g2l_queue/s_aresetn] \
  [get_bd_pins g2s_queue/s_aresetn] \
  [get_bd_pins s2g_queue/s_aresetn]
connect_bd_net -net processing_system_clk \
  [get_bd_pins pll_clk/clk_out1] \
  [get_bd_pins proc_sys_reset/slowest_sync_clk] \
  [get_bd_pins axi_smc0/aclk] \
  [get_bd_pins axi_xbar/ACLK] \
  [get_bd_pins axi_xbar/M00_ACLK] \
  [get_bd_pins axi_xbar/M01_ACLK] \
  [get_bd_pins axi_xbar/M02_ACLK] \
  [get_bd_pins axi_xbar/M03_ACLK] \
  [get_bd_pins axi_xbar/S00_ACLK] \
  [get_bd_pins fetch_0/ap_clk] \
  [get_bd_pins load_0/ap_clk] \
  [get_bd_pins compute_0/ap_clk] \
  [get_bd_pins store_0/ap_clk] \
  [get_bd_pins load_queue/s_aclk] \
  [get_bd_pins gemm_queue/s_aclk] \
  [get_bd_pins store_queue/s_aclk] \
  [get_bd_pins l2g_queue/s_aclk] \
  [get_bd_pins g2l_queue/s_aclk] \
  [get_bd_pins g2s_queue/s_aclk] \
  [get_bd_pins s2g_queue/s_aclk] \
  $maxi_clk \
  $saxi_clk

# Create address segments
create_bd_addr_seg -range $ip_reg_map_range -offset $fetch_base_addr [get_bd_addr_spaces processing_system/Data] [get_bd_addr_segs fetch_0/s_axi_CONTROL_BUS/Reg] SEG_fetch_0_Reg
create_bd_addr_seg -range $ip_reg_map_range -offset $load_base_addr [get_bd_addr_spaces processing_system/Data] [get_bd_addr_segs load_0/s_axi_CONTROL_BUS/Reg] SEG_load_0_Reg
create_bd_addr_seg -range $ip_reg_map_range -offset $compute_base_addr [get_bd_addr_spaces processing_system/Data] [get_bd_addr_segs compute_0/s_axi_CONTROL_BUS/Reg] SEG_compute_0_Reg
create_bd_addr_seg -range $ip_reg_map_range -offset $store_base_addr [get_bd_addr_spaces processing_system/Data] [get_bd_addr_segs store_0/s_axi_CONTROL_BUS/Reg] SEG_store_0_Reg
if { $device_family eq "zynq-7000" } {
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 [get_bd_addr_spaces compute_0/Data_m_axi_uop_port] [get_bd_addr_segs processing_system/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_processing_system_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 [get_bd_addr_spaces compute_0/Data_m_axi_data_port] [get_bd_addr_segs processing_system/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_processing_system_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 [get_bd_addr_spaces fetch_0/Data_m_axi_ins_port] [get_bd_addr_segs processing_system/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_processing_system_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 [get_bd_addr_spaces load_0/Data_m_axi_data_port] [get_bd_addr_segs processing_system/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_processing_system_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 [get_bd_addr_spaces store_0/Data_m_axi_data_port] [get_bd_addr_segs processing_system/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_processing_system_ACP_DDR_LOWOCM
} elseif { $device_family eq "zynq-ultrascale+"} {
  create_bd_addr_seg -range 0x80000000 -offset 0x00000000 [get_bd_addr_spaces fetch_0/Data_m_axi_ins_port] [get_bd_addr_segs processing_system/SAXIGP0/HPC0_DDR_LOW] SEG_processing_system_HPC0_DDR_LOW
  create_bd_addr_seg -range 0x80000000 -offset 0x00000000 [get_bd_addr_spaces load_0/Data_m_axi_data_port] [get_bd_addr_segs processing_system/SAXIGP0/HPC0_DDR_LOW] SEG_processing_system_HPC0_DDR_LOW
  create_bd_addr_seg -range 0x80000000 -offset 0x00000000 [get_bd_addr_spaces compute_0/Data_m_axi_uop_port] [get_bd_addr_segs processing_system/SAXIGP0/HPC0_DDR_LOW] SEG_processing_system_HPC0_DDR_LOW
  create_bd_addr_seg -range 0x80000000 -offset 0x00000000 [get_bd_addr_spaces compute_0/Data_m_axi_data_port] [get_bd_addr_segs processing_system/SAXIGP0/HPC0_DDR_LOW] SEG_processing_system_HPC0_DDR_LOW
  create_bd_addr_seg -range 0x80000000 -offset 0x00000000 [get_bd_addr_spaces store_0/Data_m_axi_data_port] [get_bd_addr_segs processing_system/SAXIGP0/HPC0_DDR_LOW] SEG_processing_system_HPC0_DDR_LOW
}

save_bd_design


##################################################################
# COMPILATION FLOW
##################################################################

# Create top-level wrapper file
make_wrapper -files \
  [get_files $proj_path/$proj_name.srcs/sources_1/bd/$proj_name/$proj_name.bd] -top
add_files -norecurse $proj_path/$proj_name.srcs/sources_1/bd/$proj_name/hdl/${proj_name}_wrapper.v
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# Run bistream generation on 8 threads with performance oriented P&R strategy
set num_threads 8
launch_runs impl_1 -to_step write_bitstream -jobs $num_threads
wait_on_run impl_1

# Export hardware description file and bitstream files to export/ dir
if {[file exist $proj_path/$proj_name.runs/impl_1/${proj_name}_wrapper.bit]} {
  file mkdir $proj_path/export
  file copy -force $proj_path/$proj_name.runs/impl_1/${proj_name}_wrapper.bit \
    $proj_path/export/vta.bit
}

exit
