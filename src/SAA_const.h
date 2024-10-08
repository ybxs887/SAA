// 本代码定义了SAA的诸多硬件常量
#ifndef SAA_CONST_H
#define SAA_CONST_H
// 定义物理地址位宽
#define PHY_ADDR_WIDTH 64

/*!
* \brief 模块的寄存器映射范围和模块控制寄存器基地址
* 因为器件是ultra scale+，因此基地址从0xA0000000开始
*/
#define IP_REG_MAP_RANGE   0x1000     // 定义每个模块寄存器地址范围大小
#define FETCH_BASE_ADDR    0xA0000000 // ftech模块的基地址
#define LOAD_BASE_ADDR     0xA0001000 // load模块的基地址
#define COMPUTE_BASE_ADDR  0xA0002000 // compute模块的基地址
#define STORE_BASE_ADDR    0xA0003000 // store模块的基地址

/*!
* \brief IP中控制寄存器相对于基地址的偏移
* 在HLS中，0x00-0x0C被保留用于块级I/O协议。也就是控制寄存器CONTROL_BUS控制启动暂停
* 下面的每个偏移在同一个模块中间隔8字节也就是64位，这是为了兼容64位系统
*/
#define REGISTER_OFFSET 0x08 // 定义每个寄存器间的间隔为8字节
//fetch模块
#define FETCH_INSN_COUNT_OFFSET  0x10                                       // fetch模块的指令数量寄存器
#define FETCH_INSN_ADDR_OFFSET   FETCH_INSN_COUNT_OFFSET + REGISTER_OFFSET  // fetch模块的指令地址寄存器
//load模块
#define LOAD_INP_ADDR_OFFSET     0x10                                       // load模块的输入缓冲区地址8字节64位
#define LOAD_WGT_ADDR_OFFSET     LOAD_INP_ADDR_OFFSET + REGISTER_OFFSET     // load模块的权重缓冲区地址8字节64位
//compute模块
#define COMPUTE_DONE_OFFSET      0x10                                       // compute模块的done信号
//store模块
#define STORE_OUT_ADDR_OFFSET    0x10                                       // store模块的输出缓冲区地址8字节64位


#endif

