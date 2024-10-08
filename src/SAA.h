//本文件定义了SAA硬件的一些常用结构体以及数据类型
#ifndef SAA_HPP
#define SAA_HPP

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <assert.h>
#include <cstdint> // 对于标准 C++ (如果可用)
#include <iostream>// 使用c++输出口
#include <cstring> // 使用memcpy

//----------------------------脉动阵列定义--------------------------------//

//脉动阵列大小定义，其得到的块必须要是TRANSFER_WIDTH的倍数，因此一般是2的倍数
#define MATRIX_WIDTH 16  //定义脉动阵列大小为32*32，该参数影响缓冲区权重排布、读取以及累加器排布等所有与数据形状相关的部分

//脉动阵列数据位宽定义
#define WEIGHT_DATA_WIDTH 8  //定义读取的权重的数据位宽
#define INPUT_DATA_WIDTH 8  //定义读取的输入的数据位宽
#define PSUM_DATA_WIDTH 32  //定义脉动阵列的PE的部分和数据位宽，起码大于两倍权重位宽防止溢出,如果脉动阵列过大，也要相应加大
#define OUTPUT_DATA_WIDTH 32  //定义输出缓冲区的数据位宽,可以小于累加器的位宽
#define SCALE_DATA_WIDTH 8  //定义输出缓冲区的定点整数大小
#define SCALE_SUM_WIDTH 16  //定义anu操作中的sum变量定点整数大小
#define UOP_WIDTH 32// 定义uop微操作缓冲区的一个微操作数据的位宽

// 最大使用BRAM的大小定义
#define MAX_BRAM 18
#define BRAM_SIZE (36 * 1024)
#define BLOCK_SIZE (MATRIX_WIDTH * MATRIX_WIDTH) // 定义块的大小

//脉动阵列数据类型定义
typedef ap_int<PSUM_DATA_WIDTH> Psum_DataType;  //定义脉动阵列的PE的数据类型,都是有符号
typedef ap_int<WEIGHT_DATA_WIDTH> Weight_DataType;  //定义读取的权重的数据类型,都是有符号
typedef ap_int<INPUT_DATA_WIDTH> Input_DataType;  //定义读取的输入的数据类型,都是有符号
typedef ap_int<OUTPUT_DATA_WIDTH> Output_DataType;  //定义输出缓冲区的数据类型
// typedef ap_fixed<32, 24> Output_DataType;  //定义输出缓冲区的定点小数数据类型,24位表示整数
typedef ap_uint<UOP_WIDTH> Uop_DataType;  //定义微操作缓冲区的数据类型

// 定义ap_fixed定点小数类型，用于进行缩放和计算归一化
typedef ap_fixed<32, SCALE_DATA_WIDTH> Norm_DataType;// 输出缓冲区进行归一化的float数据类型,定点类型,使用8位表示整数
typedef ap_fixed<32, SCALE_DATA_WIDTH> Scale_DataType;// 使用进行缩放的scale系数
typedef ap_fixed<32, SCALE_SUM_WIDTH> Sum_DataType;// 在计算layernorm和softmax时，用于存储sum和var的中间变量，如果计算的小数偏多，小数位越大精度越高，如果计算的整数偏多，整数位越大精度越高

// // 定义输出缓冲区,使用联合体,这样可以使用同样的地址空间转换不同的数据类型
// union OutputBufferUnion {
//     int32_t Output : OUTPUT_DATA_WIDTH;; //这是输出数据类型,int32类型
//     Norm_DataType Norm_t; // 这是用于非线性计算的float类型
// };

//----------------------------缓冲区定义--------------------------------//
//指令队列定义
#define STREAM_IN_DEPTH 512 //指令队列的深度
//缓冲区的深度定义，定义为能够存储多少个缓冲区数据类型的块
// #define WEIGHT_BUFFER_WIDTH       512 // 能够存储多少个传输块
// #define INPUT_BUFFER_WIDTH        512 // 能够存储多少个传输块
// #define OUTPUT_BUFFER_WIDTH       512 // 能够存储多少个传输块
// #define ACCUMULATOR_BUFFER_WIDTH  512 // 能够存储多少个传输块

#define WEIGHT_BUFFER_WIDTH       ((MAX_BRAM * BRAM_SIZE)/(BLOCK_SIZE * INPUT_DATA_WIDTH)) // 能够存储多少个传输块
#define INPUT_BUFFER_WIDTH        ((MAX_BRAM * BRAM_SIZE)/(BLOCK_SIZE * WEIGHT_DATA_WIDTH)) // 能够存储多少个传输块
#define OUTPUT_BUFFER_WIDTH       (((OUTPUT_DATA_WIDTH / INPUT_DATA_WIDTH) * MAX_BRAM * BRAM_SIZE)/(BLOCK_SIZE * OUTPUT_DATA_WIDTH)) // 能够存储多少个传输块
#define ACCUMULATOR_BUFFER_WIDTH  (((PSUM_DATA_WIDTH / INPUT_DATA_WIDTH) * MAX_BRAM * BRAM_SIZE)/(BLOCK_SIZE * PSUM_DATA_WIDTH)) // 能够存储多少个传输块

// 微操作内核
// 能够存多少个块的uop,为了保证UOP能够存储符合缓冲区大小硬件分块的计算
// 我们估计输出和输入缓冲区能够存储的最大的I，选择其中最大的值乘上4以适应不同硬件分块的计算
#define UOP_BUFFER_WIDTH         (4 * ((OUTPUT_BUFFER_WIDTH > INPUT_BUFFER_WIDTH) ? OUTPUT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH))


// //缓冲区声明
// static Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH]; //使用静态变量定义
// static Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH]; //使用静态变量定义
// static Psum_DataType accumulator_buffer[ACCUMULATOR_BUFFER_WIDTH][MATRIX_WIDTH]; //使用静态变量定义
// static Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH]; //使用静态变量定义

//----------------------------传输向量定义--------------------------------//

//定义总线
#define TRANSFER_WIDTH 128  //总线位宽，代表我们访问DDR的位宽大小，最好是MATRIX_WIDTH的整数倍
typedef ap_uint<TRANSFER_WIDTH> Transfer_DataType; // 总线数据类型

// 基于总线位宽计算BRAM深度
#define WEIGHT_BUFFER_DEPTH       ((MAX_BRAM * BRAM_SIZE)/(TRANSFER_WIDTH)) 
#define INPUT_BUFFER_DEPTH         ((MAX_BRAM * BRAM_SIZE)/(TRANSFER_WIDTH)) 
#define OUTPUT_BUFFER_DEPTH        (((OUTPUT_DATA_WIDTH / INPUT_DATA_WIDTH) * MAX_BRAM * BRAM_SIZE)/(TRANSFER_WIDTH)) 
#define ACCUMULATOR_BUFFER_DEPTH   (((PSUM_DATA_WIDTH / INPUT_DATA_WIDTH) * MAX_BRAM * BRAM_SIZE)/(TRANSFER_WIDTH)) 

//定义块
typedef Input_DataType InputBlock[MATRIX_WIDTH][MATRIX_WIDTH];
typedef Weight_DataType WeightBlock[MATRIX_WIDTH][MATRIX_WIDTH];
typedef Output_DataType Output_Block[MATRIX_WIDTH][MATRIX_WIDTH];

//定义一次传输的块的大小
#define TRANSFER_WEIGHT_BLOCK_SIZE (BLOCK_SIZE*WEIGHT_DATA_WIDTH)  // 权重块的大小
#define TRANSFER_INPUT_BLOCK_SIZE (BLOCK_SIZE*INPUT_DATA_WIDTH)  // 输入块的大小
#define TRANSFER_ACCUMULATOR_BLOCK_SIZE (BLOCK_SIZE*PSUM_DATA_WIDTH)  // 累加器块的大小
#define TRANSFER_OUTPPUT_BLOCK_SIZE (BLOCK_SIZE*OUTPUT_DATA_WIDTH)  // 输出块的大小

//传输块的大小占多少字节
#define TRANSFER_WEIGHT_DATA_BYTES (TRANSFER_WEIGHT_BLOCK_SIZE / 8)
#define TRANSFER_INPUT_DATA_BYTES (TRANSFER_INPUT_BLOCK_SIZE / 8)
#define TRANSFER_ACCUMULATOR_DATA_BYTES (TRANSFER_ACCUMULATOR_BLOCK_SIZE / 8)
#define TRANSFER_OUTPUT_DATA_BYTES (TRANSFER_OUTPPUT_BLOCK_SIZE / 8)


//传输块的大小和总线位宽的比（代表需要总线传输多少次）
#define TRANSFER_WEIGHT_AXI_RATIO (TRANSFER_WEIGHT_BLOCK_SIZE / TRANSFER_WIDTH)  // 权重块size和总线宽度的比率
#define TRANSFER_INPUT_AXI_RATIO (TRANSFER_INPUT_BLOCK_SIZE / TRANSFER_WIDTH)  // 输入块size和总线宽度的比率
#define TRANSFER_ACCUMULATOR_AXI_RATIO (TRANSFER_ACCUMULATOR_BLOCK_SIZE / TRANSFER_WIDTH)  // 累加器块size和总线宽度的比率
#define TRANSFER_OUTPPUT_AXI_RATIO (TRANSFER_OUTPPUT_BLOCK_SIZE / TRANSFER_WIDTH)  // 输出块size和总线宽度的比率

// 片上缓冲区的大小，是块大小*存储多少个块/缓冲区数据类型，实际上代表的深度是以TRANSFER_WIDTH为单位
// 如果需要转换成实际存储多少个数据，需要乘以总线类型和元素类型的比
#define WEIGHT_BUFFER_SIZE (WEIGHT_BUFFER_WIDTH * TRANSFER_WEIGHT_BLOCK_SIZE / TRANSFER_WIDTH)         // 权重缓冲区大小
#define INPUT_BUFFER_SIZE (INPUT_BUFFER_WIDTH * TRANSFER_INPUT_BLOCK_SIZE / TRANSFER_WIDTH)            // 输入缓冲区大小#define OUTPUT_BUFFER_SIZE (OUTPUT_BUFFER_WIDTH*MATRIX_WIDTH*OUTPUT_DATA_WIDTH)         // 输出缓冲区大小
#define OUTPUT_BUFFER_SIZE (OUTPUT_BUFFER_WIDTH * TRANSFER_OUTPPUT_BLOCK_SIZE / TRANSFER_WIDTH)         // 输出缓冲区大小
#define ACCUMULATOR_BUFFER_SIZE (ACCUMULATOR_BUFFER_WIDTH * TRANSFER_ACCUMULATOR_BLOCK_SIZE / TRANSFER_WIDTH) // 累加器大小

//元素的字节大小
#define WEIGHT_DATA_BYTES (WEIGHT_DATA_WIDTH / 8)
#define INPUT_DATA_BYTES (INPUT_DATA_WIDTH / 8)
#define ACCUMULATOR_DATA_BYTES (PSUM_DATA_WIDTH / 8)
#define OUTPUT_DATA_BYTES (OUTPUT_DATA_WIDTH / 8)

//总线类型和元素类型的比，用于总线传输后提取元素到缓冲区，总线地址一次偏移一个总线大小，等于多少个传输数据类型
#define TRANSFER_WEIGHT_RATIO ( TRANSFER_WIDTH / (WEIGHT_DATA_WIDTH ) ) // 计算一个总线数据等于多少个权重数据
#define TRANSFER_INPUT_RATIO ( TRANSFER_WIDTH / (INPUT_DATA_WIDTH ) )
#define TRANSFER_ACCUMULATOR_RATIO ( TRANSFER_WIDTH / (PSUM_DATA_WIDTH ) )
#define TRANSFER_OUTPPUT_RATIO ( TRANSFER_WIDTH / (OUTPUT_DATA_WIDTH ) )

//----------------------------控制类型定义--------------------------------//
// 指令数据定义
#define INSTRUCT_WIDTH 128 // 指令位宽
typedef ap_uint<INSTRUCT_WIDTH> Instruct_DataType; // 

// 位宽定义
#define OP_CODE_WIDTH 3 // 操作码位宽

// 存储映射指令
#define BUFFER_ID_WIDTH 3 // 缓冲区ID位宽
#define DRAM_ADDR_WIDTH 32 // ddr地址位宽定义
#define BUFFER_ADDR_WIDTH 16 // buffer地址位宽定义
#define TRANSFER_SIZE_WIDTH 16 // 传输尺寸位宽定义
#define TRANSFER_STRIDE_WIDTH 16 // 传输的步进位宽定义

// 计算指令
#define COMPUTE_TYPE_WIDTH 3 // 计算类型
#define COMPUTE_ACCUMULATE_WIDTH 1 // 是否进行累加
#define WEIGHT_SWITCH_WIDTH 1 // 权重寄存器选择
#define COMPUTE_SWITCH_WIDTH 1 // 计算寄存器选择
#define ITER_WIDTH 14 // 定义了分块计算的循环最大值
#define SCALE_DEQUANT_WIDTH 32 // 反量化缩放因子的位宽
#define SCALE_REQUANT_WIDTH 16 // 重量化缩放因子的位宽
#define UOP_ADDR_WIDTH 14 // UOP寻址位宽定义，与UOP缓冲区深度有关

// 激活和归一化位宽定义
#define ANU_TYPE_WIDTH 3 // 激活和归一化类型
#define ANU_IMM_BIT_WIDTH 16// ANU立即数的位宽

// 操作码定义
#define OPCODE_LOAD 0 // 定义加载指令的操作码
#define OPCODE_COMPUTE 1 // 定义计算指令的操作码
#define OPCODE_STORE 2 // 定义存储指令的操作码
#define OPCODE_DONE 3 // 定义计算完成指令的操作码
#define OPCODE_ANU 4 // 定义ANU指令的操作码
#define OPCODE_GEMM 5 // gemm指令,直接在硬件上完成分块矩阵乘法

// ANU操作码
#define UNUSE_ANU 0 // 不使用ANU操作
#define ANU_SOFTMAX 1 // 进行softmax，如果对行进行归一化
#define ANU_LAYERNORM 2 // 进行layernorm,沿着特征维度对样本进行归一化
#define ANU_SIGMOID 3 // 进行非线性激活
#define ANU_RELU 4 // 进行RELU激活

// buffer id 定义
#define WEIGHT_BUFFER_ID 0
#define INPUT_BUFFER_ID 1
#define ACCUMULATOR_BUFFER_ID 2
#define OUTPUT_BUFFER_ID 3
#define UOP_BUFFER_ID 4

// compute_type 计算类型定义
// #define WEIGHT_PRELOAD 0 // 权重预加载
// #define COMPUTE 1 // 使用当前脉动阵列权重计算
// #define COMPUTE_WEIGHT_PRELOAD 2 // 加载权重同时进行计算，用于双缓冲操作
// scale_type 缩放类型定义
#define NO_QUANT 0 // 不进行量化
#define DE_QUANT 1 // 进行反量化
#define RE_QUANT 2 // 进行重量化


// 指令内部数据类型定义
typedef ap_uint<OP_CODE_WIDTH> Opcode_DataType; // 操作码数据类型
typedef ap_uint<BUFFER_ID_WIDTH> Buffer_Id_DataType; // 缓冲区ID数据类型
typedef ap_uint<DRAM_ADDR_WIDTH> Dram_Addr_DataType; // Dram地址类型定义
typedef ap_uint<BUFFER_ADDR_WIDTH> Buffer_Addr_DataType; // 缓冲区地址类型定义
typedef ap_uint<TRANSFER_SIZE_WIDTH> Transfer_Size_DataType; // 传输尺寸类型定义
typedef ap_uint<TRANSFER_STRIDE_WIDTH> Transfer_Stride_DataType; // 传输步进类型定义

typedef ap_uint<COMPUTE_TYPE_WIDTH> Compute_Type_DataType; // 计算指令类型
typedef ap_uint<COMPUTE_ACCUMULATE_WIDTH> Compute_Accumulate_DataType; // 是否进行累加数据类型
typedef ap_uint<WEIGHT_SWITCH_WIDTH> Weight_Switch_DataType; // 使用哪一个权重加载
typedef ap_uint<COMPUTE_SWITCH_WIDTH> Compute_Switch_DataType; // 使用哪一个权重加载

//通用指令类型(128位)
typedef struct {
    uint64_t opcode : OP_CODE_WIDTH; // 定义操作码位宽
    uint64_t pop_pre_raw   : 1;  // 是否有对前面模块的RAW依赖
    uint64_t pop_next_war  : 1; // 是否有对后面模块的WAR依赖
    uint64_t push_pre_war  : 1;  // 是否有对前面模块的WAR影响
    uint64_t push_next_raw : 1; // 是否有对后面模块的RAW影响
    uint64_t pad_0 : 64-OP_CODE_WIDTH-4; 
    uint64_t pad_1 : 64;
} GenericIns;

//Memory存储(LOAD/STORE)指令类型（位宽128）（下面的有效位宽和为102bit）
typedef struct {
    uint64_t opcode : OP_CODE_WIDTH; // 定义操作码位宽
    uint64_t pop_pre_raw   : 1;  // 是否有对前面模块的RAW依赖
    uint64_t pop_next_war   : 1; // 是否有对后面模块的WAR依赖
    uint64_t push_pre_war  : 1;  // 是否有对前面模块的WAR影响
    uint64_t push_next_raw  : 1; // 是否有对后面模块的RAW影响

    uint64_t buffer_id : BUFFER_ID_WIDTH; // load/store指令的目标 bufffer ID
    uint64_t dram_base : DRAM_ADDR_WIDTH; // DRAM的偏移地址，是相对于输入数据库的偏移
    uint64_t buffer_base : BUFFER_ADDR_WIDTH; // buffer的基地址
    uint64_t y_size : TRANSFER_SIZE_WIDTH; // 需要加载的矩阵行数
    uint64_t x_size : TRANSFER_SIZE_WIDTH; // 需要加载的矩阵列数
    uint64_t x_stride : TRANSFER_STRIDE_WIDTH; // 需要加载的矩阵列方向上的步进
    // uint64_t read_scale : TRANSFER_STRIDE_WIDTH; // 读取缩放，对读取输出缓冲区时决定读取的类型，方便对store读取后输入下一层
} MemIns;

//Compute计算指令类型（位宽128）（下面的有效位宽和为57bit）(暂时只代表GEMM)
typedef struct {
// 7位
    uint64_t opcode : OP_CODE_WIDTH; // 定义操作码位宽
    uint64_t pop_pre_raw   : 1; // 是否有对load的RAW依赖
    uint64_t pop_next_war  : 1; // 是否有对store的WAR依赖
    uint64_t push_pre_war  : 1; // 是否有对load的WAR影响
    uint64_t push_next_raw : 1; // 是否有对store的WAR影响
// gemm指令操作字段 7+4*14+1=7+56+1=64
    uint64_t uop_bgn : UOP_ADDR_WIDTH; // 该指令从微操作缓冲区的uop_bgn处开始读取微操作序列,完成的是I循环
    uint64_t uop_end : UOP_ADDR_WIDTH; // 微操作结束位置
    uint64_t dim_K_block : ITER_WIDTH; // (外循环)指令执行的大块的K维度循环次数,就是大分块的K维度被MATRIX_WIDTH的分块
    uint64_t dim_J_block : ITER_WIDTH; // (内循环)指令执行的大块的J维度循环次数
    uint64_t bias_use : 1; // 是否使用偏置,如果使用,所有分块都进行累加,前提是bias已经加载
// 第二个64位
    uint64_t scale : SCALE_DEQUANT_WIDTH; // 最后32位,可以用来存储一个反量化因子和两个输入矩阵的量化因子或者一个重量化因子
    uint64_t input_base : UOP_ADDR_WIDTH; // 读取input的基地址，以行为单位
    uint64_t weight_base : UOP_ADDR_WIDTH; // 读取weight的基地址
    uint64_t relu_use : 1; // 是否进行relu
    uint64_t scale_type : COMPUTE_TYPE_WIDTH; // scale的形式
} ComIns;





//anu计算指令类型（位宽128，）用于计算激活和归一化等操作，是对矩阵结果进行处理，按照
typedef struct {
    uint64_t opcode : OP_CODE_WIDTH; // 定义操作码位宽
    uint64_t pop_pre_raw   : 1; // 是否有对load的RAW依赖
    uint64_t pop_next_war  : 1; // 是否有对store的WAR依赖
    uint64_t push_pre_war  : 1; // 是否有对load的WAR影响
    uint64_t push_next_raw : 1; // 是否有对store的WAR影响
// anu指令操作字段
    uint64_t anu_type : ANU_TYPE_WIDTH; // ANU操作的类型，决定对累加器中的矩阵使用什么操作
    uint64_t iter_uop : BUFFER_ADDR_WIDTH; // 微操作的循环,可以用来遍历行块，决定对J维度执行多少次块计算
    uint64_t iter_I : BUFFER_ADDR_WIDTH; // I维度遍历的次数
    uint64_t imm : ANU_IMM_BIT_WIDTH; // 立即数用于输入归一化的N数目
// 第二个64位
} AnuIns;

//使用联合体便捷转换指令，节省内存
union SAAInsn {
  GenericIns generic; // 通用指令，长度为128位，用于解析操作码
  MemIns mem; // 存储器指令，用于LOAD和STORE
  ComIns com; // 矩阵乘法指令，用于gemm
  AnuIns anu; // ANU指令，用于其余算子
};

/*! 微操作的起始字段, */
#define UOP_INPUT_IDX_0 0
#define UOP_INPUT_IDX_1 (UOP_INPUT_IDX_0 + BUFFER_ADDR_WIDTH -1)
#define UOP_OUTPUT_IDX_0 (UOP_INPUT_IDX_1 + 1)
#define UOP_OUTPUT_IDX_1 (UOP_OUTPUT_IDX_0 + BUFFER_ADDR_WIDTH -1)

/*! \brief 用于 GEMM/ANU 指令的 SAA 微操作序列,提取了I循环的地址参数 */
typedef struct {
  uint32_t input_idx    : BUFFER_ADDR_WIDTH; // 微操作指令计算的每个i循环的输入地址
  uint32_t output_idx    : BUFFER_ADDR_WIDTH; // 微操作指令计算的每个i循环的输出地址
} Uop;



//----------------------------SAA硬件模块定义--------------------------------//


/*!
* \brief fetch模块
* 获取指令，判断指令类型，将其填入任务队列
* \param instruct_count 总指令数。 AXI-lite 内存映射寄存器。
* \param instruct  DRAM 中的指令数据库基址。 用于读取指令
* \param load_queue 加载指令队列。 AXI 流 FIFO。
* \param gemm_queue GEMM 指令队列。 AXI 流 FIFO。
* \param store_queue 存储指令队列。 AXI 流 FIFO。
*/
void fetch(
    uint32_t instruct_count,
    volatile Instruct_DataType *instruct,
    hls::stream<Instruct_DataType> &load_queue,
    hls::stream<Instruct_DataType> &gemm_queue,
    hls::stream<Instruct_DataType> &store_queue);


/*!
* \brief load模块
* 接收load指令，并将其解码然后通过m_axi接口访问DDR读取权重和数据
* \param inputs DRAM 中的输入数据库基地址。 用于读取输入/权重
* \param load_queue 加载指令队列。 AXI 流 FIFO。
* \param input_buffer  片上输入缓冲区，只写。
* \param weight_buffer  片上权重缓冲区，只写。
*/
void load(
    hls::stream<Instruct_DataType> &load_queue,
    volatile Transfer_DataType *inputs,
    volatile Transfer_DataType *weights,
	hls::stream<bool> &l2c_raw_queue,
	hls::stream<bool> &c2l_war_queue,
    Input_DataType input_buffer[INPUT_BUFFER_WIDTH*MATRIX_WIDTH],
    Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH*MATRIX_WIDTH]) ; 


/*!
* \brief store模块
* 接收store指令，并将其解码然后通过m_axi接口访问DDR写入权重和数据，刚好与load是相反操作
* \param outputs DRAM 中的输出数据库基地址。 用于写入结果
* \param store_queue 存储指令队列。 AXI 流 FIFO。
* \param output_buffer  片上输出缓冲区，只读。
*/
// void store(
//   hls::stream<Instruct_DataType> &store_queue,
//   volatile Transfer_DataType *outputs,
//   Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH]);

void store(
  hls::stream<Instruct_DataType> &store_queue,
  volatile Transfer_DataType *outputs,
  hls::stream<bool> &s2c_war_queue,
  hls::stream<bool> &c2s_raw_queue,
  Input_DataType input_buffer[INPUT_BUFFER_WIDTH*MATRIX_WIDTH],
  Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH*MATRIX_WIDTH]) ;

/*!
* \brief compute模块
* 接收compute指令，并将其解码然后通过m_axi接口访问DDR读取权重和数据
* \param done 该信号由完成指令控制，当执行到完成指令时，代表完成计算
* \param gemm_queue 加载指令队列。 AXI 流 FIFO。
* \param input_buffer  片上输入缓冲区，只读。
* \param weight_buffer  片上权重缓冲区，只读。
* \param output_buffer  片上输出缓冲区，只写。
*/
void compute(
  volatile uint32_t &done,
  volatile Uop_DataType *uops,
  volatile Transfer_DataType *biases,
  hls::stream<Instruct_DataType> &gemm_queue,
  hls::stream<bool> &l2c_raw_queue,
  hls::stream<bool> &s2c_war_queue,
  hls::stream<bool> &c2l_war_queue,
  hls::stream<bool> &c2s_raw_queue,
  Input_DataType input_buffer[INPUT_BUFFER_WIDTH*MATRIX_WIDTH],
  Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH*MATRIX_WIDTH],
  Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH*MATRIX_WIDTH]) ;


/*!
* \brief SAA模块
* 组合了load和fetch模块，用于流处理加速器架构
* \param insn_count 总指令数。 AXI-lite 内存映射寄存器。
* \param insns DRAM 中的指令数据库基址。 用于读取指令
* \param inputs DRAM 中的输入数据基址。 用于读取输入数据
* \param input_buffer 片上输入缓冲区，只写。
* \param weight_buffer 片上权重缓冲区，只写。
* \param outputs DRAM 中的输出数据基址。 用于存储输出数据
*/
// void saa_top(
//   uint32_t insn_count,
//   volatile Instruct_DataType *insns,
//   volatile Transfer_DataType *inputs,
//   volatile Transfer_DataType *weights,
//   volatile Transfer_DataType *outputs,
//   Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
//   Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
//   Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH]);

void saa_top(
  uint32_t insn_count,
  volatile Instruct_DataType *insns,
  volatile Uop_DataType *uops,
  volatile Transfer_DataType *inputs,
  volatile Transfer_DataType *weights,
  volatile Transfer_DataType *biases,
  volatile Transfer_DataType *outputs,
  volatile uint32_t &done)  ;




//---------------------------------辅助函数--------------------------------//
#define PRAGMA_SUB(x) _Pragma (#x)  // 这两个函数将Pragma当字符串处理，这样就可以使用宏定义改变偏移
#define PRAGMA_HLS(x) PRAGMA_SUB(x)

//---------------------------------debug--------------------------------//
#include <cstdlib> // 包含 rand 函数的声明
#include <cstdio> // 包含 printf 函数的声明
#include <algorithm>// 包含 min max 函数的声明
#include <chrono> // 包含计时库头文件
#include <ctime> // 包含 clock_gettime 函数的头文件

//矩阵赋值
template<typename T>
void init_matrix(T input_matrix[][MATRIX_WIDTH])
{
    for(int i = 0; i < MATRIX_WIDTH; i++)
    {
        for(int j = 0; j < MATRIX_WIDTH; j++)
        {
			int data=(rand() % 40 - 20);
            input_matrix[i][j] = data;
        }
    }
}

//软件矩阵乘法
template<typename T, typename U>
void matrix_dot(T a_matrix[][MATRIX_WIDTH], T b_matrix[][MATRIX_WIDTH], U c_matrix[][MATRIX_WIDTH])
{
    for (int i = 0; i < MATRIX_WIDTH; i++)
    {
        for (int j = 0; j < MATRIX_WIDTH; j++)
        {
            // 初始化 c_matrix[i][j] 为零
            c_matrix[i][j] = 0;
            // 计算 c_matrix[i][j] 的值
            for (int k = 0; k < MATRIX_WIDTH; k++)
            {
                c_matrix[i][j] += a_matrix[i][k] * b_matrix[k][j];
            }
        }
    }
}

//打印向量
template<typename T>
void print_vec(T input_vec[],int length)
{
    for(int i = 0; i < length; i++)
    {
        // printf("%d,", (int)input_vec[i]);
        std::cout << input_vec[i] << " ";
    }
	printf("\n");
	printf("\n");
}

//打印MATRIX_WIDTH大小矩阵
template<typename T>
void print_matrix(T input_matrix[][MATRIX_WIDTH])
{
    for(int i = 0; i < MATRIX_WIDTH; i++)
    {
        for(int j = 0; j < MATRIX_WIDTH; j++)
        {
            // printf("%d,",(int)input_matrix[i][j]);
            std::cout << input_matrix[i][j] << " ";
        }
        printf("\n");
    }
	printf("\n");
}


//打印缓冲区
template<typename T>
void print_buffer(T output_buffer[][MATRIX_WIDTH] ,int start_addr,int row)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < MATRIX_WIDTH; j++)
        {
            printf("%d,",(float)output_buffer[i+start_addr][j]);
        }
        printf("\n");
    }
	printf("\n");
}

template<typename T>
void print_buffer1(T* output_buffer ,int row)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < MATRIX_WIDTH; j++)
        {
            // printf("%d,",(int)output_buffer[i*MATRIX_WIDTH+j]);
            std::cout << (float)output_buffer[i*MATRIX_WIDTH+j] << " ";
        }
        printf("\n");
    }
	printf("\n");
}


//打印任意缓冲区任意大的矩阵（相当于把大矩阵从output_buffer提取出来，所有矩阵都是以块方式存储）
template<typename T>
void print_buffer_matrix(T output_buffer[][MATRIX_WIDTH] ,int start_addr,int I0 ,int J0)
{

    T buffer[I0][J0] = {{0}};

    //按块循环大矩阵
	const int I = I0 / MATRIX_WIDTH; //计算行的分块数
	const int J = J0 / MATRIX_WIDTH; //计算列的分块数

    //填充buffer矩阵
    for (int i = 0; i < I; i++) { // 对输出矩阵的行块循环
        for (int j = 0; j < J; j++) { // 对输出矩阵的列块循环
        const int ob_addr = start_addr+(i*J + j)*MATRIX_WIDTH; // 当前块行寻址的起始地址
            //遍历当前块
            for(int r = 0; r < MATRIX_WIDTH; r++)
                {
                    for(int c = 0; c < MATRIX_WIDTH; c++)
                    {
                        int row_index = i*MATRIX_WIDTH + r;
                        int col_index = j*MATRIX_WIDTH + c;
                        buffer[row_index][col_index] = output_buffer[r+ob_addr][c];
                    }
                }
        }
    }

    //输出buffer矩阵
    for(int i = 0; i < I0; i++)
    {
        for(int j = 0; j < J0; j++)
        {
            printf("%d,",(int)buffer[i][j]);
        }
        printf("\n");
    }
	printf("\n");
}

//打印任意缓冲区任意大的矩阵（相当于把大矩阵从output_buffer提取出来，所有矩阵都是以块方式存储）
template<typename T>
void print_buffer_matrix_pad(T output_buffer[][MATRIX_WIDTH] ,int start_addr,int I0 ,int J0,int I_padded ,int J_padded)
{

    //输出矩阵大小
    T buffer[I0][J0] = {{0}};

    //按块循环大矩阵
	const int I = I_padded / MATRIX_WIDTH; //计算行的分块数
	const int J = J_padded / MATRIX_WIDTH; //计算列的分块数

    //计算填充的部分
    const int pad_I = I_padded - I0;
    const int pad_J = J_padded - J0;

    //填充buffer矩阵
    for (int i = 0; i < I; i++) { // 对输出矩阵的行块循环
        for (int j = 0; j < J; j++) { // 对输出矩阵的列块循环
        const int ob_addr = start_addr+(i*J + j)*MATRIX_WIDTH; // 当前块行寻址的起始地址
            int rows = MATRIX_WIDTH - (i == I-1 ? pad_I : 0); // 计算当前块有效行数
            int clos = MATRIX_WIDTH - (j == J-1 ? pad_J : 0); // 计算当前块有效列数
            //遍历当前块
            for(int r = 0; r < rows; r++)
                {
                    for(int c = 0; c < clos; c++)
                    {
                        int row_index = i*MATRIX_WIDTH + r;
                        int col_index = j*MATRIX_WIDTH + c;
                        buffer[row_index][col_index] = output_buffer[r+ob_addr][c];
                    }
                }
        }
    }

    //输出buffer矩阵
    for(int i = 0; i < I0; i++)
    {
        for(int j = 0; j < J0; j++)
        {
            printf("%d,",(int)buffer[i][j]);
        }
        printf("\n");
    }
	printf("\n");
}


//------------------------------------saa测试-------------------------------------//

//矩阵生成
template<typename T>
T** init_matrix(int rows, int cols)
{

    // 在堆上分配连续的内存块以存储所有元素
    T* data = new T[rows * cols];

    // 在堆上创建行指针数组
    T** matrix = new T*[rows];
    for(int i = 0; i < rows; i++)
    {
        // 设置行指针指向对应的内存块位置
        matrix[i] = &data[i * cols];
    }

    // 初始化矩阵元素
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            int data = (rand() % 40 - 20); // 生成随机数范围在-20到20之间
            matrix[i][j] = static_cast<T>(data);
        }
    }
    return matrix;
}

//矩阵乘法
template<typename T0,typename T1,typename T>
T** matrix_dot(T0** matrix1, T1** matrix2, int row, int col, int col1)
{
    // 创建结果矩阵
    T** result = init_matrix<T>(row, col1);

    // 计算矩阵乘法
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col1; j++) {
            result[i][j] = 0;
            for (int k = 0; k < col; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}


//矩阵打印
template<typename T>
void print_matrix(T** matrix, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    printf("\n");
}

//二进制打印
template<typename T>
void printBinary(T data,int length) {
    for (int i = length - 1; i >= 0; i--) {
        std::cout << ((data >> i) & 1);
    }
    std::cout << std::endl;
    printf("\n");
}

//指令打印
template<typename T>
void printBinary_instruction(T instruction) {
    size_t size = sizeof(instruction);
    unsigned char *ptr = (unsigned char *)&instruction;
    for (int i = size - 1; i >= 0; i--) {
        for (int j = 7; j >= 0; j--) {
            printf("%d", (ptr[i] >> j) & 1);
        }
        if(i%8==0)
            printf("\n"); //每过64位换行一次
    }
    printf("\n");
}

// //转换为指令打印
// template<typename T>
// void print_instruction(T instruction, const std::string& type) {
//     if (type == "MEMORY") {
//         MemIns* mem_instruction = reinterpret_cast<MemIns*>(&instruction); // 将指令数据转换为内存指令结构体
//         std::cout << "Memory Instruction:" << std::endl;
//         std::cout << "opcode: " << mem_instruction->opcode << std::endl;
//         std::cout << "buffer_id: " << mem_instruction->buffer_id << std::endl;
//         std::cout << "dram_base: " << mem_instruction->dram_base << std::endl;
//         std::cout << "buffer_base: " << mem_instruction->buffer_base << std::endl;
//         std::cout << "y_size: " << mem_instruction->y_size << std::endl;
//         std::cout << "x_size: " << mem_instruction->x_size << std::endl;
//         std::cout << "x_stride: " << mem_instruction->x_stride << std::endl;
//         printf("\n");
//     } else if (type == "COMPUTE") {
//         ComIns* com_instruction = reinterpret_cast<ComIns*>(&instruction); // 将指令数据转换为计算指令结构体
//         std::cout << "Compute Instruction:" << std::endl;
//         std::cout << "opcode: " << com_instruction->opcode << std::endl;
//         std::cout << "compute_type: " << com_instruction->compute_type << std::endl;
//         std::cout << "weigth_addr: " << com_instruction->weigth_addr << std::endl;
//         std::cout << "input_addr: " << com_instruction->input_addr << std::endl;
//         std::cout << "output_addr: " << com_instruction->output_addr << std::endl;
//         std::cout << "weight_switch: " << com_instruction->weight_switch << std::endl;
//         std::cout << "compute_switch: " << com_instruction->compute_switch << std::endl;
//         std::cout << "accumulate: " << com_instruction->accumulate << std::endl;
//         printf("\n");
//     } else {
//         std::cout << "Invalid instruction type" << std::endl;
//         printf("\n");
//     }
// }

//一维数组按二维打印
template<typename T>
void print1D_2DArray(T arr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << arr[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    printf("\n");
}


// 读取二进制文件并填充到二维数组，rows是读取的行，cols是每一行读取的数目也就是列
template<typename T>
void readBinaryFileToMatrix(const char* filename, T** matrix, size_t rows, size_t cols) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fputs("File error", stderr);
        exit(1);
    }
    for (size_t i = 0; i < rows; ++i) {
        if (fread(matrix[i], sizeof(T), cols, file) != cols) { // 连续读取一行cols列，如果fread返回没有读满cols，说明有错误
            fputs("Reading error", stderr);
            fclose(file);
            exit(2);
        }
    }
    fclose(file);
}

// 读取二进制文件并填充到一维数组

template<typename T>
void readBinaryFileToArray(const char* filename, T* array, size_t num_elements) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fputs("File error", stderr);
        exit(1);
    }
    if (fread(array, sizeof(T), num_elements, file) != num_elements) { // 尝试读取num_elements个元素
        fputs("Reading error", stderr);
        fclose(file);
        exit(2);
    }
    fclose(file);
}


// 将二维数组写入到二进制文件
template<typename T>
void writeBinaryFileFromMatrix(const char* filename, T** matrix, size_t rows, size_t cols) {
    FILE* file = fopen(filename, "wb"); // 使用 "wb"（写入二进制）模式打开文件
    if (file == NULL) {
        std::cerr << "File error" << std::endl;
        exit(1);
    }
    for (size_t i = 0; i < rows; ++i) {
        if (fwrite(matrix[i], sizeof(T), cols, file) != cols) { // 连续写入一行cols列
            std::cerr << "Writing error" << std::endl;
            fclose(file);
            exit(2);
        }
    }
    fclose(file);
}

//------------------------------------模拟内存分配-------------------------------------//





#endif





















