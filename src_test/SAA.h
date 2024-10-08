//本文件定义了SAA硬件的一些常用结构体以及数据类型
#ifndef SAA_HPP
#define SAA_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <assert.h>
//----------------------------脉动阵列定义--------------------------------//

//脉动阵列大小定义
#define MATRIX_WIDTH 4  //定义脉动阵列大小为32*32，该参数影响缓冲区权重排布、读取以及累加器排布等所有与数据形状相关的部分

//脉动阵列数据位宽定义
#define WEIGHT_DATA_WIDTH 8  //定义读取的权重的数据位宽
#define INPUT_DATA_WIDTH 8  //定义读取的输入的数据位宽
#define PSUM_DATA_WIDTH 32  //定义脉动阵列的PE的部分和数据位宽，起码大于两倍权重位宽防止溢出,如果脉动阵列过大，也要相应加大

//脉动阵列数据类型定义
typedef ap_int<PSUM_DATA_WIDTH> Psum_DataType;  //定义脉动阵列的PE的数据类型,都是有符号
typedef ap_int<WEIGHT_DATA_WIDTH> Weight_DataType;  //定义读取的权重的数据类型,都是有符号
typedef ap_int<INPUT_DATA_WIDTH> Input_DataType;  //定义读取的输入的数据类型,都是有符号

//----------------------------工作状态定义--------------------------------//

//


//----------------------------缓冲区定义--------------------------------//

//权重缓冲区
#define WEIGHT_BUFFER_WIDTH 1024 // 能够存多少行 MATRIX_WIDTH*Weight_DataType

//输入缓冲区
#define INPUT_BUFFER_WIDTH 1024 // 能够存多少行 MATRIX_WIDTH*Input_DataType

//累加器
#define ACCUMULATOR_BUFFER_WIDTH 2*1024 //能够存多少行 MATRIX_WIDTH*Output_DataType

//输出缓冲区
#define OUTPUT_DATA_WIDTH 32  //定义输出缓冲区的数据位宽
typedef ap_int<OUTPUT_DATA_WIDTH> Output_DataType;  //定义输出缓冲区的数据类型
#define OUTPUT_BUFFER_WIDTH 2*512 //能够存多少行 MATRIX_WIDTH*Output_DataType

//缓冲区声明
static Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH]; //使用静态变量定义
static Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH]; //使用静态变量定义
static Psum_DataType accumulator_buffer[ACCUMULATOR_BUFFER_WIDTH][MATRIX_WIDTH]; //使用静态变量定义
static Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH]; //使用静态变量定义


//----------------------------传输向量定义--------------------------------//
////地址的写入总是指向一个完整的向量，向量长度为脉动阵列的大小，使用联合体共享内存
//typedef union tpu_vector {
//	uint8_t byte_vector[PSUM_DATA_WIDTH];   //以字节为单位，一共PE_DATA_WIDTH长度的向量
//	uint32_t transfer_vector[TPU_VECTOR_PADDING/sizeof(uint32_t)];  //进行字节对齐后的
//} tpu_vector_t;

//定义总线
#define TRANSFER_WIDTH 128  //总线位宽，代表我们访问DDR的位宽大小，最好是MATRIX_WIDTH的整数倍
typedef ap_uint<TRANSFER_WIDTH> Transfer_DataType; // 总线数据类型

//定义一次传输的块的大小
#define TRANSFER_WEIGHT_BLOCK_SIZE (MATRIX_WIDTH*MATRIX_WIDTH*WEIGHT_DATA_WIDTH)  // 权重块的大小
#define TRANSFER_INPUT_BLOCK_SIZE (MATRIX_WIDTH*MATRIX_WIDTH*INPUT_DATA_WIDTH)  // 输入块的大小
#define TRANSFER_ACCUMULATOR_BLOCK_SIZE (MATRIX_WIDTH*MATRIX_WIDTH*PSUM_DATA_WIDTH)  // 累加器块的大小
#define TRANSFER_OUTPPUT_BLOCK_SIZE (MATRIX_WIDTH*MATRIX_WIDTH*OUTPUT_DATA_WIDTH)  // 输出块的大小

//传输块的大小和总线位宽的比（代表需要总线传输多少次）
#define TRANSFER_WEIGHT_AXI_RATIO (TRANSFER_WEIGHT_BLOCK_SIZE / TRANSFER_WIDTH)  // 权重块size和总线宽度的比率
#define TRANSFER_INPUT_AXI_RATIO (TRANSFER_INPUT_BLOCK_SIZE / TRANSFER_WIDTH)  // 输入块size和总线宽度的比率
#define TRANSFER_ACCUMULATOR_AXI_RATIO (TRANSFER_ACCUMULATOR_BLOCK_SIZE / TRANSFER_WIDTH)  // 累加器块size和总线宽度的比率
#define TRANSFER_OUTPPUT_AXI_RATIO (TRANSFER_OUTPPUT_BLOCK_SIZE / TRANSFER_WIDTH)  // 输出块size和总线宽度的比率

//传输块的大小占多少字节
#define TRANSFER_WEIGHT_DATA_BYTES (TRANSFER_WEIGHT_BLOCK_SIZE / 8)
#define TRANSFER_INPUT_DATA_BYTES (TRANSFER_INPUT_BLOCK_SIZE / 8)
#define TRANSFER_ACCUMULATOR_DATA_BYTES (TRANSFER_ACCUMULATOR_BLOCK_SIZE / 8)
#define TRANSFER_OUTPUT_DATA_BYTES (TRANSFER_OUTPPUT_BLOCK_SIZE / 8)

//片上缓冲区的大小
#define WEIGHT_BUFFER_SIZE (WEIGHT_BUFFER_WIDTH*MATRIX_WIDTH*WEIGHT_DATA_WIDTH)         // 权重缓冲区大小
#define INPUT_BUFFER_SIZE (INPUT_BUFFER_WIDTH*MATRIX_WIDTH*INPUT_DATA_WIDTH)           // 输入缓冲区大小
#define ACCUMULATOR_BUFFER_SIZE (ACCUMULATOR_BUFFER_WIDTH*MATRIX_WIDTH*PSUM_DATA_WIDTH) // 累加器大小
#define OUTPUT_BUFFER_SIZE (OUTPUT_BUFFER_WIDTH*MATRIX_WIDTH*OUTPUT_DATA_WIDTH)         // 输出缓冲区大小

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
// #define TRANSFER_WEIGHT_RATIO ( TRANSFER_WIDTH / (WEIGHT_DATA_WIDTH * MATRIX_WIDTH) )
// #define TRANSFER_INPUT_RATIO ( TRANSFER_WIDTH / (INPUT_DATA_WIDTH * MATRIX_WIDTH) )
// #define TRANSFER_ACCUMULATOR_RATIO ( TRANSFER_WIDTH / (PSUM_DATA_WIDTH * MATRIX_WIDTH) )
// #define TRANSFER_OUTPPUT_RATIO ( TRANSFER_WIDTH / (OUTPUT_DATA_WIDTH * MATRIX_WIDTH) )


//----------------------------控制类型定义--------------------------------//
//指令数据定义
#define INSTRUCT_WIDTH 128 // 指令位宽
typedef ap_uint<INSTRUCT_WIDTH> Instruct_DataType; // 

//位宽定义
#define OP_CODE_WIDTH 3 // 操作码位宽

//存储映射指令
#define BUFFER_ID_WIDTH 3 // 缓冲区ID位宽
#define DRAM_ADDR_WIDTH 32 // ddr地址位宽定义
#define BUFFER_ADDR_WIDTH 16 // buffer地址位宽定义
#define TRANSFER_SIZE_WIDTH 16 // 传输尺寸位宽定义
#define TRANSFER_STRIDE_WIDTH 16 // 传输的步进位宽定义

//计算指令
#define COMPUTE_TYPE_WIDTH 3 // 计算类型
#define COMPUTE_ACCUMULATE_WIDTH 1 // 是否进行累加
#define WEIGHT_SWITCH_WIDTH 1 // 权重寄存器选择
#define COMPUTE_SWITCH_WIDTH 1 // 计算寄存器选择

// 操作码定义
#define OPCODE_LOAD 0 // 定义加载指令的操作码
#define OPCODE_COMPUTE 1 // 定义计算指令的操作码
#define OPCODE_STORE 2 // 定义存储指令的操作码
#define OPCODE_DONE 3 // 定义计算完成指令的操作码

// buffer id 定义
#define WEIGHT_BUFFER_ID 0
#define INPUT_BUFFER_ID 1
#define ACCUMULATOR_BUFFER_ID 2
#define OUTPUT_BUFFER_ID 3

// compute_type 计算类型定义
#define WEIGHT_PRELOAD 0 // 权重预加载
#define COMPUTE 1 // 使用当前脉动阵列权重计算
#define COMPUTE_WEIGHT_PRELOAD 2 // 加载权重同时进行计算，用于双缓冲操作


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
    uint64_t pad_0 : 64-OP_CODE_WIDTH; // load/store指令的目标 bufffer ID
    uint64_t pad_1 : 64; // DRAM 的基地址
} GenericIns;

//Memory存储(LOAD/STORE)指令类型（位宽128）（下面的有效位宽和为102bit）
typedef struct {
    uint64_t opcode : OP_CODE_WIDTH; // 定义操作码位宽
    uint64_t buffer_id : BUFFER_ID_WIDTH; // load/store指令的目标 bufffer ID
    uint64_t dram_base : DRAM_ADDR_WIDTH; // DRAM的偏移地址，是相对于输入数据库的偏移
    uint64_t buffer_base : BUFFER_ADDR_WIDTH; // buffer的基地址
    uint64_t y_size : TRANSFER_SIZE_WIDTH; // 需要加载的矩阵行数
    uint64_t x_size : TRANSFER_SIZE_WIDTH; // 需要加载的矩阵列数
    uint64_t x_stride : TRANSFER_STRIDE_WIDTH; // 需要加载的矩阵列方向上的步进
} MemIns;

//Compute计算指令类型（位宽128）（下面的有效位宽和为57bit）(暂时只代表GEMM)
typedef struct {
    uint64_t opcode : OP_CODE_WIDTH; // 定义操作码位宽
    uint64_t compute_type : COMPUTE_TYPE_WIDTH; // compute指令执行的内容
    uint64_t weigth_addr : BUFFER_ADDR_WIDTH; // 权重读取起始行
    uint64_t input_addr : BUFFER_ADDR_WIDTH; // 输入读取起始行
    uint64_t output_addr : BUFFER_ADDR_WIDTH; // 输出写入起始行
    uint64_t weight_switch : WEIGHT_SWITCH_WIDTH; // 使用哪一个权重加载
    uint64_t compute_switch : COMPUTE_SWITCH_WIDTH; // 使用哪一个权重计算
    uint64_t accumulate : COMPUTE_ACCUMULATE_WIDTH; // 计算结果是否累加
} ComIns;


//使用联合体便捷转换指令，节省内存
union SAAInsn {
  GenericIns generic; // 通用指令，长度为128位，用于解析操作码
  MemIns mem; // 存储器指令，用于LOAD和STORE
  ComIns com; // 计算指令，用于gemm
};


//----------------------------常用函数定义--------------------------------//


//初始化指令类型


//转换指令


//---------------------------------debug--------------------------------//
#include <cstdlib> // 包含 rand 函数的声明
#include <cstdio> // 包含 printf 函数的声明
#include <algorithm>// 包含 min max 函数的声明
#include <chrono> // 包含计时库头文件

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
        printf("%d,", (int)input_vec[i]);
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
            printf("%d,",(int)input_matrix[i][j]);
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
            printf("%d,",(int)output_buffer[i+start_addr][j]);
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

//转换为指令打印
template<typename T>
void print_instruction(T instruction, const std::string& type) {
    if (type == "MEMORY") {
        MemIns* mem_instruction = reinterpret_cast<MemIns*>(&instruction); // 将指令数据转换为内存指令结构体
        std::cout << "Memory Instruction:" << std::endl;
        std::cout << "opcode: " << mem_instruction->opcode << std::endl;
        std::cout << "buffer_id: " << mem_instruction->buffer_id << std::endl;
        std::cout << "dram_base: " << mem_instruction->dram_base << std::endl;
        std::cout << "buffer_base: " << mem_instruction->buffer_base << std::endl;
        std::cout << "y_size: " << mem_instruction->y_size << std::endl;
        std::cout << "x_size: " << mem_instruction->x_size << std::endl;
        std::cout << "x_stride: " << mem_instruction->x_stride << std::endl;
        printf("\n");
    } else if (type == "COMPUTE") {
        ComIns* com_instruction = reinterpret_cast<ComIns*>(&instruction); // 将指令数据转换为计算指令结构体
        std::cout << "Compute Instruction:" << std::endl;
        std::cout << "opcode: " << com_instruction->opcode << std::endl;
        std::cout << "compute_type: " << com_instruction->compute_type << std::endl;
        std::cout << "weigth_addr: " << com_instruction->weigth_addr << std::endl;
        std::cout << "input_addr: " << com_instruction->input_addr << std::endl;
        std::cout << "output_addr: " << com_instruction->output_addr << std::endl;
        std::cout << "weight_switch: " << com_instruction->weight_switch << std::endl;
        std::cout << "compute_switch: " << com_instruction->compute_switch << std::endl;
        std::cout << "accumulate: " << com_instruction->accumulate << std::endl;
        printf("\n");
    } else {
        std::cout << "Invalid instruction type" << std::endl;
        printf("\n");
    }
}

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




//------------------------------------模拟内存分配-------------------------------------//



#endif



























