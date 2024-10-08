#include "../src/SAA.h"
#include <stdio.h>
#include "SAA_SIM.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

//1 二进制输出指令

// void printBinary(unsigned char *ptr, size_t size) {
//     for (int i = size - 1; i >= 0; i--) {
//         for (int j = 7; j >= 0; j--) {
//             printf("%d", (ptr[i] >> j) & 1);
//         }
//     }
//     printf("\n");
// }


// int main() {
//     MemIns memIns;
//     SAAInsn c;
//     memIns.opcode = 0;
//     memIns.buffer_id = 0;
//     memIns.dram_base = 0;
//     memIns.buffer_base = 0;
//     memIns.y_size = 1;
//     memIns.x_size = 0;
//     memIns.x_stride = 0;

//     printf("MemIns length: %d",OP_CODE_WIDTH+BUFFER_ID_WIDTH+DRAM_ADDR_WIDTH
//     +BUFFER_ADDR_WIDTH+2*TRANSFER_SIZE_WIDTH+TRANSFER_STRIDE_WIDTH);

//     unsigned char *ptr = (unsigned char *)&memIns;
//     size_t size = sizeof(MemIns);
    
//     printf("Binary representation of MemIns: ");
//     printBinary(ptr, size);

//     printf("Size of MemIns: %zu bytes\n", sizeof(MemIns));// 输出指令的位宽
//     return 0;
// }




// // 2 测试load指令，单个指令
// #include <iostream>

// int main() {

//     // 根据参数生成指令
//     int insn_count = 1; // 生成一个load指令 
//     MemIns instruction;
//     instruction.opcode = OPCODE_LOAD;        // LOAD操作码3位
//     instruction.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区3位
//     instruction.dram_base = 0;               // DRAM基地址32位
//     instruction.buffer_base = 0;             // buffer基地址16位
//     instruction.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位
//     printBinary_instruction(instruction);    // 输出指令的二进制

//     // 转换为指令类型
//     Instruct_DataType instruction_data;    
//     std::memcpy(&instruction_data, &instruction, sizeof(MemIns)); //转换指令结构体为128位指令数据类型
//     printBinary(instruction_data,INSTRUCT_WIDTH);                 // 输出指令的二进制

//     // 生成输入矩阵
//     const int ROWS = MATRIX_WIDTH; // 矩阵行数
//     const int COLS = MATRIX_WIDTH; // 矩阵列数
//     Input_DataType **inputs_matrix= init_matrix<Input_DataType>(ROWS,COLS); //生成矩阵
//     print_matrix(inputs_matrix, ROWS, COLS); // 打印生成的矩阵
     
//     // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输
//     Input_DataType *inputs_matrix1 = *inputs_matrix;
//     print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
//     Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
//     print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组

//     // 执行saa_top硬件，加载指令，只执行load函数，注意，传入的是指针类型
//     Transfer_DataType* output; // 输出给总线
//     saa_top(insn_count,&instruction_data,transfer_inputs_matrix, input_buffer, weight_buffer,output); 
    
//     //检查缓冲区
//     print_buffer(input_buffer,instruction.buffer_base,10);


//     return 0;
// }


// // 3 测试load指令，多指令
// #include <iostream>

// int main() {

//     // 根据参数生成指令
//     int insn_count = 3; // 生成两个load指令 
//     MemIns instruction[insn_count]; // 指令结构体，用于赋值指令操作数
//     Instruct_DataType instruction_data[insn_count]; // 指令数据，用于传输指令给SAA

//     //第一个指令
//     instruction[0].opcode = OPCODE_LOAD;        // LOAD操作码3位
//     instruction[0].buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区3位
//     instruction[0].dram_base = 0;               // DRAM基地址32位
//     instruction[0].buffer_base = 0;             // buffer基地址16位
//     instruction[0].y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[0].x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[0].x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位
//     printBinary_instruction(instruction[0]);    // 输出指令的二进制

//     //第二个指令
//     instruction[1].opcode = OPCODE_LOAD;        // LOAD操作码3位
//     instruction[1].buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区3位
//     instruction[1].dram_base = 0;               // DRAM基地址32位
//     instruction[1].buffer_base = 4;             // buffer基地址16位 ,从第四行开始
//     instruction[1].y_size = MATRIX_WIDTH-1;       // 加载MATRIX_WIDTH行16位
//     instruction[1].x_size = MATRIX_WIDTH-1;       // 加载MATRIX_WIDTH列16位
//     instruction[1].x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位
//     printBinary_instruction(instruction[1]);    // 输出指令的二进制

//    //第三个指令
//     instruction[2].opcode = OPCODE_LOAD;        // LOAD操作码3位
//     instruction[2].buffer_id = WEIGHT_BUFFER_ID; // 存入输入缓冲区3位
//     instruction[2].dram_base = 0;               // DRAM基地址32位
//     instruction[2].buffer_base = 0;             // buffer基地址16位 ,从第四行开始
//     instruction[2].y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[2].x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[2].x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位
//     printBinary_instruction(instruction[2]);    // 输出指令的二进制

//     // 转换为指令类型，并输出查看
//     for (int i = 0; i < insn_count; i++) 
//     {
//         std::memcpy(&instruction_data[i], &instruction[i], sizeof(MemIns)); //转换指令结构体为128位指令数据类型
//         printBinary(instruction_data[i],INSTRUCT_WIDTH);                 // 输出指令的二进制
//         printf("\n");
//     }

//     // 生成输入矩阵
//     const int ROWS = MATRIX_WIDTH; // 矩阵行数
//     const int COLS = MATRIX_WIDTH; // 矩阵列数
//     Input_DataType **inputs_matrix= init_matrix<Input_DataType>(ROWS,COLS); //生成矩阵
//     print_matrix(inputs_matrix, ROWS, COLS); // 打印生成的矩阵
     
//     // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输
//     Input_DataType *inputs_matrix1 = *inputs_matrix;
//     print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
//     Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
//     print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组

//     // 执行saa_top硬件，加载指令，只执行load函数，注意，传入的是指针类型
//     Transfer_DataType* output; // 输出给总线
//     saa_top(insn_count,instruction_data,transfer_inputs_matrix, input_buffer, weight_buffer,output); 
    
//     //检查缓冲区
//     print_buffer(input_buffer,0,10);
//     print_buffer(weight_buffer,0,10);


//     return 0;
// }



// // 4 测试load和compute指令，计算一个分块大小的矩阵
// #include <iostream>

// int main() {

//     // 根据参数生成指令

//     int insn_count = 5; // 生成两个load指令 
//     SAAInsn instruction[insn_count]; // 指令独联体，用于赋值指令操作数，通用
//     Instruct_DataType instruction_data[insn_count]; // 指令数据，用于传输指令给SAA
    
//     //第一个指令
//     instruction[0].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
//     instruction[0].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[0].mem.dram_base = 0;               // DRAM索引 32位
//     instruction[0].mem.buffer_base = 0;             // buffer行索引 16位
//     instruction[0].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[0].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[0].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位
//     printBinary_instruction(instruction[0]);    // 输出指令的二进制

//     //第二个指令
//     instruction[1].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
//     instruction[1].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[1].mem.dram_base = 0;               // DRAM索引偏移1个总线位宽，以总线为步长
//     instruction[1].mem.buffer_base = 4;             // buffer行索引 16位 ,从第四行开始
//     instruction[1].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[1].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[1].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位

//    //第三个指令
//     instruction[2].mem.opcode = OPCODE_LOAD;        // LOAD操作码3位
//     instruction[2].mem.buffer_id = WEIGHT_BUFFER_ID; // 存入输入缓冲区3位
//     instruction[2].mem.dram_base = 0;               // DRAM基地址32位
//     instruction[2].mem.buffer_base = 0;             // buffer基地址16位 ,从第四行开始
//     instruction[2].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[2].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[2].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位

//    //第四个指令（权重预加载指令）
//     instruction[3].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
//     instruction[3].com.compute_type = WEIGHT_PRELOAD;  // 权重预加载 3位
//     instruction[3].com.weigth_addr = 0;                // 权重读取起始行 32位
//     instruction[3].com.input_addr = 0;                 // 没用到，因为是权重预加载指令
//     instruction[3].com.output_addr = 0;                // 没用到，因为是权重预加载指令
//     instruction[3].com.weight_switch = 0;              // 使用第一个权重加载
//     instruction[3].com.compute_switch = 0;             // 没用到，因为是权重预加载指令
//     instruction[3].com.accumulate = 0;                 // 没用到，因为是权重预加载指令

//    //第五个指令（计算指令）
//     instruction[4].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
//     instruction[4].com.compute_type = COMPUTE;         // 计算指令 3位
//     instruction[4].com.weigth_addr = 0;                // 没用到，因为是权重预加载指令
//     instruction[4].com.input_addr = 0;                 // 输入读取起始行
//     instruction[4].com.output_addr = 0;                // 输出写入起始行
//     instruction[4].com.weight_switch = 0;              // 没用到，因为是权重预加载指令
//     instruction[4].com.compute_switch = 0;             // 使用前面的权重预加载寄存器计算
//     instruction[4].com.accumulate = 0;                 // 第一次计算不累加，刷新累加器

//     // 转换为指令类型，并输出查看

//     for (int i = 0; i < insn_count; i++) 
//     {
//         std::memcpy(&instruction_data[i], &instruction[i], sizeof(MemIns)); //转换指令结构体为128位指令数据类型
//         // printBinary(instruction_data[i],INSTRUCT_WIDTH);                 // 输出指令的二进制
//         // printf("\n");
//     }

//     // 生成输入矩阵和权重矩阵

//     const int ROWS = MATRIX_WIDTH; // 矩阵行数
//     const int COLS = MATRIX_WIDTH; // 公共维度
//     const int COL1S = MATRIX_WIDTH; // 矩阵列数
//     Input_DataType **inputs_matrix = init_matrix<Input_DataType>(ROWS,COLS); //输入矩阵
//     Weight_DataType **weights_matrix = init_matrix<Weight_DataType>(COL1S,COLS); //权重矩阵
//     Output_DataType **ouputs_matrix = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                       (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
//     printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//     printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//     printf("ouputs_matrix:\n");print_matrix(ouputs_matrix, ROWS, COL1S); // 打印输出矩阵

//     // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输

//     Input_DataType *inputs_matrix1 = *inputs_matrix;
//     Weight_DataType *weights_matrix1 = *weights_matrix;
//     Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
//     Transfer_DataType *transfer_weights_matrix= (Transfer_DataType*)weights_matrix1;
//     print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
//     print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组


//     // 执行saa_top硬件，加载指令，只执行load函数，注意，传入的是指针类型

//     Transfer_DataType* output; // 输出给总线
//     saa_top(insn_count,
//         instruction_data,
//         transfer_inputs_matrix, 
//         transfer_weights_matrix, 
//         input_buffer, 
//         weight_buffer,
//         output_buffer,
//         output); 
    
//     //检查缓冲区

//     printf("input_buffer:\n");print_buffer(input_buffer,0,10);
//     printf("weight_buffer:\n");print_buffer(weight_buffer,0,10);
//     printf("output_buffer:\n");print_buffer(output_buffer,0,10);

//     return 0;
// }



// // 5 测试分配内存
// #include <windows.h>
// #include <iostream>

// // 缓冲区管理结构
// struct BufferManager {
//     void* base;    // 缓冲区基地址
//     size_t size;   // 缓冲区总大小
//     size_t offset; // 当前分配到的偏移量，是递增的
// };

// // 分配缓冲区
// void* allocate_buffer(size_t size) {
//     void* buffer = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
//     if (buffer == NULL) {
//         std::cerr << "Error: Memory allocation failed." << std::endl;
//         return NULL;
//     }
//     return buffer;
// }

// // 在缓冲区中分配矩阵空间
// void* allocate_matrix_in_buffer(BufferManager& manager, size_t matrix_size, size_t alignment) {
//     size_t aligned_offset = (manager.offset + alignment - 1) & ~(alignment - 1);
//     if (aligned_offset + matrix_size > manager.size) {
//         std::cerr << "Error: Buffer overflow." << std::endl;
//         return NULL;
//     }

//     void* matrix_addr = static_cast<char*>(manager.base) + aligned_offset;
//     manager.offset = aligned_offset + matrix_size; // 更新偏移量
//     return matrix_addr;
// }

// // 释放缓冲区
// void free_buffer(void* buffer) {
//     if (buffer) {
//         VirtualFree(buffer, 0, MEM_RELEASE);
//     }
// }

// int main() {
//     // 假设总线宽度为64字节
//     const size_t bus_width = 64;
    
//     // 假设我们需要的缓冲区大小
//     const size_t buffer_size = 4 * 1024 * 1024; // 4MB

//     // 创建缓冲区管理器
//     BufferManager buffer_manager;
//     buffer_manager.base = allocate_buffer(buffer_size);
//     buffer_manager.size = buffer_size;
//     buffer_manager.offset = 0;

//     if (buffer_manager.base == NULL) {
//         return 1;
//     }

//     // 打印分配的内存地址
//     std::cout << "Allocated memory address: " << buffer << std::endl;

//     const int ROWS = MATRIX_WIDTH; // 矩阵行数
//     const int COLS = MATRIX_WIDTH; // 公共维度
//     const int COL1S = MATRIX_WIDTH; // 矩阵列数

//     // 分配矩阵空间
//     size_t input_matrix_size = ROWS * COLS; // 输入矩阵大小
//     size_t weight_matrix_size = COLS * COL1S; // 权重矩阵大小

//     void* input_matrix = allocate_matrix_in_buffer(buffer_manager, input_matrix_size, bus_width);
//     void* weight_matrix = allocate_matrix_in_buffer(buffer_manager, weight_matrix_size, bus_width);

//     // 打印分配的内存地址
//     std::cout << "Allocated memory address: " << buffer << std::endl;

    
//     if (input_matrix == NULL || weight_matrix == NULL) {
//         free_buffer(buffer_manager.base);
//         return 1;
//     }

//     // ... 使用input_matrix和weight_matrix...

//     // 清理资源
//     free_buffer(buffer_manager.base);
    
//     return 0;
// }



// // 6 输入矩阵封装为自动生成指令，测试load和compute指令
// #include <iostream>

// //总的指令缓冲区，可以容纳1000条指令
// #define INSTRUCTION_BUFFER_WEIGHT 1000

// // 统计指令信息的结构体
// struct InstructionStruct {
//     Instruct_DataType instruction_data[INSTRUCTION_BUFFER_WEIGHT]; // 指令数据缓冲区，用于传输指令给SAA
//     int total_count; // 总共生成了多少指令
// };

// // 生成加载指令的函数，根据当前矩阵的行列生成一批加载指令，暂时只能加载MATRIX_WIDTH倍数的矩阵
// void generate_load_instructions(
//     InstructionStruct* instruction_struct, // 传入结构体的地址
//     int total_rows, // 总行数
//     int total_cols, // 总列数
//     Buffer_Id_DataType buffer_id, // 当前矩阵加载到哪个缓冲区
//     Dram_Addr_DataType dram_start, // 读取位置相对于缓冲区的偏移，如果矩阵直接就从0存储，那这就是0，以总线为偏移基本单位
//     Buffer_Addr_DataType buffer_start) // 写入位置相对于buffer起始行的偏移，如果直接从0行存储，那就是0
// {
//     // 计算分多少个块，就生成多少个指令
//     const int row_block = total_rows / MATRIX_WIDTH; // 行分多少个块
//     const int col_block = total_cols / MATRIX_WIDTH; // 列分多少个块
//     int insn_count =  row_block * col_block;  // 总的分块数等于行*列
//     SAAInsn instruction[insn_count]; // 使用总的分块数生成指令数组

//      // 计算当前结构体指针的赋值位置

//     //循环块生成加载指令
//     for (int row = 0; row < row_block; ++row) {
//         for (int col = 0; col < col_block; ++col) {
//             const int block = row*col_block+col; // 第几个块
//             const int buffer_base = (buffer_start + (block)*MATRIX_WIDTH); // 计算buffer输入地址
//             const int dram_base = dram_start + row * col_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                              + col * MATRIX_WIDTH; // 计算dram读取地址
//             // 写入指令结构体
//             instruction[block].mem.opcode = OPCODE_LOAD;  // 加载指令
//             instruction[block].mem.buffer_id = buffer_id; // 存储在哪个缓冲区
//             instruction[block].mem.dram_base = dram_base; // 缓冲区偏移+矩阵内部偏移
//             instruction[block].mem.buffer_base = buffer_base; // buffer偏移+矩阵内部偏移
//             instruction[block].mem.y_size = MATRIX_WIDTH; // 每次加载MATRIX_WIDTH行
//             instruction[block].mem.x_size = MATRIX_WIDTH; // 每次加载MATRIX_WIDTH列
//             instruction[block].mem.x_stride = total_cols; // 假设每行数据在DRAM中是连续存储的，那么步长就是列宽

//             // 转换指令结构体为128位指令数据类型
//             std::memcpy(&instruction_struct->instruction_data[instruction_struct->total_count+block], 
//                         &instruction[block], sizeof(SAAInsn));
//         }
//     }
//     instruction_struct->total_count = instruction_struct->total_count + insn_count ; // 计算当前总指令数
// }


// int main() {

//     // 生成输入矩阵和权重矩阵
//     const int ROWS = 2*MATRIX_WIDTH; // 矩阵行数
//     const int COLS = 2*MATRIX_WIDTH; // 公共维度
//     const int COL1S = 2*MATRIX_WIDTH; // 矩阵列数
//     Input_DataType **inputs_matrix = init_matrix<Input_DataType>(ROWS,COLS); //输入矩阵
//     Weight_DataType **weights_matrix = init_matrix<Weight_DataType>(COL1S,COLS); //权重矩阵
//     Output_DataType **ouputs_matrix = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                       (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
//     printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//     printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//     printf("ouputs_matrix:\n");print_matrix(ouputs_matrix, ROWS, COL1S); // 打印输出矩阵

//     // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输

//     Input_DataType *inputs_matrix1 = *inputs_matrix;
//     Weight_DataType *weights_matrix1 = *weights_matrix;
//     Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
//     Transfer_DataType *transfer_weights_matrix= (Transfer_DataType*)weights_matrix1;
//     print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
//     print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组

//     // 根据矩阵参数生成指令
//     InstructionStruct instruction_struct = { 0 }; // 生成指令缓冲区结构体
//     generate_load_instructions(&instruction_struct,ROWS,
//                                COLS,INPUT_BUFFER_ID,0,0); // 生成指令缓冲区的load指令
//     generate_load_instructions(&instruction_struct,COLS,
//                                COL1S,WEIGHT_BUFFER_ID,0,0); // 生成权重缓冲区的load指令
//     printf("total_count:%d\n",instruction_struct.total_count); // 查看生成的指令数

//     for (int i = 0; i < instruction_struct.total_count; i++) 
//         print_instruction(instruction_struct.instruction_data[i],"MEMORY"); //查看生成的LOAD指令

//     // 执行saa_top硬件，加载指令，只执行load函数，注意，传入的是指针类型

//     Transfer_DataType output[ROWS*COL1S]; // 输出数组

//     saa_top(instruction_struct.total_count,
//         instruction_struct.instruction_data,
//         transfer_inputs_matrix, 
//         transfer_weights_matrix, 
//         input_buffer, 
//         weight_buffer,
//         output_buffer,
//         output); 
    
//     //检查缓冲区

//     printf("input_buffer:\n");print_buffer(input_buffer,0,10);
//     printf("weight_buffer:\n");print_buffer(weight_buffer,0,10);
//     printf("output_buffer:\n");print_buffer(output_buffer,0,10);

//     return 0;
// }


// // 7 输入矩阵封装为自动生成指令，测试load和compute指令以及store指令
// #include <iostream>

// //总的指令缓冲区，可以容纳1000条指令
// #define INSTRUCTION_BUFFER_WEIGHT 1000

// // 统计指令信息的结构体
// struct InstructionStruct {
//     Instruct_DataType instruction_data[INSTRUCTION_BUFFER_WEIGHT]; // 指令数据缓冲区，用于传输指令给SAA
//     int total_count; // 总共生成了多少指令
// };

// // 生成加载指令的函数，根据当前矩阵的行列生成一批加载指令，暂时只能加载MATRIX_WIDTH倍数的矩阵
// void generate_load_instructions(
//     InstructionStruct* instruction_struct, // 传入结构体的地址
//     int total_rows, // 总行数
//     int total_cols, // 总列数
//     Buffer_Id_DataType buffer_id, // 当前矩阵加载到哪个缓冲区
//     Dram_Addr_DataType dram_start, // 读取位置相对于缓冲区的偏移，如果矩阵直接就从0存储，那这就是0，以总线为偏移基本单位
//     Buffer_Addr_DataType buffer_start) // 写入位置相对于buffer起始行的偏移，如果直接从0行存储，那就是0
// {
//     // 计算分多少个块，就生成多少个指令
//     const int row_block = total_rows / MATRIX_WIDTH; // 行分多少个块
//     const int col_block = total_cols / MATRIX_WIDTH; // 列分多少个块
//     int insn_count =  row_block * col_block;  // 总的分块数等于行*列
//     SAAInsn instruction[insn_count]; // 使用总的分块数生成指令数组

//      // 计算当前结构体指针的赋值位置

//     //循环块生成加载指令
//     for (int row = 0; row < row_block; ++row) {
//         for (int col = 0; col < col_block; ++col) {
//             const int block = row*col_block+col; // 第几个块
//             const int buffer_base = (buffer_start + (block)*MATRIX_WIDTH); // 计算buffer输入地址
//             const int dram_base = dram_start + row * col_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                              + col * MATRIX_WIDTH; // 计算dram读取地址
//             // 写入指令结构体
//             instruction[block].mem.opcode = OPCODE_LOAD;  // 加载指令
//             instruction[block].mem.buffer_id = buffer_id; // 存储在哪个缓冲区
//             instruction[block].mem.dram_base = dram_base; // 缓冲区偏移+矩阵内部偏移
//             instruction[block].mem.buffer_base = buffer_base; // buffer偏移+矩阵内部偏移
//             instruction[block].mem.y_size = MATRIX_WIDTH; // 每次加载MATRIX_WIDTH行
//             instruction[block].mem.x_size = MATRIX_WIDTH; // 每次加载MATRIX_WIDTH列
//             instruction[block].mem.x_stride = total_cols; // 假设每行数据在DRAM中是连续存储的，那么步长就是列宽

//             // 转换指令结构体为128位指令数据类型
//             std::memcpy(&instruction_struct->instruction_data[instruction_struct->total_count+block], 
//                         &instruction[block], sizeof(SAAInsn));
//         }
//     }
//     instruction_struct->total_count = instruction_struct->total_count + insn_count ; // 计算当前总指令数
// }


// int main() {

//     // 生成输入矩阵和权重矩阵
//     const int ROWS = 2*MATRIX_WIDTH; // 矩阵行数
//     const int COLS = 2*MATRIX_WIDTH; // 公共维度
//     const int COL1S = 2*MATRIX_WIDTH; // 矩阵列数
//     Input_DataType **inputs_matrix = init_matrix<Input_DataType>(ROWS,COLS); //输入矩阵
//     Weight_DataType **weights_matrix = init_matrix<Weight_DataType>(COL1S,COLS); //权重矩阵
//     Output_DataType **ouputs_matrix = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                       (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
//     printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//     printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//     printf("ouputs_matrix:\n");print_matrix(ouputs_matrix, ROWS, COL1S); // 打印输出矩阵

//     // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输

//     Input_DataType *inputs_matrix1 = *inputs_matrix;
//     Weight_DataType *weights_matrix1 = *weights_matrix;
//     Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
//     Transfer_DataType *transfer_weights_matrix= (Transfer_DataType*)weights_matrix1;
//     print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
//     print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组

//     // 根据参数生成指令
//     int insn_count = 6; // 生成两个load指令 
//     SAAInsn instruction[insn_count]; // 指令独联体，用于赋值指令操作数，通用
//     Instruct_DataType instruction_data[insn_count]; // 指令数据，用于传输指令给SAA
    
//     //第一个指令
//     instruction[0].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
//     instruction[0].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[0].mem.dram_base = 0;               // DRAM索引 32位
//     instruction[0].mem.buffer_base = 0;             // buffer行索引 16位
//     instruction[0].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[0].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[0].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位
//     printBinary_instruction(instruction[0]);    // 输出指令的二进制

//     //第二个指令
//     instruction[1].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
//     instruction[1].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[1].mem.dram_base = 0;               // DRAM索引偏移1个总线位宽，以总线为步长
//     instruction[1].mem.buffer_base = 4;             // buffer行索引 16位 ,从第四行开始
//     instruction[1].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[1].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[1].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位

//    //第三个指令
//     instruction[2].mem.opcode = OPCODE_LOAD;        // LOAD操作码3位
//     instruction[2].mem.buffer_id = WEIGHT_BUFFER_ID; // 存入输入缓冲区3位
//     instruction[2].mem.dram_base = 0;               // DRAM基地址32位
//     instruction[2].mem.buffer_base = 0;             // buffer基地址16位 ,从第四行开始
//     instruction[2].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[2].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[2].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位

//    //第四个指令（权重预加载指令）
//     instruction[3].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
//     instruction[3].com.compute_type = WEIGHT_PRELOAD;  // 权重预加载 3位
//     instruction[3].com.weigth_addr = 0;                // 权重读取起始行 32位
//     instruction[3].com.input_addr = 0;                 // 没用到，因为是权重预加载指令
//     instruction[3].com.output_addr = 0;                // 没用到，因为是权重预加载指令
//     instruction[3].com.weight_switch = 0;              // 使用第一个权重加载
//     instruction[3].com.compute_switch = 0;             // 没用到，因为是权重预加载指令
//     instruction[3].com.accumulate = 0;                 // 没用到，因为是权重预加载指令

//    //第五个指令（计算指令）
//     instruction[4].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
//     instruction[4].com.compute_type = COMPUTE;         // 计算指令 3位
//     instruction[4].com.weigth_addr = 0;                // 没用到，因为是权重预加载指令
//     instruction[4].com.input_addr = 0;                 // 输入读取起始行
//     instruction[4].com.output_addr = 0;                // 输出写入起始行
//     instruction[4].com.weight_switch = 0;              // 没用到，因为是权重预加载指令
//     instruction[4].com.compute_switch = 0;             // 使用前面的权重预加载寄存器计算
//     instruction[4].com.accumulate = 0;                 // 第一次计算不累加，刷新累加器

//    //第六个指令（存储指令）
//     instruction[5].mem.opcode = OPCODE_STORE;        // STORE操作码3位
//     instruction[5].mem.buffer_id = 0;                //没用到，默认是以输出buffer输出 3位
//     instruction[5].mem.dram_base = 0;               // DRAM基地址32位
//     instruction[5].mem.buffer_base = 0;             // buffer基地址16位 
//     instruction[5].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[5].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[5].mem.x_stride = 2*MATRIX_WIDTH;     // 假设步进与列数相同16位

//     // 转换为指令类型，并输出查看

//     for (int i = 0; i < insn_count; i++) 
//     {
//         std::memcpy(&instruction_data[i], &instruction[i], sizeof(MemIns)); //转换指令结构体为128位指令数据类型
//         // printBinary(instruction_data[i],INSTRUCT_WIDTH);                 // 输出指令的二进制
//         // printf("\n");
//     }
    
//     // 定义输出一维数组，注意自定义类型无法自动初始化为0，除非定义为静态变量或者全局变量
//     static Output_DataType outputs_matrix_hw[ROWS*COL1S] = {0}; // 输出数组，注意应该以Output_DataType定义
//     Transfer_DataType *transfer_outputs_matrix =(Transfer_DataType *) outputs_matrix_hw;// 转换为Transfer_DataType类型给SAA

//     // 执行saa_top硬件，加载指令，只执行load函数，注意，传入的是指针类型
//     saa_top(insn_count,
//         instruction_data,
//         transfer_inputs_matrix, 
//         transfer_weights_matrix, 
//         transfer_outputs_matrix,
//         input_buffer, 
//         weight_buffer,
//         output_buffer); 
    
//     //检查缓冲区
//     printf("input_buffer:\n");print_buffer(input_buffer,0,10);
//     printf("weight_buffer:\n");print_buffer(weight_buffer,0,10);
//     printf("output_buffer:\n");print_buffer(output_buffer,0,10);

//     // 检查输出
//     print_vec((Output_DataType*)transfer_outputs_matrix,ROWS*COL1S); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组
//     printf("outputs_matrix_hw:\n");print1D_2DArray(outputs_matrix_hw,ROWS,COL1S); // 将一维数组按二维打印

//     return 0;
// }


// // 7 测试load和compute指令以及store指令的自动生成
// #include <iostream>

// int main() {

//     // 生成输入矩阵和权重矩阵
//     const int ROWS = MATRIX_WIDTH; // 矩阵行数
//     const int COLS = MATRIX_WIDTH; // 公共维度
//     const int COL1S = MATRIX_WIDTH; // 矩阵列数
//     Input_DataType **inputs_matrix = init_matrix<Input_DataType>(ROWS,COLS); //输入矩阵
//     Weight_DataType **weights_matrix = init_matrix<Weight_DataType>(COL1S,COLS); //权重矩阵
//     Output_DataType **ouputs_matrix = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                       (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
//     printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//     printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//     printf("ouputs_matrix:\n");print_matrix(ouputs_matrix, ROWS, COL1S); // 打印输出矩阵

//     // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输

//     Input_DataType *inputs_matrix1 = *inputs_matrix;
//     Weight_DataType *weights_matrix1 = *weights_matrix;
//     Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
//     Transfer_DataType *transfer_weights_matrix= (Transfer_DataType*)weights_matrix1;
//     print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
//     print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组

//     // 根据参数生成指令
//     int insn_count = 7; // 生成两个load指令 
//     SAAInsn instruction[insn_count]; // 指令独联体，用于赋值指令操作数，通用
//     Instruct_DataType instruction_data[insn_count]; // 指令数据，用于传输指令给SAA
    
//     //第一个指令
//     instruction[0].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
//     instruction[0].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[0].mem.dram_base = 0;               // DRAM索引 32位
//     instruction[0].mem.buffer_base = 0;             // buffer行索引 16位
//     instruction[0].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[0].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[0].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位
//     printBinary_instruction(instruction[0]);    // 输出指令的二进制

//     //第二个指令
//     instruction[1].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
//     instruction[1].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[1].mem.dram_base = 0;               // DRAM索引偏移1个总线位宽，以总线为步长
//     instruction[1].mem.buffer_base = 4;             // buffer行索引 16位 ,从第四行开始
//     instruction[1].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[1].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[1].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位

//    //第三个指令
//     instruction[2].mem.opcode = OPCODE_LOAD;        // LOAD操作码3位
//     instruction[2].mem.buffer_id = WEIGHT_BUFFER_ID; // 存入输入缓冲区3位
//     instruction[2].mem.dram_base = 0;               // DRAM基地址32位
//     instruction[2].mem.buffer_base = 0;             // buffer基地址16位 ,从第四行开始
//     instruction[2].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[2].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[2].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位

//    //第四个指令（权重预加载指令）
//     instruction[3].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
//     instruction[3].com.compute_type = WEIGHT_PRELOAD;  // 权重预加载 3位
//     instruction[3].com.weigth_addr = 0;                // 权重读取起始行 32位
//     instruction[3].com.input_addr = 0;                 // 没用到，因为是权重预加载指令
//     instruction[3].com.output_addr = 0;                // 没用到，因为是权重预加载指令
//     instruction[3].com.weight_switch = 0;              // 使用第一个权重加载
//     instruction[3].com.compute_switch = 0;             // 没用到，因为是权重预加载指令
//     instruction[3].com.accumulate = 0;                 // 没用到，因为是权重预加载指令

//    //第五个指令（计算指令）
//     instruction[4].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
//     instruction[4].com.compute_type = COMPUTE;         // 计算指令 3位
//     instruction[4].com.weigth_addr = 0;                // 没用到，因为是权重预加载指令
//     instruction[4].com.input_addr = 0;                 // 输入读取起始行
//     instruction[4].com.output_addr = 0;                // 输出写入起始行
//     instruction[4].com.weight_switch = 0;              // 没用到，因为是权重预加载指令
//     instruction[4].com.compute_switch = 0;             // 使用前面的权重预加载寄存器计算
//     instruction[4].com.accumulate = 0;                 // 第一次计算不累加，刷新累加器

//    //第六个指令（存储指令）
//     instruction[5].mem.opcode = OPCODE_STORE;        // STORE操作码3位
//     instruction[5].mem.buffer_id = 0;                //没用到，默认是以输出buffer输出 3位
//     instruction[5].mem.dram_base = 0;               // DRAM基地址32位
//     instruction[5].mem.buffer_base = 0;             // buffer基地址16位 
//     instruction[5].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[5].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[5].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位

//    //第七个指令（结束指令）
//     instruction[6].mem.opcode = OPCODE_DONE;        // STORE操作码3位
//     instruction[6].mem.buffer_id = 0;                //没用到，默认是以输出buffer输出 3位
//     instruction[6].mem.dram_base = 0;               // DRAM基地址32位
//     instruction[6].mem.buffer_base = 0;             // buffer基地址16位 
//     instruction[6].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[6].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[6].mem.x_stride = 2*MATRIX_WIDTH;     // 假设步进与列数相同16位


//     // 转换为指令类型，并输出查看

//     for (int i = 0; i < insn_count; i++) 
//     {
//         std::memcpy(&instruction_data[i], &instruction[i], sizeof(MemIns)); //转换指令结构体为128位指令数据类型
//         printf("instruction_data:\n%d",(int)instruction_data[i]);
//         // printBinary(instruction_data[i],INSTRUCT_WIDTH);                 // 输出指令的二进制
//         // printf("\n");
//     }

    
//     // 定义输出一维数组，注意自定义类型无法自动初始化为0，除非定义为静态变量或者全局变量
//     static Output_DataType outputs_matrix_hw[ROWS*COL1S] = {0}; // 输出数组，注意应该以Output_DataType定义
//     Transfer_DataType *transfer_outputs_matrix =(Transfer_DataType *) outputs_matrix_hw;// 转换为Transfer_DataType类型给SAA

//     // 执行saa_top硬件，加载指令，只执行load函数，注意，传入的是指针类型
//     saa_top(
//         insn_count,
//         instruction_data,
//         transfer_inputs_matrix, 
//         transfer_weights_matrix, 
//         transfer_outputs_matrix,
//         input_buffer, 
//         weight_buffer,
//         output_buffer); 

//     //检查缓冲区
//     printf("input_buffer:\n");print_buffer(input_buffer,0,10);
//     printf("weight_buffer:\n");print_buffer(weight_buffer,0,10);
//     printf("output_buffer:\n");print_buffer(output_buffer,0,10);

//     // 检查输出
//     print_vec((Output_DataType*)transfer_outputs_matrix,ROWS*COL1S); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组
//     printf("outputs_matrix_hw:\n");print1D_2DArray(outputs_matrix_hw,ROWS,COL1S); // 将一维数组按二维打印

//     return 0;
// }




// // 7 环回测试load和store
// #include <iostream>

// int main() {

//     // 生成输入矩阵和权重矩阵
//     const int ROWS = MATRIX_WIDTH; // 矩阵行数
//     const int COLS = MATRIX_WIDTH; // 公共维度
//     const int COL1S = MATRIX_WIDTH; // 矩阵列数
//     Input_DataType **inputs_matrix = init_matrix<Input_DataType>(ROWS,COLS); //输入矩阵
//     Weight_DataType **weights_matrix = init_matrix<Weight_DataType>(COL1S,COLS); //权重矩阵
//     Output_DataType **ouputs_matrix = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                       (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
//     printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//     printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//     printf("ouputs_matrix:\n");print_matrix(ouputs_matrix, ROWS, COL1S); // 打印输出矩阵

//     // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输

//     Input_DataType *inputs_matrix1 = *inputs_matrix;
//     Weight_DataType *weights_matrix1 = *weights_matrix;
//     Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
//     Transfer_DataType *transfer_weights_matrix= (Transfer_DataType*)weights_matrix1;
//     print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
//     print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组

//     // 根据参数生成指令
//     int insn_count = 4; // 生成两个load指令 
//     SAAInsn instruction[insn_count]; // 指令独联体，用于赋值指令操作数，通用
//     Instruct_DataType instruction_data[insn_count]; // 指令数据，用于传输指令给SAA
    
//     //第一个指令
//     instruction[0].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
//     instruction[0].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[0].mem.dram_base = 0;               // DRAM索引 32位
//     instruction[0].mem.buffer_base = 0;             // buffer行索引 16位
//     instruction[0].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[0].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[0].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位
//     printBinary_instruction(instruction[0]);    // 输出指令的二进制

//     //第二个指令
//     instruction[1].mem.opcode = OPCODE_STORE;        // LOAD操作码 3位
//     instruction[1].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[1].mem.dram_base = 0;               // DRAM索引偏移1个总线位宽，以总线为步长
//     instruction[1].mem.buffer_base = 0;             // buffer行索引 16位 ,从第四行开始
//     instruction[1].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[1].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[1].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位

//    //第三个指令
//     instruction[2].mem.opcode = OPCODE_LOAD;        // LOAD操作码3位
//     instruction[2].mem.buffer_id = WEIGHT_BUFFER_ID; // 存入输入缓冲区3位
//     instruction[2].mem.dram_base = 0;               // DRAM基地址32位
//     instruction[2].mem.buffer_base = 0;             // buffer基地址16位 ,从第四行开始
//     instruction[2].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[2].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[2].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位

//    //第六个指令（存储指令）
//     instruction[3].mem.opcode = OPCODE_STORE;        // STORE操作码3位
//     instruction[3].mem.buffer_id = WEIGHT_BUFFER_ID;  //没用到，默认是以输出buffer输出 3位
//     instruction[3].mem.dram_base = 16;               // DRAM基地址32位
//     instruction[3].mem.buffer_base = 0;             // buffer基地址16位 
//     instruction[3].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[3].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[3].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位

 

//     // 转换为指令类型，并输出查看

//     for (int i = 0; i < insn_count; i++) 
//     {
//         std::memcpy(&instruction_data[i], &instruction[i], sizeof(MemIns)); //转换指令结构体为128位指令数据类型
//         printf("instruction_data:\n%d",(int)instruction_data[i]);
//         // printBinary(instruction_data[i],INSTRUCT_WIDTH);                 // 输出指令的二进制
//         // printf("\n");
//     }

    
//     // 定义输出一维数组，注意自定义类型无法自动初始化为0，除非定义为静态变量或者全局变量
//     static Input_DataType outputs_matrix_hw[(ROWS+2)*COL1S] = {0}; // 输出数组，注意应该以Output_DataType定义
//     Transfer_DataType *transfer_outputs_matrix =(Transfer_DataType *) outputs_matrix_hw;// 转换为Transfer_DataType类型给SAA

//     // 执行saa_top硬件，加载指令，只执行load函数，注意，传入的是指针类型
//     saa_top(
//         insn_count,
//         instruction_data,
//         transfer_inputs_matrix, 
//         transfer_weights_matrix, 
//         transfer_outputs_matrix,
//         input_buffer, 
//         weight_buffer,
//         output_buffer); 

//     //检查缓冲区
//     printf("input_buffer:\n");print_buffer(input_buffer,0,10);
//     printf("weight_buffer:\n");print_buffer(weight_buffer,0,10);
//     printf("output_buffer:\n");print_buffer(output_buffer,0,10);

//     // 检查输出
//     print_vec((Input_DataType*)transfer_outputs_matrix,(ROWS+2)*COL1S); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组
//     printf("outputs_matrix_hw:\n");print1D_2DArray(outputs_matrix_hw,(ROWS+2),COL1S); // 将一维数组按二维打印

//     return 0;
// }


// // 7 测试使用数据依赖完成并行，比较使用依赖和不适用依赖进行并行是否会造成执行错误
// #include <iostream>

// int main() {

//     return 0;
// }



// // 10 测试saa_top IP使用，不打印buffer
// #include <iostream>

// int main() {

//     // 生成输入矩阵和权重矩阵
//     const int ROWS = MATRIX_WIDTH; // 矩阵行数
//     const int COLS = MATRIX_WIDTH; // 公共维度
//     const int COL1S = MATRIX_WIDTH; // 矩阵列数
//     Input_DataType **inputs_matrix = init_matrix<Input_DataType>(ROWS,COLS); //输入矩阵
//     Weight_DataType **weights_matrix = init_matrix<Weight_DataType>(COL1S,COLS); //权重矩阵
//     Output_DataType **ouputs_matrix = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                       (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
//     printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//     printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//     printf("ouputs_matrix:\n");print_matrix(ouputs_matrix, ROWS, COL1S); // 打印输出矩阵

//     // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输

//     Input_DataType *inputs_matrix1 = *inputs_matrix;
//     Weight_DataType *weights_matrix1 = *weights_matrix;
//     Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
//     Transfer_DataType *transfer_weights_matrix= (Transfer_DataType*)weights_matrix1;
//     print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
//     print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组

//     // 根据参数生成指令
//     int insn_count = 7; // 生成两个load指令 
//     SAAInsn instruction[insn_count]; // 指令独联体，用于赋值指令操作数，通用
//     Instruct_DataType instruction_data[insn_count]; // 指令数据，用于传输指令给SAA
    
//     //第一个指令
//     instruction[0].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
//     instruction[0].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[0].mem.dram_base = 0;               // DRAM索引 32位
//     instruction[0].mem.buffer_base = 0;             // buffer行索引 16位
//     instruction[0].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[0].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[0].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位
//     printBinary_instruction(instruction[0]);    // 输出指令的二进制

//     //第二个指令
//     instruction[1].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
//     instruction[1].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
//     instruction[1].mem.dram_base = 0;               // DRAM索引偏移1个总线位宽，以总线为步长
//     instruction[1].mem.buffer_base = 4;             // buffer行索引 16位 ,从第四行开始
//     instruction[1].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
//     instruction[1].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
//     instruction[1].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位


// //    //第三个指令
// //     instruction[2].mem.opcode = OPCODE_LOAD;        // LOAD操作码3位
// //     instruction[2].mem.buffer_id = WEIGHT_BUFFER_ID; // 存入输入缓冲区3位
// //     instruction[2].mem.dram_base = 0;               // DRAM基地址32位
// //     instruction[2].mem.buffer_base = 0;             // buffer基地址16位 ,从第四行开始
// //     instruction[2].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
// //     instruction[2].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
// //     instruction[2].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位

// //    //第四个指令（权重预加载指令）
// //     instruction[3].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
// //     instruction[3].com.compute_type = WEIGHT_PRELOAD;  // 权重预加载 3位
// //     instruction[3].com.weigth_addr = 0;                // 权重读取起始行 32位
// //     instruction[3].com.input_addr = 0;                 // 没用到，因为是权重预加载指令
// //     instruction[3].com.output_addr = 0;                // 没用到，因为是权重预加载指令
// //     instruction[3].com.weight_switch = 0;              // 使用第一个权重加载
// //     instruction[3].com.compute_switch = 0;             // 没用到，因为是权重预加载指令
// //     instruction[3].com.accumulate = 0;                 // 没用到，因为是权重预加载指令

// //    //第五个指令（计算指令）
// //     instruction[4].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
// //     instruction[4].com.compute_type = COMPUTE;         // 计算指令 3位
// //     instruction[4].com.weigth_addr = 0;                // 没用到，因为是权重预加载指令
// //     instruction[4].com.input_addr = 0;                 // 输入读取起始行
// //     instruction[4].com.output_addr = 0;                // 输出写入起始行
// //     instruction[4].com.weight_switch = 0;              // 没用到，因为是权重预加载指令
// //     instruction[4].com.compute_switch = 0;             // 使用前面的权重预加载寄存器计算
// //     instruction[4].com.accumulate = 0;                 // 第一次计算不累加，刷新累加器


//    //第六个指令（存储指令）
//     instruction[5].mem.opcode = OPCODE_STORE;        // STORE操作码3位
//     instruction[5].mem.buffer_id = INPUT_BUFFER_ID;                //没用到，默认是以输出buffer输出 3位
//     instruction[5].mem.dram_base = 0;               // DRAM基地址32位
//     instruction[5].mem.buffer_base = 0;             // buffer基地址16位 
//     instruction[5].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[5].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[5].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位

//    //第七个指令（结束指令）
//     instruction[6].mem.opcode = OPCODE_DONE;        // STORE操作码3位
//     instruction[6].mem.buffer_id = 0;                //没用到，默认是以输出buffer输出 3位
//     instruction[6].mem.dram_base = 0;               // DRAM基地址32位
//     instruction[6].mem.buffer_base = 0;             // buffer基地址16位 
//     instruction[6].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
//     instruction[6].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
//     instruction[6].mem.x_stride = 2*MATRIX_WIDTH;     // 假设步进与列数相同16位


//     // 转换为指令类型，并输出查看

//     for (int i = 0; i < insn_count; i++) 
//     {
//         std::memcpy(&instruction_data[i], &instruction[i], sizeof(MemIns)); //转换指令结构体为128位指令数据类型
//         printf("instruction_data:\n%d",(int)instruction_data[i]);
//         // printBinary(instruction_data[i],INSTRUCT_WIDTH);                 // 输出指令的二进制
//         // printf("\n");
//     }

    
//     // 定义输出一维数组，注意自定义类型无法自动初始化为0，除非定义为静态变量或者全局变量
//     static Input_DataType outputs_matrix_hw[ROWS*COL1S] = {0}; // 输出数组，注意应该以Output_DataType定义
//     Transfer_DataType *transfer_outputs_matrix =(Transfer_DataType *) outputs_matrix_hw;// 转换为Transfer_DataType类型给SAA

//     // 执行saa_top硬件，加载指令，只执行load函数，注意，传入的是指针类型
//     uint32_t done;
//     saa_top(
//         insn_count,
//         instruction_data,
//         transfer_inputs_matrix, 
//         transfer_weights_matrix, +
//         transfer_outputs_matrix,
//         done); 

//     //检查缓冲区
//     // printf("input_buffer:\n");print_buffer(input_buffer,0,10);
//     // printf("weight_buffer:\n");print_buffer(weight_buffer,0,10);
//     // printf("output_buffer:\n");print_buffer(output_buffer,0,10);

//     // 检查输出
//     print_vec((Input_DataType*)transfer_outputs_matrix,ROWS*COL1S); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组
//     printf("outputs_matrix_hw:\n");print1D_2DArray(outputs_matrix_hw,ROWS,COL1S); // 将一维数组按二维打印

//     return 0;
// }




// // 10 测试saa_top IP进行分块矩阵乘法
// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <ctime>

// int main() {
//   const int rows = 8;
//   const int cols = 8;
//   const int y_block = 4;
//   const int x_block = 4;

//   // 分配并初始化一个2D矩阵
//   Input_DataType **matrix = allocInit2dArray<Input_DataType>(rows, cols);

//   // 打印原始矩阵内容
//   std::cout << "Original 2D Matrix:" << std::endl;
//   for (int i = 0; i < rows; i++) {
//     for (int j = 0; j < cols; j++) {
//       std::cout << matrix[i][j] << " ";
//     }
//     std::cout << std::endl;
//   }

//   // 分配打包后的1D数组
//   uint32_t* packed_buffer = static_cast<uint32_t*>(allocBuffer(rows * cols * sizeof(Input_DataType)));

//   // 打包矩阵
//   packBuffer<uint32_t, 32, Input_DataType, INPUT_DATA_WIDTH>(packed_buffer, matrix, rows, cols, y_block, x_block);


//   // 打印打包后矩阵在内存中的存储方式
//   std::cout << "Packed Buffer (1D Array):" << std::endl;
//   for (int i = 0; i < rows * cols; i++) {
//     std::cout << packed_buffer[i] << " ";
//     if ((i + 1) % cols == 0) {
//       std::cout << std::endl;
//     }
//   }
//   std::cout << std::endl;

//   // 打印打包后的1D数组，转换回原始数据类型
//   std::cout << "Packed Buffer (1D Array representing 2D Matrix):" << std::endl;
//   for (size_t i = 0; i < rows * cols; ++i) {
//     std::cout << *(reinterpret_cast<Input_DataType*>(&packed_buffer[0])+i)<< " ";
//     if ((i + 1) % cols == 0) {
//       std::cout << std::endl;
//     }
//   }
//   std::cout << std::endl;

// //   // 解包矩阵
// //   Input_DataType **unpacked_matrix = allocInit2dArray<Input_DataType>(rows, cols);
// //   unpackBuffer<Input_DataType, INPUT_DATA_WIDTH, uint32_t, 32>(unpacked_matrix, packed_buffer, rows, cols, y_block, x_block);

// //   // 打印解包后矩阵的内容
// //   std::cout << "Unpacked 2D Matrix (Original Data Type):" << std::endl;
// //   for (int i = 0; i < rows; i++) {
// //     for (int j = 0; j < cols; j++) {
// //       std::cout << unpacked_matrix[i][j] << " ";
// //     }
// //     std::cout << std::endl;
// //   }

//   // 释放2D矩阵和1D数组内存
//   free2dArray(matrix, rows, cols);
//   freeBuffer(packed_buffer);

//   return 0;
// }



// // 10 测试saa_top IP进行分块矩阵乘法
// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <ctime>

// int main() {

//     // // 检查指令
//     // Instruct_DataType instruction_data;
//     // GenericIns instruction;
    
//     // // instruction = getWeightPreloadComputeInsn(
//     // //                                 1, // 权重块加载偏移
//     // //                                 1, // 权重块加载偏移
//     // //                                 1, // 输出存储偏移
//     // //                                 1, // （这个指令中没有用，加载的是计算寄存器的相反寄存器）
//     // //                                 1, // 计算寄存器依然是当前寄存器
//     // //                                 1); // 当前计算是否进行累加


//     // // instruction = getWeightPreloadInsn(
//     // //                             1, // 从Buffer中0行加载权重块
//     // //                             1);     // 使用初始pingpang权重寄存器

//     //   // instruction =  getComputeInsn(
//     //   //                         1,  // 权重块加载偏移
//     //   //                         1, // 输出存储偏移
//     //   //                         1,      // i循环内不需要切换寄存器进行计算
//     //   //                         1);   // 当前计算是否在输出块进行累加
//     //   instruction = get2DLoadStoreInsn(
//     //         1,     // 加载指令
//     //         1, // 加载到输入Buffer
//     //         1,   // buffer偏移+矩阵内部偏移
//     //         1,     // 缓冲区偏移+矩阵内部偏移
//     //         1,    // 每次加载MATRIX_WIDTH行
//     //         1,    // 每次加载MATRIX_WIDTH列
//     //         1);          // input矩阵总列数作为2D跨步DMA步进

//     // std::memcpy(&instruction_data, &instruction, sizeof(MemIns));
//     // printBinary(instruction_data,INSTRUCT_WIDTH);


//     // 生成输入矩阵和权重矩阵
//     const int ROWS =  3*MATRIX_WIDTH; // 矩阵行数
//     const int COLS =  3*MATRIX_WIDTH; // 公共维度
//     const int COL1S = 3*MATRIX_WIDTH; // 矩阵列数
//     Input_DataType **inputs_matrix = init_matrix<Input_DataType>(ROWS,COLS); //输入矩阵
//     Weight_DataType **weights_matrix = init_matrix<Weight_DataType>(COL1S,COLS); //权重矩阵
//     Output_DataType **ouputs_matrix_ref = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                       (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
//     printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//     printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//     printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//     // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输

//     Input_DataType *inputs_matrix1 = *inputs_matrix;
//     Weight_DataType *weights_matrix1 = *weights_matrix;
//     Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
//     Transfer_DataType *transfer_weights_matrix= (Transfer_DataType*)weights_matrix1;
//     print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
//     print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组


//     // 定义输出一维数组，注意自定义类型无法自动初始化为0，除非定义为静态变量或者全局变量
//     static Output_DataType outputs_matrix[ROWS*COL1S] = {0}; // 输出数组，注意应该以Output_DataType定义
//     Transfer_DataType *transfer_outputs_matrix =(Transfer_DataType *) outputs_matrix;// 转换为Transfer_DataType类型给SAA


//     // 执行分块矩阵乘法
//     blocked_gemm_test(ROWS,  // input矩阵的I行
//                       COL1S, // weight矩阵的J列
//                       COLS,  // input矩阵的K列，weight矩阵的K行
//                       transfer_inputs_matrix,  // 输入
//                       transfer_weights_matrix, // 权重
//                       0,   // 偏置
//                       transfer_outputs_matrix, // 输出
//                       0,     // 分块系数，将矩阵分为多少块 
//                       0); // 是否使用偏置  

//     // 检查输出
//     print_vec((Output_DataType*)transfer_outputs_matrix,ROWS*COL1S); //转换为Output_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组
//     printf("outputs_matrix:\n");print1D_2DArray(outputs_matrix,ROWS,COL1S); // 将一维数组按二维打印

//     blocked_gemm_test(ROWS,  // input矩阵的I行
//                       COL1S, // weight矩阵的J列
//                       COLS,  // input矩阵的K列，weight矩阵的K行
//                       transfer_inputs_matrix,  // 输入
//                       transfer_weights_matrix, // 权重
//                       0,   // 偏置
//                       transfer_outputs_matrix, // 输出
//                       0,     // 分块系数，将矩阵分为多少块 
//                       0); // 是否使用偏置  
//     // 检查输出
//     print_vec((Output_DataType*)transfer_outputs_matrix,ROWS*COL1S); //转换为Output_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组
//     printf("outputs_matrix:\n");print1D_2DArray(outputs_matrix,ROWS,COL1S); // 将一维数组按二维打印

//   return 0;
// }


// // 10 测试saa_top IP进行分块矩阵乘法
// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <ctime>



// int main() {

//   // 生成输入矩阵和权重矩阵
//   const int ROWS =  2*MATRIX_WIDTH; // 矩阵行数
//   const int COLS =  2*MATRIX_WIDTH; // 公共维度
//   const int COL1S = 2*MATRIX_WIDTH; // 矩阵列数
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **ouputs_matrix_ref = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                     (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//   // 将输入矩阵和权重矩阵进行数据重排，本质上是将其进行分块然后转换为一维数组
//   // 分配打包后的1D数组
//   uint32_t* input_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COLS * sizeof(Input_DataType)));
//   uint32_t* weight_buffer = static_cast<uint32_t*>(allocBuffer(COLS * COL1S * sizeof(Weight_DataType)));
//   uint32_t* output_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));

//   // 打包矩阵
//   packBuffer<uint32_t, 32, Input_DataType, INPUT_DATA_WIDTH>(input_buffer, inputs_matrix, ROWS, COLS, MATRIX_WIDTH, MATRIX_WIDTH);
//   packBuffer<uint32_t, 32, Weight_DataType, WEIGHT_DATA_WIDTH>(weight_buffer, weights_matrix, COLS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 按照原来的行列查看打包后的矩阵存储方式
//   print_pack_buffer<Input_DataType>(input_buffer,ROWS, COLS);
//   print_pack_buffer<Weight_DataType>(weight_buffer,COLS, COL1S);


//   // 执行分块矩阵乘法
//   blocked_gemm_test(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     input_buffer,  // 输入
//                     weight_buffer, // 权重
//                     0,   // 偏置
//                     output_buffer, // 输出
//                     8,     // 加载分块系数，一个块的大小
//                     0); // 是否使用偏置  

//   // 检查输出                  
//   print_pack_buffer<Output_DataType>(output_buffer,ROWS, COL1S);


//   // 释放2D矩阵和1D数组内存
//   free2dArray(inputs_matrix, ROWS, COLS);
//   free2dArray(weights_matrix, COLS, COL1S);
//   freeBuffer(output_buffer);
//   freeBuffer(input_buffer);
//   freeBuffer(weight_buffer);

//   return 0;
// }



// // 12 测试saa进行bias加载

// int main() {

//   // 生成输入矩阵和权重矩阵
//   const int ROWS =  6*MATRIX_WIDTH; // 矩阵行数
//   const int COLS =  6*MATRIX_WIDTH; // 公共维度
//   const int COL1S = 6*MATRIX_WIDTH; // 矩阵列数
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **biases_matrix = allocInit2dArray<Output_DataType>(ROWS,COL1S); //biase矩阵
//   Output_DataType **ouputs_matrix_ref = matrix_biase_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                     (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S); // 计算矩阵乘法
//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("biases_matrix:\n");print_matrix(biases_matrix, ROWS, COL1S); // 打印偏置矩阵
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//   // 将输入矩阵和权重矩阵进行数据重排，本质上是将其进行分块然后转换为一维数组
//   // 分配打包后的1D数组
//   uint32_t* input_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COLS * sizeof(Input_DataType)));
//   uint32_t* weight_buffer = static_cast<uint32_t*>(allocBuffer(COLS * COL1S * sizeof(Weight_DataType)));
//   uint32_t* biases_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));
//   uint32_t* output_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));

//   // 打包矩阵
//   packBuffer<uint32_t, 32, Input_DataType, INPUT_DATA_WIDTH>(input_buffer, inputs_matrix, ROWS, COLS, MATRIX_WIDTH, MATRIX_WIDTH);
//   packBuffer<uint32_t, 32, Weight_DataType, WEIGHT_DATA_WIDTH>(weight_buffer, weights_matrix, COLS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);
//   packBuffer<uint32_t, 32, Output_DataType, OUTPUT_DATA_WIDTH>(biases_buffer, biases_matrix, ROWS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 按照原来的行列查看打包后的矩阵存储方式
//   print_pack_buffer<Input_DataType>(input_buffer,ROWS, COLS);
//   print_pack_buffer<Weight_DataType>(weight_buffer,COLS, COL1S);
//   print_pack_buffer<Output_DataType>(biases_buffer,ROWS, COL1S);

//   // 执行分块矩阵乘法
//   blocked_gemm_test(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     input_buffer,  // 输入
//                     weight_buffer, // 权重
//                     biases_buffer, // 偏置
//                     output_buffer, // 输出
//                     8,     // 加载分块系数，一个块的大小
//                     1); // 是否使用偏置  

//   // 检查输出                  
//   printf("ouputs_matrix:\n");print_pack_buffer<Output_DataType>(output_buffer,ROWS, COL1S);


//   // 释放2D矩阵和1D数组内存
//   free2dArray(inputs_matrix, ROWS, COLS);
//   free2dArray(weights_matrix, COLS, COL1S);
//   free2dArray(biases_matrix, ROWS, COL1S);
//   freeBuffer(output_buffer);
//   freeBuffer(biases_buffer);
//   freeBuffer(input_buffer);
//   freeBuffer(weight_buffer);

//   return 0;
// }


// 13 测试ANU进行layernorm 和 softmax


// int main() {

//   // 生成MATRIX_WIDTH行矩阵直接加载到输出Buffer中进行layernorm
//   const int ROWS =  MATRIX_WIDTH; // 矩阵行数固定为MATRIX_WIDTH
//   const int COLS =  2*MATRIX_WIDTH; // 矩阵列数

//   Norm_DataType **layernorm_matrix = allocInit2dArray<Norm_DataType>(ROWS,COLS); //输入矩阵
//   float eps = 1e-5f;
//   Norm_DataType **layernorm_output_ref = layer_norm(layernorm_matrix, ROWS, COLS, eps);
//   Norm_DataType **softmax_output_ref = softmax(layernorm_matrix, ROWS, COLS);
//   printf("layernorm_matrix:\n");print_matrix(layernorm_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("layernorm_output_ref:\n");print_matrix(layernorm_output_ref, ROWS, COLS); // 打印layernorm输出矩阵
//   printf("softmax_output_ref:\n");print_matrix(softmax_output_ref, ROWS, COLS); // 打印softmax输出矩阵

//   // 将输入矩阵和权重矩阵进行数据重排，本质上是将其进行分块然后转换为一维数组
//   // 分配打包后的1D数组
//   Norm_DataType* layernorm_buffer = static_cast<Norm_DataType*>(allocBuffer(ROWS * COLS * sizeof(Norm_DataType)));
//   Norm_DataType* output_buffer = static_cast<Norm_DataType*>(allocBuffer(ROWS * COLS * sizeof(Norm_DataType)));

//   // 打包矩阵
//   packData<Norm_DataType>(layernorm_buffer, layernorm_matrix, ROWS, COLS, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 按照原来的行列查看打包后的矩阵存储方式
//   printf("layernorm_buffer:\n");print_pack_buffer<Norm_DataType>(layernorm_buffer,ROWS, COLS);

//   // 执行layerNorm    
//   anu_test(ANU_LAYERNORM,   // 进行什么ANU操作
//            ROWS,  // input矩阵的I行
//            COLS,  // input矩阵的J列
//            NULL,
//            NULL,
//            layernorm_buffer, // 偏置项输入等待layernorm矩阵到输出缓冲区
//            output_buffer);  // 输入待归一化矩阵

//   // 执行softmax 
//   anu_test(ANU_SOFTMAX,   // 进行什么ANU操作
//            ROWS,  // input矩阵的I行
//            COLS,  // input矩阵的J列
//            NULL,
//            NULL,
//            layernorm_buffer, // 偏置项输入等待softmax矩阵到输出缓冲区
//            output_buffer);  // 输入待归一化矩阵


//   // 用打包格式检查检查输出                  
//   printf("output_buffer:\n");print_pack_buffer<Norm_DataType>(output_buffer,ROWS, COLS);

//   // 解包输出矩阵
//   Norm_DataType **ouputs_unpacked_matrix = allocInit2dArray<Norm_DataType>(ROWS, COLS);
//   unpackData<Norm_DataType>(ouputs_unpacked_matrix, output_buffer, ROWS, COLS, MATRIX_WIDTH, MATRIX_WIDTH);
//   printf("ouputs_unpacked_matrix:\n");print_matrix(ouputs_unpacked_matrix, ROWS, COLS); 
  
//   // 释放2D矩阵和1D数组内存
//   free2dArray(layernorm_matrix, ROWS, COLS);
//   free2dArray(ouputs_unpacked_matrix, ROWS, COLS);
//   freeBuffer(output_buffer);
//   freeBuffer(layernorm_buffer);
//   return 0;
// }







// // 13 测试使用uop的gemm
// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <ctime>

// int main() {

//   // 生成输入矩阵和权重矩阵
//   const int ROWS =  2*MATRIX_WIDTH; // 矩阵行数
//   const int COLS =  2*MATRIX_WIDTH; // 公共维度
//   const int COL1S = 2*MATRIX_WIDTH; // 矩阵列数
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **ouputs_matrix_ref = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                     (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//   // 将输入矩阵和权重矩阵进行数据重排，本质上是将其进行分块然后转换为一维数组
//   // 分配打包后的1D数组
//   uint32_t* input_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COLS * sizeof(Input_DataType)));
//   uint32_t* weight_buffer = static_cast<uint32_t*>(allocBuffer(COLS * COL1S * sizeof(Weight_DataType)));
//   uint32_t* output_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));

//   // 打包矩阵
//   packBuffer<uint32_t, 32, Input_DataType, INPUT_DATA_WIDTH>(input_buffer, inputs_matrix, ROWS, COLS, MATRIX_WIDTH, MATRIX_WIDTH);
//   packBuffer<uint32_t, 32, Weight_DataType, WEIGHT_DATA_WIDTH>(weight_buffer, weights_matrix, COLS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 按照原来的行列查看打包后的矩阵存储方式
//   print_pack_buffer<Input_DataType>(input_buffer,ROWS, COLS);
//   print_pack_buffer<Weight_DataType>(weight_buffer,COLS, COL1S);


//   // 执行分块矩阵乘法
//   blocked_gemm_test(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     input_buffer,  // 输入
//                     weight_buffer, // 权重
//                     0,   // 偏置
//                     output_buffer, // 输出
//                     8,     // 加载分块系数，一个块的大小
//                     0); // 是否使用偏置  

//   // 检查输出                  
//   print_pack_buffer<Output_DataType>(output_buffer,ROWS, COL1S);


//   // 释放2D矩阵和1D数组内存
//   free2dArray(inputs_matrix, ROWS, COLS);
//   free2dArray(weights_matrix, COLS, COL1S);
//   freeBuffer(output_buffer);
//   freeBuffer(input_buffer);
//   freeBuffer(weight_buffer);

//   return 0;
// }






// // 15 测试依赖关系计算二级分块矩阵乘法

// int main() {

//   // 生成输入矩阵和权重矩阵
//   const int ROWS =  2*MATRIX_WIDTH; // 矩阵行数
//   const int COL1S = 2*MATRIX_WIDTH; // 矩阵列数
//   const int COLS =  2*MATRIX_WIDTH; // 公共维度
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **biases_matrix = allocInit2dArray<Output_DataType>(ROWS,COL1S); //biase矩阵
//   Output_DataType **ouputs_matrix_ref = matrix_biase_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                     (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,1); // 计算矩阵乘法
//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("biases_matrix:\n");print_matrix(biases_matrix, ROWS, COL1S); // 打印偏置矩阵
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//   // 将输入矩阵和权重矩阵进行数据重排，本质上是将其进行分块然后转换为一维数组
//   // 分配打包后的1D数组
//   uint32_t* input_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COLS * sizeof(Input_DataType)));
//   uint32_t* weight_buffer = static_cast<uint32_t*>(allocBuffer(COLS * COL1S * sizeof(Weight_DataType)));
//   uint32_t* biases_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));
//   uint32_t* output_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));

//   // 打包矩阵
//   packBuffer<uint32_t, 32, Input_DataType, INPUT_DATA_WIDTH>(input_buffer, inputs_matrix, ROWS, COLS, MATRIX_WIDTH, MATRIX_WIDTH);
//   packBuffer<uint32_t, 32, Weight_DataType, WEIGHT_DATA_WIDTH>(weight_buffer, weights_matrix, COLS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);
//   packBuffer<uint32_t, 32, Output_DataType, OUTPUT_DATA_WIDTH>(biases_buffer, biases_matrix, ROWS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 按照原来的行列查看打包后的矩阵存储方式
//   print_pack_buffer<Input_DataType>(input_buffer,ROWS, COLS);
//   print_pack_buffer<Weight_DataType>(weight_buffer,COLS, COL1S);
//   print_pack_buffer<Output_DataType>(biases_buffer,ROWS, COL1S);

//   // 执行分块矩阵乘法
//   blocked_gemm_test(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     input_buffer,  // 输入
//                     weight_buffer, // 权重
//                     biases_buffer, // 偏置
//                     output_buffer, // 输出
//                     1,     // tile_I
//                     COL1S,     // tile_J
//                     1,     // tile_K
//                     1); // 是否使用偏置  

//   // 检查输出                  
//   printf("ouputs_matrix:\n");print_pack_buffer<Output_DataType>(output_buffer,ROWS, COL1S);
//   // 解包输出矩阵
//   uint32_t **ouputs_unpacked_matrix = allocInit2dArray<uint32_t>(ROWS, COL1S);
//   unpackData<uint32_t>(ouputs_unpacked_matrix, output_buffer, ROWS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);
//   printf("ouputs_unpacked_matrix:\n");print_matrix(ouputs_unpacked_matrix, ROWS, COL1S); 

//   // 释放2D矩阵和1D数组内存
//   free2dArray(inputs_matrix, ROWS, COLS);
//   free2dArray(weights_matrix, COLS, COL1S);
//   free2dArray(biases_matrix, ROWS, COL1S);
//   freeBuffer(output_buffer);
//   freeBuffer(biases_buffer);
//   freeBuffer(input_buffer);
//   freeBuffer(weight_buffer);

//   return 0;
// }







// // 16 测试自动分块的矩阵乘法

// int main() {

//   // 生成输入矩阵和权重矩阵
//   const int ROWS =  1*MATRIX_WIDTH; // 矩阵行数
//   const int COL1S = 1*MATRIX_WIDTH; // 矩阵列数
//   const int COLS =  1*MATRIX_WIDTH; // 公共维度
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **biases_matrix = allocInit2dArray<Output_DataType>(ROWS,COL1S); //biase矩阵
//   Output_DataType **ouputs_matrix_ref = matrix_biase_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                     (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,1); // 计算矩阵乘法
//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("biases_matrix:\n");print_matrix(biases_matrix, ROWS, COL1S); // 打印偏置矩阵
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//   // 将输入矩阵和权重矩阵进行数据重排，本质上是将其进行分块然后转换为一维数组
//   // 分配打包后的1D数组
//   uint32_t* input_buffer = static_cast<uint32_t*>(allocBuffer(ROWS * COLS * sizeof(Input_DataType)));
//   uint32_t* weight_buffer = static_cast<uint32_t*>(allocBuffer(COLS * COL1S * sizeof(Weight_DataType)));
//   Output_DataType* biases_buffer = static_cast<Output_DataType*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));

//   // 打包矩阵
//   packBuffer<uint32_t, 32, Input_DataType, INPUT_DATA_WIDTH>(input_buffer, inputs_matrix, ROWS, COLS, MATRIX_WIDTH, MATRIX_WIDTH);
//   packBuffer<uint32_t, 32, Weight_DataType, WEIGHT_DATA_WIDTH>(weight_buffer, weights_matrix, COLS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);
//   // packBuffer<uint32_t, 32, Output_DataType, OUTPUT_DATA_WIDTH>(biases_buffer, biases_matrix, ROWS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);

//   packData<Output_DataType>(biases_buffer, biases_matrix, ROWS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);
  
//   // 按照原来的行列查看打包后的矩阵存储方式
//   print_pack_buffer<Input_DataType>(input_buffer,ROWS, COLS);
//   print_pack_buffer<Weight_DataType>(weight_buffer,COLS, COL1S);
//   print_pack_buffer<Output_DataType>(biases_buffer,ROWS, COL1S);

//   // 执行分块矩阵乘法
//   tiled_matmul_auto(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     input_buffer,  // 输入
//                     weight_buffer, // 权重
//                     biases_buffer, // 偏置
//                     output_buffer, // 输出
//                     1, // 是否使用偏置  
//                     0); // 是否使用激活，同时使用何种激活

//   // 检查输出                  
//   printf("ouputs_matrix:\n");print_pack_buffer<Output_DataType>(output_buffer,ROWS, COL1S);
//   // 解包输出矩阵
//   Output_DataType **ouputs_unpacked_matrix = allocInit2dArray<Output_DataType>(ROWS, COL1S);
//   unpackData<Output_DataType>(ouputs_unpacked_matrix, output_buffer, ROWS, COL1S, MATRIX_WIDTH, MATRIX_WIDTH);
//   printf("ouputs_unpacked_matrix:\n");print_matrix(ouputs_unpacked_matrix, ROWS, COL1S); 

//   // 释放2D矩阵和1D数组内存
//   free2dArray(inputs_matrix, ROWS, COLS);
//   free2dArray(weights_matrix, COLS, COL1S);
//   free2dArray(biases_matrix, ROWS, COL1S);
//   freeBuffer(output_buffer);
//   freeBuffer(biases_buffer);
//   freeBuffer(input_buffer);
//   freeBuffer(weight_buffer);

//   return 0;
// }



// // 17 测试自动填充,自动pack和自动反填充反pack的矩阵乘法
// int main() {

//   // 生成输入矩阵和权重矩阵
//   const int ROWS =  4; // 矩阵行数
//   const int COL1S = 1; // 矩阵列数
//   const int COLS =  5; // 公共维度
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **biases_matrix = allocInit2dArray<Output_DataType>(ROWS,COL1S); //biase矩阵
//   Output_DataType **ouputs_matrix_ref = matrix_biase_dot<Input_DataType,Weight_DataType,Output_DataType>
//                                     (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,1); // 计算矩阵乘法
//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("biases_matrix:\n");print_matrix(biases_matrix, ROWS, COL1S); // 打印偏置矩阵
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//   // 输出缓冲区
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));

//   // 执行分块矩阵乘法
//   tiled_matmul_auto(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     inputs_matrix[0],  // 输入,传入二维数组首地址
//                     weights_matrix[0], // 权重
//                     biases_matrix[0], // 偏置
//                     output_buffer, // 输出
//                     1, // 是否使用偏置  
//                     0); // 是否使用激活，同时使用何种激活

//   // 检查输出                  
//   printf("ouputs_matrix:\n");print_pack_buffer<Output_DataType>(output_buffer,ROWS, COL1S);

//   return 0;
// }


// // 18 测试虚拟线程/双缓冲
// int main() {

//   // 生成输入矩阵和权重矩阵
//   const int ROWS =  8; // 矩阵行数
//   const int COL1S = 8; // 矩阵列数
//   const int COLS =  8; // 公共维度
//   float scale = 0.05; // 输出的缩放系数
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **biases_matrix = allocInit2dArray<Output_DataType>(ROWS,COL1S); //biase矩阵
//   // Output_DataType **ouputs_matrix_ref = matrix_biase_dot<Input_DataType,Weight_DataType,Output_DataType>
//   //                                   (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,0); // 计算矩阵乘法
//   // Scale_DataType **ouputs_matrix_ref = matrix_scale_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//   //                                   (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,0,1,scale); // 计算矩阵乘法
//   Scale_DataType **ouputs_matrix_ref = matrix_relu_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//                                     (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,0,1,1,scale); // 计算矩阵乘法

//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("biases_matrix:\n");print_matrix(biases_matrix, ROWS, COL1S); // 打印偏置矩阵
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//   // 从文件中加载线性层权重参数和偏置参数

//   // 输出缓冲区
//   Scale_DataType* output_buffer = static_cast<Scale_DataType*>(allocBuffer(ROWS * COL1S * sizeof(Scale_DataType)));

//   // 执行分块矩阵乘法
//   tiled_matmul_auto(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     inputs_matrix[0],  // 输入,传入二维数组首地址
//                     weights_matrix[0], // 权重
//                     biases_matrix[0], // 偏置
//                     output_buffer, // 输出
//                     0, // 是否使用偏置  
//                     1, // 是否进行relu操作
//                     1, // scale的类型(0代表不进行scale,1代表进行反量化,2代表进行重量化)
//                     scale,
//                     0); // 是否使用激活，同时使用何种激活

//   // 检查输出                  
//   printf("ouputs_matrix:\n");print_pack_buffer<Scale_DataType>(output_buffer, ROWS, COL1S);

//   return 0;
// }


// // 20 正常的int32形式的矩阵乘法
// int main() {

//   // 生成输入矩阵和权重矩阵
//   const int ROWS =  8; // 矩阵行数
//   const int COL1S = 1; // 矩阵列数
//   const int COLS =  9; // 公共维度
//   float scale = 0.05; // 输出的缩放系数
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **biases_matrix = allocInit2dArray<Output_DataType>(ROWS,COL1S); //biase矩阵
//   // Output_DataType **ouputs_matrix_ref = matrix_biase_dot<Input_DataType,Weight_DataType,Output_DataType>
//   //                                   (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,0); // 计算矩阵乘法
//   // Scale_DataType **ouputs_matrix_ref = matrix_scale_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//   //                                   (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,0,1,scale); // 计算矩阵乘法
//   Output_DataType **ouputs_matrix_ref = matrix_relu_dot<Input_DataType,Weight_DataType,Output_DataType,Output_DataType>
//                                     (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,0,0,0,scale); // 计算矩阵乘法

//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("biases_matrix:\n");print_matrix(biases_matrix, ROWS, COL1S); // 打印偏置矩阵
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//   // 从文件中加载线性层权重参数和偏置参数

//   // 输出缓冲区
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(ROWS * COL1S * sizeof(Output_DataType)));

//   // 执行分块矩阵乘法
//   tiled_matmul_auto(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     inputs_matrix[0],  // 输入,传入二维数组首地址
//                     weights_matrix[0], // 权重
//                     biases_matrix[0], // 偏置
//                     output_buffer, // 输出
//                     0, // 是否使用偏置  
//                     0, // 是否进行relu操作
//                     0, // scale的类型(0代表不进行scale,1代表进行反量化,2代表进行重量化)
//                     scale,
//                     0); // 是否使用激活，同时使用何种激活

//   // 检查输出                  
//   printf("ouputs_matrix:\n");print_pack_buffer<Output_DataType>(output_buffer, ROWS, COL1S);

//   return 0;
// }


// // 13 测试ANU进行layernorm 和 softmax

// int main() {

//   // 生成MATRIX_WIDTH行矩阵直接加载到输出Buffer中进行layernorm
//   const int ROWS =  8; // 矩阵行数固定为MATRIX_WIDTH
//   const int COLS =  5; // 矩阵列数

//   Scale_DataType **layernorm_matrix = allocInit2dArray<Scale_DataType>(ROWS,COLS); //输入矩阵
//   float eps = 1e-5f;
//   Scale_DataType **layernorm_output_ref = layer_norm(layernorm_matrix, ROWS, COLS, eps);
//   Scale_DataType **softmax_output_ref = softmax(layernorm_matrix, ROWS, COLS);
//   printf("layernorm_matrix:\n");print_matrix(layernorm_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("layernorm_output_ref:\n");print_matrix(layernorm_output_ref, ROWS, COLS); // 打印layernorm输出矩阵
//   printf("softmax_output_ref:\n");print_matrix(softmax_output_ref, ROWS, COLS); // 打印softmax输出矩阵

//   // 输出缓冲区
//   Scale_DataType* output_buffer = static_cast<Scale_DataType*>(allocBuffer(ROWS * COLS * sizeof(Scale_DataType)));

//   // 执行layerNorm    
//   // anu_test(ANU_LAYERNORM,   // 进行什么ANU操作
//   //          ROWS,  // input矩阵的I行
//   //          COLS,  // input矩阵的J列
//   //          layernorm_matrix[0], // 偏置项输入等待layernorm矩阵到输出缓冲区
//   //          output_buffer);  // 输入待归一化矩阵


//   // 执行softmax 
//   anu_test(ANU_SOFTMAX,   // 进行什么ANU操作
//            ROWS,  // input矩阵的I行
//            COLS,  // input矩阵的J列
//            layernorm_matrix[0], // 偏置项输入等待softmax矩阵到输出缓冲区
//            output_buffer);  // 输入待归一化矩阵

//   // 检查输出                  
//   printf("ouputs_matrix:\n");print_pack_buffer<Scale_DataType>(output_buffer, ROWS, COLS);

//   return 0;
// }


// // 20 测试融合了relu/layernorm/softmax的矩阵乘法
// int main() {

//   // 生成输入矩阵和权重矩阵
//   const int ROWS =  1; // 矩阵行数
//   const int COL1S = 16; // 矩阵列数
//   const int COLS =  24; // 公共维度
//   float scale = 0.05; // 输出的缩放系数
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **biases_matrix = allocInit2dArray<Output_DataType>(ROWS,COL1S); //biase矩阵
//   // Output_DataType **ouputs_matrix_ref = matrix_biase_dot<Input_DataType,Weight_DataType,Output_DataType>
//   //                                   (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,0); // 计算矩阵乘法
//   // Scale_DataType **ouputs_matrix_ref = matrix_scale_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//   //                                   (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,0,1,scale); // 计算矩阵乘法
//   // Scale_DataType **ouputs_matrix_ref = matrix_relu_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//   //                                   (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,1,1,1,scale); // 计算矩阵乘法
//   Scale_DataType **ouputs_matrix_ref1 = matrix_relu_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//                                     (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,1,0,1,scale); // 计算矩阵乘法


//   // Scale_DataType **ouputs_matrix_ref = matrix_act_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//   //                                   (inputs_matrix,
//   //                                   weights_matrix,
//   //                                   biases_matrix,
//   //                                   ROWS, 
//   //                                   COLS, 
//   //                                   COL1S,
//   //                                   0, //bias_use
//   //                                   0, //relu_use
//   //                                   1, //scale_type,1代表进行反量化
//   //                                   scale, // scale
//   //                                   ANU_LAYERNORM); // 融合layernorm
  
//   Scale_DataType **ouputs_matrix_ref = matrix_act_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//                                     (inputs_matrix, // int8输入
//                                     weights_matrix, // int8输入
//                                     biases_matrix, // int32输入/定点输入
//                                     ROWS, 
//                                     COLS, 
//                                     COL1S,
//                                     1, //bias_use
//                                     0, //relu_use
//                                     1, //scale_type,1代表进行反量化
//                                     scale, // scale系数，float输入
//                                     ANU_SOFTMAX); // 融合softmax

//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("biases_matrix:\n");print_matrix(biases_matrix, ROWS, COL1S); // 打印偏置矩阵
//   printf("ouputs_matrix_scale:\n");print_matrix(ouputs_matrix_ref1, ROWS, COL1S); // 打印输出矩阵
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印输出矩阵

//   // 从文件中加载线性层权重参数和偏置参数

//   // 输出缓冲区
//   Scale_DataType* output_buffer = static_cast<Scale_DataType*>(allocBuffer(ROWS * COL1S * sizeof(Scale_DataType)));

//   // 执行分块矩阵乘法
//   tiled_matmul_auto(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     inputs_matrix[0],  // 输入,传入二维数组首地址
//                     weights_matrix[0], // 权重
//                     biases_matrix[0], // 偏置
//                     output_buffer, // 输出
//                     1, // 是否使用偏置  
//                     0, // 是否进行relu操作
//                     1, // scale的类型(0代表不进行scale,1代表进行反量化,2代表进行重量化)
//                     scale,
//                     ANU_SOFTMAX); // 是否使用激活，同时使用何种激活
//   // 检查输出                  
//   printf("ouputs_matrix:\n");print_pack_buffer<Scale_DataType>(output_buffer, ROWS, COL1S);

//   return 0;
// }


// // 仿真测试小数整数的定点乘法

// #include <cmath>
// #include <cstdint>
// #include <iostream>
// #include <bitset>

// #include <string>

// template<typename T>
// std::string toBinary(T val) {
//     std::string binary = "";
//     for (int i = (sizeof(unsigned int) * 8) - 1; i >= 0; i--) {
//         binary += (val & (1 << i)) ? '1' : '0';
//     }
//     return binary;
// }

// Scale_DataType scale_dot(Scale_DataType *scale ,Psum_DataType data)
// {
//   Scale_DataType result = *(scale) * data;
//   return result;
// }
// int main() {

//   float f = 0.00009054165275301784;
//   std::cout << "float : " << f << std::endl;

//   // 将f转为整数，我们存储的的Norm_DataType有多少位小数,我们就左移多少位,定点8位小数最小分辨率是1/256=0.00390625,小于这个值的数是无法表示的
//   int32_t f_int = static_cast<int32_t>(f * (1LL << 24)); // 
//   std::cout << "int : " << f_int << std::endl;

//   //进行右移转换回定点小数类型
//   ComIns insn = {};
//   insn.scale = f_int;  // GEMM指令
//   int32_t scale_int = insn.scale;
//   std::cout << "scale_int: " << scale_int  << std::endl;

//   // scale的赋值,上面都是在PC处做的转换
//   Scale_DataType scale; 
//   scale.range() = insn.scale; //按照位模式赋值,而不是值模式,达到类似memcpy的效果
//   std::cout << " scale: " << scale << std::endl;

//   // 实际上定点小数可以直接乘以整数,得到的结果再换为定点小数,这样就不用把整数转换为小数了
//   Psum_DataType a = 5; //模拟int32的值
//   Scale_DataType result1;
//   Psum_DataType result;
//   result.range() = ((Scale_DataType)(scale * a)).range(); // 将计算结果强制转换为32位定点，同时使用位模式赋值
//   // 结果是位模式的
//   result1 = scale_dot((Scale_DataType *)(&result),a);
//   // result.range() = result1.range(31,0);
//   // memcpy(&result, &result1, sizeof(result1)); //转换指令结构体为128位指令数据类型

//   std::bitset<32> binary(result); // 假设你的整数是32位的
//   std::bitset<64> binary1(result1); // 假设你的整数是32位的


//   std::cout << " result bin: " << binary << std::endl;
//   std::cout << " result1 bin: " << toBinary(result1) << std::endl;

//   std::cout << " truth_result: " << (a*f) << std::endl;
//   std::cout << " result1: " << result1 << std::endl;
//   std::cout << " fixed_result(32): " << *((Scale_DataType *)(&result)) << std::endl;
//   std::cout << " fixed_result: " << scale * a << std::endl;


//   std::cout << " sizeof(32): " << sizeof(result) << std::endl;
//   std::cout << " sizeof: " << sizeof(scale * a) << std::endl;
//   std::cout << " sizeof result1: " << sizeof(result1) << std::endl;
//     return 0;
// }


// // 20 测试实际推理权重和输入

// int main() {

//   // 读取量化的输入矩阵和权重矩阵
//   const int ROWS =  100; // 矩阵行数
//   const int COL1S = 100; // 矩阵列数
//   const int COLS =  100; // 公共维度
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **biases_matrix = allocInit2dArray<Output_DataType>(ROWS,COL1S); //biase矩阵

//   // 读取缩放系数
//   float scale = 2.6712603357736953e-05 ; // 输出的缩放系数
//   // 读取矩阵数据
//   readBinaryFileToMatrix<Input_DataType>("quant_input.bin", inputs_matrix, ROWS, COLS);  // 读取输入矩阵
//   readBinaryFileToMatrix<Weight_DataType>("quant_weight.bin", weights_matrix, COLS, COL1S);  // 读取权重矩阵
//   readBinaryFileToMatrix<Output_DataType>("quant_bias.bin", biases_matrix, ROWS, COL1S);  // 读取偏置矩阵

//   // 软件计算矩阵乘法
//   Scale_DataType **ouputs_matrix_ref1 = matrix_relu_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//                                     (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,1,0,1,scale); // 计算矩阵乘法
//   // 硬件计算矩阵乘法
//   // Scale_DataType **ouputs_matrix_ref = matrix_act_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//   //                                   (inputs_matrix, // int8输入
//   //                                   weights_matrix, // int8输入
//   //                                   biases_matrix, // int32输入/定点输入
//   //                                   ROWS, 
//   //                                   COLS, 
//   //                                   COL1S,
//   //                                   1, //bias_use
//   //                                   0, //relu_use
//   //                                   1, //scale_type,1代表进行反量化
//   //                                   scale, // scale系数，float输入
//   //                                   ANU_LAYERNORM); // 融合softmax

//   Scale_DataType **ouputs_matrix_ref = matrix_act_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//                                     (inputs_matrix, // int8输入
//                                     weights_matrix, // int8输入
//                                     biases_matrix, // int32输入/定点输入
//                                     ROWS, 
//                                     COLS, 
//                                     COL1S,
//                                     1, //bias_use
//                                     0, //relu_use
//                                     1, //scale_type,1代表进行反量化
//                                     scale, // scale系数，float输入
//                                     ANU_SOFTMAX); // 融合softmax

//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("biases_matrix:\n");print_matrix(biases_matrix, ROWS, COL1S); // 打印偏置矩阵
//   printf("ouputs_matrix_scale:\n");print_matrix(ouputs_matrix_ref1, ROWS, COL1S); // 打印软件计算int8矩阵乘法并应用反量化
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印硬件计算int8矩阵乘法并应用反量化

//   // 输出缓冲区
//   Scale_DataType* output_buffer = static_cast<Scale_DataType*>(allocBuffer(ROWS * COL1S * sizeof(Scale_DataType)));

//   // 执行分块矩阵乘法
//   tiled_matmul_auto(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     inputs_matrix[0],  // 输入,传入二维数组首地址
//                     weights_matrix[0], // 权重
//                     biases_matrix[0], // 偏置
//                     output_buffer, // 输出
//                     1, // 是否使用偏置  
//                     0, // 是否进行relu操作
//                     1, // scale的类型(0代表不进行scale,1代表进行反量化,2代表进行重量化)
//                     scale,
//                     ANU_SOFTMAX); // 是否使用激活，同时使用何种激活
//   // 检查定点小数输出                  
//   printf("ouputs_matrix_fixed:\n");print_pack_buffer<Scale_DataType>(output_buffer, ROWS, COL1S);

//   // 将输出转换为float类型，也就是直接将output_buffer除以float 2的24次方（Scale_DataType小数就是24位）保存为float
//   // 转换为浮点数组
//   float* float_output = (float*)malloc(ROWS * COL1S * sizeof(float));
//   Output_DataType* output_buffer_int = (Output_DataType*)malloc(ROWS * COL1S * sizeof(Output_DataType));
//   for (int i = 0; i < ROWS * COL1S; ++i) {
//       output_buffer_int[i].range() = output_buffer[i].range(); // 位复制到int32缓冲区
//   }

//   const float scale_factor = (float)(1 << 24);  // 2 的 24 次方,这桑因为定点类型使用24位表示小数
//   for (int i = 0; i < ROWS * COL1S; ++i) {
//       float_output[i] = output_buffer_int[i] / scale_factor; // 将int32右移24位转换为float
//   }
//   printf("ouputs_matrix_float:\n");print_pack_buffer<float>(float_output, ROWS, COL1S);
 
//   return 0;
// }


// // 20 测试实际推理权重和输入

// int main() {

//   // 读取量化的输入矩阵和权重矩阵
//   const int ROWS =  100; // 矩阵行数
//   const int COL1S = 100; // 矩阵列数
//   const int COLS =  100; // 公共维度
//   Input_DataType **inputs_matrix = allocInit2dArray<Input_DataType>(ROWS,COLS); //输入矩阵
//   Weight_DataType **weights_matrix = allocInit2dArray<Weight_DataType>(COLS,COL1S); //权重矩阵
//   Output_DataType **biases_matrix = allocInit2dArray<Output_DataType>(ROWS,COL1S); //biase矩阵
//   Scale_DataType **output_matrix = allocInit2dArray<Scale_DataType>(ROWS,COL1S); //输出矩阵

//   // 读取缩放系数
//   float scale = 2.6712603357736953e-05 ; // 输出的缩放系数
//   // 读取矩阵数据
//   readBinaryFileToMatrix<Input_DataType>("quant_input.bin", inputs_matrix, ROWS, COLS);  // 读取输入矩阵
//   readBinaryFileToMatrix<Weight_DataType>("quant_weight.bin", weights_matrix, COLS, COL1S);  // 读取权重矩阵
//   readBinaryFileToMatrix<Output_DataType>("quant_bias.bin", biases_matrix, ROWS, COL1S);  // 读取偏置矩阵

//   // 软件计算矩阵乘法
//   Scale_DataType **ouputs_matrix_ref1 = matrix_relu_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//                                     (inputs_matrix,weights_matrix,biases_matrix,ROWS, COLS, COL1S,1,0,1,scale); // 计算矩阵乘法
//   // 硬件计算矩阵乘法
//   // Scale_DataType **ouputs_matrix_ref = matrix_act_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//   //                                   (inputs_matrix, // int8输入
//   //                                   weights_matrix, // int8输入
//   //                                   biases_matrix, // int32输入/定点输入
//   //                                   ROWS, 
//   //                                   COLS, 
//   //                                   COL1S,
//   //                                   1, //bias_use
//   //                                   0, //relu_use
//   //                                   1, //scale_type,1代表进行反量化
//   //                                   scale, // scale系数，float输入
//   //                                   ANU_LAYERNORM); // 融合softmax

//   Scale_DataType **ouputs_matrix_ref = matrix_act_dot<Input_DataType,Weight_DataType,Output_DataType,Scale_DataType>
//                                     (inputs_matrix, // int8输入
//                                     weights_matrix, // int8输入
//                                     biases_matrix, // int32输入/定点输入
//                                     ROWS, 
//                                     COLS, 
//                                     COL1S,
//                                     1, //bias_use
//                                     0, //relu_use
//                                     1, //scale_type,1代表进行反量化
//                                     scale, // scale系数，float输入
//                                     ANU_SOFTMAX); // 融合softmax

//   printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
//   printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
//   printf("biases_matrix:\n");print_matrix(biases_matrix, ROWS, COL1S); // 打印偏置矩阵
//   printf("ouputs_matrix_scale:\n");print_matrix(ouputs_matrix_ref1, ROWS, COL1S); // 打印软件计算int8矩阵乘法并应用反量化
//   printf("ouputs_matrix_ref:\n");print_matrix(ouputs_matrix_ref, ROWS, COL1S); // 打印硬件计算int8矩阵乘法并应用反量化

//   // 输出缓冲区
//   Scale_DataType* output_buffer = static_cast<Scale_DataType*>(allocBuffer(ROWS * COL1S * sizeof(Scale_DataType)));

//   // 执行分块矩阵乘法
//   tiled_matmul_auto(ROWS,  // input矩阵的I行
//                     COL1S, // weight矩阵的J列
//                     COLS,  // input矩阵的K列，weight矩阵的K行
//                     inputs_matrix[0],  // 输入,传入二维数组首地址
//                     weights_matrix[0], // 权重
//                     biases_matrix[0], // 偏置
//                     output_matrix[0], // 输出
//                     1, // 是否使用偏置  
//                     0, // 是否进行relu操作
//                     1, // scale的类型(0代表不进行scale,1代表进行反量化,2代表进行重量化)
//                     scale,
//                     ANU_SOFTMAX); // 是否使用激活，同时使用何种激活
//   // 检查定点小数输出                  
//   // printf("ouputs_matrix_fixed:\n");print_pack_buffer<Scale_DataType>(output_buffer, ROWS, COL1S);
//   printf("ouputs_matrix_fixed:\n");print_matrix(output_matrix, ROWS, COL1S); // 打印硬件计算int8矩阵乘法并应用反量化

//   // 将输出转换为float类型，也就是直接将output_buffer除以float 2的24次方（Scale_DataType小数就是24位）保存为float
//   // 转换为浮点数组
//   float **output_matrix_float = allocInit2dArray<float>(ROWS,COL1S); //输出矩阵
//   Output_DataType **output_matrix_int = allocInit2dArray<Output_DataType>(ROWS,COL1S); //输出矩阵
//   for (int i = 0; i < ROWS ; ++i) {
//       for (int j = 0; j < COL1S; ++j) {
//         output_matrix_int[i][j].range() = output_matrix[i][j].range(); // 位复制到int32缓冲区
//       }
//   }

//   const float scale_factor = (float)(1 << 24);  // 2 的 24 次方,这桑因为定点类型使用24位表示小数
//   for (int i = 0; i < ROWS ; ++i) {
//       for (int j = 0; j < COL1S; ++j) {
//         output_matrix_float[i][j] = output_matrix_int[i][j] / scale_factor;  // 将int32右移24位转换为float
//       }
//   }

//   printf("output_matrix_float:\n");print_matrix(output_matrix_float, ROWS, COL1S); // 打印硬件计算int8矩阵乘法并应用反量化
//   return 0;
// }

// 21 测试推理CNN网络
// 用于比较两个float值的宏
#define FLOAT_EQUAL(a, b) (fabs((a) - (b)) < 1e-6)

int compare_floats(float a, float b) {
    if (FLOAT_EQUAL(a, b)) return 1;
    return 0;
}

// 对称量化给定的矩阵到INT8
template<typename T0>
void symmetric_quantize(float **input, T0 **output, float scale, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // 对矩阵中的每个元素进行量化i
            output[i][j] = (T0)roundf(input[i][j] / scale);
        }
    }
}

// QuantizedMLP的定点前向推理函数

// QuantizedMLP的软件前向推理函数

int main() {
  // 维度参数
  int Batch_size = 1000; // 有1000个784大小的输入图像
  int fc1_row = Batch_size;
  int fc1_col = 784;
  int fc1_col1 = 512;
  int fc2_row = Batch_size;
  int fc2_col = 512;
  int fc2_col1 = 10;

  // 初始化矩阵
  float **fc1_input = allocInit2dArray<float>(fc1_row,fc1_col); //输入矩阵
  Input_DataType **quant_fc1_input = allocInit2dArray<Input_DataType>(fc1_row,fc1_col); //输入矩阵
  Weight_DataType **quant_fc1_weight = allocInit2dArray<Weight_DataType>(fc1_col,fc1_col1); //权重矩阵
  Output_DataType **quant_fc1_bias = allocInit2dArray<Output_DataType>(fc1_row,fc1_col1); //biase矩阵
  Scale_DataType **fc1_output = allocInit2dArray<Scale_DataType>(fc1_row,fc1_col1); //输出矩阵
  float **fc2_input = allocInit2dArray<float>(fc2_row,fc2_col); //输入矩阵
  Input_DataType **quant_fc2_input = allocInit2dArray<Input_DataType>(fc2_row,fc2_col); //输入矩阵
  Weight_DataType **quant_fc2_weight = allocInit2dArray<Weight_DataType>(fc2_col,fc2_col1); //权重矩阵
  Output_DataType **quant_fc2_bias = allocInit2dArray<Output_DataType>(fc2_row,fc2_col1); //biase矩阵
  Scale_DataType **fc2_output = allocInit2dArray<Scale_DataType>(fc2_row,fc2_col1); //输出矩阵
  float **output = allocInit2dArray<float>(fc2_row,fc2_col1); //输入矩阵

  // 读取scale
  float fc1_input_scale  = 0.022363794967532158;
  float fc2_input_scale  = 0.08953838050365448;
  float fc1_weight_scale = 0.00457571167498827;
  float fc2_weight_scale = 0.0025448380038142204;
  float fc1_output_scale = 0.00010233027569483966;
  float fc2_output_scale = 0.00022786066983826458;

  // 读取权重和偏置到我们的矩阵中
  readBinaryFileToMatrix<Weight_DataType>("quant_fc1_weight.bin", quant_fc1_weight, fc1_col, fc1_col1);  // 读取权重矩阵
  readBinaryFileToMatrix<Output_DataType>("quant_fc1_bias.bin", quant_fc1_bias, fc1_row, fc1_col1);  // 读取偏置矩阵
  readBinaryFileToMatrix<Weight_DataType>("quant_fc2_weight.bin", quant_fc2_weight, fc2_col, fc2_col1);  // 读取权重矩阵
  readBinaryFileToMatrix<Output_DataType>("quant_fc2_bias.bin", quant_fc2_bias, fc2_row, fc2_col1);  // 读取偏置矩阵

  // // 执行定点推理
  // // fc1推理，融合relu
  // // 加载输入
  // readBinaryFileToMatrix<float>("data_0.bin", fc1_input, fc1_row, fc1_col);  // 读取fc1的输入
  // symmetric_quantize<Input_DataType>(fc1_input,quant_fc1_input,fc1_input_scale,fc1_row,fc1_col); // 量化输入
  // // printf("fc1_input:\n");print_matrix(fc1_input, fc1_row, fc1_col); // 打印输出矩阵
  // // printf("quant_fc1_input:\n");print_matrix(quant_fc1_input, fc1_row, fc1_col); // 打印输出矩阵
  // tiled_matmul_auto(fc1_row,  // input矩阵的I行
  //                   fc1_col1, // weight矩阵的J列
  //                   fc1_col,  // input矩阵的K列，weight矩阵的K行
  //                   quant_fc1_input[0],  // 输入,传入二维数组首地址
  //                   quant_fc1_weight[0], // 权重
  //                   quant_fc1_bias[0], // 偏置
  //                   fc1_output[0], // 输出
  //                   1, // 是否使用偏置  
  //                   1, // 是否进行relu操作
  //                   1, // scale的类型(0代表不进行scale,1代表进行反量化,2代表进行重量化)
  //                   fc1_output_scale, // 反量化f1层输出为定点，然后将其转换为float
  //                   0); // 是否使用激活，同时使用何种激活

  // //将定点类型转换为浮点
  // for (int i = 0; i < fc1_row ; ++i) {
  //   for (int j = 0; j < fc1_col1; ++j) {
  //     fc2_input[i][j] = (float)fc1_output[i][j]; 
  //   }
  // }

  // // fc2层
  // symmetric_quantize<Input_DataType>(fc2_input,quant_fc2_input,fc2_input_scale,fc2_row,fc2_col); // 量化f1输出flaot给fc2
  // tiled_matmul_auto(fc2_row,  // input矩阵的I行
  //                   fc2_col1, // weight矩阵的J列
  //                   fc2_col,  // input矩阵的K列，weight矩阵的K行
  //                   quant_fc2_input[0],  // 输入,传入二维数组首地址
  //                   quant_fc2_weight[0], // 权重
  //                   quant_fc2_bias[0], // 偏置
  //                   fc2_output[0], // 输出
  //                   1, // 是否使用偏置  
  //                   0, // 是否进行relu操作
  //                   1, // scale的类型(0代表不进行scale,1代表进行反量化,2代表进行重量化)
  //                   fc2_output_scale, // 反量化f2层输出为定点，然后将其转换为float
  //                   0); // 是否使用激活，同时使用何种激活

  // //将定点类型转换为浮点
  // for (int i = 0; i < fc2_row ; ++i) {
  //   for (int j = 0; j < fc2_col1; ++j) {
  //     output[i][j] = (float)fc2_output[i][j]; 
  //   }
  // }
  
  // printf("interface output:\n");print_matrix(output, fc2_row, fc2_col1); // 打印输出矩阵
  // writeBinaryFileFromMatrix<float>("output.bin", output, fc2_row, fc2_col1);// 写入二维数组
  readBinaryFileToMatrix<float>("output.bin", output, fc2_row, fc2_col1);  // 读取偏置矩阵
  printf("interface output:\n");print_matrix(output, fc2_row, fc2_col1); // 打印输出矩阵

  // 对比结果,读取target
  int* lable = static_cast<int*>(allocBuffer(Batch_size * sizeof(int)));
  readBinaryFileToArray<>("target_0.bin", lable, Batch_size);// 存储结果
  printf("target_0:\n");print_pack_buffer<int>(lable, 1, Batch_size);

  // 对output输出求出标签和lable标签进行对比
  // 计算预测标签
  int correct = 0;
  for (int i = 0; i < fc2_row; ++i) {
      // 找到output中最大值的索引，作为预测标签
      int max_idx = 0;
      for (int j = 1; j < fc2_col1; ++j) {
          if (output[i][j] > output[i][max_idx]) {
              max_idx = j;
          }
      }

      // 比较预测标签和真实标签
      if (compare_floats((float)max_idx, lable[i]) == 1) {
          correct++;
      }
  }

  // 计算准确率
  float accuracy = 100.0f * correct / Batch_size;
  printf("Accuracy of the quantized inference: %.2f%%\n", accuracy);

  return 0;
}