#include "LOAD.hpp"
#include <stdio.h>

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


// 7 测试load和compute指令以及store指令的自动生成
#include <iostream>

int main() {

    // 生成输入矩阵和权重矩阵
    const int ROWS = 2*MATRIX_WIDTH; // 矩阵行数
    const int COLS = 2*MATRIX_WIDTH; // 公共维度
    const int COL1S = 2*MATRIX_WIDTH; // 矩阵列数
    Input_DataType **inputs_matrix = init_matrix<Input_DataType>(ROWS,COLS); //输入矩阵
    Weight_DataType **weights_matrix = init_matrix<Weight_DataType>(COL1S,COLS); //权重矩阵
    Output_DataType **ouputs_matrix = matrix_dot<Input_DataType,Weight_DataType,Output_DataType>
                                      (inputs_matrix,weights_matrix, ROWS, COLS, COL1S); // 计算矩阵乘法
    printf("inputs_matrix:\n");print_matrix(inputs_matrix, ROWS, COLS); // 打印输入矩阵
    printf("weights_matrix:\n");print_matrix(weights_matrix, COLS, COL1S); // 打印输入矩阵
    printf("ouputs_matrix:\n");print_matrix(ouputs_matrix, ROWS, COL1S); // 打印输出矩阵

    // 转换为一维数组然后再转换为总线类型，类似DDR，其数据为总线类型数据，保证按照总线方式传输

    Input_DataType *inputs_matrix1 = *inputs_matrix;
    Weight_DataType *weights_matrix1 = *weights_matrix;
    Transfer_DataType *transfer_inputs_matrix= (Transfer_DataType*)inputs_matrix1;
    Transfer_DataType *transfer_weights_matrix= (Transfer_DataType*)weights_matrix1;
    print_vec(inputs_matrix1,ROWS*COLS); //转换为Input_DataType指针打印一维数组
    print_vec((Input_DataType*)transfer_inputs_matrix,ROWS*COLS); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组

    // 根据参数生成指令
    int insn_count = 6; // 生成两个load指令 
    SAAInsn instruction[insn_count]; // 指令独联体，用于赋值指令操作数，通用
    Instruct_DataType instruction_data[insn_count]; // 指令数据，用于传输指令给SAA
    
    //第一个指令
    instruction[0].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
    instruction[0].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
    instruction[0].mem.dram_base = 0;               // DRAM索引 32位
    instruction[0].mem.buffer_base = 0;             // buffer行索引 16位
    instruction[0].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
    instruction[0].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
    instruction[0].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位
    printBinary_instruction(instruction[0]);    // 输出指令的二进制

    //第二个指令
    instruction[1].mem.opcode = OPCODE_LOAD;        // LOAD操作码 3位
    instruction[1].mem.buffer_id = INPUT_BUFFER_ID; // 存入输入缓冲区 3位
    instruction[1].mem.dram_base = 0;               // DRAM索引偏移1个总线位宽，以总线为步长
    instruction[1].mem.buffer_base = 4;             // buffer行索引 16位 ,从第四行开始
    instruction[1].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行 16位
    instruction[1].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列 16位
    instruction[1].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同 16位

   //第三个指令
    instruction[2].mem.opcode = OPCODE_LOAD;        // LOAD操作码3位
    instruction[2].mem.buffer_id = WEIGHT_BUFFER_ID; // 存入输入缓冲区3位
    instruction[2].mem.dram_base = 0;               // DRAM基地址32位
    instruction[2].mem.buffer_base = 0;             // buffer基地址16位 ,从第四行开始
    instruction[2].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
    instruction[2].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
    instruction[2].mem.x_stride = MATRIX_WIDTH;     // 假设步进与列数相同16位

   //第四个指令（权重预加载指令）
    instruction[3].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
    instruction[3].com.compute_type = WEIGHT_PRELOAD;  // 权重预加载 3位
    instruction[3].com.weigth_addr = 0;                // 权重读取起始行 32位
    instruction[3].com.input_addr = 0;                 // 没用到，因为是权重预加载指令
    instruction[3].com.output_addr = 0;                // 没用到，因为是权重预加载指令
    instruction[3].com.weight_switch = 0;              // 使用第一个权重加载
    instruction[3].com.compute_switch = 0;             // 没用到，因为是权重预加载指令
    instruction[3].com.accumulate = 0;                 // 没用到，因为是权重预加载指令

   //第五个指令（计算指令）
    instruction[4].com.opcode = OPCODE_COMPUTE;        // COMPUTE操作码 3位
    instruction[4].com.compute_type = COMPUTE;         // 计算指令 3位
    instruction[4].com.weigth_addr = 0;                // 没用到，因为是权重预加载指令
    instruction[4].com.input_addr = 0;                 // 输入读取起始行
    instruction[4].com.output_addr = 0;                // 输出写入起始行
    instruction[4].com.weight_switch = 0;              // 没用到，因为是权重预加载指令
    instruction[4].com.compute_switch = 0;             // 使用前面的权重预加载寄存器计算
    instruction[4].com.accumulate = 0;                 // 第一次计算不累加，刷新累加器

   //第六个指令（存储指令）
    instruction[5].mem.opcode = OPCODE_STORE;        // STORE操作码3位
    instruction[5].mem.buffer_id = 0;                //没用到，默认是以输出buffer输出 3位
    instruction[5].mem.dram_base = 0;               // DRAM基地址32位
    instruction[5].mem.buffer_base = 0;             // buffer基地址16位 
    instruction[5].mem.y_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH行16位
    instruction[5].mem.x_size = MATRIX_WIDTH;       // 加载MATRIX_WIDTH列16位
    instruction[5].mem.x_stride = 2*MATRIX_WIDTH;     // 假设步进与列数相同16位

    // 转换为指令类型，并输出查看

    for (int i = 0; i < insn_count; i++) 
    {
        std::memcpy(&instruction_data[i], &instruction[i], sizeof(MemIns)); //转换指令结构体为128位指令数据类型
        // printBinary(instruction_data[i],INSTRUCT_WIDTH);                 // 输出指令的二进制
        // printf("\n");
    }
    
    // 定义输出一维数组，注意自定义类型无法自动初始化为0，除非定义为静态变量或者全局变量
    static Output_DataType outputs_matrix_hw[ROWS*COL1S] = {0}; // 输出数组，注意应该以Output_DataType定义
    Transfer_DataType *transfer_outputs_matrix =(Transfer_DataType *) outputs_matrix_hw;// 转换为Transfer_DataType类型给SAA

    // 执行saa_top硬件，加载指令，只执行load函数，注意，传入的是指针类型
    saa_top(insn_count,
        instruction_data,
        transfer_inputs_matrix, 
        transfer_weights_matrix, 
        transfer_outputs_matrix,
        input_buffer, 
        weight_buffer,
        output_buffer); 
    
    //检查缓冲区
    printf("input_buffer:\n");print_buffer(input_buffer,0,10);
    printf("weight_buffer:\n");print_buffer(weight_buffer,0,10);
    printf("output_buffer:\n");print_buffer(output_buffer,0,10);

    // 检查输出
    print_vec((Output_DataType*)transfer_outputs_matrix,ROWS*COL1S); //转换为Input_DataType指针打印一维数组//转换为Input_DataType指针打印一维数组
    printf("outputs_matrix_hw:\n");print1D_2DArray(outputs_matrix_hw,ROWS,COL1S); // 将一维数组按二维打印

    return 0;
}



// // 7 测试使用数据依赖完成并行，比较使用依赖和不适用依赖进行并行是否会造成执行错误
// #include <iostream>

// int main() {

//     return 0;
// }
