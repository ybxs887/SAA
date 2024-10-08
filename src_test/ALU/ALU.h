#ifndef ALU_H
#define ALU_H

#include <ap_int.h>

// 定义操作类型
enum Operation { OP_ADD, OP_MAX, OP_MIN, OP_SHR, OP_SHL };

// 定义数据宽度和向量长度
#define DATA_WIDTH 32
#define VECTOR_LENGTH 4

// HLS针对FPGA的数据类型，这里使用32位整数
typedef ap_int<DATA_WIDTH> data_t;
typedef ap_int<DATA_WIDTH> vec_t[VECTOR_LENGTH];

void vector_operation(vec_t a, vec_t b, vec_t &result, Operation op, int shift_amount);


#endif
