#include <iostream>
#include <cstdlib>

// 包含向量运算单元的头文件
#include "ALU.h"

//使用ALU实现各操作
//relu
void apply_relu(vec_t input, vec_t &output) {
    vec_t zero = {0, 0, 0, 0};
    vector_operation(input, zero, output, OP_MAX, 0);
}

//normalization
void apply_normalization(vec_t input, vec_t &output, int shift_amount) {
    vector_operation(input, input, output, OP_SHR, shift_amount);
}

//maxpool
void apply_max_pooling(vec_t input, vec_t &output) {
    vec_t temp_output; // 临时存储每个2个元素的最大值
    
    for(int i = 0; i < 4; i += 2) {
        vector_operation(input + i, input + i + 1, temp_output, OP_MAX, 0);
        output[i / 2] = temp_output[0]; // 将每个2个元素的最大值存入output数组
    }
}


int main() {
    // 初始化输入向量，负值用于测试ReLU激活
    vec_t input = {1600,-100, 800, 400};
    vec_t zero = {0, 0, 0, 0}; // 用于ReLU激活
    vec_t output; // 存储各个操作的结果

    // 应用ReLU激活
    apply_relu(input,output);

	// 输出结果
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        std::cout << "relu Output[" << i << "] = " << output[i] << std::endl;
    }
	
    // 应用归一化，这里使用右移操作来模拟除以某个2的幂次
    int normalization_factor = 8; // 模拟除以256
    vec_t normalized_output = {0, 0, 0, 0}; // 注意：归一化结果的长度也是VECTOR_LENGTH/2

    apply_normalization(output,normalized_output,normalization_factor);

	// 输出结果
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        std::cout << "Normalized  Output[" << i << "] = " << normalized_output[i] << std::endl;
    }
	
    // 应用最大池化，池化窗口大小为2
    vec_t pooled_output = {0, 0}; // 注意：池化结果的长度是VECTOR_LENGTH/2

    apply_max_pooling(normalized_output,pooled_output);

	// 输出结果
    for (int i = 0; i < VECTOR_LENGTH / 2; i++) {
        std::cout << "pooled Output[" << i << "] = " << pooled_output[i] << std::endl;
    }

    return 0;
}

//测试
