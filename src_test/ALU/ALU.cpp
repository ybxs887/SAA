#include "ALU.h"

// 向量运算函数，可以完成最小值、最大值、加法、右移和左移运算
void vector_operation(vec_t a, vec_t b, vec_t &result, Operation op, int shift_amount) {
    // #pragma HLS INTERFACE s_axilite port=a bundle=BUS_A
    // #pragma HLS INTERFACE s_axilite port=b bundle=BUS_B
    // #pragma HLS INTERFACE s_axilite port=result bundle=BUS_R
    // #pragma HLS INTERFACE s_axilite port=op bundle=CTRL
    // #pragma HLS INTERFACE s_axilite port=shift_amount bundle=CTRL
    // #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS ARRAY_PARTITION variable=a complete
    #pragma HLS ARRAY_PARTITION variable=b complete
    #pragma HLS ARRAY_PARTITION variable=result complete
    
    Loop1:for (int i = 0; i < VECTOR_LENGTH; i++) {
        #pragma HLS UNROLL//完全展开循环，直接向量层级进行运算，降低延时
        switch (op) {
            case OP_ADD:
                result[i] = a[i] + b[i];
                break;
            case OP_MAX:
                result[i] = (a[i] > b[i]) ? a[i] : b[i];
                break;
            case OP_MIN:
                result[i] = (a[i] < b[i]) ? a[i] : b[i];
                break;
            case OP_SHR:
                result[i] = a[i] >> shift_amount;
                break;
            case OP_SHL:
                result[i] = a[i] << shift_amount;
                break;
            default:
                result[i] = 0;
                break;
        }
    }
}


