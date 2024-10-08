#include "GEMM.hpp"

int main()
{
    Weight_DataType input[MATRIX_WIDTH][MATRIX_WIDTH] = {0};
    Weight_DataType weight[MATRIX_WIDTH][MATRIX_WIDTH] = {0};
    Psum_DataType out[MATRIX_WIDTH][MATRIX_WIDTH] = {0};
    Psum_DataType out_ref[MATRIX_WIDTH][MATRIX_WIDTH] = {0};

    // 初始化输入和权重
    init_matrix(input);
    init_matrix(weight);

    printf("input:\n");
    print_matrix(input);

    printf("weight:\n");
    print_matrix(weight);

    // 进行软件矩阵乘法运算
    matrix_dot(input, weight, out_ref);

    // 调用gemm_kernel函数进行矩阵乘法计算
    gemm_kernel(input, weight, out);

    // 输出结果
    printf("ref:\n");
    print_matrix(out_ref);

    printf("res:\n");
    print_matrix(out);

    return 0;
}




// #include <stdio.h>

// #define LEN 4

// void print_diagonal(int matrix[LEN][LEN]) {
//     int max_pe_num = 2 * LEN - 1;
//     int max_len = (max_pe_num > LEN) ? max_pe_num : LEN;

//     for (int i = 0; i < max_len; i++) {
//         int pe_num = (i < LEN) ? (i + 1) : (2 * LEN - 1 - i);

//         int output_vector[LEN] = {0}; // 傻逼

//         for (int j = 0; j < pe_num; j++) {
//             int pos_y = (i >= LEN) ? i - LEN + j + 1 : j;
//             int pos_x = (i >= LEN) ? LEN - 1 - j : i - j;
//             if(i>=LEN)
//                 output_vector[j+i-LEN+1] = matrix[pos_y][pos_x];
//             else
//                 output_vector[j] = matrix[pos_y][pos_x];
//         }

//         // �������ΪLEN������
//         for (int k = 0; k < LEN; k++) {
//             printf("%d ", output_vector[k]);
//         }
//         printf("\n");
//     }
// }

// int main() {
//     int matrix[LEN][LEN] = {
//         {1, 2, 3, 4},
//         {5, 6, 7, 8},
//         {9, 10, 11, 12},
//         {13, 14, 15, 16}
//     };

//     printf("Diagonal elements of the matrix:\n");
//     print_diagonal(matrix);

//     return 0;
// }

