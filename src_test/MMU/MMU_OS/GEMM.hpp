#ifndef GEMM_HPP
#define GEMM_HPP

#include "../SAA.h"
#include <cstdlib> // 包含 rand 函数的声明
#include <cstdio> // 包含 printf 函数的声明


void gemm_kernel(Weight_DataType input[][MATRIX_WIDTH], Weight_DataType weight[][MATRIX_WIDTH],Psum_DataType out[][MATRIX_WIDTH]);


//-----------------------------------------debug--------------------------------------//
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
void print_vec(T input_vec[])
{
    for(int i = 0; i < MATRIX_WIDTH; i++)
    {
        printf("%d,", (int)input_vec[i]);
    }
	printf("\n");
	printf("\n");
}

//打印矩阵
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

#endif
