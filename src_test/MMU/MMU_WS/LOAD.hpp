#ifndef LOAD_HPP
#define LOAD_HPP

#include "../../SAA.h"
#include <fstream>
#include <iostream>

void load_weights_from_file(const char* filename, Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < MATRIX_WIDTH; ++i) {
        for (int j = 0; j < MATRIX_WIDTH; ++j) {
            if (file.eof()) break;  // 文件结束，退出
            file >> weight_buffer[i][j];
        }
    }
    file.close();
}


//从文件中读取任意大的矩阵到input/weight_buffer中（相当于把大矩阵以块的形式存放到缓冲区中）
template<typename T>
void load_from_file(const char* filename, T input_buffer[][MATRIX_WIDTH] ,int start_addr,int I0 ,int J0)
{

    //中间矩阵
    T buffer[I0][J0] = {{0}};

    //读取文件内容到矩阵
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < I0; ++i) {
        for (int j = 0; j < J0; ++j) {
            if (file.eof()) break;  // 文件结束，退出
            file >> buffer[i][j];
        }
    }
    file.close();

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
                        input_buffer[r+ob_addr][c] = buffer[row_index][col_index];
                    }
                }
        }
    }
}


//从文件中读取任意大的矩阵到input/weight_buffer中并自动对矩阵进行padding
template<typename T>
void load_from_file_pad(const char* filename, T input_buffer[][MATRIX_WIDTH] ,int start_addr,int I0 ,int J0,int I_padded ,int J_padded )
{

    //中间矩阵
    T buffer[I_padded][J_padded] = {{0}};

    //读取文件内容到矩阵
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < I_padded; ++i) {
        for (int j = 0; j < J_padded; ++j) {
            if (file.eof()) break;  // 文件结束，退出
            if (i<I0 && j<J0)
                file >> buffer[i][j];
            else
                buffer[i][j] = 0;
        }
    }
    printf("\n");
    file.close();

    //按块循环大矩阵
	const int I = I_padded / MATRIX_WIDTH; //计算行的分块数
	const int J = J_padded / MATRIX_WIDTH; //计算列的分块数

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
                        input_buffer[r+ob_addr][c] = buffer[row_index][col_index];
                    }
                }
        }
    }
}



#endif