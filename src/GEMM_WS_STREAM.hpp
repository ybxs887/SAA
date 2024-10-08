#ifndef GEMM_WS_HPP
#define GEMM_WS_HPP


#include "SAA.h"
//-----------------------------------------脉动阵列--------------------------------------//

//使用这种定义，容易导致HLS难以优化，因此完成功能后需要将函数去除
//定义WS模式的PE处理单元，有两个预加载寄存器，T是输入和权重的数据类型，U是部分和的数据类型
template<typename T0, typename T, typename U>
struct Process_Element
{
	T0 input;        // 输入
    T weight1;      // 第一个权重寄存器,预加载寄存器
    T weight2;      // 第二个权重寄存器,预加载寄存器
	U psum;         // 部分和

    // PE单元复位

	void reset()
	{
#pragma HLS INLINE
		input = 0;
		psum = 0;
		weight1 = 0;
		weight2 = 0;
	}

	// 设置PE权重，sw_load代表设置哪个权重，那么就使用另一个权重计算
	void set_pe_weights(T new_weight , Weight_Switch_DataType sw_load)
	{
#pragma HLS INLINE
		weight1 = sw_load ? weight1 : new_weight; // 如果sw_load=0，那就加载新权重到weight1，否则保持
		weight2 = sw_load ? new_weight : weight2; // 如果sw_load=1，那就加载新权重到weight2，否则保持
	}

	// 执行一次脉动，进行乘法累加操作，参数是输入和部分和，保存输入
	void process(T0 new_input,U last_sum , Compute_Switch_DataType sw_calc)
	{
#pragma HLS INLINE
        input = new_input; // 加载输入
        T weight= sw_calc ? weight2 : weight1; //如果sw_calc=0，那就使用weight1计算，否则使用weight2
		psum = input * weight + last_sum; // 乘累加操作
	}
};


//定义脉动阵列以及脉动函数
template<typename T0, typename T, typename U, int LEN>
struct Systolic_Array
{
    // 定义由若干PE单元组成的脉动阵列方阵
    Process_Element<T0,T,U> pe[LEN][LEN];

    // 脉动阵列复位操作
	void reset()
	{
    reset_loop:
		for (int r = 0; r < LEN; r++)
        {
#pragma HLS UNROLL //循环展开
			for (int c = 0; c < LEN; c++)
				pe[r][c].reset();
        }
	}

    // 定义脉动阵列的权重设置函数，入口参数为weight_buffer用于读取权重，是转置读取权重矩阵
    //wb_start_addr代表权重矩阵开始地址，data_length代表需要读取多少行MATRIX_WIDTH，sw表明设置哪个权重，en代表启用
    void set_array_weights(Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
                          Buffer_Addr_DataType wb_start_addr ,Weight_Switch_DataType sw_load )
	{
// printf("wb_start_addr:%d\n",(int)wb_start_addr);

// #pragma HLS INLINE recursive
        // 遍历权重缓冲区指定长度，并将权重分配给脉动阵列，按行寻址
    set_weights_loop:
        for (int i = 0; i < MATRIX_WIDTH; ++i) {
// #pragma HLS UNROLL //循环展开
#pragma HLS PIPELINE
            for (int j = 0; j < MATRIX_WIDTH; ++j) {
#pragma HLS UNROLL //循环展开
                    // 为每个PE单元设置权重
                    pe[i][j].set_pe_weights(weight_buffer[wb_start_addr + i][j], sw_load);
                    // pe[i][j].weight1 = sw_load ? pe[i][j].weight1 : weight_buffer[buffer_idx][j]; // 如果sw_load=0，那就加载新权重到weight1，否则保持
                    // pe[i][j].weight2 = sw_load ? weight_buffer[buffer_idx][j] : pe[i][j].weight2; // 如果sw_load=1，那就加载新权重到weight2，否则保持
            }
        }
    }

    //--------------------------------未优化HLS------------------------------//
    // 定义BMM块矩阵乘法核函数
    void BMM_kernel(Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
            Output_DataType accumulator_buffer[ACCUMULATOR_BUFFER_WIDTH][MATRIX_WIDTH],
            Buffer_Addr_DataType ib_start_addr,Buffer_Addr_DataType ob_start_addr, 
            Compute_Switch_DataType sw_calc, Compute_Accumulate_DataType accumulate)
    {
// #pragma HLS INLINE recursive

        // 完成一次矩阵乘法计算需要的脉动次数,在这里row=col=col1=MATRIX_WIDTH
        int total_pulse = 3 * MATRIX_WIDTH - 1; // 为了不频繁reset，多脉动一次除去最后一个PE的psum
    total_pulse_loop:
        for (int t = 0; t < total_pulse; t++) {
#pragma HLS PIPELINE II=1//流水线

        // 为当前脉动生成输入向量

            //初始化输入向量
            T0 input_vector[MATRIX_WIDTH] = {0};
        input_vec_init_loop:
            for(int k = 0; k < MATRIX_WIDTH; k++)// 对角输出向量,初始化为0
            {
#pragma HLS UNROLL //循环展开
                input_vector[k]=0;  
            }
            

            // 计算输入向量的起始索引和结束索引
            int start_index = std::max(0, t - MATRIX_WIDTH + 1); //计算得到右上角元素的起始横坐标
            int end_index = std::min(t, MATRIX_WIDTH - 1);//计算得到右上角元素的起始纵坐标

            // 根据脉动的当前步骤填充输入向量
        input_vec_loop:
            for (int k = 0; k <= MATRIX_WIDTH-1; k++) {
#pragma HLS UNROLL //循环展开
                if (k >= start_index && k <= end_index) {
                    int row_index = k;      //横坐标递增
                    int col_index = t - k;  //纵坐标递减
                    input_vector[row_index] = input_buffer[ib_start_addr+col_index][row_index];//反向对角化
                }
            }

        // 遍历整个脉动阵列执行脉动计算,从右下角更新脉动阵列
        pulse_loop:
            for (int r = MATRIX_WIDTH - 1; r >= 0; r--) {
// #pragma HLS UNROLL //循环展开
                for (int c = MATRIX_WIDTH - 1; c >= 0; c--) {
#pragma HLS PIPELINE II=1//流水线
                    // 获取新的输入值,如果是第一列则是新的输入,否则是上一列的输入
                    T0 new_input = (c == 0) ? input_vector[r] : pe[r][c-1].input;
                    
                    //获取新的部分和，如果是第一行，则为0，否则是上一行的部分和
                    U last_psum = (r == 0) ? (U)0 : pe[r - 1][c].psum;
                    
                    // 执行脉动计算
                    pe[r][c].process(new_input, last_psum, sw_calc);
                    // T weight= sw_calc ? pe[r][c].weight2 : pe[r][c].weight1; //如果sw_calc=0，那就使用weight1计算，否则使用weight2
                    // pe[r][c].psum =new_input * weight + last_psum;// 乘累加操作
                }
            }

        // 写入输出buffer,与生成输入向量是相反操作
            int t1=t - MATRIX_WIDTH + 1; // t减去 MATRIX_WIDTH -1 使得第一个输出时，t1为0

            //初始化输出向量
            U output_vector[MATRIX_WIDTH] = {0};
        output_vec_init_loop:
            for(int k = 0; k < MATRIX_WIDTH; k++) // 从脉动阵列最后一行提取psum结果
            {
#pragma HLS UNROLL //循环展开
                output_vector[k]=pe[MATRIX_WIDTH-1][k].psum;
            }

            // 计算输入向量的起始索引和结束索引
            start_index = std::max(0, t1 - MATRIX_WIDTH + 1); //计算得到右上角元素的起始横坐标
            end_index = std::min(t1, MATRIX_WIDTH - 1);//计算得到右上角元素的起始纵坐标

            // 根据脉动的当前步骤填充输出矩阵，accumulate标志是选择刷新还是继续累加
        output_vec_loop:
            for (int k = 0; k <= MATRIX_WIDTH - 1; k++) {
// #pragma HLS UNROLL //循环展开
#pragma HLS PIPELINE II=1//流水线
                if (k >= start_index && k <= end_index) {
                    int row_index = k;
                    int col_index = t1 - k;
                    // 写入输出矩阵,同时将结果转换为Output_DataType
                    accumulator_buffer[ob_start_addr+col_index][row_index] = accumulate ? 
                    (U)(accumulator_buffer[ob_start_addr+col_index][row_index] + output_vector[row_index]): output_vector[row_index];
                }
            }
        }
    }

	// // debug
	// void print_pe()
	// {
	// 	printf("Pe now is below:\n");
	// 	for(int i = 0; i < LEN; i++)
	// 	{
	// 		for(int j = 0; j < LEN; j++)
	// 		{
	// 			printf("(%d, %d, %d, %d)\t", (int)pe[i][j].input, (int)pe[i][j].weight1, (int)pe[i][j].weight2,(int)pe[i][j].psum);
	// 		}
	// 		printf("\n");
	// 	}
	// }
};

// 脉动阵列例化,声明表示在源文件定义
extern Systolic_Array<Input_DataType, Weight_DataType, Psum_DataType, MATRIX_WIDTH> systolic_array;




