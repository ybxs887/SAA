//本代码实现了脉动阵列的权重固定模式
#include "GEMM_WS.hpp"


// 脉动阵列例化
Systolic_Array<Input_DataType,Weight_DataType, Psum_DataType, MATRIX_WIDTH> systolic_array;

//调用BMM_kernel函数，实现BMM函数，调用了双缓冲
void BMM(Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
        Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
        Psum_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH],
        int wb_start_addr,int ib_start_addr,int ac_start_addr,
        bool sw_load, bool sw_calc,bool accumulate, bool enable)
{
#pragma HLS INTERFACE bram port=input_buffer
#pragma HLS INTERFACE bram port=weight_buffer
#pragma HLS INTERFACE bram port=output_buffer
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_buffer complete dim=2

    //如果未启用直接返回
    if(enable==0) return;

    //设置权重
    systolic_array.set_array_weights(weight_buffer, wb_start_addr, sw_load, enable); //设置sw_load
    
    //进行计算
    systolic_array.BMM_kernel(input_buffer, output_buffer, ib_start_addr, ac_start_addr, sw_calc, accumulate,enable);//使用sw_calc计算

}


// //GEMM_OS函数,复用的是输出矩阵，进行累加(无双缓冲，只使用一个权重寄存器)
// //注意，该函数不包括将input\weight移入缓冲区,因此调用此函数的前提是已经把矩阵载入
// //I0、K0、J0代表输入input矩阵、weight矩阵的行列
// void GEMM_OS(Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
//         Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
//         Psum_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH],
//         int wb_start_addr,int ib_start_addr,int ac_start_addr,int I0, int K0, int J0, 
//         bool enable)
// {
//     //如果未启用直接返回
//     if(enable==0) return;

// 	//计算输入矩阵的块，没有考虑padding情况
// 	const int I = I0 / MATRIX_WIDTH; //计算行的分块数
// 	const int J = J0 / MATRIX_WIDTH; //计算列的分块数
// 	const int K = K0 / MATRIX_WIDTH; //计算公共维度的分块数

//     for (int i = 0; i < I; i++) { // 对输出矩阵的行循环
//         for (int j = 0; j < J; j++) { // 对输出矩阵的列循环
//         	const int ac_addr = ac_start_addr + (i*J + j)*MATRIX_WIDTH; // 计算输出矩阵的地址i行J列加上当前的j列
// 			for (int k = 0; k < K; k++) {
//         		const int ib_addr = ib_start_addr + (i*K + k)*MATRIX_WIDTH; // 计算输入矩阵的地址i行K列加上当前的k列
// 				const int wb_addr = wb_start_addr + (k*J + j)*MATRIX_WIDTH; // 计算权重矩阵的地址k行J列加上当前的j列

// 				//设置权重(无双缓冲，只使用一个权重寄存器)
//     			systolic_array.set_array_weights(weight_buffer, wb_addr, 0, enable); //设置weight1
// 				if (k == 0) // 如果是第一个块，刷新累加器中的累加值
// 					systolic_array.BMM_kernel(input_buffer, output_buffer, ib_addr, ac_addr, 0, 0,enable);//使用weight1计算
// 				else // 其他块进行累加
// 					systolic_array.BMM_kernel(input_buffer, output_buffer, ib_addr, ac_addr, 0, 1,enable);//使用weight1计算
// 			}
// 		}
// 	}
// }


//GEMM_OS函数,复用的是输出矩阵，进行累加(双缓冲，使用两个权重寄存器)(在I和J内进行双缓冲)
// void GEMM_OS(Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
//         Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
//         Psum_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH],
//         int wb_start_addr, int ib_start_addr, int ac_start_addr, int I0, int K0, int J0,
//         bool enable)
// {
//     if(enable==0) return;

//     const int I = I0 / MATRIX_WIDTH;
//     const int J = J0 / MATRIX_WIDTH;
//     const int K = K0 / MATRIX_WIDTH;

//     bool pingpang = 0; // 用于切换权重寄存器,最先使用weight1

//     for (int i = 0; i < I; i++) {
//         for (int j = 0; j < J; j++) {
//             const int ac_addr = ac_start_addr + (i*J + j)*MATRIX_WIDTH;

//             // 预加载第一个权重块
//             const int wb_addr_first = wb_start_addr + (0*J + j)*MATRIX_WIDTH;
//             systolic_array.set_array_weights(weight_buffer, wb_addr_first, pingpang , enable); //最先加载weight1

//             for (int k = 0; k < K; k++) {
//                 const int ib_addr = ib_start_addr + (i*K + k)*MATRIX_WIDTH;
//                 const int wb_addr_next = wb_start_addr + ((k+1)*J + j)*MATRIX_WIDTH; //按列增1
//                 // 切换权重寄存器
//                 pingpang = !pingpang;

//                 // 在计算当前块的同时，预加载下一个权重块,当k=K-2时将最后一个权重块加载完毕
//                 if (k < K - 1) {
//                     systolic_array.set_array_weights(weight_buffer, wb_addr_next, pingpang, enable);
//                 }

//                 // 计算使用当前权重寄存器中的权重
// 				bool accumulate = k == 0 ? 0 : 1; // 如果是第一个块，刷新累加器中的累加值,其他块进行累加
//                 systolic_array.BMM_kernel(input_buffer, output_buffer, ib_addr, ac_addr,
//                                           !pingpang, accumulate , enable); //计算上一次加载的权重
//             }
//         }
//     }
// }


// //GEMM_OS函数,复用的是输出矩阵，进行累加
// //(双缓冲，使用两个权重寄存器)(三个循环内都进行双缓冲)(权重和输入都是行块存储)
// void GEMM_OS(Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
//         Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
//         Psum_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH],
//         int wb_start_addr, int ib_start_addr, int ac_start_addr, int I0, int K0, int J0,
//         bool enable)
// {
//     if(enable==0) return;

//     const int I = I0 / MATRIX_WIDTH;
//     const int J = J0 / MATRIX_WIDTH;
//     const int K = K0 / MATRIX_WIDTH;

//     bool pingpang = 0; // 用于切换权重寄存器,最先使用weight1
// 	int k1 = 0; // 超前k值一位，当k值超过最大值时归零
// 	int t=0; // 激励跳转了多少次列
	
// 	// 预加载第一个权重块
// 	const int wb_addr_first = wb_start_addr;
// 	systolic_array.set_array_weights(weight_buffer, wb_addr_first, pingpang , enable); //最先加载weight1

//     for (int i = 0; i < I; i++) {
//         for (int j = 0; j < J; j++) {
//             const int ac_addr = ac_start_addr + (i*J + j)*MATRIX_WIDTH;
//             for (int k = 0; k < K; k++) {
//                 const int ib_addr = ib_start_addr + (i*K + k)*MATRIX_WIDTH;
//                 // 计算下一个权重块的地址
// 				k1 = ((k+1) % K);
// 				t = (k1 == 0) ? ((t + 1 == J) ? 0 : t + 1) : t; //如果满足跳转就加1,最多达到J时归零代表权重重新从0开始加载
//                 const int wb_addr_next = wb_start_addr + (k1 * J+ t)*MATRIX_WIDTH; //t代表跳转了多少列

// 				// 切换权重寄存器
//                 pingpang = !pingpang;

//                 // 在计算当前块的同时，预加载下一个权重块
//                 if (!(i == I - 1 && j == J - 1 && k == K - 1 )) { // 最后一行少一个,排除特例
//                     systolic_array.set_array_weights(weight_buffer, wb_addr_next, pingpang, enable);
//                 }

//                 // 计算使用当前权重寄存器中的权重
// 				bool accumulate = k == 0 ? 0 : 1; // 如果是第一个块，刷新累加器中的累加值,其他块进行累加
//                 systolic_array.BMM_kernel(input_buffer, output_buffer, ib_addr, ac_addr,
//                                           !pingpang, accumulate , enable); //计算上一次加载的权重
//             }
//         }
//     }
// }

// //GEMM_OS函数,复用的是输出矩阵，进行累加，在片上进行计算
// //(双缓冲，使用两个权重寄存器)(三个循环内都进行双缓冲)(权重和输入都是行块存储)(任意大小矩阵输入)
// void GEMM_OS(Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
//         Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
//         Psum_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH],
//         int wb_start_addr, int ib_start_addr, int ac_start_addr, int I0, int K0, int J0,
//         bool enable)
// {
//     if(enable==0) return;

//     const int I = I0 / MATRIX_WIDTH;
//     const int J = J0 / MATRIX_WIDTH;
//     const int K = K0 / MATRIX_WIDTH;

//     bool pingpang = 0; // 用于切换权重寄存器,最先使用weight1
// 	int k1 = 0; // 超前k值一位，当k值超过最大值时归零
// 	int t=0; // 激励跳转了多少次列
	
// 	// 预加载第一个权重块
// 	const int wb_addr_first = wb_start_addr;
// 	systolic_array.set_array_weights(weight_buffer, wb_addr_first, pingpang , enable); //最先加载weight1

//     for (int i = 0; i < I; i++) {
//         for (int j = 0; j < J; j++) {
//             const int ac_addr = ac_start_addr + (i*J + j)*MATRIX_WIDTH;
//             for (int k = 0; k < K; k++) {
//                 const int ib_addr = ib_start_addr + (i*K + k)*MATRIX_WIDTH;
//                 // 计算下一个权重块的地址
// 				k1 = ((k+1) % K);
// 				t = (k1 == 0) ? ((t + 1 == J) ? 0 : t + 1) : t; //如果满足跳转就加1,最多达到J时归零代表权重重新从0开始加载
//                 const int wb_addr_next = wb_start_addr + (k1 * J+ t)*MATRIX_WIDTH; //t代表跳转了多少列

// 				// 切换权重寄存器
//                 pingpang = !pingpang;

//                 // 在计算当前块的同时，预加载下一个权重块
//                 if (!(i == I - 1 && j == J - 1 && k == K - 1 )) { // 最后一行少一个,排除特例
//                     systolic_array.set_array_weights(weight_buffer, wb_addr_next, pingpang, enable);
//                 }

//                 // 计算使用当前权重寄存器中的权重
// 				bool accumulate = k == 0 ? 0 : 1; // 如果是第一个块，刷新累加器中的累加值,其他块进行累加
//                 systolic_array.BMM_kernel(input_buffer, output_buffer, ib_addr, ac_addr,
//                                           !pingpang, accumulate , enable); //计算上一次加载的权重
//             }
//         }
//     }
// }



//GEMM_OS函数,复用的是权重矩阵，进行累加，在片上进行计算
//(双缓冲，使用两个权重寄存器)(三个循环内都进行双缓冲)(权重和输入都是行块存储)(任意大小矩阵输入)
void GEMM_OS(Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
        Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
        Psum_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH],
        int wb_start_addr, int ib_start_addr, int ac_start_addr, int I0, int K0, int J0,
        bool enable)
{
    if(enable==0) return;

    const int I = I0 / MATRIX_WIDTH;
    const int J = J0 / MATRIX_WIDTH;
    const int K = K0 / MATRIX_WIDTH;

    bool pingpang = 0; // 用于切换权重寄存器,最先使用weight1
	int k1 = 0; // 超前k值一位，当k值超过最大值时归零
	int t=0; // 激励跳转了多少次列
	
	// 预加载第一个权重块
	const int wb_addr_first = wb_start_addr;
	systolic_array.set_array_weights(weight_buffer, wb_addr_first, pingpang , enable); //最先加载weight1

    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            const int ac_addr = ac_start_addr + (i*J + j)*MATRIX_WIDTH;
            for (int k = 0; k < K; k++) {
                const int ib_addr = ib_start_addr + (i*K + k)*MATRIX_WIDTH;
                // 计算下一个权重块的地址
				k1 = ((k+1) % K);
				t = (k1 == 0) ? ((t + 1 == J) ? 0 : t + 1) : t; //如果满足跳转就加1,最多达到J时归零代表权重重新从0开始加载
                const int wb_addr_next = wb_start_addr + (k1 * J+ t)*MATRIX_WIDTH; //t代表跳转了多少列

				// 切换权重寄存器
                pingpang = !pingpang;

                // 在计算当前块的同时，预加载下一个权重块
                if (!(i == I - 1 && j == J - 1 && k == K - 1 )) { // 最后一行少一个,排除特例
                    systolic_array.set_array_weights(weight_buffer, wb_addr_next, pingpang, enable);
                }

                // 计算使用当前权重寄存器中的权重
				bool accumulate = k == 0 ? 0 : 1; // 如果是第一个块，刷新累加器中的累加值,其他块进行累加
                systolic_array.BMM_kernel(input_buffer, output_buffer, ib_addr, ac_addr,
                                          !pingpang, accumulate , enable); //计算上一次加载的权重
            }
        }
    }
}