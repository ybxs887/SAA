#include "GEMM_WS.hpp"
#include "LOAD.hpp"



// //1 用于测试原始的脉动阵列的BMM_kernel函数

// int main()
// {
//     // 脉动阵列例化
//     Systolic_Array<Weight_DataType, Psum_DataType, MATRIX_WIDTH> systolic_array;    

//     //权重缓冲区的地址设置
//     int wb_start_addr=0;
//     int ib_start_addr=0;
//     int ob_start_addr=0;

//     //进行矩阵乘法的两个矩阵
//     int row=3;
//     int col=3;
//     int col1=3;
    
//     // 从文件加载权重
//     load_weights_from_file("weights.txt", weight_buffer);
//     load_weights_from_file("inputs.txt", input_buffer);

//     //查看加载的权重
//     print_iw_matrix(input_buffer,row,col);
//     print_iw_matrix(weight_buffer,col,col1);

//     // 调用set_array_weights函数来设置权重
//     // 执行GEMM操作
//     systolic_array.reset(); //只在第一次使用GEMM时reset就行，因为脉动阵列多脉动了一次，除weight外的
//     systolic_array.set_array_weights(weight_buffer, wb_start_addr, 0, 1); //设置weight1
//     systolic_array.BMM_kernel(input_buffer, output_buffer, ib_start_addr, ob_start_addr, 0, 1);//使用weight1计算
//     print_ao_matrix(output_buffer,ob_start_addr,3,3);

//     // 执行GEMM操作

//     systolic_array.set_array_weights(weight_buffer, wb_start_addr, 1, 1); //设置weight2
//     ob_start_addr=3;//从第四行开始写入输出buffer
//     systolic_array.BMM_kernel(input_buffer, output_buffer, ib_start_addr, ob_start_addr, 1, 1);//使用weight2计算

//     //检查输出
//     print_ao_matrix(output_buffer,0,3,3);
    
//     //查看pe状态
//     systolic_array.print_pe(); 
// }



// //2 用于测试BMM函数，完成pingpang操作的矩阵乘法

// int main()
// {

//     //权重缓冲区的地址设置
//     int wb_start_addr=0;
//     int ib_start_addr=0;
//     int ob_start_addr=0;

//     //进行矩阵乘法的两个矩阵
//     int row=6;
//     int col=6;
//     int col1=6;
    
//     // 从文件加载权重
//     load_weights_from_file("weights.txt", weight_buffer);
//     load_weights_from_file("inputs.txt", input_buffer);

//     //查看加载的权重

//     // 调用set_array_weights函数来设置权重
    
//     //初始化脉动阵列
//     systolic_array.reset(); //只在第一次使用GEMM时reset就行，因为脉动阵列多脉动了一次，除weight外的都清零

//     // 执行BMM操作
//     BMM(input_buffer,weight_buffer,output_buffer,wb_start_addr,ib_start_addr,ob_start_addr, 0,0,0,1);

//     // 执行BMM操作
//     BMM(input_buffer,weight_buffer,output_buffer,wb_start_addr,ib_start_addr,ob_start_addr, 0,0,1,1);

//     //检查输出
//     // print_ao_matrix(output_buffer,0,3,3);
    
//     //查看pe状态
//     systolic_array.print_pe(); 
// }


// //3 用于测试load函数，将脉动阵列倍数大小矩阵加载到缓冲区

// int main()
// {
//     //权重缓冲区的地址设置
//     int wb_start_addr=0;
//     int ib_start_addr=0;
//     int ob_start_addr=0;

//     //进行矩阵乘法的两个矩阵
//     int row=6;
//     int col=6;
//     int col1=6;
    
//     // 从文件加载权重
//     load_from_file("weights.txt", weight_buffer,wb_start_addr,6,6);
//     load_from_file("inputs.txt", input_buffer,ib_start_addr,6,6);

//     //查看加载的权重
//     print_buffer(input_buffer,ib_start_addr,10);
//     print_buffer(weight_buffer,wb_start_addr,10);

//     print_buffer_matrix(input_buffer,ib_start_addr,9,9);
//     print_buffer_matrix(weight_buffer,wb_start_addr,9,9);

//     print_buffer_matrix(output_buffer,ob_start_addr,12,12);


// }


// //4 用于测试GEMM_OS函数

// int main()
// {
//     //权重缓冲区的地址设置
//     int wb_start_addr=0;
//     int ib_start_addr=0;
//     int ob_start_addr=0;

//     //进行矩阵乘法的两个矩阵
//     int row=6;
//     int col=6;
//     int col1=6;
    
//     // 从文件加载权重
//     load_from_file("weights.txt", weight_buffer,wb_start_addr,6,6);
//     load_from_file("inputs.txt", input_buffer,ib_start_addr,6,6);

//     //查看加载的权重
//     print_buffer(input_buffer,ib_start_addr,12);
//     print_buffer(weight_buffer,wb_start_addr,12);
//     print_buffer_matrix(input_buffer,ib_start_addr,6,6);
//     print_buffer_matrix(weight_buffer,wb_start_addr,6,6);

//     // 开始计时
//     auto start = std::chrono::high_resolution_clock::now();

//     //进行矩阵乘法
//     GEMM_OS(input_buffer,weight_buffer,output_buffer,
//         wb_start_addr,ib_start_addr,ob_start_addr,row, col, col1, 1);

//     // 结束计时
//     auto end = std::chrono::high_resolution_clock::now();

//     // 计算持续时间
//     std::chrono::duration<double> duration = end - start;
//     double duration_in_seconds = duration.count();
//     std::cout << "GEMM_OS execution time: " << duration_in_seconds << " s" << std::endl;

//     //打印output_buffer中的输出矩阵
//     print_buffer_matrix(output_buffer,ob_start_addr,6,6);
// }



//5 用于测试能够进行任意大小矩阵乘法的GEMM函数

int main()
{
    //权重缓冲区的地址设置
    int wb_start_addr=0;
    int ib_start_addr=0;
    int ob_start_addr=0;

    //进行矩阵乘法的两个矩阵
    int I0=6;
    int K0=6;
    int J0=6;
    
    int I_pad,J_pad,K_pad;


	//计算矩阵填充后分块数
	const int I = I0 / MATRIX_WIDTH + (I0 % MATRIX_WIDTH != 0);
    const int J = J0 / MATRIX_WIDTH + (J0 % MATRIX_WIDTH != 0);
    const int K = K0 / MATRIX_WIDTH + (K0 % MATRIX_WIDTH != 0);
    
	//计算填充后矩阵维度
	const int I_padded = I * MATRIX_WIDTH;
	const int J_padded = J * MATRIX_WIDTH;
	const int K_padded = K * MATRIX_WIDTH;

    printf("(%d,%d,%d)\n",I_padded,K_padded,J_padded);


    // 从文件加载权重
    load_from_file_pad("inputs.txt", input_buffer,ib_start_addr,I0,K0,I_padded,K_padded);
    load_from_file_pad("weights.txt", weight_buffer,wb_start_addr,K0,J0,K_padded,J_padded);

    //查看加载的权重
    print_buffer(input_buffer,ib_start_addr,10);
    print_buffer(weight_buffer,wb_start_addr,10);
    print_buffer_matrix_pad(input_buffer,ib_start_addr,I0,K0,I_padded,K_padded);
    print_buffer_matrix_pad(weight_buffer,wb_start_addr,K0,J0,K_padded,J_padded);

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    //进行矩阵乘法
    GEMM_OS(input_buffer,weight_buffer,output_buffer,
        wb_start_addr,ib_start_addr,ob_start_addr,I_padded, K_padded, J_padded, 1);

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    // 计算持续时间
    std::chrono::duration<double> duration = end - start;
    double duration_in_seconds = duration.count();
    std::cout << "GEMM_OS execution time: " << duration_in_seconds << " s" << std::endl;

    //打印output_buffer中的输出矩阵
    print_buffer_matrix_pad(output_buffer,ob_start_addr,I0,J0,I_padded,J_padded);
}
