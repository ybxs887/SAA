#include "GEMM.hpp"

//定义PE处理单元
template<typename T, typename U>
struct Process_Element
{
	T input, weight;    // 输入和权重
    U psum;             // 部分和

    // PE单元复位
	void reset()
	{
		input = 0;
		weight = 0;
		psum = 0;
	}

    // 调用复位
	Process_Element()
	{
		reset();
	}

    // 乘累加操作
	void process(T input_i, T weight_i)
	{
		input = input_i;			// 输入从左往右传递
		weight = weight_i;			// 权重从上往下传递
		psum += input_i*weight_i;   // 计算当前输入和权重的乘累加 
	}
};

//定义脉动阵列以及脉动函数
template<typename T, typename U, int LEN>
struct Systolic_Array
{
	// 定义由若干PE单元组成的脉动阵列方阵
	Process_Element<T,U> pe[LEN][LEN];

    // 脉动阵列复位操作
	void reset(int row, int col)
	{
		for (int r = 0; r < row; r++)
			for (int c = 0; c < col; c++)
				pe[r][c].reset();
	}

	// pulse脉动函数，数据在脉动阵列的水平方向和竖直方向上均走了一步，即整个阵列的脉搏跳了一次
	void pulse(T input_vec[LEN], T weight_vec[LEN])
	{
		systolic_array_outer_loop: // 脉动外循环
		for (int i = 2*LEN - 2; i >= 0; i--) // 方阵从左上角走到右下角，总共需要2*LEN-1个step，沿着矩阵的边从一个对角走到另一个对角
		{
#pragma HLS UNROLL // 循环展开
			int pe_num = (i > LEN - 1)?(2 * LEN - 1 - i):(i + 1); // 计算每一个step需要处理的PE数，判断依据是是否越过主对角线
			systolic_array_inner_loop: // 脉动内循环
			for (int j = 0; j < pe_num; j++)
			{
//#pragma HLS PIPELINE II=1
				// 获取当前PE的坐标
				int pos_y = (i >= LEN)?LEN-1-j:i-j;
				int pos_x = (i >= LEN)?i-LEN+j+1:j;

				// 获取当前PE的左侧输入input和上方输入weight，当坐标为0时，就是外部向量输入，否则是相邻PE输入
				T input_get = (pos_y == 0)?input_vec[pos_x]:pe[pos_x][pos_y-1].input;
				T weight_get = (pos_x == 0)?weight_vec[pos_y]:pe[pos_x-1][pos_y].weight;
				// 利用a和b更新PE
				pe[pos_x][pos_y].process(input_get, weight_get);
			}
		}
	}

	// debug
	void print_pe()
	{
		printf("Pe now is below:\n");
		for(int i = 0; i < LEN; i++)
		{
			for(int j = 0; j < LEN; j++)
			{
				printf("(%d, %d, %d)\t", (int)pe[i][j].input, (int)pe[i][j].weight, (int)pe[i][j].psum);
			}
			printf("\n");
		}
	}

};


// 脉动阵列例化
Systolic_Array<Weight_DataType, Psum_DataType, MATRIX_WIDTH> systolic_array;

//读入矩阵，根据当前脉动步数获取每次的输入向量，相当于将传输过来的列向量对角化然后把向量输入
void Systolic_data_setup(Weight_DataType in_matrix[][MATRIX_WIDTH], Weight_DataType out_vec[MATRIX_WIDTH],bool sw,int i )
{
    for(int k = 0; k < MATRIX_WIDTH; k++)// 对角输出向量,初始化为0
        out_vec[k]=0;
    int out_step = 2*MATRIX_WIDTH-1; //有效输出次数
    if(i<out_step)
    {
        // out_vec[MATRIX_WIDTH]={0}; // 对角输出向量,初始化为0        
        int out_num = (i > MATRIX_WIDTH - 1) ? (out_step - i) : (i + 1);
        for(int j = 0; j < out_num; j++)
            {
				#pragma HLS UNROLL
                int index_x = 0;
                int index_y = 0;
                if(sw==0)//如果是输入向量
                {
                    index_x= (i >= MATRIX_WIDTH) ? i - MATRIX_WIDTH + j + 1 : j;
                    index_y= (i >= MATRIX_WIDTH) ? MATRIX_WIDTH - 1 - j : i - j;
                }
                else//如果是权重向量
                {
                    index_y= (i >= MATRIX_WIDTH) ? i - MATRIX_WIDTH + j + 1 : j;
                    index_x= (i >= MATRIX_WIDTH) ? MATRIX_WIDTH - 1 - j : i - j;
                }

                if(i>=MATRIX_WIDTH)
                    out_vec[j+i-MATRIX_WIDTH+1] =in_matrix[index_x][index_y];//剩下没有赋值的就是0
                else
                    out_vec[j] =in_matrix[index_x][index_y];//剩下没有赋值的就是0
            }
    }
}

//读取PE中的结果
void read_result(Psum_DataType out[][MATRIX_WIDTH])
{
	for(int i = 0; i < MATRIX_WIDTH; i++)
	{
		for(int j = 0; j < MATRIX_WIDTH; j++)
			out[i][j]=systolic_array.pe[i][j].psum;
	}
}


//进行块的矩阵乘法的部分
void gemm_kernel(Weight_DataType input[][MATRIX_WIDTH], Weight_DataType weight[][MATRIX_WIDTH],Psum_DataType out[][MATRIX_WIDTH])
{
    //初始化脉动阵列
	systolic_array.reset(MATRIX_WIDTH, MATRIX_WIDTH);

	// 脉动阵列水平方向的输入向量、竖直方向的输入向量
	Weight_DataType a_vec[MATRIX_WIDTH], b_vec[MATRIX_WIDTH];

	// 计算脉动阵列计算完成时，脉搏需要跳动的次数
	int total_pulse = 3*MATRIX_WIDTH -2;
	gemm_outer_loop:  // 脉动阵列总脉动次数
	for (int i = 0; i < total_pulse; i++)
	{
		// 计算得到脉动阵列的每次输入向量
        Systolic_data_setup(input,a_vec,0,i);
        Systolic_data_setup(weight,b_vec,1,i);

        // print_matrix(input);
        // print_matrix(weight);
        // print_vec(a_vec);
        // print_vec(b_vec);

        //进行一次脉动
		systolic_array.pulse(a_vec, b_vec); 

        //读取结果
        read_result(out);
	}
}




