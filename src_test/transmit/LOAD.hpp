#ifndef LOAD_HPP
#define LOAD_HPP

#include "../SAA.h"
#include "GEMM_WS.hpp"


//实例化脉动阵列
Systolic_Array<Input_DataType, Weight_DataType, Psum_DataType, MATRIX_WIDTH> systolic_array;

//----------------------------Fetch模块--------------------------------//
/*!
* \brief fetch模块
* 获取指令，判断指令类型，将其填入任务队列
* \param instruct_count 总指令数。 AXI-lite 内存映射寄存器。
* \param instruct  DRAM 中的指令数据库基址。 用于读取指令
* \param load_queue 加载指令队列。 AXI 流 FIFO。
* \param gemm_queue GEMM 指令队列。 AXI 流 FIFO。
* \param store_queue 存储指令队列。 AXI 流 FIFO。
*/
void fetch(
    uint32_t instruct_count,
    volatile Instruct_DataType *instruct,
    hls::stream<Instruct_DataType> &load_queue,
    hls::stream<Instruct_DataType> &gemm_queue,
    hls::stream<Instruct_DataType> &store_queue)
  {
    #pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port = instruct_count offset = 0x10 bundle = CONTROL_BUS // 指令数绑定到CONTROL_BUS端口
    #pragma HLS INTERFACE m_axi port = instruct offset = slave bundle = instruct_port // 指令地址
    #pragma HLS INTERFACE axis port = load_queue // 流式接口，接FIFO
    #pragma HLS INTERFACE axis port = gemm_queue
    #pragma HLS INTERFACE axis port = store_queue

        //获取指令并解码
    INSN_DECODE: 
    for (int pc = 0; pc < instruct_count; pc++) {
#pragma HLS PIPELINE // 流水线优化
        // 从指令地址读取指令
        Instruct_DataType raw_instruct = instruct[pc];
        SAAInsn instruct_t; // 使用独联体初始化指令
        instruct_t.generic = *((GenericIns *) &raw_instruct); // 将其转换为通用指令进行解码
        // 部分解码
        Opcode_DataType opcode = instruct_t.generic.opcode;
        Buffer_Id_DataType buffer_id = instruct_t.mem.buffer_id;
        // 根据解码内容推送到正确的指令对列
        if (opcode == OPCODE_STORE) 
            store_queue.write(raw_instruct); // 写入FIFO
        else if (opcode == OPCODE_LOAD) 
            load_queue.write(raw_instruct);
        else 
            gemm_queue.write(raw_instruct); // 不属于LOAD和STORE的写入计算队列
    }

  }


//----------------------------Load模块--------------------------------//

/*!
* \brief 重置内存模块，用于添加padding，如果
* 通过将片上内存初始化为零来重置内存。
* \param sram_idx 片上内存缓冲区的索引，用于定位存储位置
* \param range 要重置的内存范围大小
* \param mem 目标内存数组，用于存储重置的数据
*
* \tparam DATA_T 输入数据类型，即总线数据类型
* \tparam MAT_AXI_RATIO 传输块大小/总线位宽，代表传输一共元素需要传输的次数
*/
template <typename DATA_T>
void reset_mem(
  Buffer_Addr_DataType &buffer_idx,
  Buffer_Addr_DataType range,
  DATA_T mem[][MATRIX_WIDTH]) {

  for (int i = 0; i < range; i ++) {
    for (int j = 0; j < MATRIX_WIDTH; j ++) {
#pragma HLS UNROLL
      mem[buffer_idx][j] = 0;
    }
    buffer_idx ++;
  }
}

/*!
* \brief load_2d模块
* 加载任意大小矩阵到片上，每次对一行进行突发，使用强制类型转换保证总线数据和缓冲区数据对齐
* 缺点是没有充分利用突发，只有当一行的数据够多效率才比较高，因此
* \param src DRAM 中的数据库基地址。 用于读取输入/权重
* \param dst 目标buffer，用于存储加载的矩阵数据，列大小固定为脉动阵列大小
* \param buffer_idx 片上buffer索引，用于定位存储位置，以缓冲区行为最小偏移
* \param dram_idx 输入数据库的偏移，用于定位读取位置,dram_idx按照总线大小进行偏移
* \param y_size Y 方向大小，表示要加载的行数，以缓冲区数据类型为最小偏移
* \param x_size X 方向大小，表示每行要加载的元素个数，必须是MATRIX_WIDTH的整数倍，以缓冲区数据类型为最小偏移
* \param x_stride X 方向步长，表示每行数据在外部存储器中的间隔，以缓冲区数据类型为最小偏移
* \tparam DATA_T 输入数据地址,在这里是总线数据类型地址
* \tparam BUFFER_DATA_TYPE 缓冲区数据类型
* \tparam ELEM_BYTES 传输元素的字节数
*/
template <typename DATA_T,typename BUFFER_DATA_TYPE, int ELEM_BYTES>
void load_2d(
  volatile DATA_T *src,
  BUFFER_DATA_TYPE dst[][MATRIX_WIDTH],
  Buffer_Addr_DataType buffer_idx,
  Dram_Addr_DataType dram_idx,
  Transfer_Size_DataType y_size,
  Transfer_Size_DataType x_size,
  Transfer_Stride_DataType x_stride) 
{
#pragma HLS INLINE // 函数内联减少调用影响
	Dram_Addr_DataType dram_offset = 0;
	const BUFFER_DATA_TYPE *src1 =(const BUFFER_DATA_TYPE *)src;// 提前将src转换为BUFFER_DATA_TYPE类型指针，然后再使用dram_idx进行偏移
	for (int y = 0; y < y_size; y++) // 循环y_size要加载的行
	{
		// 目标是片上buffer的行首地址，直接赋值一整行。src是dram的地址，length是一行的元素字节数
		// 为了防止src的索引偏移过大，将其强制转换为BUFFER_DATA_TYPE类型指针，使得每次偏移与缓冲区对齐
		memcpy(&dst[buffer_idx][0] , &src1[dram_idx] + dram_offset ,x_size * ELEM_BYTES);
	#pragma HLS RESOURCE variable = dram_offset core = Mul_LUT
		dram_offset += x_stride; // 相对于起始地址的偏移量
		buffer_idx += 1; // buffer换行
	}
}


/*!
* \brief load模块
* 接收load指令，并将其解码然后通过m_axi接口访问DDR读取权重和数据
* \param inputs DRAM 中的输入数据库基地址。 用于读取输入/权重
* \param load_queue 加载指令队列。 AXI 流 FIFO。
* \param input_buffer  片上输入缓冲区，只写。
* \param weight_buffer  片上权重缓冲区，只写。
*/
void load(
    hls::stream<Instruct_DataType> &load_queue,
    volatile Transfer_DataType *inputs,
    volatile Transfer_DataType *weights,
    Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
    Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH]) 
  {
#pragma HLS INTERFACE axis port = load_queue // FIFO接口连接load指令队列
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = data_port
#pragma HLS INTERFACE bram port = input_buffer  // 定义缓冲区接口为bram接口
#pragma HLS INTERFACE bram port = weight_buffer
#pragma HLS RESOURCE variable = input_buffer core = RAM_1P // 使用单端口ram，只写
#pragma HLS RESOURCE variable = weight_buffer core = RAM_1P
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

    //从加载指令队列读取指令
    Instruct_DataType raw_instruct = load_queue.read();
    // 将原始指令转换为memory指令
    Instruct_DataType raw_copy = raw_instruct;
    MemIns load_insn = *((MemIns *) &raw_copy);

	// printf("buffer_id:%d\n",(int)load_insn.buffer_id);
	// printf("buffer_base:%d\n",(int)load_insn.buffer_base);

    // 预处理（padding的预处理，暂时没有）

    // 根据目标buffer不同进行加载
    if (load_insn.buffer_id == WEIGHT_BUFFER_ID) {
		load_2d<Transfer_DataType,Weight_DataType, WEIGHT_DATA_BYTES>(
			weights,
			weight_buffer,
			load_insn.buffer_base,
			load_insn.dram_base,
			load_insn.y_size,
			load_insn.x_size,
			load_insn.x_stride);
    }
    else if (load_insn.buffer_id == INPUT_BUFFER_ID) {
		load_2d<Transfer_DataType,Input_DataType, INPUT_DATA_BYTES>(
			inputs,
			input_buffer,
			load_insn.buffer_base,
			load_insn.dram_base,
			load_insn.y_size,
			load_insn.x_size,
			load_insn.x_stride);
    }
  }

/*!
* \brief store_2d模块
* 将任意大小矩阵从片上存储回DRAM，每次对一行进行突发，使用强制类型转换保证总线数据和缓冲区数据对齐
* \param src 源buffer，存储要写回的矩阵数据，列大小固定为脉动阵列大小
* \param dst DRAM 中的数据库基地址。 用于写回输入/权重
* \param buffer_idx 片上buffer索引，用于定位读取位置，以缓冲区行为最小偏移
* \param dram_idx 输出数据库的偏移，用于定位存储位置,dram_idx按照总线大小进行偏移
* \param y_size Y 方向大小，表示要存储的行数，以缓冲区数据类型为最小偏移
* \param x_size X 方向大小，表示每行要存储的元素个数，必须是MATRIX_WIDTH的整数倍，以缓冲区数据类型为最小偏移
* \param x_stride X 方向步长，表示每行数据在外部存储器中的间隔，以缓冲区数据类型为最小偏移
* \tparam DATA_T 输出数据地址,在这里是总线数据类型地址
* \tparam BUFFER_DATA_TYPE 缓冲区数据类型
* \tparam ELEM_BYTES 传输元素的字节数
*/
template <typename DATA_T, typename BUFFER_DATA_TYPE, int ELEM_BYTES>
void store_2d(
  volatile DATA_T *dst,
  BUFFER_DATA_TYPE src[][MATRIX_WIDTH],
  Buffer_Addr_DataType buffer_idx,
  Dram_Addr_DataType dram_idx,
  Transfer_Size_DataType y_size,
  Transfer_Size_DataType x_size,
  Transfer_Stride_DataType x_stride) 
{
#pragma HLS INLINE // 函数内联减少调用影响
	Dram_Addr_DataType dram_offset = 0;
	BUFFER_DATA_TYPE *dst1 = (BUFFER_DATA_TYPE *)dst; // 提前将dst转换为BUFFER_DATA_TYPE类型指针，然后再使用dram_idx进行偏移
	for (int y = 0; y < y_size; y++) // 循环y_size要存储的行
	{
		// 源是片上buffer的行首地址，直接赋值一整行。dst是dram的地址，length是一行的元素字节数
		// 为了防止dst的索引偏移过大，将其强制转换为BUFFER_DATA_TYPE类型指针，使得每次偏移与缓冲区对齐
		memcpy(&dst1[dram_idx] + dram_offset, &src[buffer_idx][0], x_size * ELEM_BYTES);
#pragma HLS RESOURCE variable = dram_offset core = Mul_LUT
		dram_offset += x_stride; // 相对于起始地址的偏移量
		buffer_idx += 1; // buffer换行
	}
}


/*!
* \brief store模块
* 接收store指令，并将其解码然后通过m_axi接口访问DDR写入权重和数据，刚好与load是相反操作
* \param outputs DRAM 中的输出数据库基地址。 用于写入结果
* \param store_queue 存储指令队列。 AXI 流 FIFO。
* \param output_buffer  片上输出缓冲区，只读。
*/
void store(
  hls::stream<Instruct_DataType> &store_queue,
  volatile Transfer_DataType *outputs,
  Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH]) {
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE bram port = output_buffer
#pragma HLS RESOURCE variable = output_buffer core = RAM_1P

    //从存储指令队列读取指令
    Instruct_DataType raw_instruct = store_queue.read();
    // 将原始指令转换为memory指令
    Instruct_DataType raw_copy = raw_instruct;
    MemIns store_insn = *((MemIns *) &raw_copy);

	// 调用2d存储函数
	store_2d<Transfer_DataType, Output_DataType, OUTPUT_DATA_BYTES>(
		outputs,
		output_buffer,
		store_insn.buffer_base,
		store_insn.dram_base,
		store_insn.y_size,
		store_insn.x_size,
		store_insn.x_stride);
}

/*!
* \brief compute模块
* 接收compute指令，并将其解码然后通过m_axi接口访问DDR读取权重和数据
* \param done 该信号由完成指令控制，当执行到完成指令时，代表完成计算
* \param gemm_queue 加载指令队列。 AXI 流 FIFO。
* \param input_buffer  片上输入缓冲区，只读。
* \param weight_buffer  片上权重缓冲区，只读。
* \param output_buffer  片上输出缓冲区，只写。
*/
void compute(
  volatile uint32_t &done,
  hls::stream<Instruct_DataType> &gemm_queue,
  Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
  Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
  Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH]) 
  {
#pragma HLS INTERFACE bram port = input_buffer
#pragma HLS INTERFACE bram port = weight_buffer
#pragma HLS INTERFACE bram port = output_buffer
#pragma HLS RESOURCE variable = input_buffer core = RAM_1P
#pragma HLS RESOURCE variable = weight_buffer core = RAM_1P
#pragma HLS RESOURCE variable = output_buffer core = RAM_1P
#pragma HLS INTERFACE axis port = gemm_queue // FIFO接口连接compute指令队列
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = done offset = 0x20 bundle = CONTROL_BUS // done信号

    //从计算指令队列读取指令
    Instruct_DataType raw_instruct = gemm_queue.read();
    // 将原始指令转换为compute指令
    Instruct_DataType raw_copy = raw_instruct;
    ComIns compute_insn = *((ComIns *) &raw_copy);

	//判断指令类型
    if (compute_insn.compute_type == WEIGHT_PRELOAD) { // 权重预加载
	    systolic_array.set_array_weights(
			weight_buffer, // 权重缓冲区
			compute_insn.weigth_addr, // 权重缓冲区起始地址
			compute_insn.weight_switch); // 加载到哪个权重寄存器
    }
    else if (compute_insn.compute_type == COMPUTE) { // 使用当前脉动阵列权重计算
		systolic_array.BMM_kernel(
			input_buffer,  				// 输入bram
			output_buffer,  			// 输出bram
			compute_insn.input_addr,    // 输入起始地址
			compute_insn.output_addr,   // 输出起始地址
			compute_insn.compute_switch,// 使用哪一个权重寄存器进行计算
			compute_insn.accumulate);   // 是否对结果进行累加
		}
    else if (compute_insn.compute_type == COMPUTE_WEIGHT_PRELOAD) { // 加载权重同时进行计算，用于双缓冲操作
#pragma HLS dataflow // 使得设置权重和进行脉动阵列数据流并行
		// 设置权重，计算哪一个就加载另一个
		systolic_array.set_array_weights(
			weight_buffer, // 权重缓冲区
			compute_insn.weigth_addr, // 权重缓冲区起始地址
			!compute_insn.compute_switch); // 与计算权重的寄存器相反
			
		// 计算矩阵
		systolic_array.BMM_kernel(
			input_buffer,  				// 输入bram
			output_buffer,  			// 输出bram
			compute_insn.input_addr,    // 输入起始地址
			compute_insn.output_addr,   // 输出起始地址
			compute_insn.compute_switch,// 使用哪一个权重寄存器进行计算
			compute_insn.accumulate);   // 是否对结果进行累加
	}
  }


//--------------------------------SAA模块--------------------------------//
/*!
* \brief SAA模块
* 组合了load和fetch模块，用于流处理加速器架构
* \param insn_count 总指令数。 AXI-lite 内存映射寄存器。
* \param insns DRAM 中的指令数据库基址。 用于读取指令
* \param inputs DRAM 中的输入数据基址。 用于读取输入数据
* \param input_buffer 片上输入缓冲区，只写。
* \param weight_buffer 片上权重缓冲区，只写。
* \param outputs DRAM 中的输出数据基址。 用于存储输出数据
*/
void saa_top(
  uint32_t insn_count,
  volatile Instruct_DataType *insns,
  volatile Transfer_DataType *inputs,
  volatile Transfer_DataType *weights,
  volatile Transfer_DataType *outputs,
  Input_DataType input_buffer[INPUT_BUFFER_WIDTH][MATRIX_WIDTH],
  Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH][MATRIX_WIDTH],
  Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH][MATRIX_WIDTH]) 
{
#pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = data_port
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

    // 实例化加载和获取队列
    hls::stream<Instruct_DataType> load_queue;
    hls::stream<Instruct_DataType> gemm_queue;
    hls::stream<Instruct_DataType> store_queue;

    // 只在第一次使用脉动阵列时reset就行，目的是清除PSUM寄存器中的值
    systolic_array.reset(); 

    // 将所有指令压入队列
    fetch(insn_count, insns, load_queue, gemm_queue, store_queue);
    
    // 全局计算完成信号
    uint32_t done = 0;

    // 执行加载命令
    while (true) {
        if(load_queue.empty() && gemm_queue.empty()) //如果加载队列和计算队列空就跳出
            break;
        // 首先尽可能的执行加载命令
        while(!load_queue.empty())
        {
            load(load_queue, inputs, weights, input_buffer, weight_buffer); // 执行加载队列的指令
        }
        // 然后尽可能的执行计算命令
        while(!gemm_queue.empty())
        {
            compute(done, gemm_queue, input_buffer, weight_buffer, output_buffer); // 执行计算队列指令
        }
        // 然后尽可能的执行存储命令
        while(!store_queue.empty())
        {
            store(store_queue, outputs, output_buffer); // 执行存储队列指令
        }
    }
}





//---------------------------------debug--------------------------------//



#endif