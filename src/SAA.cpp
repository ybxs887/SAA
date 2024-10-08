#include "SAA_const.h"
#include "SAA.h"
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
    // 指令数绑定到CONTROL_BUS端口
    PRAGMA_HLS(HLS INTERFACE s_axilite port = instruct_count bundle = CONTROL_BUS offset = FETCH_INSN_COUNT_OFFSET)
    #pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port = instruct bundle = CONTROL_BUS //强行将instruct地址寄存器绑定到CONTROL_BUS
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
        else if (opcode == OPCODE_LOAD)  // 只有load input 和load output才fetch到加载队列
        {
            if (buffer_id == INPUT_BUFFER_ID || buffer_id == WEIGHT_BUFFER_ID) 
              load_queue.write(raw_instruct);
            else 
              gemm_queue.write(raw_instruct);
        }
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
  BUFFER_DATA_TYPE *dst,
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
		memcpy(&dst[buffer_idx*MATRIX_WIDTH] , src1 + dram_idx + dram_offset ,x_size * ELEM_BYTES);
	#pragma HLS BIND_OP  variable = dram_offset op=mul
		dram_offset += x_stride; // 相对于起始地址的偏移量
		buffer_idx += 1; // buffer换行
	}
}

// 以块为最小单位进行传输
template <typename DATA_T,typename BLOCK_DATA_TYPE, int BLOCK_ELEM_BYTES>
void load_2d_block(
  volatile DATA_T *src,
  BLOCK_DATA_TYPE *dst,
  Buffer_Addr_DataType buffer_idx,
  Dram_Addr_DataType dram_idx,
  Transfer_Size_DataType y_size,
  Transfer_Size_DataType x_size,
  Transfer_Stride_DataType x_stride) 
{
#pragma HLS INLINE // 函数内联减少调用影响
	Dram_Addr_DataType dram_offset = 0;
	const BLOCK_DATA_TYPE *src1 =(const BLOCK_DATA_TYPE *)src;//将src转换为块类型,以块类型加载
	for (int y = 0; y < y_size; y++) // 循环y_size要加载的行
	{
		// 目标是片上buffer的行首地址，直接赋值一整行。src是dram的地址，length是一行的元素字节数
		// 为了防止src的索引偏移过大，将其强制转换为BUFFER_DATA_TYPE类型指针，使得每次偏移与缓冲区对齐
		memcpy(&dst[buffer_idx] , src1 + dram_idx + dram_offset ,x_size * BLOCK_ELEM_BYTES);
	#pragma HLS BIND_OP  variable = dram_offset op=mul
		dram_offset += x_stride; // 相对于起始地址的偏移量,该偏移量按照块类型计算
		buffer_idx += x_size; // 复制多少个之后切换
	}
}

template <typename DATA_T,typename BLOCK_DATA_TYPE,int MAT_AXI_RATIO,int BLOCK_ELEM_BYTES>
void load_2d_block1(
  volatile DATA_T *src,
  BLOCK_DATA_TYPE *dst,
  Buffer_Addr_DataType buffer_idx,
  Dram_Addr_DataType dram_idx,
  Transfer_Size_DataType y_size,
  Transfer_Size_DataType x_size,
  Transfer_Stride_DataType x_stride) 
{
#pragma HLS INLINE // 函数内联减少调用影响

    //强行将src转换为块类型,以块类型加载
	for (int y = 0; y < y_size; y++) // 循环y_size要加载的行
	{
		// 目标是片上buffer的行首地址，直接赋值一整行。src是dram的地址，length是一行的元素字节数
		// 为了防止src的索引偏移过大，将其强制转换为BUFFER_DATA_TYPE类型指针，使得每次偏移与缓冲区对齐
		memcpy(&dst[buffer_idx] ,(const DATA_T*) &src[dram_idx*MAT_AXI_RATIO] ,x_size * BLOCK_ELEM_BYTES); // 一次加载x_size个块，dram偏移使用比例转换为块偏移
	#pragma HLS BIND_OP  variable = dram_idx op=mul
		dram_idx += x_stride; // 实际上偏移按总线算，但是由于在上面乘上了比例因子，因此可以按照块来算
		buffer_idx += x_size; // 复制多少个之后切换
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
    hls::stream<bool> &l2c_raw_queue,
    hls::stream<bool> &c2l_war_queue,
    Transfer_DataType input_buffer[INPUT_BUFFER_SIZE],
    Transfer_DataType weight_buffer[WEIGHT_BUFFER_SIZE]) 
  {
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = data_port
#pragma HLS INTERFACE s_axilite port = inputs bundle = CONTROL_BUS 
#pragma HLS INTERFACE s_axilite port = weights bundle = CONTROL_BUS 
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS INTERFACE axis port = load_queue // FIFO接口连接load指令队列
#pragma HLS INTERFACE axis port = l2c_raw_queue // FIFO接口连接依赖队列
#pragma HLS INTERFACE axis port = c2l_war_queue // FIFO接口连接依赖队列
#pragma HLS INTERFACE bram port = input_buffer storage_type = RAM_1P // 定义缓冲区接口为bram接口
#pragma HLS INTERFACE bram port = weight_buffer storage_type = RAM_1P 
// #pragma HLS ARRAY_RESHAPE variable=input_buffer complete  dim=2
// #pragma HLS ARRAY_RESHAPE variable=weight_buffer complete  dim=2

    //从加载指令队列读取指令
    Instruct_DataType raw_instruct = load_queue.read();
    // 将原始指令转换为memory指令
    Instruct_DataType raw_copy = raw_instruct;
    MemIns load_insn = *((MemIns *) &raw_copy);

	// printf("buffer_id:%d\n",(int)load_insn.buffer_id);
	// printf("buffer_base:%d\n",(int)load_insn.buffer_base);

	// 如果指令有指示存在依赖
	if (load_insn.pop_next_war) { // 如果有对下一个模块的war依赖，等待下一个模块完成才能写入
		c2l_war_queue.read(); // 使用阻塞读取方法读取war队列，如果存在war令牌，表示下一个模块完成
	}

    // 预处理（padding的预处理，暂时没有）

    // 根据目标buffer不同进行加载
    if (load_insn.buffer_id == WEIGHT_BUFFER_ID) {
      load_2d_block1<Transfer_DataType, WeightBlock,TRANSFER_WEIGHT_AXI_RATIO,TRANSFER_WEIGHT_DATA_BYTES>(
        weights,
        (WeightBlock *)weight_buffer,
        load_insn.buffer_base,
        load_insn.dram_base,
        load_insn.y_size,
        load_insn.x_size,
        load_insn.x_stride);  
        // printf("weight_buffer:\n");print_buffer1((Weight_DataType *)weight_buffer,12);
    }
    else if (load_insn.buffer_id == INPUT_BUFFER_ID) {
      load_2d_block1<Transfer_DataType, InputBlock,TRANSFER_INPUT_AXI_RATIO,TRANSFER_INPUT_DATA_BYTES>(
        inputs,
        (InputBlock *)input_buffer,
        load_insn.buffer_base,
        load_insn.dram_base,
        load_insn.y_size,
        load_insn.x_size,
        load_insn.x_stride); 
        // printf("input_buffer:\n");print_buffer1((Input_DataType *)input_buffer,12);
    }

	// 如果指令有指示存在影响
	if (load_insn.push_next_raw) { // 如果对下一个模块有raw影响，那么执行完后，告诉下一个模块我已经写入完毕，你可以读取
		l2c_raw_queue.write(1);; // 写入一个raw令牌
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
  BUFFER_DATA_TYPE *src,
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
		memcpy(dst1 + dram_idx + dram_offset, &src[buffer_idx*MATRIX_WIDTH], x_size * ELEM_BYTES);
#pragma HLS BIND_OP  variable = dram_offset op=mul
		dram_offset += x_stride; // 相对于起始地址的偏移量
		buffer_idx += 1; // buffer换行
	}
}

// 以块为最小单位进行传输
template <typename DATA_T,typename BLOCK_DATA_TYPE, int BLOCK_ELEM_BYTES>
void store_2d_block(
  volatile DATA_T *dst,
  BLOCK_DATA_TYPE *src,
  Buffer_Addr_DataType buffer_idx,
  Dram_Addr_DataType dram_idx,
  Transfer_Size_DataType y_size,
  Transfer_Size_DataType x_size,
  Transfer_Stride_DataType x_stride)
{
#pragma HLS INLINE // 函数内联减少调用影响
	Dram_Addr_DataType dram_offset = 0;
	BLOCK_DATA_TYPE *dst1 = (BLOCK_DATA_TYPE *)dst; // 提前将dst转换为BUFFER_DATA_TYPE类型指针，然后再使用dram_idx进行偏移
	for (int y = 0; y < y_size; y++) // 循环y_size要存储的行
	{
		// 源是片上buffer的行首地址，直接赋值一整行。dst是dram的地址，length是一行的元素字节数
		// 为了防止dst的索引偏移过大，将其强制转换为BUFFER_DATA_TYPE类型指针，使得每次偏移与缓冲区对齐
		memcpy(dst1 + dram_idx + dram_offset, &src[buffer_idx], x_size * BLOCK_ELEM_BYTES);
#pragma HLS BIND_OP  variable = dram_offset op=mul
		dram_offset += x_stride; // 每次存储完x_size个块,进行偏移x_stride个块,继续加载下一行
		buffer_idx += x_size; // 按照块类型偏移x_size个块,因为memcpy存储了x_size个块
	}
}


template <typename DATA_T,typename BLOCK_DATA_TYPE,int MAT_AXI_RATIO,int BLOCK_ELEM_BYTES>
void store_2d_block1(
  volatile DATA_T *dst,
  BLOCK_DATA_TYPE *src,
  Buffer_Addr_DataType buffer_idx,
  Dram_Addr_DataType dram_idx,
  Transfer_Size_DataType y_size,
  Transfer_Size_DataType x_size,
  Transfer_Stride_DataType x_stride)
{
#pragma HLS INLINE // 函数内联减少调用影响

    //强行将src转换为块类型,以块类型加载
	for (int y = 0; y < y_size; y++) // 循环y_size要加载的行
	{
		// 目标是片上buffer的行首地址，直接赋值一整行。src是dram的地址，length是一行的元素字节数
		// 为了防止src的索引偏移过大，将其强制转换为BUFFER_DATA_TYPE类型指针，使得每次偏移与缓冲区对齐
		memcpy(const_cast<DATA_T*>(&dst[dram_idx*MAT_AXI_RATIO]),(const BLOCK_DATA_TYPE*) &src[buffer_idx], x_size * BLOCK_ELEM_BYTES); // 一次加载x_size个块，dram偏移使用比例转换为块偏移
	#pragma HLS BIND_OP  variable = dram_idx op=mul
		dram_idx += x_stride; // 实际上偏移按总线算，但是由于在上面乘上了比例因子，因此可以按照块来算
		buffer_idx += x_size; // 复制多少个之后切换
	}
}

/*!
* \brief store模块
* 接收store指令，并将其解码然后通过m_axi接口访问DDR写入权重和数据，刚好与load是相反操作，同时在写回时执行maxpooling操作
* \param outputs DRAM 中的输出数据库基地址。 用于写入结果
* \param store_queue 存储指令队列。 AXI 流 FIFO。
* \param output_buffer  片上输出缓冲区，只读。
*/
void store(
  hls::stream<Instruct_DataType> &store_queue,
  volatile Transfer_DataType *outputs,
  hls::stream<bool> &s2c_war_queue,
  hls::stream<bool> &c2s_raw_queue,
  Transfer_DataType  input_buffer[INPUT_BUFFER_SIZE],
  Transfer_DataType  output_buffer[OUTPUT_BUFFER_SIZE]) {
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = outputs bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE axis port = s2c_war_queue
#pragma HLS INTERFACE axis port = c2s_raw_queue
#pragma HLS INTERFACE bram port = output_buffer storage_type = RAM_1P
#pragma HLS INTERFACE bram port = input_buffer storage_type = RAM_1P
// #pragma HLS ARRAY_RESHAPE variable=output_buffer complete  dim=2

    //从存储指令队列读取指令
    Instruct_DataType raw_instruct = store_queue.read();
    // 将原始指令转换为memory指令
    Instruct_DataType raw_copy = raw_instruct;
    MemIns store_insn = *((MemIns *) &raw_copy);

	// printf("buffer_id:%d\n",(int)store_insn.buffer_id);
	// printf("buffer_base:%d\n",(int)load_insn.buffer_base);

	// 如果指令有指示存在依赖
	if (store_insn.pop_pre_raw) { // 如果有对上一个模块的raw依赖，等待上一个模块完成才能读取
		c2s_raw_queue.read(); // 使用阻塞读取方法读取raw队列，如果存在raw令牌，表示上一个模块完成
	}

	// 调用2d存储函数
    if (store_insn.buffer_id == OUTPUT_BUFFER_ID) {
      store_2d_block1<Transfer_DataType, Output_Block,TRANSFER_OUTPPUT_AXI_RATIO,TRANSFER_OUTPUT_DATA_BYTES>(
        outputs,
        (Output_Block *)output_buffer,
        store_insn.buffer_base,
        store_insn.dram_base,
        store_insn.y_size,
        store_insn.x_size,
        store_insn.x_stride);
    }
    else if (store_insn.buffer_id == INPUT_BUFFER_ID) {
      store_2d_block1<Transfer_DataType, InputBlock,TRANSFER_INPUT_AXI_RATIO,TRANSFER_INPUT_DATA_BYTES>(
        outputs,
        (InputBlock *)input_buffer,
        store_insn.buffer_base,
        store_insn.dram_base,
        store_insn.y_size,
        store_insn.x_size,
        store_insn.x_stride);
    }

	// 如果指令有指示存在影响
	if (store_insn.push_pre_war) { // 如果对上一个模块有war影响，那么执行完后，告诉上一个模块我已经读取完毕，你可以写入
		s2c_war_queue.write(1); // 写入一个war令牌
	}
}


#include <cmath> 
// #include "hls_math.h"
// // 指数函数计算单元，用于计算指数函数，可以用于softmax函数
// template <typename DATA_T>
// DATA_T expf(DATA_T x) {
//     x = 1.0 + x / 1024;
//     x *= x; x *= x; x *= x; x *= x; x *= x; 
//     x *= x; x *= x; x *= x; x *= x; x *= x; // 10次乘法，总次方数是2的10次方
//     return x;
// }
// // 指数函数计算单元，基于分段线性函数计算指数

// // 指数函数计算单元，基于查表法计算指数

// 从缓冲区中读取块
template <typename DATA_SRC_DataType,typename DATA_DST_DataType,int DATA_DST_W,int Y_DIM,int X_DIM>
void read_block(
    Buffer_Addr_DataType buffer_idx,
    DATA_SRC_DataType src[][MATRIX_WIDTH],
    DATA_DST_DataType dst[Y_DIM][X_DIM]) 
{
// 使用for循环从缓冲区中读取矩阵到块中
  for (int i = 0; i < Y_DIM; i++) {
    for (int j = 0; j < X_DIM; j++) {
      dst[i][j] = src[buffer_idx+i][j];
    }
  }
}

template <typename DATA_SRC_DataType, typename DATA_DST_DataType, int DATA_DST_W, int Y_DIM, int X_DIM>
void write_block(
    Buffer_Addr_DataType buffer_idx,
    DATA_SRC_DataType src[Y_DIM][X_DIM],
    DATA_DST_DataType dst[][MATRIX_WIDTH])
{
    // 使用嵌套循环将块数据写入缓冲区中
    for (int i = 0; i < Y_DIM; i++) {
        for (int j = 0; j < X_DIM; j++) {
            // 假设dst是一个足够大的一维数组
            dst[buffer_idx + i][j] = src[i][j];
        }
    }
}

/*!
* \brief ANU模块
* 作为非线性张量计算单元，完成包括归一化，激活等操作，包括普通的norm、relu、softmax等
* 运算的基本单位是脉动阵列大小，基于微操作完成多个块的计算
* \param done 
*/

void anu(
  Instruct_DataType insn_raw,
  Input_DataType input_buffer[][MATRIX_WIDTH],
  Weight_DataType weight_buffer[][MATRIX_WIDTH],
  Scale_DataType output_buffer[][MATRIX_WIDTH]) {
#pragma HLS INLINE 
#pragma HLS allocation operation instances=mul limit=8// 需要限制这个函数中的DSP资源消耗，一个32位乘法消耗3个DSP
// #pragma HLS ALLOCATION operation instances=fsqrt limit=1
// #pragma HLS ALLOCATION operation instances=sdiv limit=1
// #pragma HLS ALLOCATION instances=sdiv limit=1 operation

    // 将原始指令转换为anu指令类型
    AnuIns insn = *((AnuIns *) &insn_raw); 

    // 初始化外循环索引量
    Buffer_Addr_DataType out_offset = 0;

    // 初始化循环外的变量,必须手动初始化，只有基础类型才能自动被初始化为0
    Sum_DataType sum_mean[MATRIX_WIDTH] ; //  求和/均值变量，用于对每一行的列进行求和（可同步进行指数求和），也可以在求完和后直接求均值
    Sum_DataType var_max[MATRIX_WIDTH] ; // 方差变量/均值的平方,同时也可以用来求最大值
    Buffer_Addr_DataType src_offset;
    Buffer_Addr_DataType dst_offset;
    Buffer_Addr_DataType J_idx; // 用于表明J循环到第几列

    // 判断指令类型
    if(insn.anu_type == ANU_LAYERNORM)
    {
      // 执行layernorm的操作
      LAYERNORM_OUT_LOOP: 
      for (int iter_out = 0; iter_out < insn.iter_I; iter_out++) {
        //必须手动初始化，只有基础类型才能自动被初始化为0
        for (int i = 0; i < MATRIX_WIDTH; i++) {
            sum_mean[i] = 0;
            var_max[i] = 0;
        }
        // 内循环，layer需要重复遍历两次
        // 初始化索引量
        src_offset = out_offset;
        dst_offset = out_offset;

        // LAYERNORM的第一个循环计算元素的和以及元素平方的和
        LAYERNORM_LOOP1: 
        for (int iter_uop = 0; iter_uop < insn.iter_uop; iter_uop++) {
          // 从输出缓冲区/累加器中读取一个矩阵/可以视为一个MATRIX_WIDTH行的MATRIX_WIDTH大小向量拼成/MATRIX_WIDTH * MATRIX_WIDTH大小的向量
          Scale_DataType src_block[MATRIX_WIDTH][MATRIX_WIDTH];
          read_block<Scale_DataType,Scale_DataType,OUTPUT_DATA_WIDTH,MATRIX_WIDTH,MATRIX_WIDTH>(src_offset,output_buffer,src_block);
          // 进入最基础的块循环内部
          LAYERNORM_LOOP1_block: 
          for (int j = 0; j < MATRIX_WIDTH; j++) { // 先计算列
            for (int i = 0; i < MATRIX_WIDTH; i++) { //循环块中的行，由于所有块按行排序，因此这也是矩阵的行
#pragma HLS PIPELINE II = 1 // 通过控制pipelen控制资源消耗
                sum_mean[i] += src_block[i][j];// 计算和
                var_max[i] += (Sum_DataType)(src_block[i][j] * src_block[i][j]);// 计算元素的平方的和,要转换为结果Sum_DataType保存
            }
          }
          // 更新偏移,一次偏移MATRIX_WIDTH行
          src_offset += MATRIX_WIDTH;
          dst_offset += MATRIX_WIDTH;
        }

        // 在第一次外循环结束时，通过元素的和以及元素平方的和，计算均值和方差
        LAYERNORM_LOOP2: 
        for (int i = 0; i < MATRIX_WIDTH; i++) {
#pragma HLS PIPELINE II = 1 // 通过控制pipelen控制资源消耗
          sum_mean[i] /= insn.imm; // 求均值
          var_max[i] /= insn.imm; // 求平方的均值
          var_max[i] -= sum_mean[i] * sum_mean[i]; // 平方的均值减去均值的平方
          var_max[i] = sqrtf(var_max[i]); // 计算平方根
          var_max[i] = (Sum_DataType)1 / var_max[i]; // 计算平方根的倒数
        }

        //第二次循环，一样的起点
        src_offset = out_offset;
        dst_offset = out_offset;

        // LAYERNORM的第二个循环计算元素的和以及元素平方的和
        LAYERNORM_LOOP3: 
        for (int iter_uop = 0; iter_uop < insn.iter_uop; iter_uop++) {
          Scale_DataType src_block[MATRIX_WIDTH][MATRIX_WIDTH];
          read_block<Scale_DataType,Scale_DataType,OUTPUT_DATA_WIDTH,MATRIX_WIDTH,MATRIX_WIDTH>(src_offset,output_buffer,src_block);
          // 进入最基础的块循环内部
          LAYERNORM_LOOP2_block:
          for (int j = 0; j < MATRIX_WIDTH; j++) { // 先计算列
            for (int i = 0; i < MATRIX_WIDTH; i++) { //循环块中的行，由于所有块按行排序，因此这也是矩阵的行
// #pragma HLS unroll factor=4 // 指明UNROOL，防止HLS自动完全展开
#pragma HLS PIPELINE II = 1 // 通过控制pipelen控制资源消耗
                  src_block[i][j] = (src_block[i][j] - sum_mean[i]); // 进行归一化计算每个元素减去均值除以标准差
                  src_block[i][j] = (Scale_DataType)(src_block[i][j] * var_max[i]);
            }
          }
          // 写回src_block，layernorm第一次循环只是计算均值和方差，不用改变src_block，不用写回
          // 也就是layernorm只需要最后一次计算写回，而softmax两次计算都要写回
          write_block<Scale_DataType,Scale_DataType,OUTPUT_DATA_WIDTH,MATRIX_WIDTH,MATRIX_WIDTH>(dst_offset,src_block,output_buffer);
          src_offset += MATRIX_WIDTH;
          dst_offset += MATRIX_WIDTH;
        }
        out_offset += insn.iter_uop * MATRIX_WIDTH; // I循环内每一次增加整个J循环计算的行，切换到下一个I行
      }
    }
    // 如果是ANU_SOFTMAX
    else if(insn.anu_type == ANU_SOFTMAX)
    {
      // // 执行softmax的操作
      // SOFTMAX_OUT_LOOP: 
      // for (int iter_out = 0; iter_out < insn.iter_I; iter_out++) {
      //   //必须手动初始化，只有基础类型才能自动被初始化为0
      //   for (int i = 0; i < MATRIX_WIDTH; i++) {
      //       sum_mean[i] = 0;
      //       var_max[i] = 0;
      //   }
      //   // 第一次初始化索引量
      //   src_offset = out_offset;
      //   dst_offset = out_offset;
      //   J_idx = 0; // 用于表明J循环到第几列
      //   // SOFTMAX的第一次循环用来找最大值
      //   SOFTMAX_LOOP1: 
      //   for (int iter_uop = 0; iter_uop < insn.iter_uop; iter_uop++) {
      //     Scale_DataType src_block[MATRIX_WIDTH][MATRIX_WIDTH];
      //     read_block<Scale_DataType,Scale_DataType,OUTPUT_DATA_WIDTH,MATRIX_WIDTH,MATRIX_WIDTH>(src_offset,output_buffer,src_block);
      //     for (int j = 0; j < MATRIX_WIDTH; j++) { // 先计算列
      //       if(J_idx < insn.imm)
      //         for (int i = 0; i < MATRIX_WIDTH; i++) { //循环块中的行，由于所有块按行排序，因此这也是矩阵的行
      //             var_max[i] = src_block[i][j] > var_max[i] ? (Sum_DataType)src_block[i][j]: var_max[i]; // 转换src_block为var_max一样的类型
      //         }
      //       J_idx++; // 循环到第几列++,因为i和j一样是MATRIX_WIDTH
      //     }
      //     src_offset += MATRIX_WIDTH;
      //     dst_offset += MATRIX_WIDTH;
      //   }

      //   // 第二次初始化索引量
      //   src_offset = out_offset;
      //   dst_offset = out_offset;
      //   J_idx = 0; // 用于表明J循环到第几列
      //   // SOFTMAX的第二次循环用来计算指数和指数的和,减去了max防止溢出
      //   SOFTMAX_LOOP2: 
      //   for (int iter_uop = 0; iter_uop < insn.iter_uop; iter_uop++) {
      //     Scale_DataType src_block[MATRIX_WIDTH][MATRIX_WIDTH];
      //     read_block<Scale_DataType,Scale_DataType,OUTPUT_DATA_WIDTH,MATRIX_WIDTH,MATRIX_WIDTH>(src_offset,output_buffer,src_block);
      //     for (int j = 0; j < MATRIX_WIDTH; j++) { // 先计算列
      //       if(J_idx < insn.imm)
      //         for (int i = 0; i < MATRIX_WIDTH; i++) { //循环块中的行，由于所有块按行排序，因此这也是矩阵的行
      //             src_block[i][j] = (Scale_DataType)expf(src_block[i][j] - var_max[i]); // 使用hls数学库计算指数，可以替换为分段线性函数或者查表
      //             sum_mean[i] += src_block[i][j]; // 添加进入sum计算exp的和            
      //         }
      //       J_idx++; // 循环到第几列++,因为i和j一样是MATRIX_WIDTH
      //     }
      //     write_block<Scale_DataType,Scale_DataType,OUTPUT_DATA_WIDTH,MATRIX_WIDTH,MATRIX_WIDTH>(dst_offset,src_block,output_buffer);
      //     src_offset += MATRIX_WIDTH;
      //     dst_offset += MATRIX_WIDTH;
      //   }

      //   // 计算指数的和的倒数
      //   for (int i = 0; i < MATRIX_WIDTH; i++) { 
      //     sum_mean[i] =(Sum_DataType)1 / sum_mean[i]; // 每个值除以每行指数的和
      //   }


      //   // 第三次初始化索引量
      //   src_offset = out_offset;
      //   dst_offset = out_offset;
      //   J_idx = 0; // 用于表明J循环到第几列
      //   // SOFTMAX的第三次循环用来计算每个指数除以指数的和
      //   SOFTMAX_LOOP3: 
      //   for (int iter_uop = 0; iter_uop < insn.iter_uop; iter_uop++) {
      //     Scale_DataType src_block[MATRIX_WIDTH][MATRIX_WIDTH];
      //     read_block<Scale_DataType,Scale_DataType,OUTPUT_DATA_WIDTH,MATRIX_WIDTH,MATRIX_WIDTH>(src_offset,output_buffer,src_block);
      //     for (int j = 0; j < MATRIX_WIDTH; j++) { // 先计算列
      //       if(J_idx < insn.imm)
      //         for (int i = 0; i < MATRIX_WIDTH; i++) { //循环块中的行，由于所有块按行排序，因此这也是矩阵的行
      //           src_block[i][j] = (Scale_DataType)(src_block[i][j] * sum_mean[i]); // 每个值除以每行指数的和
      //         }
      //       J_idx++; // 循环到第几列++,因为i和j一样是MATRIX_WIDTH
      //     }
      //     write_block<Scale_DataType,Scale_DataType,OUTPUT_DATA_WIDTH,MATRIX_WIDTH,MATRIX_WIDTH>(dst_offset,src_block,output_buffer);
      //     src_offset += MATRIX_WIDTH;
      //     dst_offset += MATRIX_WIDTH;
      //   }
      //   out_offset += insn.iter_uop * MATRIX_WIDTH; // I循环内每一次增加整个J循环计算的行，切换到下一个I行
      // }
    }
}
/*!
* \brief gemm模块
* 作为矩阵乘法单元,可以完成基于权重固定的脉动阵列的运算
* 运算的基本单位是脉动阵列大小，基于微操作完成多个块的计算,这样就不用使用软件生成大量指令浪费空间存储指令了
* 微操作指令uop缓冲区,作为从
*/
// #define max(a,b) ((a>b)?a:b) // 定义了一个max函数用于返回最大值（relu）
void gemm(
  Instruct_DataType insn_raw,
  Uop_DataType uop_buffer[UOP_BUFFER_WIDTH],
  Input_DataType input_buffer[][MATRIX_WIDTH],
  Weight_DataType weight_buffer[][MATRIX_WIDTH],
  Output_DataType output_buffer[][MATRIX_WIDTH]) {
#pragma HLS INLINE

  // 将原始指令转换为compute指令类型
  ComIns insn = *((ComIns *) &insn_raw);

  // 初始化偏移量,input_offset和weigth_offset有k维度不需要进行重置,因此每次在这里重置
  Buffer_Addr_DataType input_offset = 0;
  Buffer_Addr_DataType weigth_offset = 0;
  Buffer_Addr_DataType output_offset = 0;
  // 初始化索引量
  Buffer_Addr_DataType input_idx = insn.input_base;  // 一开始就加载输入基地址
  Buffer_Addr_DataType weigth_idx = insn.weight_base;// 一开始就加载权重基地址
  Buffer_Addr_DataType output_idx = 0;
  // 初始化乒乓操作变量
  bool pingpang = 0;
  bool accumulate = 0;
  bool accumulate_delay = 0;  

  // 执行第一次权重预加载
  systolic_array.set_array_weights(
      weight_buffer, // 权重缓冲区
      weigth_idx, // 权重缓冲区起始索引
      pingpang); // 加载到权重寄存器0

  // 外循环,计算K维度
  K_LOOP: for (int k = 0; k < insn.dim_K_block; k++) {
    accumulate = (insn.bias_use==1)? 1 :(k == 0 ? 0 : 1); // 从指令判断是否使用bias,使用了就全部累加,没用就k=0时累加
    output_offset = 0; // 在j循环外初始化输出偏置
    // 内循环,计算J维度
    J_LOOP: for (int j = 0; j < insn.dim_J_block; j++) {
      // 执行预加载计算指令
      if(k != 0 || j != 0){
        // 设置权重，计算哪一个就加载另一个
        systolic_array.set_array_weights(
          weight_buffer, // 权重缓冲区
          weigth_idx, // 权重缓冲区起始地址
          !pingpang); // 与计算权重的寄存器相反
          
        // 计算矩阵
        systolic_array.BMM_kernel(
          input_buffer,  				// 输入bram
          output_buffer,  			// 输出bram
          input_idx,    // 输入起始地址
          output_idx,   // 输出起始地址
          pingpang,// 使用哪一个权重寄存器进行计算
          accumulate_delay);   // 是否对结果进行累加

        pingpang = !pingpang; // 计算完成后，切换加载寄存器和计算寄存器
      }

      // 迭代微操作,执行计算指令,少执行一个,少的用预加载计算执行,流水线每个周期执行一次
      I_UOP: for (int upc = insn.uop_bgn; upc < insn.uop_end -1; upc++) {
#pragma HLS PIPELINE II = 1

        // 从微操作缓冲区中提取微操作
        Uop_DataType uop = uop_buffer[upc];

        // 利用仿射解码索引
        input_idx = uop.range(UOP_INPUT_IDX_1, UOP_INPUT_IDX_0) + insn.input_base + input_offset;
        output_idx = uop.range(UOP_OUTPUT_IDX_1, UOP_OUTPUT_IDX_0) + output_offset;
        accumulate_delay = accumulate; // 累加信号延迟一位

        // 执行计算
        systolic_array.BMM_kernel(
            input_buffer,  				// 输入bram
            output_buffer,  			// 输出bram
            input_idx,    // 输入起始地址
            output_idx,   // 输出起始地址
            pingpang,// 使用哪一个权重寄存器进行计算
            accumulate);   // 是否对结果进行累加
      }

      // 计算最后一行的索引
      accumulate_delay = accumulate; // 在外面也要累加信号延迟一位
      input_idx=uop_buffer[insn.uop_end -1].range(UOP_INPUT_IDX_1, UOP_INPUT_IDX_0) + insn.input_base +  input_offset; // (i*dim_K_block + k)*MATRIX_WIDTH
      output_idx=uop_buffer[insn.uop_end -1].range(UOP_OUTPUT_IDX_1, UOP_OUTPUT_IDX_0) + output_offset;// (i*dim_J_block + j)*MATRIX_WIDTH ，i*dim_J_block*MATRIX_WIDTH在微操作指令中
      output_offset += MATRIX_WIDTH; // 提前预加载下一个权重块
      weigth_idx = insn.weight_base + weigth_offset + output_offset; // (k*dim_J_block + j)*MATRIX_WIDTH, weigth_offset是k*dim_J_block*MATRIX_WIDTH，j*MATRIX_WIDTH是output_offset 
    }
    input_offset += MATRIX_WIDTH; // 更新偏移
    weigth_offset += insn.dim_J_block*MATRIX_WIDTH;  // 如果想要去除这个，那么需要指令中带有偏移大小/利用dim_J_block计算
    // printf("input_idx:%d , weigth_idx:%d ,output_idx:%d\n",(int)input_offset,(int)weigth_idx,(int)output_idx);
    // printf("output_buffer:\n");print_buffer1((Output_DataType *)output_buffer,4);
  }

  // 执行最后一次计算
  systolic_array.BMM_kernel(
      input_buffer,  				// 输入bram
      output_buffer,  			// 输出bram
      input_idx,    // 输入起始地址
      output_idx,   // 输出起始地址
      pingpang,// 使用哪一个权重寄存器进行计算
      accumulate);   // 是否对结果进行累加
  // printf("input_offset:%d , weigth_offset:%d ,output_offset:%d\n",(int)input_offset,(int)weigth_offset,(int)output_offset);
  // printf("output_buffer:\n");print_buffer1((Output_DataType *)output_buffer,4);
  // printf("input_base:%d ,weight_base:%d\n",(int)insn.input_base,(int)insn.weight_base);

  // 转换outputbuff的指针为scale_type，用于处理relu的情况
  // T2** scale_buffer = reinterpret_cast<T2**>(result);

  // 是否对计算结果进行缩放?
  if(insn.scale_type != NO_QUANT) // 如果scale_type为NO_QUANT(0),代表不进行缩放,如果scale有值,代表进行缩放(可能是重量化也可能是量化)
  {
    // 进行反量化
    // 得到本次的运算参数
    Scale_DataType scale; // 定点的scale参数,有24位小数精度
    scale.range() = insn.scale; //按照位模式赋值,而不是值模式,达到类似memcpy的效果,不会转换类型
    output_offset = 0; // 重新初始化偏移参数

    // 下面通过for循环计算scale缩放,按顺序计算缩放
    // 外循环,计算I维度
    SCALE_I_UOP: for (int i0 = 0; i0 < insn.uop_end - insn.uop_bgn; i0++) {
      // 内循环,计算J维度
      SCALE_J_LOOP: for (int j0 = 0; j0 < insn.dim_J_block; j0++) {
        // 直接遍历输出缓冲区进行计算SCALE
        for (int i = 0; i <= MATRIX_WIDTH-1; i++) {
          for (int j = 0; j <= MATRIX_WIDTH-1; j++) {
            // 计算int32 * fixed<32,8>,将output_buffer中的计算int32转换为fixed<32,8>
            // 计算完一个直接转换为32位的Scale_DataType截断,然后写回,位模式赋值
            output_buffer[output_offset+ i][j].range() = ((Scale_DataType)(scale * output_buffer[output_offset+ i][j])).range();
            if (insn.relu_use) // 如果指明了反量化后进行relu操作，那么直接执行relu操作,需要转换为scale类型进行max然后再位复制
              output_buffer[output_offset+ i][j].range() = (std::max(*((Scale_DataType*)(&output_buffer[output_offset+ i][j])),(Scale_DataType)(0))).range(); 
          }
        }
        output_offset += MATRIX_WIDTH; // j一次增加MATRIX_WIDTH,因为一次处理了MATRIX_WIDTH行MATRIX_WIDTH列
      }
    }
  }
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
  volatile Uop_DataType *uops,
  volatile Transfer_DataType *biases,
  hls::stream<Instruct_DataType> &gemm_queue,
  hls::stream<bool> &l2c_raw_queue,
  hls::stream<bool> &s2c_war_queue,
  hls::stream<bool> &c2l_war_queue,
  hls::stream<bool> &c2s_raw_queue,
  Transfer_DataType input_buffer[INPUT_BUFFER_SIZE],
  Transfer_DataType weight_buffer[WEIGHT_BUFFER_SIZE],
  Transfer_DataType output_buffer[OUTPUT_BUFFER_SIZE]) 
  {
// PRAGMA_HLS(HLS INTERFACE s_axilite port = done bundle = CONTROL_BUS offset = COMPUTE_DONE_OFFSET) // done信号
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = done bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = uops bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = biases bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = uops offset = slave bundle = uops_port 
#pragma HLS INTERFACE m_axi port = biases offset = slave bundle = biases_port
#pragma HLS INTERFACE bram port = input_buffer storage_type = RAM_1P
#pragma HLS INTERFACE bram port = weight_buffer storage_type = RAM_1P
#pragma HLS INTERFACE bram port = output_buffer storage_type = RAM_1P
#pragma HLS INTERFACE axis port = l2c_raw_queue
#pragma HLS INTERFACE axis port = s2c_war_queue
#pragma HLS INTERFACE axis port = c2l_war_queue
#pragma HLS INTERFACE axis port = c2s_raw_queue
#pragma HLS INTERFACE axis port = gemm_queue // FIFO接口连接compute指令队列
// #pragma HLS ARRAY_RESHAPE variable=input_buffer complete  dim=2
// #pragma HLS ARRAY_RESHAPE variable=weight_buffer complete  dim=2
// #pragma HLS ARRAY_RESHAPE variable=output_buffer complete  dim=2

#pragma HLS dataflow // 使得设置权重和进行脉动阵列数据流并行

    // 微操作缓冲区
    static Uop_DataType uop_buffer[UOP_BUFFER_WIDTH];

    //从计算指令队列读取指令
    Instruct_DataType raw_instruct = gemm_queue.read();
    // 将原始指令转换为generic指令
    SAAInsn insn;
    Instruct_DataType raw_copy = raw_instruct;
    insn.generic = *((GenericIns *) &raw_copy);

	// 如果指令有指示存在依赖
	if (insn.generic.pop_pre_raw) { // 如果有对上一个模块的raw依赖，等待上一个模块完成才能读取
		l2c_raw_queue.read(); // 使用阻塞读取方法读取raw队列，如果存在raw令牌，表示上一个模块完成
	}
	if (insn.generic.pop_next_war) { // 如果有对下一个模块的war依赖，等待下一个模块完成才能写入
		s2c_war_queue.read(); // 使用阻塞读取方法读取war队列，如果存在war令牌，表示下一个模块完成
	}

    // 每次执行compute自动设置done信号未完成，直到最后执行DONE指令才设置为1，此时DONE指令后没有别的指令，保持done=1
    done = 0;
    // 判断指令类型
    if (insn.generic.opcode == OPCODE_DONE) {
      done = 1; // 如果我们到达DONE指令，则设置完成标志
    } 
    else if (insn.generic.opcode == OPCODE_LOAD) { // 计算模块的加载指令
      if (insn.mem.buffer_id == UOP_BUFFER_ID) { // 如果要加载uop微操作到微操作缓冲区
        // 将PS处微操作缓冲区数据复制到uop缓冲区
        memcpy(&uop_buffer[insn.mem.buffer_base],(const Uop_DataType*) &uops[insn.mem.dram_base],insn.mem.x_size * sizeof(Uop_DataType));
      }
      else if (insn.mem.buffer_id == OUTPUT_BUFFER_ID) { // 加载到输出缓冲区
        load_2d_block1<Transfer_DataType, Output_Block,TRANSFER_OUTPPUT_AXI_RATIO,TRANSFER_OUTPUT_DATA_BYTES>(
          biases,
          (Output_Block *)output_buffer,
          insn.mem.buffer_base,
          insn.mem.dram_base,
          insn.mem.y_size,
          insn.mem.x_size,
          insn.mem.x_stride);
      }
    }
    else if (insn.generic.opcode == OPCODE_GEMM) { // 执行gemm指令使用微操作进行硬件块矩阵乘法
      // gemm(
      //   raw_copy,
      //   uop_buffer,
      //   (Input_DataType (*)[MATRIX_WIDTH])input_buffer,
      //   (Weight_DataType (*)[MATRIX_WIDTH])weight_buffer,
      //   (Output_DataType (*)[MATRIX_WIDTH])output_buffer); // 被转换为数组指针，一维数组变成二维数组进行索引
    }
    else if (insn.generic.opcode == OPCODE_ANU) { // 执行ANU指令进行激活和归一化
      // anu(
      //   raw_copy,
      //   (Input_DataType (*)[MATRIX_WIDTH])input_buffer,
      //   (Weight_DataType (*)[MATRIX_WIDTH])weight_buffer,
      //   (Scale_DataType (*)[MATRIX_WIDTH])output_buffer); // 转换为归一化类型读取
    }

	// 如果指令有指示存在影响
	if (insn.generic.push_pre_war) { // 如果对上一个模块有war影响，那么执行完后，告诉上一个模块我已经读取完毕，你可以写入
		c2l_war_queue.write(1); // 写入一个war令牌
	}
	if (insn.generic.push_next_raw) { // 如果对下一个模块有raw影响，那么执行完后，告诉下一个模块我已经写入完毕，你可以读取
		c2s_raw_queue.write(1);; // 写入一个raw令牌
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
// void saa_top(
//   uint32_t insn_count,
//   volatile Instruct_DataType *insns,
//   volatile Uop_DataType *uops,
//   volatile Transfer_DataType *inputs,
//   volatile Transfer_DataType *weights,
//   volatile Transfer_DataType *biases,
//   volatile Transfer_DataType *outputs,
//   volatile uint32_t &done) 
// {
// #pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = done bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

// #pragma HLS INTERFACE s_axilite port = insns bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = uops bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = inputs bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = weights bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = outputs bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = biases bundle = CONTROL_BUS

// #pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port 
// #pragma HLS INTERFACE m_axi port = uops offset = slave bundle = uops_port 
// #pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = inputs_port
// #pragma HLS INTERFACE m_axi port = weights offset = slave bundle = weights_port
// #pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = outputs_port
// #pragma HLS INTERFACE m_axi port = biases offset = slave bundle = biases_port

//   // // 只在第一次使用脉动阵列时reset就行，目的是清除PSUM寄存器中的值
//   // systolic_array.reset(); 

//   // 实例化命令队列并设置队列深度
//   hls::stream<Instruct_DataType> load_queue;
//   hls::stream<Instruct_DataType> gemm_queue;
//   hls::stream<Instruct_DataType> store_queue;
//   PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=load_queue)
//   PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=gemm_queue)
//   PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=store_queue)

//   // 依赖队列，顺着数据流就是RAW，逆着数据流就是WAR
//   hls::stream<bool> l2c_raw_queue; // RAW，load模块到compute模块
//   hls::stream<bool> s2c_war_queue; // WAR，store模块到compute模块
//   hls::stream<bool> c2l_war_queue; // WAR,compute模块到load模块
//   hls::stream<bool> c2s_raw_queue; // RAW,compute模块到store模块
//   PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=l2c_raw_queue)
//   PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=s2c_war_queue)
//   PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=c2l_war_queue)
//   PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=c2s_raw_queue)

//   //缓冲区buffer声明
//   static Transfer_DataType  weight_buffer[WEIGHT_BUFFER_SIZE]; //使用静态变量定义
//   static Transfer_DataType  input_buffer[INPUT_BUFFER_SIZE];   //使用静态变量定义
//   static Transfer_DataType  output_buffer[OUTPUT_BUFFER_SIZE]; //使用静态变量定义

//     // 优化buffer
// // PRAGMA_HLS(HLS ARRAY_RESHAPE variable = output_buffer type=block factor=MATRIX_WIDTH dim=1) // 对累加器BUffer进行优化存储效率
// // #pragma HLS DEPENDENCE variable = output_buffer inter false // 指明依赖关系,表示不同迭代不会相互影响,使得II可以为1

// 	// 临时队列用于查看指令中的依赖决定是否执行指令
// 	hls::stream<Instruct_DataType> tmp_load_queue;
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_load_queue)
// 	hls::stream<Instruct_DataType> tmp_gemm_queue;
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_gemm_queue)
// 	hls::stream<Instruct_DataType> tmp_store_queue;
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_store_queue)

// 	// 临时指令，同样用于提取指令中的依赖，如果满足，则压入正式队列执行该公式
// 	Instruct_DataType tmp_load;
// 	Instruct_DataType tmp_gemv;
// 	Instruct_DataType tmp_store;

// 	// 查看状态，用于查看临时队列中的指令是否被取出，取出则为真，压入正式队列又为假
// 	bool tmp_load_popped = false;
// 	bool tmp_gemm_popped = false;
// 	bool tmp_store_popped = false;

//     //顺序执行模块
// printf("\n=========================SAA Start===========================\n");

//     // 将所有指令压入队列，这是执行一批指令的大小，可以是执行一个二级分块矩阵乘法的指令流
//     fetch(insn_count, insns, tmp_load_queue, tmp_gemm_queue, tmp_store_queue);

//     // 执行命令
//     while (true) {

//         // printf("- Execute");

//         // 首先尽可能的执行加载命令
//         // 当临时队列不为空或者已经获得一个加载指令时，需要判断这个指令是否满足依赖，如果满足则将指令压入正式队列执行
//         while(!tmp_load_queue.empty() || tmp_load_popped == true)
//         {
//           // 如果队列不为空同时，没有获得一个加载指令，那么从队列中取出一个加载指令，同时设置标志位为已获得加载指令
//           if (!tmp_load_popped) {
//             tmp_load_queue.read(tmp_load);
//             tmp_load_popped = true;
//           }
//           // 检查获得的加载指令的依赖关系，若满足则压入正式load队列
//           // 分两种情况,
//           // 1 如果存在依赖，同时c2l_war_queue队列不为空，代表此时c2l_war_queue存在令牌，也就是compute完成，依赖满足可以执行
//           // 2 如果不存在依赖，则直接执行
//           GenericIns insn = *((GenericIns *) &tmp_load);
//           MemIns mem_insn = *((MemIns *) &tmp_load); // 查看load指令内容
//           if ((insn.pop_next_war && !c2l_war_queue.empty()) || 
//               !insn.pop_next_war)
//           {
//             // 将指令压入加载队列
//             load_queue.write(tmp_load);
//             tmp_load_popped = false; // 压入后，标志位变成没有压入

//             // 执行加载指令
//             load(load_queue, 
//                   inputs, 
//                   weights,
//                   l2c_raw_queue,
//                   c2l_war_queue,
//                   input_buffer, 
//                   weight_buffer); // 执行加载队列的指令

//             // printf(" load ");
//             // 打印指令情况
//             printf("- Execute  load");
//             if (mem_insn.buffer_id == INPUT_BUFFER_ID) printf("(inp):");
//             if (mem_insn.buffer_id == WEIGHT_BUFFER_ID) printf("(wgt):");
//             printf("pop_next_war:%d , push_next_raw:%d\n",insn.pop_next_war,insn.push_next_raw);
//           }
//           else 
//             break;//如果不满足上面的条件，证明加载阶段的执行等待其他阶段的完成（能执行到这里说明存在依赖但是compute模块未执行完）
//         }

//         // 然后尽可能的执行计算命令（包含uop加载和bias加载）
//         while(!tmp_gemm_queue.empty() || tmp_gemm_popped == true) 
//         {
//           if (!tmp_gemm_popped) {
//             tmp_gemm_queue.read(tmp_gemv);
//             tmp_gemm_popped = true;
//           }

//           // 检查计算指令的依赖关系，如果满足则压入正式gemm队列
//           // 分为四种情况，这些情况有着先后顺序问题
//           // 1 存在对load的依赖、同时load执行完毕，l2c中有raw令牌、同时存在对sotre的依赖、同时store执行完，s2c中存在war令牌
//           // 2 存在对store的依赖不存在对load的依赖,同时s2c中存在war令牌代表store执行完毕
//           // 3 存在对load的依赖不存在对store的依赖,同时l2c中存在raw令牌代表load执行完毕
//           // 4 以上条件都不成立，不存在对load的依赖和对store的依赖，可以直接执行
//           GenericIns insn = *((GenericIns *) &tmp_gemv);
//           MemIns mem_insn = *((MemIns *) &tmp_gemv); // 查看load指令内容
//           ComIns com_insn = *((ComIns *) &tmp_gemv); // 查看gemm指令内容
//           if (
//             (insn.pop_pre_raw && !l2c_raw_queue.empty() && insn.pop_next_war && !s2c_war_queue.empty()) ||
//             (!insn.pop_pre_raw && insn.pop_next_war && !s2c_war_queue.empty()) ||
//             (insn.pop_pre_raw && !l2c_raw_queue.empty() && !insn.pop_next_war) ||
//             (!insn.pop_pre_raw && !insn.pop_next_war)) 
//           {
//             // 将指令压入计算队列
//             gemm_queue.write(tmp_gemv);
//             tmp_gemm_popped = false; // 压入后，标志位变成没有压入

//             // 执行计算指令
//             compute(done,
//                     uops,
//                     biases, 
//                     gemm_queue, 
//                     l2c_raw_queue,
//                     s2c_war_queue,
//                     c2l_war_queue,
//                     c2s_raw_queue,
//                     input_buffer, 
//                     weight_buffer,
//                     output_buffer); // 执行计算队列指令

//             // printf(" compute ");
//             // 打印指令情况
//             printf("- Execute  compute");
//             if(insn.opcode == OPCODE_DONE) printf("(done):");
//             if(insn.opcode == OPCODE_LOAD) 
//             {
//               if (mem_insn.buffer_id == OUTPUT_BUFFER_ID)
//                 printf("(bias):");
//               else if(mem_insn.buffer_id == UOP_BUFFER_ID)
//                 printf("(uop):");
//             }
//             if(insn.opcode == OPCODE_GEMM) printf("(gemm):");
//             if(insn.opcode == OPCODE_ANU) printf("(anu):");
//             printf("pop_pre_raw:%d , pop_next_war:%d , push_pre_war:%d , push_next_raw:%d\n",insn.pop_pre_raw,insn.pop_next_war,insn.push_pre_war,insn.push_next_raw);
//           }
//           else 
//             break;//如果不满足上面的条件，证明计算阶段的执行等待其他阶段的完成（可能是load也可能是store未完成）
//         }
//         // 然后尽可能的执行存储命令
//         while(!tmp_store_queue.empty() || tmp_store_popped == true)
//         {
//           if (!tmp_store_popped) {
//             tmp_store_queue.read(tmp_store);
//             tmp_store_popped = true;
//           }
//           // 检查存储指令的依赖关系，如果满足则压入正式store队列
//           // 分为两种情况
//           // 1 存在对compute的依赖，同时c2s队列中存在raw令牌代表compute执行完毕
//           // 2 不存在依赖，直接执行
//           GenericIns insn = *((GenericIns *) &tmp_store);
//           MemIns mem_insn = *((MemIns *) &tmp_store); // 查看store指令内容
//           if ((insn.pop_pre_raw && !c2s_raw_queue.empty()) ||
//               !insn.pop_pre_raw) 
//           {
//             // 将指令压入存储队列
//             store_queue.write(tmp_store);
//             tmp_store_popped = false;

//             // 执行存储指令
//             store(store_queue, 
//                   outputs,
//                   s2c_war_queue,
//                   c2s_raw_queue,
//                   input_buffer, 
//                   output_buffer); // 执行存储队列指令

//             // printf(" store ");
//             // 打印指令情况
//             printf("- Execute  store");
//             if (mem_insn.buffer_id == OUTPUT_BUFFER_ID) printf("(out):");
//             printf("pop_pre_raw:%d , push_pre_war:%d\n",insn.pop_pre_raw,insn.push_pre_war);
//           }
//           else 
//             break;//如果不满足上面的条件，证明计算阶段的执行等待其他阶段的完成（compute未完成）
//         }
//       // printf("\n");
//       // 检查是否收到已完成的信号,如果收到就跳出循环
//       if (done) 
//       {
//         printf("\nINFO - SAA is done\n");
//         break;
//       }
//     }

//   // 检测令牌队列是否为空，保证所有依赖关系处理完成
//   bool tmp_tok;
//   int l2c_count = 0;
//   int s2c_count = 0;
//   int c2l_count = 0;
//   int c2s_count = 0;
//   while (l2c_raw_queue.read_nb(tmp_tok)) { // 如果非阻塞一直读取成功表示有令牌存在
//     l2c_count++;
//   }
//   while (s2c_war_queue.read_nb(tmp_tok)) {
//     s2c_count++;
//   }
//   while (c2l_war_queue.read_nb(tmp_tok)) {
//     c2l_count++;
//   }
//   while (c2s_raw_queue.read_nb(tmp_tok)) {
//     c2s_count++;
//   }

//   printf("INFO - SAA queue: l2c_count=%d, s2c_count=%d, c2l_count=%d, c2s_count=%d\n",
//          l2c_count, s2c_count, c2l_count,c2s_count);

// printf("\n==========================SAA End===========================\n");

// }


void saa_top(
  uint32_t insn_count,
  volatile Instruct_DataType *insns,
  volatile Uop_DataType *uops,
  volatile Transfer_DataType *inputs,
  volatile Transfer_DataType *weights,
  volatile Transfer_DataType *biases,
  volatile Transfer_DataType *outputs,
  volatile uint32_t &done) 
{
#pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = done bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

#pragma HLS INTERFACE s_axilite port = insns bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = uops bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = inputs bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = weights bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = outputs bundle = CONTROL_BUS
#pragma HLS INTERFACE s_axilite port = biases bundle = CONTROL_BUS

#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port 
#pragma HLS INTERFACE m_axi port = uops offset = slave bundle = uops_port 
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = inputs_port
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = weights_port
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = outputs_port
#pragma HLS INTERFACE m_axi port = biases offset = slave bundle = biases_port

  // // 只在第一次使用脉动阵列时reset就行，目的是清除PSUM寄存器中的值
  // systolic_array.reset(); 

  // 实例化命令队列并设置队列深度
  hls::stream<Instruct_DataType> load_queue;
  hls::stream<Instruct_DataType> gemm_queue;
  hls::stream<Instruct_DataType> store_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=load_queue)
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=gemm_queue)
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=store_queue)

  // 依赖队列，顺着数据流就是RAW，逆着数据流就是WAR
  hls::stream<bool> l2c_raw_queue; // RAW，load模块到compute模块
  hls::stream<bool> s2c_war_queue; // WAR，store模块到compute模块
  hls::stream<bool> c2l_war_queue; // WAR,compute模块到load模块
  hls::stream<bool> c2s_raw_queue; // RAW,compute模块到store模块
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=l2c_raw_queue)
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=s2c_war_queue)
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=c2l_war_queue)
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=c2s_raw_queue)

  //缓冲区buffer声明
  static Transfer_DataType  weight_buffer[WEIGHT_BUFFER_SIZE]; //使用静态变量定义
  static Transfer_DataType  input_buffer[INPUT_BUFFER_SIZE];   //使用静态变量定义
  static Transfer_DataType  output_buffer[OUTPUT_BUFFER_SIZE]; //使用静态变量定义

    // 优化buffer
// PRAGMA_HLS(HLS ARRAY_RESHAPE variable = output_buffer type=block factor=MATRIX_WIDTH dim=1) // 对累加器BUffer进行优化存储效率
// #pragma HLS DEPENDENCE variable = output_buffer inter false // 指明依赖关系,表示不同迭代不会相互影响,使得II可以为1

	// 临时队列用于查看指令中的依赖决定是否执行指令
	hls::stream<Instruct_DataType> tmp_load_queue;
	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_load_queue)
	hls::stream<Instruct_DataType> tmp_gemm_queue;
	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_gemm_queue)
	hls::stream<Instruct_DataType> tmp_store_queue;
	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_store_queue)

	// 临时指令，同样用于提取指令中的依赖，如果满足，则压入正式队列执行该公式
	Instruct_DataType tmp_load;
	Instruct_DataType tmp_gemv;
	Instruct_DataType tmp_store;

	// 查看状态，用于查看临时队列中的指令是否被取出，取出则为真，压入正式队列又为假
	bool tmp_load_popped = false;
	bool tmp_gemm_popped = false;
	bool tmp_store_popped = false;

    //顺序执行模块
printf("\n=========================SAA Start===========================\n");

    // 将所有指令压入队列，这是执行一批指令的大小，可以是执行一个二级分块矩阵乘法的指令流
    fetch(insn_count, insns, tmp_load_queue, tmp_gemm_queue, tmp_store_queue);

    // 执行命令
    while (true) {

        // printf("- Execute");

        // 首先尽可能的执行加载命令
        // 当临时队列不为空或者已经获得一个加载指令时，需要判断这个指令是否满足依赖，如果满足则将指令压入正式队列执行
        while(!tmp_load_queue.empty() || tmp_load_popped == true)
        {
          // 如果队列不为空同时，没有获得一个加载指令，那么从队列中取出一个加载指令，同时设置标志位为已获得加载指令
          if (!tmp_load_popped) {
            tmp_load_queue.read(tmp_load);
            tmp_load_popped = true;
          }
          // 检查获得的加载指令的依赖关系，若满足则压入正式load队列
          // 分两种情况,
          // 1 如果存在依赖，同时c2l_war_queue队列不为空，代表此时c2l_war_queue存在令牌，也就是compute完成，依赖满足可以执行
          // 2 如果不存在依赖，则直接执行
          GenericIns insn = *((GenericIns *) &tmp_load);
          MemIns mem_insn = *((MemIns *) &tmp_load); // 查看load指令内容
          if ((insn.pop_next_war && !c2l_war_queue.empty()) || 
              !insn.pop_next_war)
          {
            // 将指令压入加载队列
            load_queue.write(tmp_load);
            tmp_load_popped = false; // 压入后，标志位变成没有压入

            // 执行加载指令
            load(load_queue, 
                  inputs, 
                  weights,
                  l2c_raw_queue,
                  c2l_war_queue,
                  input_buffer, 
                  weight_buffer); // 执行加载队列的指令
          }
          else 
            break;//如果不满足上面的条件，证明加载阶段的执行等待其他阶段的完成（能执行到这里说明存在依赖但是compute模块未执行完）
        }

        // 然后尽可能的执行计算命令（包含uop加载和bias加载）
        while(!tmp_gemm_queue.empty() || tmp_gemm_popped == true) 
        {
          if (!tmp_gemm_popped) {
            tmp_gemm_queue.read(tmp_gemv);
            tmp_gemm_popped = true;
          }

          // 检查计算指令的依赖关系，如果满足则压入正式gemm队列
          // 分为四种情况，这些情况有着先后顺序问题
          // 1 存在对load的依赖、同时load执行完毕，l2c中有raw令牌、同时存在对sotre的依赖、同时store执行完，s2c中存在war令牌
          // 2 存在对store的依赖不存在对load的依赖,同时s2c中存在war令牌代表store执行完毕
          // 3 存在对load的依赖不存在对store的依赖,同时l2c中存在raw令牌代表load执行完毕
          // 4 以上条件都不成立，不存在对load的依赖和对store的依赖，可以直接执行
          GenericIns insn = *((GenericIns *) &tmp_gemv);
          MemIns mem_insn = *((MemIns *) &tmp_gemv); // 查看load指令内容
          ComIns com_insn = *((ComIns *) &tmp_gemv); // 查看gemm指令内容
          if (
            (insn.pop_pre_raw && !l2c_raw_queue.empty() && insn.pop_next_war && !s2c_war_queue.empty()) ||
            (!insn.pop_pre_raw && insn.pop_next_war && !s2c_war_queue.empty()) ||
            (insn.pop_pre_raw && !l2c_raw_queue.empty() && !insn.pop_next_war) ||
            (!insn.pop_pre_raw && !insn.pop_next_war)) 
          {
            // 将指令压入计算队列
            gemm_queue.write(tmp_gemv);
            tmp_gemm_popped = false; // 压入后，标志位变成没有压入

            // 执行计算指令
            compute(done,
                    uops,
                    biases, 
                    gemm_queue, 
                    l2c_raw_queue,
                    s2c_war_queue,
                    c2l_war_queue,
                    c2s_raw_queue,
                    input_buffer, 
                    weight_buffer,
                    output_buffer); // 执行计算队列指令
          }
          else 
            break;//如果不满足上面的条件，证明计算阶段的执行等待其他阶段的完成（可能是load也可能是store未完成）
        }
        // 然后尽可能的执行存储命令
        while(!tmp_store_queue.empty() || tmp_store_popped == true)
        {
          if (!tmp_store_popped) {
            tmp_store_queue.read(tmp_store);
            tmp_store_popped = true;
          }
          // 检查存储指令的依赖关系，如果满足则压入正式store队列
          // 分为两种情况
          // 1 存在对compute的依赖，同时c2s队列中存在raw令牌代表compute执行完毕
          // 2 不存在依赖，直接执行
          GenericIns insn = *((GenericIns *) &tmp_store);
          MemIns mem_insn = *((MemIns *) &tmp_store); // 查看store指令内容
          if ((insn.pop_pre_raw && !c2s_raw_queue.empty()) ||
              !insn.pop_pre_raw) 
          {
            // 将指令压入存储队列
            store_queue.write(tmp_store);
            tmp_store_popped = false;

            // 执行存储指令
            store(store_queue, 
                  outputs,
                  s2c_war_queue,
                  c2s_raw_queue,
                  input_buffer, 
                  output_buffer); // 执行存储队列指令
          }
          else 
            break;//如果不满足上面的条件，证明计算阶段的执行等待其他阶段的完成（compute未完成）
        }
      // printf("\n");
      // 检查是否收到已完成的信号,如果收到就跳出循环
      if (done) 
      {
        printf("\nINFO - SAA is done\n");
        break;
      }
    }

  // 检测令牌队列是否为空，保证所有依赖关系处理完成
  bool tmp_tok;
  int l2c_count = 0;
  int s2c_count = 0;
  int c2l_count = 0;
  int c2s_count = 0;
  while (l2c_raw_queue.read_nb(tmp_tok)) { // 如果非阻塞一直读取成功表示有令牌存在
    l2c_count++;
  }
  while (s2c_war_queue.read_nb(tmp_tok)) {
    s2c_count++;
  }
  while (c2l_war_queue.read_nb(tmp_tok)) {
    c2l_count++;
  }
  while (c2s_raw_queue.read_nb(tmp_tok)) {
    c2s_count++;
  }

  printf("INFO - SAA queue: l2c_count=%d, s2c_count=%d, c2l_count=%d, c2s_count=%d\n",
         l2c_count, s2c_count, c2l_count,c2s_count);

printf("\n==========================SAA End===========================\n");

}

//---------------------------------debug--------------------------------//

// void saa_top(
//   uint32_t insn_count,
//   volatile Instruct_DataType *insns,
//   volatile Uop_DataType *uops,
//   volatile Transfer_DataType *inputs,
//   volatile Transfer_DataType *weights,
//   volatile Transfer_DataType *biases,
//   volatile Transfer_DataType *outputs,
//   volatile uint32_t &done) 
// {
// #pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = done bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

// #pragma HLS INTERFACE s_axilite port = insns bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = uops bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = inputs bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = weights bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = outputs bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = biases bundle = CONTROL_BUS

// #pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port 
// #pragma HLS INTERFACE m_axi port = uops offset = slave bundle = uops_port 
// #pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = inputs_port
// #pragma HLS INTERFACE m_axi port = weights offset = slave bundle = weights_port
// #pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = outputs_port
// #pragma HLS INTERFACE m_axi port = biases offset = slave bundle = biases_port

//     // // 只在第一次使用脉动阵列时reset就行，目的是清除PSUM寄存器中的值
//     // systolic_array.reset(); 

//     // 实例化命令队列并设置队列深度
//     hls::stream<Instruct_DataType> load_queue;
//     hls::stream<Instruct_DataType> gemm_queue;
//     hls::stream<Instruct_DataType> store_queue;
//     PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=load_queue)
//     PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=gemm_queue)
//     PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=store_queue)

// 	// 依赖队列，顺着数据流就是RAW，逆着数据流就是WAR
// 	hls::stream<bool> l2c_raw_queue; // RAW，load模块到compute模块
// 	hls::stream<bool> s2c_war_queue; // WAR，store模块到compute模块
// 	hls::stream<bool> c2l_war_queue; // WAR,compute模块到load模块
// 	hls::stream<bool> c2s_raw_queue; // RAW,compute模块到store模块
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=l2c_raw_queue)
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=s2c_war_queue)
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=c2l_war_queue)
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=c2s_raw_queue)

//     //缓冲区buffer声明
//   static Transfer_DataType  weight_buffer[WEIGHT_BUFFER_SIZE]; //使用静态变量定义
//   static Transfer_DataType  input_buffer[INPUT_BUFFER_SIZE];   //使用静态变量定义
//   static Transfer_DataType  output_buffer[OUTPUT_BUFFER_SIZE]; //使用静态变量定义

//     // 优化buffer
// // PRAGMA_HLS(HLS ARRAY_RESHAPE variable = output_buffer type=block factor=MATRIX_WIDTH dim=1) // 对累加器BUffer进行优化存储效率
// // #pragma HLS DEPENDENCE variable = output_buffer inter false // 指明依赖关系,表示不同迭代不会相互影响,使得II可以为1

// 	// 临时队列用于查看指令中的依赖决定是否执行指令
// 	hls::stream<Instruct_DataType> tmp_load_queue;
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_load_queue)
// 	hls::stream<Instruct_DataType> tmp_gemm_queue;
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_gemm_queue)
// 	hls::stream<Instruct_DataType> tmp_store_queue;
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_store_queue)

// 	// 临时指令，同样用于提取指令中的依赖，如果满足，则压入正式队列执行该公式
// 	Instruct_DataType tmp_load;
// 	Instruct_DataType tmp_gemv;
// 	Instruct_DataType tmp_store;

// 	// 查看状态，用于查看临时队列中的指令是否被取出，取出则为真，压入正式队列又为假
// 	bool tmp_load_popped = false;
// 	bool tmp_gemm_popped = false;
// 	bool tmp_store_popped = false;

//     //顺序执行模块

//     // 将所有指令压入队列，这是执行一批指令的大小，可以是执行一个二级分块矩阵乘法的指令流
//     fetch(insn_count, insns, load_queue, gemm_queue, store_queue);

//     // 执行命令
//     while (true) {

//         // 首先尽可能的执行加载命令
//         while(!load_queue.empty())
//         {
// 			// 如果满足依赖条件，执行一个加载命令
// 			// 如果load模块WAR依赖于compute
//       		// if ((insn.pop_next_war && !c2l_war_queue.empty()) || !insn.pop_next_war)
//             load(load_queue, 
//                 inputs, 
//                 weights,
//                 l2c_raw_queue,
//                 c2l_war_queue,
//                 input_buffer, 
//                 weight_buffer); // 执行加载队列的指令
//         }
//         while(!gemm_queue.empty()) // 
//         {
//             compute(done,
//                     uops,
//                     biases, 
//                     gemm_queue, 
//                     l2c_raw_queue,
//                     s2c_war_queue,
//                     c2l_war_queue,
//                     c2s_raw_queue,
//                     input_buffer, 
//                     weight_buffer,
//                     output_buffer); // 执行计算队列指令
//         }
//         // 然后尽可能的执行存储命令
//         while(!store_queue.empty())
//         {
//             store(store_queue, 
//                   outputs,
//                   s2c_war_queue,
//                   c2s_raw_queue,
//                   input_buffer, 
//                   output_buffer); // 执行存储队列指令
//         }

// 		// 检查是否收到已完成的信号,如果收到就跳出循环
// 		if (done) 
// 		{
// 			break;
// 		}
//     }

// }


// void saa_top(
//   uint32_t insn_count,
//   volatile Instruct_DataType *insns,
//   volatile Uop_DataType *uops,
//   volatile Transfer_DataType *inputs,
//   volatile Transfer_DataType *weights,
//   volatile Transfer_DataType *biases,
//   volatile Transfer_DataType *outputs,
//   volatile uint32_t &done) 
// {
// #pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = done bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

// #pragma HLS INTERFACE s_axilite port = insns bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = uops bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = inputs bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = weights bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = outputs bundle = CONTROL_BUS
// #pragma HLS INTERFACE s_axilite port = biases bundle = CONTROL_BUS

// #pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port 
// #pragma HLS INTERFACE m_axi port = uops offset = slave bundle = uops_port 
// #pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = inputs_port
// #pragma HLS INTERFACE m_axi port = weights offset = slave bundle = weights_port
// #pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = outputs_port
// #pragma HLS INTERFACE m_axi port = biases offset = slave bundle = biases_port

//     // // 只在第一次使用脉动阵列时reset就行，目的是清除PSUM寄存器中的值
//     // systolic_array.reset(); 

//     // 实例化命令队列并设置队列深度
//     hls::stream<Instruct_DataType> load_queue;
//     hls::stream<Instruct_DataType> gemm_queue;
//     hls::stream<Instruct_DataType> store_queue;
//     PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=load_queue)
//     PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=gemm_queue)
//     PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=store_queue)

// 	// 依赖队列，顺着数据流就是RAW，逆着数据流就是WAR
// 	hls::stream<bool> l2c_raw_queue; // RAW，load模块到compute模块
// 	hls::stream<bool> s2c_war_queue; // WAR，store模块到compute模块
// 	hls::stream<bool> c2l_war_queue; // WAR,compute模块到load模块
// 	hls::stream<bool> c2s_raw_queue; // RAW,compute模块到store模块
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=l2c_raw_queue)
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=s2c_war_queue)
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=c2l_war_queue)
// 	PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=c2s_raw_queue)

//     //缓冲区buffer声明
//     static Weight_DataType weight_buffer[WEIGHT_BUFFER_WIDTH*MATRIX_WIDTH]; //使用静态变量定义
//     static Input_DataType input_buffer[INPUT_BUFFER_WIDTH*MATRIX_WIDTH]; //使用静态变量定义
//     static Output_DataType output_buffer[OUTPUT_BUFFER_WIDTH*MATRIX_WIDTH]; //使用静态变量定义

//     // 优化buffer
// // PRAGMA_HLS(HLS ARRAY_RESHAPE variable = output_buffer type=block factor=MATRIX_WIDTH dim=1) // 对累加器BUffer进行优化存储效率
// // #pragma HLS DEPENDENCE variable = output_buffer inter false // 指明依赖关系,表示不同迭代不会相互影响,使得II可以为1

//     //顺序执行模块
// printf("\n=========================SAA Start===========================\n");

//     // 将所有指令压入队列，这是执行一批指令的大小，可以是执行一个二级分块矩阵乘法的指令流
//     fetch(insn_count, insns, load_queue, gemm_queue, store_queue);

//     // 执行命令
//     while (true) {
//         // if(load_queue.empty() && store_queue.empty() && gemm_queue.empty()) //如果加载队列和计算队列空就跳出
//         //     break;
//         // 首先尽可能的执行加载命令
//         while(!load_queue.empty())
//         {
// 			// 如果满足依赖条件，执行一个加载命令
//       if ((insn.pop_next_dep && !g2l_dep_queue.empty()) || !insn.pop_next_dep)
//             load(load_queue, 
// 				 inputs, 
// 				 weights,
// 				 l2c_raw_queue,
// 				 c2l_war_queue,
// 				 input_buffer, 
// 				 weight_buffer); // 执行加载队列的指令
//         }
//         while(!gemm_queue.empty()) // 
//         {
//             compute(done,
// 					uops,
// 					biases, 
// 					gemm_queue, 
// 					l2c_raw_queue,
// 					s2c_war_queue,
// 					c2l_war_queue,
// 					c2s_raw_queue,
// 					input_buffer, 
// 					weight_buffer,
// 					output_buffer); // 执行计算队列指令
//         }
//         // 然后尽可能的执行存储命令
//         while(!store_queue.empty())
//         {
//             store(store_queue, 
// 				  outputs,
// 				  s2c_war_queue,
// 				  c2s_raw_queue,
// 				  input_buffer, 
// 				  output_buffer); // 执行存储队列指令
//         }


// 		// 如果加载队列不是空同时


// 		// 检查是否收到已完成的信号,如果收到就跳出循环
// 		if (done) 
// 		{
// 			// printf("\nsaa is done\n");
// 			break;
// 		}
//     }

//   // 检测令牌队列是否为空，保证所有依赖关系处理完成
//   bool tmp_tok;
//   int l2c_count = 0;
//   int s2c_count = 0;
//   int c2l_count = 0;
//   int c2s_count = 0;
//   while (l2c_raw_queue.read_nb(tmp_tok)) { // 如果非阻塞一直读取成功表示有令牌存在
//     l2c_count++;
//   }
//   while (s2c_war_queue.read_nb(tmp_tok)) {
//     s2c_count++;
//   }
//   while (c2l_war_queue.read_nb(tmp_tok)) {
//     c2l_count++;
//   }
//   while (c2s_raw_queue.read_nb(tmp_tok)) {
//     c2s_count++;
//   }

//   printf("INFO - SAA queue: l2c_count=%d, s2c_count=%d, c2l_count=%d, c2s_count=%d\n",
//          l2c_count, s2c_count, c2l_count,c2s_count);

// printf("\n==========================SAA End===========================\n");

// }
