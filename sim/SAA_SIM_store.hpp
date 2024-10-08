// // 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// // 并行方式:存在Bias那么load和bias并行,或者和store并行，在K维度上使用双缓冲，使用计算时间隐藏加载时间
// // 可以融合relu、反量化、layernorm、softmax
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       bool relu_use, // 是否应用relu
//                       int  scale_type, // scale的类型
//                       float scale, // scale的大小(浮点小数)
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
//   Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
//   Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
//   Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

//   // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
//   Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 声明输出缓冲区(输出的是没有去填充的输出,一维)
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

// // 假设只对K维度进行双缓冲，因此I，J不变，K维度存储大小减少一半
// #define db_max_tile_k (((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / 2) / max_tile_i_j) // 在不进行双缓冲的基础上，假设K维度减少一半的存储大小，计算得到最大K维度分块的大小

//   bool virtual_threads = 1; // 如果不存在act，就使用k维度虚拟线程/双缓冲
//   // virtual_threads = act ? 0 : 1;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲
//   // virtual_threads = 0;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲

//   // 计算每个缓冲区的最大容量，在使用双缓冲的前提下
//   const size_t max_input_buffer = virtual_threads ? INPUT_BUFFER_WIDTH / 2 :INPUT_BUFFER_WIDTH;
//   const size_t max_weight_buffer = virtual_threads ? WEIGHT_BUFFER_WIDTH / 2 :WEIGHT_BUFFER_WIDTH;
//   const size_t max_output_buffer = virtual_threads ? OUTPUT_BUFFER_WIDTH : OUTPUT_BUFFER_WIDTH;

//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算，如果有act，则首先选择act条件，如果没有就可以使用双缓冲条件
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX，首先满足act
//   {
//       tile_I = 1;
//       tile_J = dim_J_stride; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else if (virtual_threads) // 如果没有act，才考虑使用虚拟线程/双缓冲初始条件，如果是针对K维度进行的双缓冲，那么K维度的最大分块减少一倍
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < db_max_tile_k ? dim_K_stride : db_max_tile_k; 
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 先增大K维度的分块系数，提高双缓冲效率
//     if(tile_I * (tile_K+1) <= max_input_buffer &&
//        (tile_K+1) * tile_J <= max_weight_buffer &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= max_output_buffer &&
//        tile_K * (tile_J+1) <= max_weight_buffer &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }

//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= max_input_buffer &&
//        (tile_I+1) * tile_J <= max_output_buffer &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 如果得到的分块系数计算还是大于权重缓冲区的一半，那么不使用双缓冲,这一步保证了双缓冲让步于完整完成计算
//   if((tile_K * tile_J)>max_weight_buffer) virtual_threads = 0;// 如果使用双缓冲还是大于权重缓冲区大小，那么只能暂停使用双缓冲
//   if((tile_I * tile_K)>max_input_buffer) virtual_threads = 0;// 如果使用双缓冲还是大于输入缓冲区大小，那么只能暂停使用双缓冲

//   // 在上面的基础上，还是判断分块系数不使用双缓冲能不能加载上
//   assert((tile_I * tile_J)<=OUTPUT_BUFFER_WIDTH);
//   assert((tile_I * tile_K)<=INPUT_BUFFER_WIDTH);
//   assert((tile_K * tile_J)<=WEIGHT_BUFFER_WIDTH); // 输出缓冲区要计算

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("==========================================GEMM Start===========================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, double_buffer_use=%d, bias_use=%d, relu_use=%d, scale=%f, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K,virtual_threads,bias_use,relu_use,scale,act);
//   // printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
//   int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_act_size = act ? I0*J0 : 0; //计算一个输出块在store前进行anu操作,如果使用则存在该指令
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size + insn_act_size;

//   // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
//   assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size+insn_act_size) <= STREAM_IN_DEPTH); // 计算队列
//   assert((insn_input_weight_size) <= STREAM_IN_DEPTH); // 加载队列
//   assert((insn_output_size) <= STREAM_IN_DEPTH); // 存储队列

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   bool pingpang = 0; // 用于判断此次加载到哪个buffer中
//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               0,     // push_pre_war  ，如果是双缓冲使用bias，bias不对load产生影响
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0++) { // 对K维度划分虚拟线程，一次计算完两个k0块，因此增加2
//           const int K = (k0) < K0-1 ? tile_K : last_K; // 根据k0+l判断当前块，得到计算大小，判断是否是最后一个块的计算，从而选择此次计算的大小

//           // 输入地址，按照块计算,根据l当前的值判断加载到双Buffer中的哪一个
//           int input_base = pingpang * (tile_I * tile_K);// 本次计算读取输入缓冲区的基地址,这是按照块计算的
//           int weight_base = pingpang * (tile_K * tile_J); // 本次计算读取权重缓冲区的基地址,这是按照块计算的 
//           pingpang = virtual_threads ? (!pingpang) : 0 ; // 反转pingpang,如果使用双缓冲

//           // 加载输入input(dim_I,dim_K)
//           // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//           bool input_pop_next_war = (i0 > 0 || j0 > 0 || (virtual_threads ? (k0 > 1) : (k0 > 0))); // 输入依赖，第一个k0块的加载不需要依赖，也就是刚开始的两个load都不依赖，后面都有依赖
//           const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 INPUT_BUFFER_ID, // 存储到输出buffer
//                 input_base,    // input buffer偏移+矩阵内部偏移
//                 input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 I,     // 每次加载MATRIX_WIDTH行
//                 K,     // 每次加载MATRIX_WIDTH列
//                 dim_K_stride,  // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 input_pop_next_war, // pop_next_war   
//                 0,     // push_pre_war 
//                 0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//           printf("- Generate load(input)\n");

//           // 加载权重weight(dim_K,dim_J)
//           const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 WEIGHT_BUFFER_ID, // 存储到输出buffer
//                 weight_base,    // weight buffer偏移+矩阵内部偏移
//                 weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 K,     // 每次加载MATRIX_WIDTH行
//                 J,     // 每次加载MATRIX_WIDTH列
//                 dim_J_stride, // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//                 0,     // push_pre_war 
//                 1);    // push_next_raw  load weight的完成影响后面gemm的执行
//           printf("- Generate load(weight)\n");

//           // GEMM指令生成
//           bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，因为切换i,j时bias会刷新缓冲区，不使用则k0=0时刷新，而后累加
//           int scale_int = static_cast<int>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
//           int scale_type1 = (k0==K0-1) ? scale_type: 0; // 只有K循环执行到最后才进行scale，保证只对输出块进行scale
//           // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//           int uop_bgn = 0; 
//           if( J==tile_J && K==tile_K ) // 分块计算的形状
//             uop_bgn = 0;
//           else if( J==tile_J && K==last_K )
//             uop_bgn = tile_I;
//           else if( J==last_J && K==tile_K )
//             uop_bgn = 2*tile_I;
//           else if( J==last_J && K==last_K )
//             uop_bgn = 3*tile_I;
//           // 依赖
//           // gemm会一直对load产生影响，如果是最后一个GEMM就不用影响了,因为不会再回去加载input了，如果是双缓冲，那么最后两个GEMM都不影响load
//           bool gemm_push_pre_war =(j0!=J0-1) || (i0 != I0-1) || ((virtual_threads ? (k0<K0-2) : (k0!=K0-1))? 1 : 0); // 最后两个gemm都不对load产生影响
//           bool gemm_push_next_raw = act ? 0 : ((k0==K0-1) ? 1 : 0); // 在k循环执行到最后一个，对store产生影响，如果使用act，该影响被act取代
//           // 当不存在bias时，K循环第一个GEMM会受到store的影响，但是第一个K循环不受影响，因为前面还没有store,如果存在bias，那么store的影响就给了bias了，那么就不用gemm受影响了
//           bool gemm_pop_next_war = bias_use ? 0 : ((i0 > 0 || j0 > 0) && (k0==0) ); 
//           insn_buf[insn_idx++] = getGEMMInsn(
//                                     uop_bgn,
//                                     uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                     J, // J
//                                     K, // K
//                                     input_base, // 本次计算读取输入缓冲区的基地址,按块计算
//                                     weight_base, // 本次计算读取权重缓冲区的基地址 
//                                     accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                     relu_use,
//                                     scale_type1, // scale的类型，只有在k循环快结束时有效
//                                     scale_int, // scale的大小(整数)
//                                     1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                     gemm_pop_next_war,    // pop_next_war 
//                                     gemm_push_pre_war,    // push_pre_war 
//                                     gemm_push_next_raw);   // push_next_raw  
//           printf("- Generate compute(gemm)\n");
//       }
//       // 在store i,j前插入ANU操作
//       if(act == ANU_LAYERNORM) // 生成LayerNorm指令
//       {
//         insn_buf[insn_idx++] = getLayerNormInsn(dim_J_stride, // 写入当前填充矩阵的分块数
//                                                 I,  // 传入DIM_I用于计算
//                                                 dim_J,// 实际未填充的dim_J列数
//                                                 0,     // pop_pre_raw  
//                                                 0,  // pop_next_war
//                                                 0,     // push_pre_war  
//                                                 1);    // push_next_raw ，对store产生影响
//       }
//       else if(act == ANU_SOFTMAX) // 生成softmax指令
//       {
//         insn_buf[insn_idx++] = getSoftmaxInsn(dim_J_stride, // 写入当前填充矩阵的分块数
//                                               I,
//                                               dim_J, //实际未填充的dim_J列数
//                                               0,     // pop_pre_raw  
//                                               0,  // pop_next_war
//                                               0,     // push_pre_war  
//                                               1);    // push_next_raw 
//       }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             1,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }

//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input_buffer, 
//       (volatile Transfer_DataType *)weight_buffer, 
//       (volatile Transfer_DataType *)biases_buffer, 
//       (volatile Transfer_DataType *)output_buffer,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   // 解包输出提取有效部分并返回输出给output指针
//   Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
//   Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
//   memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针
  
//   printf("==========================================GEMM End===========================================\n");
//   return 0;
// }


// // 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// // 并行方式:存在Bias那么load和bias并行,或者和store并行，在K维度上使用双缓冲，使用计算时间隐藏加载时间
// // 可以融合relu、反量化、layernorm、softmax
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       bool relu_use, // 是否应用relu
//                       int  scale_type, // scale的类型
//                       float scale, // scale的大小(浮点小数)
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
//   Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
//   Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
//   Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

//   // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
//   Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 声明输出缓冲区(输出的是没有去填充的输出,一维)
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

// // 假设只对K维度进行双缓冲，因此I，J不变，K维度存储大小减少一半
// #define db_max_tile_k (((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / 2) / max_tile_i_j) // 在不进行双缓冲的基础上，假设K维度减少一半的存储大小，计算得到最大K维度分块的大小

//   bool virtual_threads = 1; // 如果不存在act，就使用k维度虚拟线程/双缓冲
//   // virtual_threads = act ? 0 : 1;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲
//   // virtual_threads = 0;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲

//   // 计算每个缓冲区的最大容量，在使用双缓冲的前提下
//   const size_t max_input_buffer = virtual_threads ? INPUT_BUFFER_WIDTH / 2 :INPUT_BUFFER_WIDTH;
//   const size_t max_weight_buffer = virtual_threads ? WEIGHT_BUFFER_WIDTH / 2 :WEIGHT_BUFFER_WIDTH;
//   const size_t max_output_buffer = virtual_threads ? OUTPUT_BUFFER_WIDTH : OUTPUT_BUFFER_WIDTH;

//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算，如果有act，则首先选择act条件，如果没有就可以使用双缓冲条件
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX，首先满足act
//   {
//       tile_I = 1;
//       tile_J = dim_J_stride; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else if (virtual_threads) // 如果没有act，才考虑使用虚拟线程/双缓冲初始条件，如果是针对K维度进行的双缓冲，那么K维度的最大分块减少一倍
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < db_max_tile_k ? dim_K_stride : db_max_tile_k; 
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 先增大K维度的分块系数，提高双缓冲效率
//     if(tile_I * (tile_K+1) <= max_input_buffer &&
//        (tile_K+1) * tile_J <= max_weight_buffer &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= max_output_buffer &&
//        tile_K * (tile_J+1) <= max_weight_buffer &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }

//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= max_input_buffer &&
//        (tile_I+1) * tile_J <= max_output_buffer &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵
//   assert((tile_I * tile_J)<=OUTPUT_BUFFER_WIDTH);
//   assert((tile_I * tile_K)<=INPUT_BUFFER_WIDTH);
//   assert((tile_K * tile_J)<=WEIGHT_BUFFER_WIDTH);

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   // printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
//   int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_act_size = act ? I0*J0 : 0; //计算一个输出块在store前进行anu操作,如果使用则存在该指令
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size + insn_act_size;

//   // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
//   assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size+insn_act_size) <= STREAM_IN_DEPTH); // 计算队列
//   assert((insn_input_weight_size) <= STREAM_IN_DEPTH); // 加载队列
//   assert((insn_output_size) <= STREAM_IN_DEPTH); // 存储队列

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   bool pingpang = 0; // 用于判断此次加载到哪个buffer中
//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               0,     // push_pre_war  ，如果是双缓冲使用bias，bias不对load产生影响
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0++) { // 对K维度划分虚拟线程，一次计算完两个k0块，因此增加2
//           const int K = (k0) < K0-1 ? tile_K : last_K; // 根据k0+l判断当前块，得到计算大小，判断是否是最后一个块的计算，从而选择此次计算的大小

//           // 输入地址，按照块计算,根据l当前的值判断加载到双Buffer中的哪一个
//           int input_base = pingpang * (tile_I * tile_K);// 本次计算读取输入缓冲区的基地址,这是按照块计算的
//           int weight_base = pingpang * (tile_K * tile_J); // 本次计算读取权重缓冲区的基地址,这是按照块计算的 
//           pingpang = virtual_threads ? (!pingpang) : 0 ; // 反转pingpang,如果使用双缓冲

//           // 加载输入input(dim_I,dim_K)
//           // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//           bool input_pop_next_war = (i0 > 0 || j0 > 0 || (virtual_threads ? (k0 > 1) : (k0 > 0))); // 输入依赖，第一个k0块的加载不需要依赖，也就是刚开始的两个load都不依赖，后面都有依赖
//           const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 INPUT_BUFFER_ID, // 存储到输出buffer
//                 input_base,    // input buffer偏移+矩阵内部偏移
//                 input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 I,     // 每次加载MATRIX_WIDTH行
//                 K,     // 每次加载MATRIX_WIDTH列
//                 dim_K_stride,  // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 input_pop_next_war, // pop_next_war   
//                 0,     // push_pre_war 
//                 0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//           printf("- Generate load(input)\n");

//           // 加载权重weight(dim_K,dim_J)
//           const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 WEIGHT_BUFFER_ID, // 存储到输出buffer
//                 weight_base,    // weight buffer偏移+矩阵内部偏移
//                 weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 K,     // 每次加载MATRIX_WIDTH行
//                 J,     // 每次加载MATRIX_WIDTH列
//                 dim_J_stride, // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//                 0,     // push_pre_war 
//                 1);    // push_next_raw  load weight的完成影响后面gemm的执行
//           printf("- Generate load(weight)\n");

//           // GEMM指令生成
//           bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，因为切换i,j时bias会刷新缓冲区，不使用则k0=0时刷新，而后累加
//           int scale_int = static_cast<int>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
//           int scale_type1 = (k0==K0-1) ? scale_type: 0; // 只有K循环执行到最后才进行scale，保证只对输出块进行scale
//           // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//           int uop_bgn = 0; 
//           if( J==tile_J && K==tile_K ) // 分块计算的形状
//             uop_bgn = 0;
//           else if( J==tile_J && K==last_K )
//             uop_bgn = tile_I;
//           else if( J==last_J && K==tile_K )
//             uop_bgn = 2*tile_I;
//           else if( J==last_J && K==last_K )
//             uop_bgn = 3*tile_I;
//           // 依赖
//           // gemm会一直对load产生影响，如果是最后一个GEMM就不用影响了,因为不会再回去加载input了，如果是双缓冲，那么最后两个GEMM都不影响load
//           bool gemm_push_pre_war =(j0!=J0-1) || (i0 != I0-1) || ((virtual_threads ? (k0<K0-2) : (k0!=K0-1))? 1 : 0); // 最后两个gemm都不对load产生影响
//           bool gemm_push_next_raw = act ? 0 : ((k0==K0-1) ? 1 : 0); // 在k循环执行到最后一个，对store产生影响，如果使用act，该影响被act取代
//           // 当不存在bias时，K循环第一个GEMM会受到store的影响，但是第一个K循环不受影响，因为前面还没有store,如果存在bias，那么store的影响就给了bias了，那么就不用gemm受影响了
//           bool gemm_pop_next_war = bias_use ? 0 : ((i0 > 0 || j0 > 0) && (k0==0) ); 
//           insn_buf[insn_idx++] = getGEMMInsn(
//                                     uop_bgn,
//                                     uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                     J, // J
//                                     K, // K
//                                     input_base, // 本次计算读取输入缓冲区的基地址,按块计算
//                                     weight_base, // 本次计算读取权重缓冲区的基地址 
//                                     accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                     relu_use,
//                                     scale_type1, // scale的类型，只有在k循环快结束时有效
//                                     scale_int, // scale的大小(整数)
//                                     1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                     gemm_pop_next_war,    // pop_next_war 
//                                     gemm_push_pre_war,    // push_pre_war 
//                                     gemm_push_next_raw);   // push_next_raw  
//           printf("- Generate compute(gemm)\n");
//       }
//       // 在store i,j前插入ANU操作
//       if(act == ANU_LAYERNORM) // 生成LayerNorm指令
//       {
//         insn_buf[insn_idx++] = getLayerNormInsn(dim_J_stride, // 写入当前填充矩阵的分块数
//                                                 I,  // 传入DIM_I用于计算
//                                                 dim_J,// 实际未填充的dim_J列数
//                                                 0,     // pop_pre_raw  
//                                                 0,  // pop_next_war
//                                                 0,     // push_pre_war  
//                                                 1);    // push_next_raw ，对store产生影响
//       }
//       else if(act == ANU_SOFTMAX) // 生成softmax指令
//       {
//         insn_buf[insn_idx++] = getSoftmaxInsn(dim_J_stride, // 写入当前填充矩阵的分块数
//                                               I,
//                                               dim_J, //实际未填充的dim_J列数
//                                               0,     // pop_pre_raw  
//                                               0,  // pop_next_war
//                                               0,     // push_pre_war  
//                                               1);    // push_next_raw 
//       }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             1,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }

//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input_buffer, 
//       (volatile Transfer_DataType *)weight_buffer, 
//       (volatile Transfer_DataType *)biases_buffer, 
//       (volatile Transfer_DataType *)output_buffer,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   // 解包输出提取有效部分并返回输出给output指针
//   Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
//   Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
//   memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针

//   return 0;
// }


// // 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// // 并行方式:存在Bias那么load和bias并行,或者和store并行，在K维度上使用双缓冲，使用计算时间隐藏加载时间
// // 可以融合relu、反量化、layernorm、softmax
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       bool relu_use, // 是否应用relu
//                       int  scale_type, // scale的类型
//                       float scale, // scale的大小(浮点小数)
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
//   Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
//   Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
//   Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

//   // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
//   Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 声明输出缓冲区(输出的是没有去填充的输出,一维)
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

// // 假设只对K维度进行双缓冲，因此I，J不变，K维度存储大小减少一半
// #define db_max_tile_k (((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / 2) / max_tile_i_j) // 在不进行双缓冲的基础上，假设K维度减少一半的存储大小，计算得到最大K维度分块的大小

//   bool virtual_threads = 0; // 如果不存在act，就使用k维度虚拟线程/双缓冲
//   virtual_threads = act ? 0 : 1;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲
//   // virtual_threads = 0;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲

//   // 计算每个缓冲区的最大容量
//   const size_t max_input_buffer = virtual_threads ? INPUT_BUFFER_WIDTH / 2 :INPUT_BUFFER_WIDTH;
//   const size_t max_weight_buffer = virtual_threads ? WEIGHT_BUFFER_WIDTH / 2 :WEIGHT_BUFFER_WIDTH;
//   const size_t max_output_buffer = virtual_threads ? OUTPUT_BUFFER_WIDTH : OUTPUT_BUFFER_WIDTH;

//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX
//   {
//       tile_I = 1;
//       tile_J = dim_J_stride; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else if (virtual_threads) // 如果使用虚拟线程/双缓冲，同时不存在act，如果是针对K维度进行的双缓冲，那么K维度的最大分块减少一倍
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < db_max_tile_k ? dim_K_stride : db_max_tile_k; 
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= max_output_buffer &&
//        tile_K * (tile_J+1) <= max_weight_buffer &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }
//     // 增大K维度的分块系数
//     if(tile_I * (tile_K+1) <= max_input_buffer &&
//        (tile_K+1) * tile_J <= max_weight_buffer &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }
//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= max_input_buffer &&
//        (tile_I+1) * tile_J <= max_output_buffer &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵
//   assert((tile_I * tile_J)<=OUTPUT_BUFFER_WIDTH);
//   assert((tile_I * tile_K)<=INPUT_BUFFER_WIDTH);
//   assert((tile_K * tile_J)<=WEIGHT_BUFFER_WIDTH);

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   // printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
//   int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_act_size = act ? I0*J0 : 0; //计算一个输出块在store前进行anu操作,如果使用则存在该指令
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size + insn_act_size;

//   // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
//   assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size+insn_act_size) <= STREAM_IN_DEPTH); // 计算队列
//   assert((insn_input_weight_size) <= STREAM_IN_DEPTH); // 加载队列
//   assert((insn_output_size) <= STREAM_IN_DEPTH); // 存储队列

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   bool pingpang = 0; // 用于判断此次加载到哪个buffer中
//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               0,     // push_pre_war  ，如果是双缓冲使用bias，bias不对load产生影响
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0++) { // 对K维度划分虚拟线程，一次计算完两个k0块，因此增加2
//           const int K = (k0) < K0-1 ? tile_K : last_K; // 根据k0+l判断当前块，得到计算大小，判断是否是最后一个块的计算，从而选择此次计算的大小

//           // 输入地址，按照块计算,根据l当前的值判断加载到双Buffer中的哪一个
//           int input_base = pingpang * (tile_I * tile_K);// 本次计算读取输入缓冲区的基地址,这是按照块计算的
//           int weight_base = pingpang * (tile_K * tile_J); // 本次计算读取权重缓冲区的基地址,这是按照块计算的 
//           pingpang = virtual_threads ? (!pingpang) : 0 ; // 反转pingpang,如果使用双缓冲

//           // 加载输入input(dim_I,dim_K)
//           // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//           bool input_pop_next_war = (i0 > 0 || j0 > 0 || (virtual_threads ? (k0 > 1) : (k0 > 0))); // 输入依赖，第一个k0块的加载不需要依赖，也就是刚开始的两个load都不依赖，后面都有依赖
//           const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 INPUT_BUFFER_ID, // 存储到输出buffer
//                 input_base,    // input buffer偏移+矩阵内部偏移
//                 input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 I,     // 每次加载MATRIX_WIDTH行
//                 K,     // 每次加载MATRIX_WIDTH列
//                 dim_K_stride,  // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 input_pop_next_war, // pop_next_war   
//                 0,     // push_pre_war 
//                 0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//           printf("- Generate load(input)\n");

//           // 加载权重weight(dim_K,dim_J)
//           const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 WEIGHT_BUFFER_ID, // 存储到输出buffer
//                 weight_base,    // weight buffer偏移+矩阵内部偏移
//                 weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 K,     // 每次加载MATRIX_WIDTH行
//                 J,     // 每次加载MATRIX_WIDTH列
//                 dim_J_stride, // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//                 0,     // push_pre_war 
//                 1);    // push_next_raw  load weight的完成影响后面gemm的执行
//           printf("- Generate load(weight)\n");

//           // GEMM指令生成
//           bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，因为切换i,j时bias会刷新缓冲区，不使用则k0=0时刷新，而后累加
//           int scale_int = static_cast<int>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
//           int scale_type1 = (k0==K0-1) ? scale_type: 0; // 只有K循环执行到最后才进行scale，保证只对输出块进行scale
//           // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//           int uop_bgn = 0; 
//           if( J==tile_J && K==tile_K ) // 分块计算的形状
//             uop_bgn = 0;
//           else if( J==tile_J && K==last_K )
//             uop_bgn = tile_I;
//           else if( J==last_J && K==tile_K )
//             uop_bgn = 2*tile_I;
//           else if( J==last_J && K==last_K )
//             uop_bgn = 3*tile_I;
//           // 依赖
//           // gemm会一直对load产生影响，如果是最后一个GEMM就不用影响了,因为不会再回去加载input了，如果是双缓冲，那么最后两个GEMM都不影响load
//           bool gemm_push_pre_war =(j0!=J0-1) || (i0 != I0-1) || ((virtual_threads ? (k0<K0-2) : (k0!=K0-1))? 1 : 0); // 最后两个gemm都不对load产生影响
//           bool gemm_push_next_raw = act ? 0 : ((k0==K0-1) ? 1 : 0); // 在k循环执行到最后一个，对store产生影响，如果使用act，该影响被act取代
//           // 当不存在bias时，K循环第一个GEMM会受到store的影响，但是第一个K循环不受影响，因为前面还没有store,如果存在bias，那么store的影响就给了bias了，那么就不用gemm受影响了
//           bool gemm_pop_next_war = bias_use ? 0 : ((i0 > 0 || j0 > 0) && (k0==0) ); 
//           insn_buf[insn_idx++] = getGEMMInsn(
//                                     uop_bgn,
//                                     uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                     J, // J
//                                     K, // K
//                                     input_base, // 本次计算读取输入缓冲区的基地址,按块计算
//                                     weight_base, // 本次计算读取权重缓冲区的基地址 
//                                     accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                     relu_use,
//                                     scale_type1, // scale的类型，只有在k循环快结束时有效
//                                     scale_int, // scale的大小(整数)
//                                     1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                     gemm_pop_next_war,    // pop_next_war 
//                                     gemm_push_pre_war,    // push_pre_war 
//                                     gemm_push_next_raw);   // push_next_raw  
//           printf("- Generate compute(gemm)\n");
//       }
//       // 在store i,j前插入ANU操作
//       if(act == ANU_LAYERNORM) // 生成LayerNorm指令
//       {
//         insn_buf[insn_idx++] = getLayerNormInsn(dim_J_stride, // 写入当前填充矩阵的分块数
//                                                 I,  // 传入DIM_I用于计算
//                                                 dim_J,// 实际未填充的dim_J列数
//                                                 0,     // pop_pre_raw  
//                                                 0,  // pop_next_war
//                                                 0,     // push_pre_war  
//                                                 1);    // push_next_raw ，对store产生影响
//       }
//       else if(act == ANU_SOFTMAX) // 生成softmax指令
//       {
//         insn_buf[insn_idx++] = getSoftmaxInsn(dim_J_stride, // 写入当前填充矩阵的分块数
//                                               I,
//                                               dim_J, //实际未填充的dim_J列数
//                                               0,     // pop_pre_raw  
//                                               0,  // pop_next_war
//                                               0,     // push_pre_war  
//                                               1);    // push_next_raw 
//       }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             1,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }

//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input_buffer, 
//       (volatile Transfer_DataType *)weight_buffer, 
//       (volatile Transfer_DataType *)biases_buffer, 
//       (volatile Transfer_DataType *)output_buffer,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   // 解包输出提取有效部分并返回输出给output指针
//   Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
//   Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
//   memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针

//   return 0;
// }

// // 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// // 并行方式:存在Bias那么load和bias并行,或者和store并行，在K维度上使用双缓冲，使用计算时间隐藏加载时间
// // 可以融合relu、反量化、layernorm、softmax
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       bool relu_use, // 是否应用relu
//                       int  scale_type, // scale的类型
//                       float scale, // scale的大小(浮点小数)
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
//   Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
//   Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
//   Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

//   // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
//   Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 声明输出缓冲区(输出的是没有去填充的输出,一维)
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

// // 假设只对K维度进行双缓冲，因此I，J不变，K维度存储大小减少一半
// #define db_max_tile_k (((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / 2) / max_tile_i_j) // 在不进行双缓冲的基础上，假设K维度减少一半的存储大小，计算得到最大K维度分块的大小

//   bool virtual_threads = 0; // 如果不存在act，就使用k维度虚拟线程/双缓冲
//   virtual_threads = act ? 0 : 1;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲
//   // virtual_threads = 0;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲

//   // 计算每个缓冲区的最大容量
//   const size_t max_input_buffer = virtual_threads ? INPUT_BUFFER_WIDTH / 2 :INPUT_BUFFER_WIDTH;
//   const size_t max_weight_buffer = virtual_threads ? WEIGHT_BUFFER_WIDTH / 2 :WEIGHT_BUFFER_WIDTH;
//   const size_t max_output_buffer = virtual_threads ? OUTPUT_BUFFER_WIDTH : OUTPUT_BUFFER_WIDTH;


//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX
//   {
//       tile_I = 1;
//       tile_J = dim_J_stride; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else if (virtual_threads) // 如果使用虚拟线程/双缓冲，同时不存在act，如果是针对K维度进行的双缓冲，那么K维度的最大分块减少一倍
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < db_max_tile_k ? dim_K_stride : db_max_tile_k; 
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= max_output_buffer &&
//        tile_K * (tile_J+1) <= max_weight_buffer &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }
//     // 增大K维度的分块系数
//     if(tile_I * (tile_K+1) <= max_input_buffer &&
//        (tile_K+1) * tile_J <= max_weight_buffer &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }
//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= max_input_buffer &&
//        (tile_I+1) * tile_J <= max_output_buffer &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   // printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
//   int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size;

//   // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
//   assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size) < STREAM_IN_DEPTH); // 计算队列
//   assert((insn_input_weight_size) < STREAM_IN_DEPTH); // 加载队列
//   assert((insn_output_size) < STREAM_IN_DEPTH); // 存储队列

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   bool pingpang = 0; // 用于判断此次加载到哪个buffer中
//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               0,     // push_pre_war  ，如果是双缓冲使用bias，bias不对load产生影响
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0++) { // 对K维度划分虚拟线程，一次计算完两个k0块，因此增加2
//           const int K = (k0) < K0-1 ? tile_K : last_K; // 根据k0+l判断当前块，得到计算大小，判断是否是最后一个块的计算，从而选择此次计算的大小

//           // 输入地址，按照块计算,根据l当前的值判断加载到双Buffer中的哪一个
//           int input_base = pingpang * (tile_I * tile_K);// 本次计算读取输入缓冲区的基地址,这是按照块计算的
//           int weight_base = pingpang * (tile_K * tile_J); // 本次计算读取权重缓冲区的基地址,这是按照块计算的 
//           pingpang = virtual_threads ? (!pingpang) : 0 ; // 反转pingpang,如果使用双缓冲

//           // 加载输入input(dim_I,dim_K)
//           // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//           bool input_pop_next_war = (i0 > 0 || j0 > 0 || (virtual_threads ? (k0 > 1) : (k0 > 0))); // 输入依赖，第一个k0块的加载不需要依赖，也就是刚开始的两个load都不依赖，后面都有依赖
//           const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增，需要增加l0
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 INPUT_BUFFER_ID, // 存储到输出buffer
//                 input_base,    // input buffer偏移+矩阵内部偏移
//                 input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 I,     // 每次加载MATRIX_WIDTH行
//                 K,     // 每次加载MATRIX_WIDTH列
//                 dim_K_stride,  // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 input_pop_next_war, // pop_next_war   
//                 0,     // push_pre_war 
//                 0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//           printf("- Generate load(input)\n");

//           // 加载权重weight(dim_K,dim_J)
//           const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增，需要增加l0
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 WEIGHT_BUFFER_ID, // 存储到输出buffer
//                 weight_base,    // weight buffer偏移+矩阵内部偏移
//                 weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 K,     // 每次加载MATRIX_WIDTH行
//                 J,     // 每次加载MATRIX_WIDTH列
//                 dim_J_stride, // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//                 0,     // push_pre_war 
//                 1);    // push_next_raw  load weight的完成影响后面gemm的执行
//           printf("- Generate load(weight)\n");

//           // GEMM指令生成
//           bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，因为切换i,j时bias会刷新缓冲区，不使用则k0=0时刷新，而后累加，需要增加l0
//           int scale_int = static_cast<int>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
//           int scale_type1 = (k0==K0-1) ? scale_type: 0; // 只有K循环执行到最后才进行scale，保证只对输出块进行scale
//           // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//           int uop_bgn = 0; 
//           if( J==tile_J && K==tile_K ) // 分块计算的形状
//             uop_bgn = 0;
//           else if( J==tile_J && K==last_K )
//             uop_bgn = tile_I;
//           else if( J==last_J && K==tile_K )
//             uop_bgn = 2*tile_I;
//           else if( J==last_J && K==last_K )
//             uop_bgn = 3*tile_I;
//           // 依赖
//           // gemm会一直对load产生影响，如果是最后一个GEMM就不用影响了,因为不会再回去加载input了，如果是双缓冲，那么最后两个GEMM都不影响load
//           bool gemm_push_pre_war =(j0!=J0-1) || (i0 != I0-1) || ((virtual_threads ? (k0<K0-2) : (k0!=K0-1))? 1 : 0); // 最后两个gemm都不对load产生影响
//           bool gemm_push_next_raw = (k0==K0-1) ? 1 : 0; // 在k循环执行到最后一个，对store产生影响
//           // 当不存在bias时，K循环第一个GEMM会受到store的影响，但是第一个K循环不受影响，因为前面还没有store,如果存在bias，那么store的影响就给了bias了，那么就不用gemm受影响了
//           bool gemm_pop_next_war = bias_use ? 0 : ((i0 > 0 || j0 > 0) && (k0==0) ); 
//           insn_buf[insn_idx++] = getGEMMInsn(
//                                     uop_bgn,
//                                     uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                     J, // J
//                                     K, // K
//                                     input_base, // 本次计算读取输入缓冲区的基地址,按块计算
//                                     weight_base, // 本次计算读取权重缓冲区的基地址 
//                                     accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                     relu_use,
//                                     scale_type1, // scale的类型，只有在k循环快结束时有效
//                                     scale_int, // scale的大小(整数)
//                                     1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                     gemm_pop_next_war,    // pop_next_war 
//                                     gemm_push_pre_war,    // push_pre_war 
//                                     gemm_push_next_raw);   // push_next_raw  
//           printf("- Generate compute(gemm)\n");
//         }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             1,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }

//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input_buffer, 
//       (volatile Transfer_DataType *)weight_buffer, 
//       (volatile Transfer_DataType *)biases_buffer, 
//       (volatile Transfer_DataType *)output_buffer,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   // 解包输出提取有效部分并返回输出给output指针
//   Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
//   Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
//   memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针

//   return 0;
// }


// // 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// // 并行方式:存在Bias那么load和bias并行,或者和store并行，在K维度上使用双缓冲，使用计算时间隐藏加载时间
// // 可以融合relu、反量化、layernorm、softmax
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       bool relu_use, // 是否应用relu
//                       int  scale_type, // scale的类型
//                       float scale, // scale的大小(浮点小数)
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
//   Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
//   Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
//   Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

//   // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
//   Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 声明输出缓冲区(输出的是没有去填充的输出,一维)
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

// // 假设只对K维度进行双缓冲，因此I，J不变，K维度存储大小减少一半
// #define db_max_tile_k (((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / 2) / max_tile_i_j) // 在不进行双缓冲的基础上，假设K维度减少一半的存储大小，计算得到最大K维度分块的大小

//   const int virtual_threads = 2; // 使用虚拟线程/双缓冲

//   // 计算每个缓冲区的最大容量
//   const size_t max_input_buffer = virtual_threads ? INPUT_BUFFER_WIDTH / 2 :INPUT_BUFFER_WIDTH;
//   const size_t max_weight_buffer = virtual_threads ? WEIGHT_BUFFER_WIDTH / 2 :WEIGHT_BUFFER_WIDTH;
//   const size_t max_output_buffer = virtual_threads ? OUTPUT_BUFFER_WIDTH : OUTPUT_BUFFER_WIDTH;


//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX
//   {
//       tile_I = 1;
//       tile_J = dim_J_stride; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else if (virtual_threads) // 如果使用虚拟线程/双缓冲，如果是针对K维度进行的双缓冲，那么K维度的最大分块减少一倍
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < db_max_tile_k ? dim_K_stride : db_max_tile_k; 
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= max_output_buffer &&
//        tile_K * (tile_J+1) <= max_weight_buffer &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }
//     // 增大K维度的分块系数
//     if(tile_I * (tile_K+1) <= max_input_buffer &&
//        (tile_K+1) * tile_J <= max_weight_buffer &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }
//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= max_input_buffer &&
//        (tile_I+1) * tile_J <= max_output_buffer &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   // printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
//   int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size;

//   // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
//   assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size) < STREAM_IN_DEPTH); // 计算队列
//   assert((insn_input_weight_size) < STREAM_IN_DEPTH); // 加载队列
//   assert((insn_output_size) < STREAM_IN_DEPTH); // 存储队列

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   bool pingpang = 0; // 用于判断此次加载到哪个buffer中
//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               1,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0++) { // 对K维度划分虚拟线程，一次计算完两个k0块，因此增加2
//           const int K = (k0) < K0-1 ? tile_K : last_K; // 根据k0+l判断当前块，得到计算大小，判断是否是最后一个块的计算，从而选择此次计算的大小

//           // 输入地址，按照块计算,根据l当前的值判断加载到双Buffer中的哪一个
//           int input_base = pingpang * (tile_I * tile_K);// 本次计算读取输入缓冲区的基地址,这是按照块计算的
//           int weight_base = pingpang * (tile_K * tile_J); // 本次计算读取权重缓冲区的基地址,这是按照块计算的 
//           pingpang = !pingpang; // 反转pingpang

//           // 加载输入input(dim_I,dim_K)
//           // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//           bool input_pop_next_war = (bias_use ? 1 : (i0 > 0 || j0 > 0 || k0 > 1)); // 输入依赖，第一个k0块的加载不需要依赖，也就是刚开始的两个load都不依赖，后面都有依赖
//           const int input_dram_offset = i0 * dim_K_stride * tile_I + (k0) * tile_K; // 在input的K方向递增，需要增加l0
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 INPUT_BUFFER_ID, // 存储到输出buffer
//                 input_base,    // input buffer偏移+矩阵内部偏移
//                 input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 I,     // 每次加载MATRIX_WIDTH行
//                 K,     // 每次加载MATRIX_WIDTH列
//                 dim_K_stride,  // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 input_pop_next_war, // pop_next_war   
//                 0,     // push_pre_war 
//                 0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//           printf("- Generate load(input)\n");

//           // 加载权重weight(dim_K,dim_J)
//           const int weight_dram_offset = (k0) * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增，需要增加l0
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 WEIGHT_BUFFER_ID, // 存储到输出buffer
//                 weight_base,    // weight buffer偏移+矩阵内部偏移
//                 weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 K,     // 每次加载MATRIX_WIDTH行
//                 J,     // 每次加载MATRIX_WIDTH列
//                 dim_J_stride, // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//                 0,     // push_pre_war 
//                 1);    // push_next_raw  load weight的完成影响后面gemm的执行
//           printf("- Generate load(weight)\n");

//           // GEMM指令生成
//           bool accumulate = bias_use ? 1 : ((k0) == 0 ? 0 : 1); // 如果使用bias则一直累加，不使用则k0=0时刷新，而后累加，需要增加l0
//           int scale_int = static_cast<int>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
//           int scale_type1 = ((k0)==K0-1) ? scale_type: 0; // 只有K循环执行到最后才进行scale，保证只对输出块进行scale
//           // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//           int uop_bgn = 0; 
//           if( J==tile_J && K==tile_K ) // 分块计算的形状
//             uop_bgn = 0;
//           else if( J==tile_J && K==last_K )
//             uop_bgn = tile_I;
//           else if( J==last_J && K==tile_K )
//             uop_bgn = 2*tile_I;
//           else if( J==last_J && K==last_K )
//             uop_bgn = 3*tile_I;
//           // 依赖
//           // 影响load input和weight 在k循环不执行到最后一个时，对load产生影响，如果不使用bias，那么整个k循环都对load产生影响，代替bias的作用
//           bool gemm_push_pre_war = bias_use ? (((k0)!=K0-1) ? 1 : 0) : 1; // 第一个就存在影响，影响第二次执行的
//           // 如果是最后一个GEMM就不用影响了,因为不会再回去加载input了，因此与上最后一个GEMM在最后一个k时信号为0,当是最后一次计算两个线程，都不用对下面的线程产生影响
//           gemm_push_pre_war = gemm_push_pre_war && (((j0!=J0-1) || (i0 != I0-1)) || (((k0)<K0-2) ? 1 : 0)); // 最后两个gemm都不对load产生影响
//           bool gemm_push_next_raw = ((k0)==K0-1) ? 1 : 0; // 在k循环执行到最后一个，对store产生影响
//           bool gemm_pop_next_war = bias_use ? 0 : ((i0 > 0 || j0 > 0) && ((k0)==0) ); // 当不存在bias时，会受到store的影响，只有在K循环切换时会受到上一个store的影响，第一个K循环不受影响,k循环中的第一个受影响
//           insn_buf[insn_idx++] = getGEMMInsn(
//                                     uop_bgn,
//                                     uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                     J, // J
//                                     K, // K
//                                     input_base, // 本次计算读取输入缓冲区的基地址,按块计算
//                                     weight_base, // 本次计算读取权重缓冲区的基地址 
//                                     accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                     relu_use,
//                                     scale_type1, // scale的类型，只有在k循环快结束时有效
//                                     scale_int, // scale的大小(整数)
//                                     1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                     gemm_pop_next_war,    // pop_next_war 
//                                     gemm_push_pre_war,    // push_pre_war 
//                                     gemm_push_next_raw);   // push_next_raw  
//           printf("- Generate compute(gemm)\n");
//         }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             1,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }

//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input_buffer, 
//       (volatile Transfer_DataType *)weight_buffer, 
//       (volatile Transfer_DataType *)biases_buffer, 
//       (volatile Transfer_DataType *)output_buffer,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   // 解包输出提取有效部分并返回输出给output指针
//   Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
//   Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
//   memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针

//   return 0;
// }





// // 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// // 并行方式:存在Bias那么load和bias并行,或者和store并行，在K维度上使用虚拟线程，使用计算时间隐藏加载时间
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       bool relu_use, // 是否应用relu
//                       int  scale_type, // scale的类型
//                       float scale, // scale的大小(浮点小数)
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
//   Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
//   Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
//   Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

//   // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
//   Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 声明输出缓冲区(输出的是没有去填充的输出,一维)
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

// // 假设只对K维度进行双缓冲，因此I，J不变，K维度存储大小减少一半
// #define db_max_tile_k (((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / 2) / max_tile_i_j) // 在不进行双缓冲的基础上，假设K维度减少一半的存储大小，计算得到最大K维度分块的大小

//   const int virtual_threads = 2; // 使用虚拟线程

//   // 计算每个缓冲区的最大容量
//   const size_t max_input_buffer = virtual_threads ? INPUT_BUFFER_WIDTH / 2 :INPUT_BUFFER_WIDTH;
//   const size_t max_weight_buffer = virtual_threads ? WEIGHT_BUFFER_WIDTH / 2 :WEIGHT_BUFFER_WIDTH;
//   const size_t max_output_buffer = virtual_threads ? OUTPUT_BUFFER_WIDTH : OUTPUT_BUFFER_WIDTH;


//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX
//   {
//       tile_I = 1;
//       tile_J = dim_J_stride; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else if (virtual_threads) // 如果使用虚拟线程/双缓冲，如果是针对K维度进行的双缓冲，那么K维度的最大分块减少一倍
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < db_max_tile_k ? dim_K_stride : db_max_tile_k; 
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= max_output_buffer &&
//        tile_K * (tile_J+1) <= max_weight_buffer &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }
//     // 增大K维度的分块系数
//     if(tile_I * (tile_K+1) <= max_input_buffer &&
//        (tile_K+1) * tile_J <= max_weight_buffer &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }
//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= max_input_buffer &&
//        (tile_I+1) * tile_J <= max_output_buffer &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   // printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
//   int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0*6; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size;

//   // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
//   assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size) < STREAM_IN_DEPTH); // 计算队列
//   assert((insn_input_weight_size) < STREAM_IN_DEPTH); // 加载队列
//   assert((insn_output_size) < STREAM_IN_DEPTH); // 存储队列

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               1,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0+=2) { // 对K维度划分虚拟线程，一次计算完两个k0块，因此增加2
//         for (int l0 = 0; l0 < virtual_threads; l0++) { // 虚拟线程计算块,分两个循环，第一个线程计算块1，第二个线程计算块2
//           if (k0 + l0 >= K0) // 判断当前k0+l0是否超出了K0的范围
//               continue; // 超出就跳出本次循环
//           const int K = (k0+l0) < K0-1 ? tile_K : last_K; // 根据k0+l判断当前块，得到计算大小，判断是否是最后一个块的计算，从而选择此次计算的大小

//           // 输入地址，按照块计算,根据l当前的值判断加载到双Buffer中的哪一个
//           int input_base = l0 * (I * K);// 本次计算读取输入缓冲区的基地址,这是按照块计算的
//           int weight_base = l0 * (K * J); // 本次计算读取权重缓冲区的基地址,这是按照块计算的 

//           // 加载输入input(dim_I,dim_K)
//           // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//           bool input_pop_next_war = (bias_use ? 1 : (i0 > 0 || j0 > 0 || k0 > 0)); // 输入依赖，第一个k0块的加载不需要依赖，也就是刚开始的两个load都不依赖，后面都有依赖
//           const int input_dram_offset = i0 * dim_K_stride * tile_I + (k0+l0) * tile_K; // 在input的K方向递增，需要增加l0
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 INPUT_BUFFER_ID, // 存储到输出buffer
//                 input_base,    // input buffer偏移+矩阵内部偏移
//                 input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 I,     // 每次加载MATRIX_WIDTH行
//                 K,     // 每次加载MATRIX_WIDTH列
//                 dim_K_stride,  // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 input_pop_next_war, // pop_next_war   
//                 0,     // push_pre_war 
//                 0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//           printf("- Generate load(input)\n");

//           // 加载权重weight(dim_K,dim_J)
//           const int weight_dram_offset = (k0+l0) * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增，需要增加l0
//           insn_buf[insn_idx++] = get2DLoadStoreInsn(
//                 OPCODE_LOAD,     // 存储指令
//                 WEIGHT_BUFFER_ID, // 存储到输出buffer
//                 weight_base,    // weight buffer偏移+矩阵内部偏移
//                 weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//                 K,     // 每次加载MATRIX_WIDTH行
//                 J,     // 每次加载MATRIX_WIDTH列
//                 dim_J_stride, // output矩阵的列的分块数作为步进
//                 0,     // pop_pre_raw  
//                 0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//                 0,     // push_pre_war 
//                 1);    // push_next_raw  load weight的完成影响后面gemm的执行
//           printf("- Generate load(weight)\n");

//           // GEMM指令生成
//           bool accumulate = bias_use ? 1 : ((k0+l0) == 0 ? 0 : 1); // 如果使用bias则一直累加，不使用则k0=0时刷新，而后累加，需要增加l0
//           int scale_int = static_cast<int>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
//           int scale_type1 = ((k0+l0)==K0-1) ? scale_type: 0; // 只有K循环执行到最后才进行scale，保证只对输出块进行scale
//           // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//           int uop_bgn = 0; 
//           if( J==tile_J && K==tile_K ) // 分块计算的形状
//             uop_bgn = 0;
//           else if( J==tile_J && K==last_K )
//             uop_bgn = tile_I;
//           else if( J==last_J && K==tile_K )
//             uop_bgn = 2*tile_I;
//           else if( J==last_J && K==last_K )
//             uop_bgn = 3*tile_I;
//           // 依赖
//           // 影响load input和weight 在k循环不执行到最后一个时，对load产生影响，如果不使用bias，那么整个k循环都对load产生影响，代替bias的作用
//           bool gemm_push_pre_war = bias_use ? (((k0+l0)!=K0-1) ? 1 : 0) : 1; // 第一个就存在影响，影响第二次执行的
//           // 如果是最后一个GEMM就不用影响了,因为不会再回去加载input了，因此与上最后一个GEMM在最后一个k时信号为0,当是最后一次计算两个线程，都不用对下面的线程产生影响
//           gemm_push_pre_war = gemm_push_pre_war && (((j0!=J0-1) || (i0 != I0-1)) || (((k0+l0)!=K0-1) ? 1 : 0));
//           bool gemm_push_next_raw = ((k0+l0)==K0-1) ? 1 : 0; // 在k循环执行到最后一个，对store产生影响
//           bool gemm_pop_next_war = bias_use ? 0 : ((i0 > 0 || j0 > 0) && ((k0+l0)==0) ); // 当不存在bias时，会受到store的影响，只有在K循环切换时会受到上一个store的影响，第一个K循环不受影响,k循环中的第一个受影响
//           insn_buf[insn_idx++] = getGEMMInsn(
//                                     uop_bgn,
//                                     uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                     J, // J
//                                     K, // K
//                                     input_base, // 本次计算读取输入缓冲区的基地址,按块计算
//                                     weight_base, // 本次计算读取权重缓冲区的基地址 
//                                     accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                     relu_use,
//                                     scale_type1, // scale的类型，只有在k循环快结束时有效
//                                     scale_int, // scale的大小(整数)
//                                     1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                     gemm_pop_next_war,    // pop_next_war 
//                                     gemm_push_pre_war,    // push_pre_war 
//                                     gemm_push_next_raw);   // push_next_raw  
//           printf("- Generate compute(gemm)\n");
//         }
//       }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             1,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }

//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input_buffer, 
//       (volatile Transfer_DataType *)weight_buffer, 
//       (volatile Transfer_DataType *)biases_buffer, 
//       (volatile Transfer_DataType *)output_buffer,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   // 解包输出提取有效部分并返回输出给output指针
//   Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
//   Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
//   memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针

//   return 0;
// }



// // 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// // 并行方式:存在Bias那么load和bias并行,或者和store并行
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       bool relu_use, // 是否应用relu
//                       int  scale_type, // scale的类型
//                       float scale, // scale的大小(浮点小数)
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
//   Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
//   Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
//   Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

//   // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
//   Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 声明输出缓冲区(输出的是没有去填充的输出,一维)
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX
//   {
//       tile_I = 1;
//       tile_J = dim_J_stride; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= OUTPUT_BUFFER_WIDTH &&
//        tile_K * (tile_J+1) <= WEIGHT_BUFFER_WIDTH &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }
//     // 增大K维度的分块系数
//     if(tile_I * (tile_K+1) <= INPUT_BUFFER_WIDTH &&
//        (tile_K+1) * tile_J <= WEIGHT_BUFFER_WIDTH &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }
//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= INPUT_BUFFER_WIDTH &&
//        (tile_I+1) * tile_J <= OUTPUT_BUFFER_WIDTH &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   // printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
//   int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size;

//   // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
//   assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size) < STREAM_IN_DEPTH); // 计算队列
//   assert((insn_input_weight_size) < STREAM_IN_DEPTH); // 加载队列
//   assert((insn_output_size) < STREAM_IN_DEPTH); // 存储队列

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               1,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;

//         // 加载输入input(dim_I,dim_K)
//         // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//         bool input_pop_next_war = (bias_use ? 1 : (i0 > 0 || j0 > 0 || k0 > 0 ));
//         const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               INPUT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               I,     // 每次加载MATRIX_WIDTH行
//               K,     // 每次加载MATRIX_WIDTH列
//               dim_K_stride,  // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               input_pop_next_war, // pop_next_war   
//               0,     // push_pre_war 
//               0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//         printf("- Generate load(input)\n");

//         // 加载权重weight(dim_K,dim_J)
//         const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               WEIGHT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               K,     // 每次加载MATRIX_WIDTH行
//               J,     // 每次加载MATRIX_WIDTH列
//               dim_J_stride, // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//               0,     // push_pre_war 
//               1);    // push_next_raw  load weight的完成影响后面gemm的执行
//         printf("- Generate load(weight)\n");

//         // GEMM指令生成
//         bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，不使用则k0=0时刷新，而后累加
//         int scale_int = static_cast<int>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
//         int scale_type1 = (k0==K0-1) ? scale_type: 0; // 只有K循环执行到最后才进行scale，保证只对输出块进行scale
//         // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//         int uop_bgn = 0; 
//         if( J==tile_J && K==tile_K ) // 分块计算的形状
//           uop_bgn = 0;
//         else if( J==tile_J && K==last_K )
//           uop_bgn = tile_I;
//         else if( J==last_J && K==tile_K )
//           uop_bgn = 2*tile_I;
//         else if( J==last_J && K==last_K )
//           uop_bgn = 3*tile_I;
//         // 依赖
//         // 影响load input和weight 在k循环不执行到最后一个时，对load产生影响，如果不使用bias，那么整个k循环都对load产生影响，代替bias的作用
//         bool gemm_push_pre_war = bias_use ? ((k0!=K0-1) ? 1 : 0) : 1; 
//         // 如果是最后一个GEMM就不用影响了,因为不会再回去加载input了，因此与上最后一个GEMM在最后一个k时信号为0
//         gemm_push_pre_war = gemm_push_pre_war && (((j0!=J0-1) || (i0 != I0-1)) || ((k0!=K0-1) ? 1 : 0));
//         bool gemm_push_next_raw = (k0==K0-1) ? 1 : 0; // 在k循环执行到最后一个，对store产生影响
//         insn_buf[insn_idx++] = getGEMMInsn(
//                                   uop_bgn,
//                                   uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                   J, // J
//                                   K, // K
//                                   accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                   relu_use,
//                                   scale_type1, // scale的类型，只有在k循环快结束时有效
//                                   scale_int, // scale的大小(整数)
//                                   1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                   0,    // pop_next_war 
//                                   gemm_push_pre_war,    // push_pre_war 
//                                   gemm_push_next_raw);   // push_next_raw  
//         printf("- Generate compute(gemm)\n");
//       }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             store_push_pre_war,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }

//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input_buffer, 
//       (volatile Transfer_DataType *)weight_buffer, 
//       (volatile Transfer_DataType *)biases_buffer, 
//       (volatile Transfer_DataType *)output_buffer,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   // 解包输出提取有效部分并返回输出给output指针
//   Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
//   Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
//   memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针

//   return 0;
// }




// // 硬件执行ANU计算矩阵的layernorm和softmax
// // 该函数使用指令控制微操作循环，使得指令分块计算数量减少 ，减少了生成的指令数量
// // 目前只能测试计算MATRIX_WIDTH行的数据,因为硬件一次只能执行这么多,如果要多次执行,需要管理依赖关系
// int anu_test(int opcode,   // 进行什么ANU操作
//             size_t dim_I,  // input矩阵的I行
//             size_t dim_J,  // input矩阵的J列
//             void * input,  // 输入
//             void * weight, // 权重
//             void * bias,   // 偏置
//             void * output) // 输出
// {
//   // 断言检查
//   assert(dim_I % MATRIX_WIDTH == 0); // I是否能被block整除
//   assert(dim_J % MATRIX_WIDTH == 0); // J是否能被block整除

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO -  ANU test: opcode=%d, dim_I=%d, dim_J=%d\n",
//          opcode, dim_I, dim_J);

//   // 加载分块和计算分块的比值
//   const int load_compute_block_ratio = dim_J / MATRIX_WIDTH; //一列的加载数量

//   // 如果要进行分块计算，行分块系数是脉动阵列大小，列就是计算矩阵的大小
//   const int dim_I_block = dim_I / MATRIX_WIDTH; // 行分多少个块

//   const int tile_I = MATRIX_WIDTH; //行的分块
//   const int tile_J = dim_J; //列的分块系数,直接加载一整列

//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = dim_I_block; // 一次加载MATRIX_WIDTH行
//   int insn_anu_size = dim_I_block; // 一次计算MATRIX_WIDTH行
//   int insn_store_size = dim_I_block; // 一次存储MATRIX_WIDTH行
//   int insn_size = insn_load_size + insn_store_size + insn_anu_size+1;

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 指令生成
//   // 迭代处理输出行块
//   for (int i = 0; i < dim_I_block; i++) {

//     // 加载指令生成
//     // 加载输入input(i*MATRIX_WIDTH+MATRIX_WIDTH,dim_J)
//     int buffer_start = 0; // 加载到输入buffer的起始位置  
//     int dram_start = 0; // 读取输入缓冲区的起始位置      
//     int A_block = i; // 第几个块
//     int buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//     int x_size = MATRIX_WIDTH*dim_J;//加载MATRIX_WIDTH行
//     int dram_offset = (dram_start + (A_block)*x_size); // 计算dram

//     insn_buf[insn_idx++] = get2DLoadStoreInsn(
//           OPCODE_LOAD,     // 加载指令
//           OUTPUT_BUFFER_ID,// 加载到输出Buffer
//           buffer_offset,   // buffer偏移+矩阵内部偏移
//           dram_offset,     // 缓冲区偏移+矩阵内部偏移
//           1,    // 每次加载MATRIX_WIDTH行
//           x_size,    // 每次加载MATRIX_WIDTH列
//           dim_J,
//           0,
//           0,
//           0,
//           0);          // input矩阵总列数作为2D跨步DMA步进,如果只加载一行连续加载,这个就没用了

//     // 计算指令生成
//     const int read_src_base = 0;  //每次计算的是同一个物理地址起始
//     const int write_dst_base = 0; // 写入buffer块的基地址,读取写入同一片地址     
//     const int src_offset =  MATRIX_WIDTH;      // 片上微操作每次的读取块偏移,就是MATRIX_WIDTH行
//     const int dst_offset =  MATRIX_WIDTH;      // 片上微操作每次的写入块偏移

//     // 生成LayerNorm指令
//     if(opcode == ANU_LAYERNORM) 
//     {
//       insn_buf[insn_idx++] = getLayerNormInsn(dim_J, // 写入当前矩阵的列数用于计算均值
//                                               read_src_base, // 读取buffer的基地址
//                                               write_dst_base, // 写入buffer的基地址
//                                               src_offset, // 每次读取块的偏移
//                                               dst_offset); // 每次写入块的偏移
//     }
//     else if(opcode == ANU_SOFTMAX)
//     {
//       insn_buf[insn_idx++] = getSoftmaxInsn(dim_J, // 写入当前矩阵的列数用于计算均值
//                                             read_src_base, // 读取buffer的基地址
//                                             write_dst_base, // 写入buffer的基地址
//                                             src_offset, // 每次读取块的偏移
//                                             dst_offset); // 每次写入块的偏移
//     }

//     // 存储指令行生成
//     buffer_start = 0; // 读取输出buffer的起始位置  
//     dram_start = 0; // 输出缓冲区的起始位置      
//     A_block = i; // 第几个块
//     buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//     x_size = MATRIX_WIDTH*dim_J;//加载MATRIX_WIDTH行
//     dram_offset = (dram_start + (A_block)*x_size); // 计算dram    // 存储输出
//     insn_buf[insn_idx++] = get2DLoadStoreInsn(
//           OPCODE_STORE,     // 存储指令
//           OUTPUT_BUFFER_ID, // 存储到输出buffer
//           buffer_offset,    // buffer偏移+矩阵内部偏移
//           dram_offset,      // 缓冲区偏移+矩阵内部偏移
//           1,     // 每次存储1行
//           x_size,     // 一次存储MATRIX_WIDTH*dim_J的数据
//           dim_J,
//           0,
//           0,
//           0,
//           0);           // output矩阵总列数作为2D跨步DMA步进
//   }

//   // 结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,0);
//   printf("insn_count:%d\n",insn_idx);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) NULL,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);

//   return 0;
// }


// // 硬件执行gemm+bias
// // 该函数完成了两级分块的分块矩阵乘法，调用了硬件计算矩阵乘法的函数
// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int tile_I,    // 分块系数，代表分了多少个小块
//                       int tile_J,    // 分块系数，代表分了多少个小块
//                       int tile_K,    // 分块系数，代表分了多少个小块
//                       bool bias_use) // 是否使用偏置  
// {
//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use);

//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 计算需要生成的指令数量
//   int insn_bias_output_size = (bias_use ? I0*J0 : 0) + I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size;

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引


//   // 断言判断加载的矩阵大小小于缓冲区大小

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // printf("bias_dram_offset:%d,I:%d,J:%d\n",bias_dram_offset,I,J);

//       // 加载偏置biases(dim_I,dim_J)
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,     // buffer为0，因为每次计算都直接加载满
//             bias_dram_offset,     // dram中偏移一个块尝试
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride, // 总矩阵的块的列数作为步进
//             0,     // pop_pre_raw  
//             (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//             1,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//             0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//       printf("- Generate compute(bias)\n");

//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;

//         // 指令生成

//         // 加载输入input(dim_I,dim_K)
//         const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//         // printf("input_dram_offset:%d,I:%d,K:%d\n",input_dram_offset,I,K);
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               INPUT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               I,     // 每次加载MATRIX_WIDTH行
//               K,     // 每次加载MATRIX_WIDTH列
//               dim_K_stride,  // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               1,     // pop_next_war   第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//               0,     // push_pre_war 
//               0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//         printf("- Generate load(input)\n");

//         // 加载权重weight(dim_K,dim_J)
//         const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//         // printf("weight_dram_offset:%d,K:%d,J:%d\n",weight_dram_offset,K,J);
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               WEIGHT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               K,     // 每次加载MATRIX_WIDTH行
//               J,     // 每次加载MATRIX_WIDTH列
//               dim_J_stride, // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//               0,     // push_pre_war 
//               1);    // push_next_raw  load weight的完成影响后面gemm的执行
//         printf("- Generate load(weight)\n");

//         // GEMM指令生成
//         bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，不使用则k0=0时刷新，而后累加
//         // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//         int uop_bgn = 0; 
//         if( J==tile_J && K==tile_K ) // 分块计算的形状
//           uop_bgn = 0;
//         else if( J==tile_J && K==last_K )
//           uop_bgn = tile_I;
//         else if( J==last_J && K==tile_K )
//           uop_bgn = 2*tile_I;
//         else if( J==last_J && K==last_K )
//           uop_bgn = 3*tile_I;

//         insn_buf[insn_idx++] = getGEMMInsn(
//                                   uop_bgn,
//                                   uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                   J, // J
//                                   K, // K
//                                   accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                   1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                   0,    // pop_next_war 
//                                   (k0!=K0-1) ? 1 : 0,    // push_pre_war 影响load input和weight 在k循环不执行到最后一个时，对load产生影响
//                                   (k0==K0-1) ? 1 : 0);   // push_next_raw  在k循环执行到最后一个，对store产生影响
//         printf("- Generate compute(gemm)\n");
//       }
//       // 存储指令生成, 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             1,     // push_pre_war , 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }


//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }







// // 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// // 并行方式:存在Bias那么load和bias并行,或者和store并行
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
//   Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
//   Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
//   Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

//   // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
//   Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 声明输出缓冲区(输出的是没有去填充的输出,一维)
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX
//   {
//       tile_I = 1;
//       tile_J = dim_J_padded; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= OUTPUT_BUFFER_WIDTH &&
//        tile_K * (tile_J+1) <= WEIGHT_BUFFER_WIDTH &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }
//     // 增大K维度的分块系数
//     if(tile_I * (tile_K+1) <= INPUT_BUFFER_WIDTH &&
//        (tile_K+1) * tile_J <= WEIGHT_BUFFER_WIDTH &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }
//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= INPUT_BUFFER_WIDTH &&
//        (tile_I+1) * tile_J <= OUTPUT_BUFFER_WIDTH &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   // printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
//   int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size;

//   // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
//   assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size) < STREAM_IN_DEPTH); // 计算队列
//   assert((insn_input_weight_size) < STREAM_IN_DEPTH); // 加载队列
//   assert((insn_output_size) < STREAM_IN_DEPTH); // 存储队列

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               1,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;

//         // 加载输入input(dim_I,dim_K)
//         // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//         bool input_pop_next_war = (bias_use ? 1 : (i0 > 0 || j0 > 0 || k0 > 0 ));
//         const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               INPUT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               I,     // 每次加载MATRIX_WIDTH行
//               K,     // 每次加载MATRIX_WIDTH列
//               dim_K_stride,  // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               input_pop_next_war, // pop_next_war   
//               0,     // push_pre_war 
//               0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//         printf("- Generate load(input)\n");

//         // 加载权重weight(dim_K,dim_J)
//         const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               WEIGHT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               K,     // 每次加载MATRIX_WIDTH行
//               J,     // 每次加载MATRIX_WIDTH列
//               dim_J_stride, // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//               0,     // push_pre_war 
//               1);    // push_next_raw  load weight的完成影响后面gemm的执行
//         printf("- Generate load(weight)\n");

//         // GEMM指令生成
//         bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，不使用则k0=0时刷新，而后累加
//         // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//         int uop_bgn = 0; 
//         if( J==tile_J && K==tile_K ) // 分块计算的形状
//           uop_bgn = 0;
//         else if( J==tile_J && K==last_K )
//           uop_bgn = tile_I;
//         else if( J==last_J && K==tile_K )
//           uop_bgn = 2*tile_I;
//         else if( J==last_J && K==last_K )
//           uop_bgn = 3*tile_I;
//         // 依赖
//         bool gemm_push_pre_war = bias_use ? ((k0!=K0-1) ? 1 : 0) : 1; // 影响load input和weight 在k循环不执行到最后一个时，对load产生影响，如果不使用bias，那么整个k循环都对load产生影响，代替bias的作用
//         bool gemm_push_next_raw = (k0==K0-1) ? 1 : 0; // 在k循环执行到最后一个，对store产生影响
//         insn_buf[insn_idx++] = getGEMMInsn(
//                                   uop_bgn,
//                                   uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                   J, // J
//                                   K, // K
//                                   accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                   1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                   0,    // pop_next_war 
//                                   gemm_push_pre_war,    // push_pre_war 
//                                   gemm_push_next_raw);   // push_next_raw  
//         printf("- Generate compute(gemm)\n");
//       }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             store_push_pre_war,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }


//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input_buffer, 
//       (volatile Transfer_DataType *)weight_buffer, 
//       (volatile Transfer_DataType *)biases_buffer, 
//       (volatile Transfer_DataType *)output_buffer,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   // 解包输出提取有效部分并返回输出给output指针
//   Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
//   Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
//   memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针

//   return 0;
// }


// // 块矩阵乘法的指令
// GenericIns getGEMMInsn(int uop_bgn, // 本次GEMM微操作缓冲区的起始位置
//                        int uop_end, // 本次GEMM微操作缓冲区的结束位置，这二者之差就是dim_I_block也就是本次分块的I
//                        int dim_J_block, // 硬件执行分块的J循环的次数
//                        int dim_K_block, // 硬件执行的K循环的次数
//                        int bias_use,    // 硬件上是否使用bias，代表硬件分块计算是否和bias累加
//                        bool pop_pre_raw,  
//                        bool pop_next_war, 
//                        bool push_pre_war,
//                        bool push_next_raw) 
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   ComIns insn = {};

//   // GEMM字段
//   insn.opcode = OPCODE_GEMM;  // GEMM指令
//   insn.uop_bgn = uop_bgn;  // 微操作缓冲区开始的地方
//   insn.uop_end = uop_end;  // 微操作缓冲区的大小
//   insn.dim_K_block = dim_K_block;  // K循环的次数
//   insn.dim_J_block = dim_J_block;  // J循环的次数
//   insn.bias_use = bias_use;  // 是否使用偏置
//   insn.scale_type = 0;  // 是否进行缩放,是进行反量化还是重量化
//   insn.scale = 0;  // 缩放系数
//   // 依赖字段
//   insn.pop_pre_raw = pop_pre_raw; // 执行前对前一个load的raw依赖
//   insn.pop_next_war = pop_next_war; //  执行前对后一个store的war依赖
//   insn.push_pre_war = push_pre_war; //  执行后是否对前一个load有war影响
//   insn.push_next_raw = push_next_raw; // 执行后是否对后一个store的raw影响

//   converter.com = insn;

//   return converter.generic;
// }


// // 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
//   Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
//   Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
//   Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

//   // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
//   Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
//   Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

//   // 声明输出缓冲区(输出的是没有去填充的输出,一维)
//   Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX
//   {
//       tile_I = 1;
//       tile_J = dim_J_padded; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= OUTPUT_BUFFER_WIDTH &&
//        tile_K * (tile_J+1) <= WEIGHT_BUFFER_WIDTH &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }
//     // 增大K维度的分块系数
//     if(tile_I * (tile_K+1) <= INPUT_BUFFER_WIDTH &&
//        (tile_K+1) * tile_J <= WEIGHT_BUFFER_WIDTH &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }
//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= INPUT_BUFFER_WIDTH &&
//        (tile_I+1) * tile_J <= OUTPUT_BUFFER_WIDTH &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   // printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
//   int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size;

//   // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
//   assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size) < STREAM_IN_DEPTH); // 计算队列
//   assert((insn_input_weight_size) < STREAM_IN_DEPTH); // 加载队列
//   assert((insn_output_size) < STREAM_IN_DEPTH); // 存储队列

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               1,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;

//         // 加载输入input(dim_I,dim_K)
//         // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//         bool input_pop_next_war = (bias_use ? 1 : (i0 > 0 || j0 > 0 || k0 > 0 ));
//         const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               INPUT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               I,     // 每次加载MATRIX_WIDTH行
//               K,     // 每次加载MATRIX_WIDTH列
//               dim_K_stride,  // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               input_pop_next_war, // pop_next_war   
//               0,     // push_pre_war 
//               0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//         printf("- Generate load(input)\n");

//         // 加载权重weight(dim_K,dim_J)
//         const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               WEIGHT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               K,     // 每次加载MATRIX_WIDTH行
//               J,     // 每次加载MATRIX_WIDTH列
//               dim_J_stride, // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//               0,     // push_pre_war 
//               1);    // push_next_raw  load weight的完成影响后面gemm的执行
//         printf("- Generate load(weight)\n");

//         // GEMM指令生成
//         bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，不使用则k0=0时刷新，而后累加
//         // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//         int uop_bgn = 0; 
//         if( J==tile_J && K==tile_K ) // 分块计算的形状
//           uop_bgn = 0;
//         else if( J==tile_J && K==last_K )
//           uop_bgn = tile_I;
//         else if( J==last_J && K==tile_K )
//           uop_bgn = 2*tile_I;
//         else if( J==last_J && K==last_K )
//           uop_bgn = 3*tile_I;
//         // 依赖
//         bool gemm_push_pre_war = bias_use ? ((k0!=K0-1) ? 1 : 0) : 1; // 影响load input和weight 在k循环不执行到最后一个时，对load产生影响，如果不使用bias，那么整个k循环都对load产生影响，代替bias的作用
//         bool gemm_push_next_raw = (k0==K0-1) ? 1 : 0; // 在k循环执行到最后一个，对store产生影响
//         insn_buf[insn_idx++] = getGEMMInsn(
//                                   uop_bgn,
//                                   uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                   J, // J
//                                   K, // K
//                                   accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                   1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                   0,    // pop_next_war 
//                                   gemm_push_pre_war,    // push_pre_war 
//                                   gemm_push_next_raw);   // push_next_raw  
//         printf("- Generate compute(gemm)\n");
//       }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             store_push_pre_war,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }


//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input_buffer, 
//       (volatile Transfer_DataType *)weight_buffer, 
//       (volatile Transfer_DataType *)biases_buffer, 
//       (volatile Transfer_DataType *)output_buffer,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   // 解包输出提取有效部分并返回输出给output指针
//   Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
//   Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
//   memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针

//   return 0;
// }



// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入(dim_I,dim_K)
//                       void * weight, // 权重(dim_K,dim_J)
//                       void * bias,   // 偏置(dim_I,dim_J)
//                       void * output, // 输出(dim_I,dim_J)
//                       bool bias_use, // 是否使用偏置  
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX
//   {
//       tile_I = 1;
//       tile_J = dim_J_padded; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= OUTPUT_BUFFER_WIDTH &&
//        tile_K * (tile_J+1) <= WEIGHT_BUFFER_WIDTH &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }
//     // 增大K维度的分块系数
//     if(tile_I * (tile_K+1) <= INPUT_BUFFER_WIDTH &&
//        (tile_K+1) * tile_J <= WEIGHT_BUFFER_WIDTH &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }
//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= INPUT_BUFFER_WIDTH &&
//        (tile_I+1) * tile_J <= OUTPUT_BUFFER_WIDTH &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_output_size = (bias_use ? I0*J0 : 0) + I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size;

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // 加载偏置biases(dim_I,dim_J)
//       if(bias_use==1 && bias!=NULL)
//       {
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 加载指令
//               OUTPUT_BUFFER_ID, // 存储到输出buffer
//               0,     // buffer为0，因为每次计算都直接加载满
//               bias_dram_offset,     // dram中偏移一个块尝试
//               I,     // 每次加载I个行块
//               J,     // 每次加载J个列块
//               dim_J_stride, // 总矩阵的块的列数作为步进
//               0,     // pop_pre_raw  
//               (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//               1,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//               0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//         printf("- Generate compute(bias)\n");
//       }
//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;

//         // 加载输入input(dim_I,dim_K)
//         // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//         bool input_pop_next_war = (bias_use ? 1 : (i0 > 0 || j0 > 0 || k0 > 0 ));
//         const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               INPUT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               I,     // 每次加载MATRIX_WIDTH行
//               K,     // 每次加载MATRIX_WIDTH列
//               dim_K_stride,  // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               input_pop_next_war, // pop_next_war   
//               0,     // push_pre_war 
//               0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//         printf("- Generate load(input)\n");

//         // 加载权重weight(dim_K,dim_J)
//         const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               WEIGHT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               K,     // 每次加载MATRIX_WIDTH行
//               J,     // 每次加载MATRIX_WIDTH列
//               dim_J_stride, // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//               0,     // push_pre_war 
//               1);    // push_next_raw  load weight的完成影响后面gemm的执行
//         printf("- Generate load(weight)\n");

//         // GEMM指令生成
//         bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，不使用则k0=0时刷新，而后累加
//         // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//         int uop_bgn = 0; 
//         if( J==tile_J && K==tile_K ) // 分块计算的形状
//           uop_bgn = 0;
//         else if( J==tile_J && K==last_K )
//           uop_bgn = tile_I;
//         else if( J==last_J && K==tile_K )
//           uop_bgn = 2*tile_I;
//         else if( J==last_J && K==last_K )
//           uop_bgn = 3*tile_I;
//         // 依赖
//         bool gemm_push_pre_war = bias_use ? ((k0!=K0-1) ? 1 : 0) : 1; // 影响load input和weight 在k循环不执行到最后一个时，对load产生影响，如果不使用bias，那么整个k循环都对load产生影响，代替bias的作用
//         bool gemm_push_next_raw = (k0==K0-1) ? 1 : 0; // 在k循环执行到最后一个，对store产生影响
//         insn_buf[insn_idx++] = getGEMMInsn(
//                                   uop_bgn,
//                                   uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                   J, // J
//                                   K, // K
//                                   accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                   1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                   0,    // pop_next_war 
//                                   gemm_push_pre_war,    // push_pre_war 
//                                   gemm_push_next_raw);   // push_next_raw  
//         printf("- Generate compute(gemm)\n");
//       }
//       // 存储指令生成, 存储输出
//       // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
//       bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             store_push_pre_war,  // push_pre_war , 
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }


//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }


// // 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       bool bias_use, // 是否使用偏置  
//                       int act) // 是否使用激活，使用何种激活
// {
//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
//   // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
// # define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
// # define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

//   size_t tile_I, tile_J, tile_K;
//   // 首先是初始分块系数的计算
//   if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX
//   {
//       tile_I = 1;
//       tile_J = dim_J_padded; // J维度完全装入片上
//       tile_K = 1;
//   }
//   else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
//   {
//     tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
//     tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
//     tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
//   }

//   // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
//   while (true) {
//     bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

//     // 增大J维度的分块系数
//     if(tile_I * (tile_J+1) <= OUTPUT_BUFFER_WIDTH &&
//        tile_K * (tile_J+1) <= WEIGHT_BUFFER_WIDTH &&
//        (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_J++;
//         increased = true;
//     }
//     // 增大K维度的分块系数
//     if(tile_I * (tile_K+1) <= INPUT_BUFFER_WIDTH &&
//        (tile_K+1) * tile_J <= WEIGHT_BUFFER_WIDTH &&
//        (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_K++;
//         increased = true;
//     }
//     // 增大I维度的分块系数
//     if((tile_I+1) * tile_K <= INPUT_BUFFER_WIDTH &&
//        (tile_I+1) * tile_J <= OUTPUT_BUFFER_WIDTH &&
//        (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
//     {
//         tile_I++;
//         increased = true;
//     }

//     if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
//       break;
//   }

//   // 断言判断缓冲区是否能够完整加载MATRIX_WIDTH行dim_J_padded列的激活矩阵

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use=%d, activate=%d\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use,act);
//   printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);

//   // 计算需要生成的指令数量
//   int insn_bias_output_size = (bias_use ? I0*J0 : 0) + I0*J0; //加载bias，存储output,需要判断是否使用bias
//   int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
//   int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
//   int insn_uop_size = 1; // 加载uop的大小
//   int insn_finish_size = 1; // 结束指令
//   int insn_size = insn_bias_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size;

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // printf("bias_dram_offset:%d,I:%d,J:%d\n",bias_dram_offset,I,J);

//       // 加载偏置biases(dim_I,dim_J)
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,     // buffer为0，因为每次计算都直接加载满
//             bias_dram_offset,     // dram中偏移一个块尝试
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride, // 总矩阵的块的列数作为步进
//             0,     // pop_pre_raw  
//             (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//             1,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//             0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//       printf("- Generate compute(bias)\n");

//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;

//         // 指令生成

//         // 加载输入input(dim_I,dim_K)
//         const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//         // printf("input_dram_offset:%d,I:%d,K:%d\n",input_dram_offset,I,K);
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               INPUT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               I,     // 每次加载MATRIX_WIDTH行
//               K,     // 每次加载MATRIX_WIDTH列
//               dim_K_stride,  // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               1,     // pop_next_war   第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//               0,     // push_pre_war 
//               0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//         printf("- Generate load(input)\n");

//         // 加载权重weight(dim_K,dim_J)
//         const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//         // printf("weight_dram_offset:%d,K:%d,J:%d\n",weight_dram_offset,K,J);
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               WEIGHT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               K,     // 每次加载MATRIX_WIDTH行
//               J,     // 每次加载MATRIX_WIDTH列
//               dim_J_stride, // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//               0,     // push_pre_war 
//               1);    // push_next_raw  load weight的完成影响后面gemm的执行
//         printf("- Generate load(weight)\n");

//         // GEMM指令生成
//         bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，不使用则k0=0时刷新，而后累加
//         // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//         int uop_bgn = 0; 
//         if( J==tile_J && K==tile_K ) // 分块计算的形状
//           uop_bgn = 0;
//         else if( J==tile_J && K==last_K )
//           uop_bgn = tile_I;
//         else if( J==last_J && K==tile_K )
//           uop_bgn = 2*tile_I;
//         else if( J==last_J && K==last_K )
//           uop_bgn = 3*tile_I;

//         insn_buf[insn_idx++] = getGEMMInsn(
//                                   uop_bgn,
//                                   uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                   J, // J
//                                   K, // K
//                                   accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                   1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                   0,    // pop_next_war 
//                                   (k0!=K0-1) ? 1 : 0,    // push_pre_war 影响load input和weight 在k循环不执行到最后一个时，对load产生影响
//                                   (k0==K0-1) ? 1 : 0);   // push_next_raw  在k循环执行到最后一个，对store产生影响
//         printf("- Generate compute(gemm)\n");
//       }
//       // 存储指令生成, 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             1,     // push_pre_war , 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }


//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_idx:%d\n",insn_idx);
//   printf("insn_size:%d\n",insn_size);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }























// #ifndef SAA_SIM_H_
// #define SAA_SIM_H_

// #include "../src/SAA.h"
// #include <cstdlib> 
// #include <stdlib.h>

// // 分配连续缓存
// void * allocBuffer(size_t num_bytes) {
//   return malloc(num_bytes);
// }

// // 释放连续缓存
// void freeBuffer(void * buffer) {
//   return free(buffer);
// }


// // 用于产生任意大小的初始化的2D矩阵（地址不连续）
// template <typename T>
// T ** allocInit2dArray(int rows, int cols) {
//   // 分配内存
//   T **array = static_cast<T **>(malloc(sizeof(T *) * rows));
//   for (int i = 0; i < rows; i++) {
//     array[i] = static_cast<T *>(malloc(sizeof(T) * cols));
//   }
//   // 初始化
//   for (int i = 0; i < rows; i++) {
//     for (int j = 0; j < cols; j++) {
//       array[i][j] = static_cast<T>(rand() % 10);
//     }
//   }
//   return array;
// }

// // 用于释放上面的2D矩阵内存
// template <typename T>
// void free2dArray(T **array, int rows, int cols) {
//   for (int i = 0; i < rows; i++) {
//     free(array[i]);
//   }
//   free(array);
// }


// //-----------打包解包只是单纯将数组变得连续和从连续数组中读取数据--------------//

// /**
//  * @brief 将数据分块从源二维数组打包到目标一维数组中。
//  *
//  * 该函数将源数据 `src` 中的数据块打包到目标数据 `dst` 中。打包过程中，会按照
//  * `y_block` 和 `x_block` 定义的块大小，将 `SRC_T` 类型的数据转换为 `DST_T` 类型，
//  * 并存储在目标数组中。这个过程涉及到按位操作，将多个源数据的位拼接到一个
//  * `DST_T` 类型的变量中，直到填满为止，然后将该变量写入目标数组，继续处理下一个数据块。
//  *
//  * 代码按照块打包数据到连续缓存中，存好第一个块然后再存第二个块
//  * 
//  * @tparam DST_T 目标数据类型。
//  * @tparam DST_T_WIDTH 目标数据类型的位宽。
//  * @tparam SRC_T 源数据类型。
//  * @tparam SRC_T_WIDTH 源数据类型的位宽。
//  * @param dst 目标一维数组的指针。
//  * @param src 源二维数组的指针。
//  * @param y_size 源数据的行数。
//  * @param x_size 源数据的列数。
//  * @param y_block 每个块的行数。
//  * @param x_block 每个块的列数。
//  */
// template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
// void packBuffer(DST_T *dst, SRC_T **src, int y_size, int x_size, int y_block, int x_block) {
//   // 确保源数据和目标数据的宽度比例正确。
//   assert((SRC_T_WIDTH * x_block * y_block) % DST_T_WIDTH == 0);
//   assert(DST_T_WIDTH <= 64);

//   int buffer_idx = 0; // 目标数组的索引。
//   int ratio = DST_T_WIDTH / SRC_T_WIDTH; // 数据类型宽度比例。
//   long long int mask = (1ULL << SRC_T_WIDTH) - 1; // 源数据类型的位掩码。
//   DST_T tmp = 0; // 用于临时存储打包数据的变量。

//   // 遍历源数据的每个块。
//   for (int i = 0; i < y_size / y_block; i++) {
//     for (int j = 0; j < x_size / x_block; j++) {
//       for (int k = 0; k < y_block; k++) {
//         for (int l = 0; l < x_block; l++) {
//           int block_idx = l + k * x_block; // 当前块内元素的索引。
//           // 将源数据的元素与掩码进行按位与操作，然后左移相应的位数，合并到tmp变量中。
//           tmp |= (src[i * y_block + k][j * x_block + l] & mask) << ((block_idx % ratio) * SRC_T_WIDTH);
//           // 当处理完一个数据类型的所有位后，将其写入目标数组。
//           if (block_idx % ratio == ratio - 1) {
//             dst[buffer_idx++] = tmp;
//             tmp = 0; // 重置tmp变量，准备下一个打包周期。
//           }
//         }
//       }
//     }
//   }
// }

// /**
//  * @brief 将数据分块从源一维数组解包到目标二维数组中。
//  *
//  * 该函数将源数据 `src` 中的数据解包到目标数据 `dst` 中。解包过程与打包过程相反，
//  * 它将 `DST_T` 类型的数据转换回 `SRC_T` 类型，并存储到目标二维数组中。
//  *
//  * @tparam DST_T 目标数据类型。
//  * @tparam DST_T_WIDTH 目标数据类型的位宽。
//  * @tparam SRC_T 源数据类型。
//  * @tparam SRC_T_WIDTH 源数据类型的位宽。
//  * @param dst 目标二维数组的指针。
//  * @param src 源一维数组的指针。
//  * @param y_size 目标数据的行数。
//  * @param x_size 目标数据的列数。
//  * @param y_block 每个块的行数。
//  * @param x_block 每个块的列数。
//  */
// template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
// void unpackBuffer(DST_T **dst, SRC_T *src, int y_size, int x_size, int y_block, int x_block) {
//   // 确保源数据和目标数据的宽度比例正确。
//   assert((DST_T_WIDTH * x_block * y_block) % SRC_T_WIDTH == 0);

//   int buffer_idx = 0; // 源数组的索引。
//   long long int mask = (1ULL << DST_T_WIDTH) - 1; // 目标数据类型的位掩码。
//   int ratio = SRC_T_WIDTH / DST_T_WIDTH; // 数据类型宽度比例。

//   // 遍历目标数据的每个块。
//   for (int i = 0; i < y_size / y_block; i++) {
//     for (int j = 0; j < x_size / x_block; j++) {
//       for (int k = 0; k < y_block; k++) {
//         for (int l = 0; l < x_block; l++) {
//           int block_idx = l + k * x_block; // 当前块内元素的索引。
//           // 从源数组中读取数据，右移相应的位数，并与掩码进行按位与操作，得到目标数据。
//           dst[i * y_block + k][j * x_block + l] = (src[buffer_idx] >> ((block_idx % ratio) * DST_T_WIDTH)) & mask;
//           // 当处理完一个数据类型的所有位后，移动到源数组的下一个位置。
//           if (block_idx % ratio == ratio - 1) {
//             buffer_idx++;
//           }
//         }
//       }
//     }
//   }
// }



// // 简化后的按块打包函数
// template <typename T>
// void packData(T *dst, T **src, int y_size, int x_size, int y_block, int x_block) {
//   int dst_idx = 0; // 目标数组的索引

//   // 遍历源数据的每个块
//   for (int i = 0; i < y_size; i += y_block) {
//     for (int j = 0; j < x_size; j += x_block) {
//       for (int k = 0; k < y_block && i + k < y_size; ++k) {
//         for (int l = 0; l < x_block && j + l < x_size; ++l) {
//           dst[dst_idx++] = src[i + k][j + l];
//         }
//       }
//     }
//   }
// }

// // 简化后的按块解包函数
// template <typename T>
// void unpackData(T **dst, T *src, int y_size, int x_size, int y_block, int x_block) {
//   int src_idx = 0; // 源数组的索引

//   // 遍历目标数据的每个块
//   for (int i = 0; i < y_size; i += y_block) {
//     for (int j = 0; j < x_size; j += x_block) {
//       for (int k = 0; k < y_block && i + k < y_size; ++k) {
//         for (int l = 0; l < x_block && j + l < x_size; ++l) {
//           dst[i + k][j + l] = src[src_idx++];
//         }
//       }
//     }
//   }
// }





// template <typename T>
// void print_pack_buffer(void* packed_buffer, int rows,int cols)
// {
//   // 打印打包后的1D数组，转换回原始数据类型
//   std::cout << "Packed Buffer (1D Array representing 2D Matrix):" << std::endl;
//   for (size_t i = 0; i < rows * cols; ++i) {
//     std::cout << *(reinterpret_cast<T*>(packed_buffer)+i)<< " ";
//     if ((i + 1) % cols == 0) {
//       std::cout << std::endl;
//     }
//   }
//   std::cout << std::endl;
// }




// //------------------------------------指令生成-------------------------------------//
// //以下函数由于操作码都不大于32，因此都是用int传参

// // 具有依赖关系的指令
// // 生成2D加载、存储指令，加载存储矩阵块，需要指定opcode是加载还是存储
// GenericIns get2DLoadStoreInsn(int opcode, 
//                               int buffer_id, 
//                               int buffer_offset, 
//                               int dram_offset,
//                               int y_size, 
//                               int x_size, 
//                               int x_stride,
//                               bool pop_pre_raw,  
//                               bool pop_next_war, 
//                               bool push_pre_war,
//                               bool push_next_raw) 
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 存储指令初始化
//   MemIns insn = {};
//   insn.opcode = opcode;  // 加载指令
//   insn.buffer_id = buffer_id; // 存储在哪个缓冲区
//   insn.dram_base = dram_offset; // 缓冲区偏移+矩阵内部偏移
//   insn.buffer_base = buffer_offset; // buffer偏移+矩阵内部偏移
//   insn.y_size = y_size; // 每次加载MATRIX_WIDTH行
//   insn.x_size = x_size; // 每次加载MATRIX_WIDTH列
//   insn.x_stride = x_stride; // 假设每行数据在DRAM中是连续存储的，那么步长就是列宽
//   insn.pop_pre_raw   = pop_pre_raw  ; // 该指令对前一个模块的RAW依赖
//   insn.pop_next_war  = pop_next_war ; // 该指令对后一个模块的WAR依赖
//   insn.push_pre_war  = push_pre_war ; // 该指令对前一个模块的WAR影响
//   insn.push_next_raw = push_next_raw; // 该指令对后一个模块的RAW影响

//   converter.mem = insn;

//   return converter.generic;
// }

// // 块矩阵乘法的指令
// GenericIns getGEMMInsn(int uop_bgn, // 本次GEMM微操作缓冲区的起始位置
//                        int uop_end, // 本次GEMM微操作缓冲区的结束位置，这二者之差就是dim_I_block也就是本次分块的I
//                        int dim_J_block, // 硬件执行分块的J循环的次数
//                        int dim_K_block, // 硬件执行的K循环的次数
//                        int bias_use,    // 硬件上是否使用bias，代表硬件分块计算是否和bias累加
//                        bool pop_pre_raw,  
//                        bool pop_next_war, 
//                        bool push_pre_war,
//                        bool push_next_raw) 
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   ComIns insn = {};

//   // GEMM字段
//   insn.opcode = OPCODE_GEMM;  // GEMM指令
//   insn.uop_bgn = uop_bgn;  // 微操作缓冲区开始的地方
//   insn.uop_end = uop_end;  // 微操作缓冲区的大小
//   insn.dim_K_block = dim_K_block;  // K循环的次数
//   insn.dim_J_block = dim_J_block;  // J循环的次数
//   insn.bias_use = bias_use;  // 是否使用偏置
//   // 依赖字段
//   insn.pop_pre_raw = pop_pre_raw; // 执行前对前一个load的raw依赖
//   insn.pop_next_war = pop_next_war; //  执行前对后一个store的war依赖
//   insn.push_pre_war = push_pre_war; //  执行后是否对前一个load有war影响
//   insn.push_next_raw = push_next_raw; // 执行后是否对后一个store的raw影响
//   converter.com = insn;

//   return converter.generic;
// }

// // 块矩阵乘法运算的微操作指令生成
// // 生成了四种形状块的微操作，使得可以适应多种形式的硬件分块计算
// Uop * getGEMMUops(int tile_I, // 硬件分块计算的最大分块大小I
//                   int tile_J, // 硬件分块计算的最大分块大小J  
//                   int tile_K, // 硬件分块计算的最大分块大小K
//                   int last_I, // 此处不使用，直接按照tile_I生成，实际GEMM指令传入时bgn到end间会动态变化适应分块
//                   int last_J, // 最后不满tile_J的分块
//                   int last_K) // 最后不满tile_K的分块 
// {
//   //如果存在异形的分块计算，那么我们需要将异形的分块计算的大小考虑进来，一共有四种情况
//   int uop_size = 4*tile_I ; // 因为压缩了微操作的数量,只计算I循环相关的两个偏移,权重偏移在片上利用仿射计算

//   Uop *uop_buf = static_cast<Uop *>(malloc(sizeof(Uop) * uop_size));

//   int uop_idx=0;
//   //tile_I、tile_J、tile_K
//   for (int i = 0; i < tile_I; i++) {
//     uop_buf[uop_idx].input_idx = i*tile_K*MATRIX_WIDTH; // 以MATRIX_WIDTH为最小分块
//     uop_buf[uop_idx].output_idx = i*tile_J*MATRIX_WIDTH;
//     // uop_buf[i].weight_idx = 0; // 不计算权重的偏移
//     uop_idx++;
//   }

//   //tile_I、tile_J、last_K
//   for (int i = 0; i < tile_I; i++) {
//     uop_buf[uop_idx].input_idx = i*last_K*MATRIX_WIDTH; 
//     uop_buf[uop_idx].output_idx = i*tile_J*MATRIX_WIDTH;
//     uop_idx++;
//   }
  
//   //tile_I、last_J、tile_K
//   for (int i = 0; i < tile_I; i++) {
//     uop_buf[uop_idx].input_idx = i*tile_K*MATRIX_WIDTH; 
//     uop_buf[uop_idx].output_idx = i*last_J*MATRIX_WIDTH;
//     uop_idx++;
//   }

//   //tile_I、last_J、last_K
//   for (int i = 0; i < tile_I; i++) {
//     uop_buf[uop_idx].input_idx = i*last_K*MATRIX_WIDTH; 
//     uop_buf[uop_idx].output_idx = i*last_J*MATRIX_WIDTH;
//     uop_idx++;
//   }
//   return uop_buf;
// }

// // 生成运算完成指令
// GenericIns getFinishInsn(bool pop_pre_raw,bool pop_next_war) 
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   ComIns insn = {};
//   insn.opcode = OPCODE_DONE;  // 完成指令
//   insn.pop_pre_raw = pop_pre_raw; // 对前一个load的raw依赖
//   insn.pop_next_war = pop_next_war; // 对后一个store的war依赖
//   insn.push_pre_war = 0;
//   insn.push_next_raw = 0;
//   converter.com = insn;
//   return converter.generic;
// }

// // 生成LayerNorm指令
// GenericIns getLayerNormInsn(int dim_J, // 写入当前矩阵的列数用于计算均值
//                             int read_src_base, // 读取buffer的基地址
//                             int write_dst_base, // 写入buffer的基地址
//                             int src_offset, // 每次读取块的偏移
//                             int dst_offset) // 每次写入块的偏移
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   AnuIns insn = {};
//   insn.opcode = OPCODE_ANU;  // ANU指令
//   insn.anu_type = ANU_LAYERNORM; // 进行layernorm
//   insn.iter_uop = dim_J/MATRIX_WIDTH; // 需要遍历多少个行块才能计算完一行，根据dim_J/MATRIX_WIDTH计算得到，需要能够整除
//   insn.read_src_base = read_src_base; // 读取buffer的基地址
//   insn.write_dst_base = write_dst_base; // 写入buffer的基地址
//   insn.src_offset = src_offset; // 每次读取块的偏移
//   insn.dst_offset = dst_offset; // 每次写入块的偏移
//   insn.imm = dim_J; // 立即数用于输入归一化的列数目
//   converter.anu = insn;

//   return converter.generic;
// }

// // 生成Softmax指令
// GenericIns getSoftmaxInsn(int dim_J, // 写入当前矩阵的列数用于计算均值
//                           int read_src_base, // 读取buffer的基地址
//                           int write_dst_base, // 写入buffer的基地址
//                           int src_offset, // 每次读取块的偏移
//                           int dst_offset) // 每次写入块的偏移
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   AnuIns insn = {};
//   insn.opcode = OPCODE_ANU;  // ANU指令
//   insn.anu_type = ANU_SOFTMAX; // 进行softmax
//   insn.iter_uop = dim_J/MATRIX_WIDTH; // 需要遍历多少个行块才能计算完一行，根据dim_J/MATRIX_WIDTH计算得到，需要能够整除
//   insn.read_src_base = read_src_base; // 读取buffer的基地址
//   insn.write_dst_base = write_dst_base; // 写入buffer的基地址
//   insn.src_offset = src_offset; // 每次读取块的偏移
//   insn.dst_offset = dst_offset; // 每次写入块的偏移
//   insn.imm = dim_J; // 立即数用于输入归一化的列数目
//   converter.anu = insn;

//   return converter.generic;
// }


// //------------------------------------高级函数，调用指令进行计算-------------------------------------//

// //软件实现gemm+bias
// template<typename T0,typename T1 ,typename T>
// T** matrix_biase_dot(T0** input, T1** weight, T** bias,int row, int col, int col1)
// {
//     // 创建结果矩阵
//     T** result = init_matrix<T>(row, col1);

//     // 计算矩阵乘法
//     for (int i = 0; i < row; i++) {
//         for (int j = 0; j < col1; j++) {
//             result[i][j] = bias[i][j];
//             for (int k = 0; k < col; k++) {
//                 result[i][j] += input[i][k] * weight[k][j];
//             }
//         }
//     }

//     return result;
// }

// // 硬件执行gemm+bias
// // 该函数完成了两级分块的分块矩阵乘法，调用了硬件计算矩阵乘法的函数
// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int tile_I,    // 分块系数，代表分了多少个小块
//                       int tile_J,    // 分块系数，代表分了多少个小块
//                       int tile_K,    // 分块系数，代表分了多少个小块
//                       bool bias_use) // 是否使用偏置  
// {
//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use);

//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = I0*J0*K0; // 计算加载A、B矩阵需要多少大分块指令
//   int insn_compute_size = I0*J0*K0; // 不使用权重复用
//   int insn_store_size = I0*J0*K0; // 输出矩阵大小
//   int insn_size = insn_load_size + insn_store_size + insn_compute_size+1 + 100;

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引


//   // 断言判断加载的矩阵大小小于缓冲区大小

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K, // K
//                       last_I,
//                       last_J,
//                       last_K); 

//   int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       printf("bias_dram_offset:%d,I:%d,J:%d\n",bias_dram_offset,I,J);

//       // 加载偏置biases(dim_I,dim_J)
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,     // buffer为0，因为每次计算都直接加载满
//             bias_dram_offset,     // dram中偏移一个块尝试
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride, // 总矩阵的块的列数作为步进
//             0,     // pop_pre_raw  
//             (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//             1,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//             0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//       printf("- Generate compute(bias)\n");

//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;

//         // 指令生成

//         // 加载输入input(dim_I,dim_K)
//         const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
//         printf("input_dram_offset:%d,I:%d,K:%d\n",input_dram_offset,I,K);
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               INPUT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               I,     // 每次加载MATRIX_WIDTH行
//               K,     // 每次加载MATRIX_WIDTH列
//               dim_K_stride,  // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               1,     // pop_next_war   第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
//               0,     // push_pre_war 
//               0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
//         // printf("- Generate load(input)\n");

//         // 加载权重weight(dim_K,dim_J)
//         const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
//         printf("weight_dram_offset:%d,K:%d,J:%d\n",weight_dram_offset,K,J);
//         insn_buf[insn_idx++] = get2DLoadStoreInsn(
//               OPCODE_LOAD,     // 存储指令
//               WEIGHT_BUFFER_ID, // 存储到输出buffer
//               0,    // buffer偏移+矩阵内部偏移
//               weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
//               K,     // 每次加载MATRIX_WIDTH行
//               J,     // 每次加载MATRIX_WIDTH列
//               dim_J_stride, // output矩阵的列的分块数作为步进
//               0,     // pop_pre_raw  
//               0,     // pop_next_war   load input完成后直接执行，因此没有依赖
//               0,     // push_pre_war 
//               1);    // push_next_raw  load weight的完成影响后面gemm的执行
//         // printf("- Generate load(weight)\n");

//         // GEMM指令生成
//         bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，不使用则k0=0时刷新，而后累加
//         int uop_bgn = 0; // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
//         if( J==tile_J && K==tile_K )
//           uop_bgn = 0;
//         else if( J==tile_J && K==last_K )
//           uop_bgn = tile_I;
//         else if( J==last_J && K==tile_K )
//           uop_bgn = 2*tile_I;
//         else if( J==last_J && K==last_K )
//           uop_bgn = 3*tile_I;

//         insn_buf[insn_idx++] = getGEMMInsn(
//                                   uop_bgn,
//                                   uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
//                                   J, // J
//                                   K, // K
//                                   accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
//                                   1,    // pop_pre_raw  需要等待load input和weight执行完毕
//                                   0,    // pop_next_war 
//                                   (k0!=K0-1) ? 1 : 0,    // push_pre_war 影响load input和weight 在k循环不执行到最后一个时，对load产生影响
//                                   (k0==K0-1) ? 1 : 0);   // push_next_raw  在k循环执行到最后一个，对store产生影响
//         // printf("- Generate compute(gemm)\n");
//       }
//       // 存储指令生成, 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
//             0,     // pop_next_war   
//             1,     // push_pre_war , 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }


//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_count:%d\n",insn_idx);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }


// // 软件计算layernorm,遍历两次
// template<typename T>
// T** layer_norm(T** x, int batch_size, int features, float eps) {
//     // 分配内存以存储每批的均值和方差
//     T* means = new T[batch_size];
//     T* variances = new T[batch_size];

//     // 计算每个样本的均值和方差
//     for (int i = 0; i < batch_size; ++i) {
//         float sum = 0.0f;
//         float sum_of_squares = 0.0f;
        
//         // 第一次遍历：计算均值和平方的均值
//         for (int j = 0; j < features; ++j) {
//             sum += x[i][j];
//             sum_of_squares += static_cast<float>(x[i][j]) * x[i][j];
//         }
        
//         // 保存每个样本的均值
//         float mean = sum / features;
//         // 计算方差
//         float variance = (sum_of_squares / features) - (mean * mean);
//         means[i] = mean;
//         variances[i] = variance;
//     }

//     // 计算所有样本的均值和方差
//     float mean_of_means = 0.0f;
//     float mean_of_variances = 0.0f;
//     for (int i = 0; i < batch_size; ++i) {
//         mean_of_means += means[i];
//         mean_of_variances += variances[i];
//     }
//     mean_of_means /= batch_size;
//     mean_of_variances /= batch_size;

//     // 第二次遍历：归一化
//     // 创建结果矩阵
//     T** result = init_matrix<T>(batch_size, features);
//     for (int i = 0; i < batch_size; ++i) {
//         for (int j = 0; j < features; ++j) {
//             T mean_diff = x[i][j] - means[i];
//             T normalized_val = mean_diff / (sqrtf(variances[i] + eps));
//             result[i][j] = normalized_val;
//         }
//     }
    
//     printf("\nsum:\n");
//     print_vec(means,batch_size);
//     printf("var:\n");
//     print_vec(variances,batch_size);

//     // 释放分配的内存
//     delete[] means;
//     delete[] variances;

//     return result;
// }

// // 软件计算softmax,遍历三次,第一次找最大值,第二次进行exp并求exp的和,第三次应用softmax
// template<typename T>
// T** softmax(T** x, int batch_size, int features) {
//     // 创建结果矩阵
//     T** result = init_matrix<T>(batch_size, features);

//     // 计算softmax，对每个样本的特征进行操作
//     for (int i = 0; i < batch_size; ++i) {
//         // 先找到最大值，用于数值稳定性
//         T max_val = x[i][0];
//         for (int j = 1; j < features; ++j) {
//             if (x[i][j] > max_val) {
//                 max_val = x[i][j];
//             }
//         }
        
//         // 计算每个特征的指数，并累计求和
//         T sum_exp = 0.0f;
//         for (int j = 0; j < features; ++j) {
//             result[i][j] = exp(x[i][j] - max_val); // 防止溢出
//             sum_exp += result[i][j];
//         }
        
//         // 归一化，使得每个样本的特征值加起来为1
//         for (int j = 0; j < features; ++j) {
//             result[i][j] /= sum_exp;
//         }
//     }

//     return result;
// }




// // 硬件执行ANU计算矩阵的layernorm和softmax
// // 该函数使用指令控制微操作循环，使得指令分块计算数量减少 ，减少了生成的指令数量
// // 目前只能测试计算MATRIX_WIDTH行的数据,因为硬件一次只能执行这么多,如果要多次执行,需要管理依赖关系
// int anu_test(int opcode,   // 进行什么ANU操作
//             size_t dim_I,  // input矩阵的I行
//             size_t dim_J,  // input矩阵的J列
//             void * input,  // 输入
//             void * weight, // 权重
//             void * bias,   // 偏置
//             void * output) // 输出
// {
//   // 断言检查
//   assert(dim_I % MATRIX_WIDTH == 0); // I是否能被block整除
//   assert(dim_J % MATRIX_WIDTH == 0); // J是否能被block整除

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO -  ANU test: opcode=%d, dim_I=%d, dim_J=%d\n",
//          opcode, dim_I, dim_J);

//   // 加载分块和计算分块的比值
//   const int load_compute_block_ratio = dim_J / MATRIX_WIDTH; //一列的加载数量

//   // 如果要进行分块计算，行分块系数是脉动阵列大小，列就是计算矩阵的大小
//   const int dim_I_block = dim_I / MATRIX_WIDTH; // 行分多少个块

//   const int tile_I = MATRIX_WIDTH; //行的分块
//   const int tile_J = dim_J; //列的分块系数,直接加载一整列

//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = dim_I_block; // 一次加载MATRIX_WIDTH行
//   int insn_anu_size = dim_I_block; // 一次计算MATRIX_WIDTH行
//   int insn_store_size = dim_I_block; // 一次存储MATRIX_WIDTH行
//   int insn_size = insn_load_size + insn_store_size + insn_anu_size+1;

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 指令生成
//   // 迭代处理输出行块
//   for (int i = 0; i < dim_I_block; i++) {

//     // 加载指令生成
//     // 加载输入input(i*MATRIX_WIDTH+MATRIX_WIDTH,dim_J)
//     int buffer_start = 0; // 加载到输入buffer的起始位置  
//     int dram_start = 0; // 读取输入缓冲区的起始位置      
//     int A_block = i; // 第几个块
//     int buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//     int x_size = MATRIX_WIDTH*dim_J;//加载MATRIX_WIDTH行
//     int dram_offset = (dram_start + (A_block)*x_size); // 计算dram

//     insn_buf[insn_idx++] = get2DLoadStoreInsn(
//           OPCODE_LOAD,     // 加载指令
//           OUTPUT_BUFFER_ID,// 加载到输出Buffer
//           buffer_offset,   // buffer偏移+矩阵内部偏移
//           dram_offset,     // 缓冲区偏移+矩阵内部偏移
//           1,    // 每次加载MATRIX_WIDTH行
//           x_size,    // 每次加载MATRIX_WIDTH列
//           dim_J,
//           0,
//           0,
//           0,
//           0);          // input矩阵总列数作为2D跨步DMA步进,如果只加载一行连续加载,这个就没用了

//     // 计算指令生成
//     const int read_src_base = 0;  //每次计算的是同一个物理地址起始
//     const int write_dst_base = 0; // 写入buffer块的基地址,读取写入同一片地址     
//     const int src_offset =  MATRIX_WIDTH;      // 片上微操作每次的读取块偏移,就是MATRIX_WIDTH行
//     const int dst_offset =  MATRIX_WIDTH;      // 片上微操作每次的写入块偏移

//     // 生成LayerNorm指令
//     if(opcode == ANU_LAYERNORM) 
//     {
//       insn_buf[insn_idx++] = getLayerNormInsn(dim_J, // 写入当前矩阵的列数用于计算均值
//                                               read_src_base, // 读取buffer的基地址
//                                               write_dst_base, // 写入buffer的基地址
//                                               src_offset, // 每次读取块的偏移
//                                               dst_offset); // 每次写入块的偏移
//     }
//     else if(opcode == ANU_SOFTMAX)
//     {
//       insn_buf[insn_idx++] = getSoftmaxInsn(dim_J, // 写入当前矩阵的列数用于计算均值
//                                             read_src_base, // 读取buffer的基地址
//                                             write_dst_base, // 写入buffer的基地址
//                                             src_offset, // 每次读取块的偏移
//                                             dst_offset); // 每次写入块的偏移
//     }

//     // 存储指令行生成
//     buffer_start = 0; // 读取输出buffer的起始位置  
//     dram_start = 0; // 输出缓冲区的起始位置      
//     A_block = i; // 第几个块
//     buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//     x_size = MATRIX_WIDTH*dim_J;//加载MATRIX_WIDTH行
//     dram_offset = (dram_start + (A_block)*x_size); // 计算dram    // 存储输出
//     insn_buf[insn_idx++] = get2DLoadStoreInsn(
//           OPCODE_STORE,     // 存储指令
//           OUTPUT_BUFFER_ID, // 存储到输出buffer
//           buffer_offset,    // buffer偏移+矩阵内部偏移
//           dram_offset,      // 缓冲区偏移+矩阵内部偏移
//           1,     // 每次存储1行
//           x_size,     // 一次存储MATRIX_WIDTH*dim_J的数据
//           dim_J,
//           0,
//           0,
//           0,
//           0);           // output矩阵总列数作为2D跨步DMA步进
//   }

//   // 结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,0);
//   printf("insn_count:%d\n",insn_idx);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) NULL,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);

//   return 0;
// }


// #endif





// 生成2D加载、存储指令，加载存储矩阵块，需要指定opcode是加载还是存储
// GenericIns get2DLoadStoreInsn(int opcode, 
//                               int buffer_id, 
//                               int buffer_offset, 
//                               int dram_offset,
//                               int y_size, 
//                               int x_size, 
//                               int x_stride) 
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 存储指令初始化
//   MemIns insn = {};
//   insn.opcode = opcode;  // 加载指令
//   insn.buffer_id = buffer_id; // 存储在哪个缓冲区
//   insn.dram_base = dram_offset; // 缓冲区偏移+矩阵内部偏移
//   insn.buffer_base = buffer_offset; // buffer偏移+矩阵内部偏移
//   insn.y_size = y_size; // 每次加载MATRIX_WIDTH行
//   insn.x_size = x_size; // 每次加载MATRIX_WIDTH列
//   insn.x_stride = x_stride; // 假设每行数据在DRAM中是连续存储的，那么步长就是列宽
//   converter.mem = insn;

//   return converter.generic;
// }


// // 生成GEMM指令，包括多种计算类型
// // 权重预加载指令，只需要权重相关参数
// GenericIns getWeightPreloadInsn(int weigth_offset, // 权重块加载偏移
//                                 int weight_switch) // 加载到哪个权重位置
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   ComIns insn = {};
//   insn.opcode = OPCODE_COMPUTE;  // 计算指令
//   insn.compute_type = WEIGHT_PRELOAD; // 权重预加载指令
//   insn.input_addr = 0; // 输入读取基地址
//   insn.weigth_addr = weigth_offset; // 权重读取基地址
//   insn.output_addr = 0; // 输出读取基地址
//   insn.weight_switch = weight_switch; // 使用哪一个权重加载
//   insn.compute_switch = 0; // 使用哪一个权重计算
//   insn.accumulate = 0; // 计算结果是否累加
//   converter.com = insn;

//   return converter.generic;
// }

// // 计算指令，只需要读取输入和存储输出，以及选择进行矩阵乘法的寄存器和累加
// GenericIns getComputeInsn(int input_offset, // 权重块加载偏移
//                           int output_offset, // 输出存储偏移
//                           int compute_switch, // 使用哪个寄存器进行计算
//                           int accumulate) // 当前计算是否进行累加
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   ComIns insn = {};
//   insn.opcode = OPCODE_COMPUTE;  // 计算指令
//   insn.compute_type = COMPUTE; // 计算指令
//   insn.input_addr = input_offset; // 输入读取基地址
//   insn.weigth_addr = 0; // 权重读取基地址
//   insn.output_addr = output_offset; // 输出存储基地址
//   insn.weight_switch = 0; // 使用哪一个权重加载
//   insn.compute_switch = compute_switch; // 使用哪一个权重计算
//   insn.accumulate = accumulate; // 计算结果是否累加
//   converter.com = insn;

//   return converter.generic;
// }


// // 计算预加载指令，同时进行计算和预加载，只不过预加载和计算的寄存器在内部就做了乒乓缓冲
// // 计算指令计算的就是参数的寄存器，预加载加载的是另一个寄存器
// // 因此预加载寄存器后，调用该函数继续计算上面那个寄存器，而这个最后执行完，调用计算计算相反寄存器
// GenericIns getWeightPreloadComputeInsn(int input_offset, // 权重块加载偏移
//                                        int weigth_offset, // 权重块加载偏移
//                                        int output_offset, // 输出存储偏移
//                                        int weight_switch, // 加载到哪个权重位置
//                                        int compute_switch, // 使用哪个寄存器进行计算
//                                        int accumulate) // 当前计算是否进行累加
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   ComIns insn = {};
//   insn.opcode = OPCODE_COMPUTE;  // 计算指令
//   insn.compute_type = COMPUTE_WEIGHT_PRELOAD; // 计算预加载指令
//   insn.input_addr = input_offset; // 输入读取基地址
//   insn.weigth_addr = weigth_offset; // 权重读取基地址
//   insn.output_addr = output_offset; // 输出存储基地址
//   insn.weight_switch = weight_switch; // 使用哪一个权重加载
//   insn.compute_switch = compute_switch; // 使用哪一个权重计算
//   insn.accumulate = accumulate; // 计算结果是否累加
//   converter.com = insn;

//   return converter.generic;
// }

// // 块矩阵乘法的指令
// GenericIns getGEMMInsn(int dim_I_block, // 
//                        int dim_J_block, // 用于
//                        int dim_K_block,
//                        int bias_use) // 
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   ComIns insn = {};

//   // GEMM字段
//   insn.opcode = OPCODE_GEMM;  // GEMM指令
//   insn.uop_bgn = 0;  // GEMM指令
//   insn.uop_end = dim_I_block;  // 微操作缓冲区的大小
//   insn.dim_K_block = dim_K_block;  // K循环的次数
//   insn.dim_J_block = dim_J_block;  // J循环的次数
//   insn.bias_use = bias_use;  // 是否使用偏置

//   converter.com = insn;

//   return converter.generic;
// }



// // 生成运算完成指令
// GenericIns getFinishInsn() 
// {
//   // 独联体进行转换
//   union SAAInsn converter;
//   // 计算指令初始化
//   ComIns insn = {};
//   insn.opcode = OPCODE_DONE;  // 完成指令
//   converter.com = insn;

//   return converter.generic;
// }











// // 硬件执行gemm+bias
// // 该函数完成了两级分块的分块矩阵乘法，调用了硬件计算矩阵乘法的函数
// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int tile_I,    // 分块系数，代表分了多少个小块
//                       int tile_J,    // 分块系数，代表分了多少个小块
//                       int tile_K,    // 分块系数，代表分了多少个小块
//                       bool bias_use) // 是否使用偏置  
// {
//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use);

//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = I0*J0*K0; // 计算加载A、B矩阵需要多少大分块指令
//   int insn_compute_size = I0*J0*K0; // 不使用权重复用
//   int insn_store_size = I0*J0*K0; // 输出矩阵大小
//   int insn_size = insn_load_size + insn_store_size + insn_compute_size+1 + 100;

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引


//   // 断言判断加载的矩阵大小小于缓冲区大小

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块的结果一样使用，根据下面I的值判断有效size
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K); // K

//   int uop_size = tile_I; //当前微操作缓冲区内有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // printf("bias_dram_offset:%d,I:%d,J:%d\n",bias_dram_offset,I,J);

//       // 加载偏置biases(dim_I,dim_J)
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,     // buffer为0，因为每次计算都直接加载满
//             bias_dram_offset,     // dram中偏移一个块尝试
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride, // 总矩阵的块的列数作为步进
//             0,     // pop_pre_raw  
//             (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//             0,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//             1);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//       printf("- Generate compute(bias)\n");

//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;
//         // printf("I:%d,J:%d,K:%d\n",I,J,K);

//         // 计算指令参数

//           // // input
//           // const int input_buffer_offset = ;
//           // const int input_dram_offset = ;

//           // // weight
//           // const int weight_buffer_offset = ;
//           // const int weight_dram_offset = ;

//           // 

//         // // 指令生成

//         // // 加载输入input(dim_I,dim_K)
//         // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         //       OPCODE_LOAD,     // 存储指令
//         //       INPUT_BUFFER_ID, // 存储到输出buffer
//         //       0,    // buffer偏移+矩阵内部偏移
//         //       0,      // 缓冲区偏移+矩阵内部偏移
//         //       I,     // 每次加载MATRIX_WIDTH行
//         //       K,     // 每次加载MATRIX_WIDTH列
//         //       dim_K_stride,  // output矩阵的列的分块数作为步进
//         //       0,     // pop_pre_raw  
//         //       0,     // pop_next_war   
//         //       0,     // push_pre_war 
//         //       0);    // push_next_raw  

//         // // 加载权重weight(dim_K,dim_J)
//         // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         //       OPCODE_LOAD,     // 存储指令
//         //       WEIGHT_BUFFER_ID, // 存储到输出buffer
//         //       0,    // buffer偏移+矩阵内部偏移
//         //       0,      // 缓冲区偏移+矩阵内部偏移
//         //       K,     // 每次加载MATRIX_WIDTH行
//         //       J,     // 每次加载MATRIX_WIDTH列
//         //       dim_J_stride, // output矩阵的列的分块数作为步进
//         //       0,     // pop_pre_raw  
//         //       0,     // pop_next_war   
//         //       0,     // push_pre_war 
//         //       0);    // push_next_raw  


//         // // GEMM指令生成
//         // insn_buf[insn_idx++] = getGEMMInsn(
//         //                           I, // I
//         //                           J, // J
//         //                           K, // K
//         //                           bias_use); // 是否使用bias

//       }
//       // // 存储指令生成
//       // // 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于bias加载
//             0,     // pop_next_war   
//             1,     // push_pre_war , 影响bias加载,会写入s2c_war_queue队列
//             0);    // push_next_raw
//       printf("- Generate store\n");
//     }
//   }


//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_count:%d\n",insn_idx);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }




// 二级分块计算
// #include <stdio.h>
// #define N 8  // 矩阵的总大小
// #define BS1 4  // 一级分块大小
// #define BS2 2  // 二级分块大小

// // 初始化矩阵
// void initMatrix(int mat[N][N], int value) {
//     for (int i = 0; i < N; i++)
//         for (int j = 0; j < N; j++)
//             mat[i][j] = value;
// }

// // 打印矩阵
// void printMatrix(int mat[N][N]) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%5d", mat[i][j]);
//         }
//         printf("\n");
//     }
// }

// // 三级分块矩阵乘法
// void blockMatrixMultiply(int A[N][N], int B[N][N], int C[N][N]) {
//     int i, j, k, ii, jj, kk, iii, jjj, kkk;

//     // 初始化结果矩阵为0
//     initMatrix(C, 0);

//     // 一级分块
//     for (ii = 0; ii < N; ii += BS1) {
//         for (jj = 0; jj < N; jj += BS1) {
//             for (kk = 0; kk < N; kk += BS1) {
                
//                 // 二级分块
//                 for (iii = ii; iii < ii + BS1; iii += BS2) {
//                     for (jjj = jj; jjj < jj + BS1; jjj += BS2) {
//                         for (kkk = kk; kkk < kk + BS1; kkk += BS2) {

//                             // 三级分块（实际计算）
//                             for (i = iii; i < iii + BS2; i++) {
//                                 for (j = jjj; j < jjj + BS2; j++) {
//                                     for (k = kkk; k < kkk + BS2; k++) {
//                                         C[i][j] += A[i][k] * B[k][j];
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// int main() {
//     int A[N][N], B[N][N], C[N][N];

//     // 初始化矩阵A和B
//     initMatrix(A, 1);
//     initMatrix(B, 2);

//     // 执行分块矩阵乘法
//     blockMatrixMultiply(A, B, C);

//     // 打印结果
//     printf("Matrix A:\n");
//     printMatrix(A);
//     printf("Matrix B:\n");
//     printMatrix(B);
//     printf("Matrix C (Result):\n");
//     printMatrix(C);

//     return 0;
// }





// // 硬件执行gemm+bias
// // 该函数完成了两级分块的分块矩阵乘法，调用了硬件计算矩阵乘法的函数
// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int tile_I,    // 分块系数，代表分了多少个小块
//                       int tile_J,    // 分块系数，代表分了多少个小块
//                       int tile_K,    // 分块系数，代表分了多少个小块
//                       bool bias_use) // 是否使用偏置  
// {
//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use);

//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = I0*J0*K0; // 计算加载A、B矩阵需要多少大分块指令
//   int insn_compute_size = I0*J0*K0; // 不使用权重复用
//   int insn_store_size = I0*J0*K0; // 输出矩阵大小
//   int insn_size = insn_load_size + insn_store_size + insn_compute_size+1 + 100;

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引


//   // 断言判断加载的矩阵大小小于缓冲区大小

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块的结果一样使用，根据下面I的值判断有效size
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K); // K

//   int uop_size = tile_I; //当前微操作缓冲区内有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 一行
//         uop_size,    // 加载微操作数量
//         0,   // 步进不使用
//         0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//         0,
//         0,
//         0);  
//   printf("- Generate compute(uop)\n");
//   // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//   //       OPCODE_LOAD,     // 加载指令
//   //       UOP_BUFFER_ID,// 加载到微操作缓冲区
//   //       0,   // buffer的0地址处加载
//   //       0,   // dram缓冲区的0地址处加载
//   //       1,   // 一行
//   //       uop_size,    // 加载微操作数量
//   //       0);  

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // printf("bias_dram_offset:%d,I:%d,J:%d\n",bias_dram_offset,I,J);

//       // 加载偏置biases(dim_I,dim_J)
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,     // buffer为0，因为每次计算都直接加载满
//             bias_dram_offset,     // dram中偏移一个块尝试
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride, // 总矩阵的块的列数作为步进
//             0,     // pop_pre_raw  
//             (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//             0,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//             1);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
//       printf("- Generate compute(bias)\n");
//       // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//       //       OPCODE_LOAD,     // 加载指令
//       //       OUTPUT_BUFFER_ID, // 存储到输出buffer
//       //       0,     // buffer为0，因为每次计算都直接加载满
//       //       bias_dram_offset,     // dram中偏移一个块尝试
//       //       I,     // 每次加载I个行块
//       //       J,     // 每次加载J个列块
//       //       dim_J_stride);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成

//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;
//         // printf("I:%d,J:%d,K:%d\n",I,J,K);

//         // 计算指令参数

//           // // input
//           // const int input_buffer_offset = ;
//           // const int input_dram_offset = ;

//           // // weight
//           // const int weight_buffer_offset = ;
//           // const int weight_dram_offset = ;

//           // 

//         // // 指令生成

//         // // 加载输入input(dim_I,dim_K)
//         // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         //       OPCODE_LOAD,     // 存储指令
//         //       INPUT_BUFFER_ID, // 存储到输出buffer
//         //       0,    // buffer偏移+矩阵内部偏移
//         //       0,      // 缓冲区偏移+矩阵内部偏移
//         //       I,     // 每次加载MATRIX_WIDTH行
//         //       K,     // 每次加载MATRIX_WIDTH列
//         //       dim_K_stride,  // output矩阵的列的分块数作为步进
//         //       0,     // pop_pre_raw  
//         //       0,     // pop_next_war   
//         //       0,     // push_pre_war 
//         //       0);    // push_next_raw  

//         // // 加载权重weight(dim_K,dim_J)
//         // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         //       OPCODE_LOAD,     // 存储指令
//         //       WEIGHT_BUFFER_ID, // 存储到输出buffer
//         //       0,    // buffer偏移+矩阵内部偏移
//         //       0,      // 缓冲区偏移+矩阵内部偏移
//         //       K,     // 每次加载MATRIX_WIDTH行
//         //       J,     // 每次加载MATRIX_WIDTH列
//         //       dim_J_stride, // output矩阵的列的分块数作为步进
//         //       0,     // pop_pre_raw  
//         //       0,     // pop_next_war   
//         //       0,     // push_pre_war 
//         //       0);    // push_next_raw  


//         // // GEMM指令生成
//         // insn_buf[insn_idx++] = getGEMMInsn(
//         //                           I, // I
//         //                           J, // J
//         //                           K, // K
//         //                           bias_use); // 是否使用bias

//       }
//       // // 存储指令生成
//       // // 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride,    // 总矩阵的块的列数作为步进
//             1,     // pop_pre_raw  , 依赖于bias加载
//             0,     // pop_next_war   
//             1,     // push_pre_war , 影响bias加载,会写入s2c_war_queue队列
//             0);    // push_next_raw
//       printf("- Generate store\n");

//       // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//       //       OPCODE_STORE,     // 存储指令
//       //       OUTPUT_BUFFER_ID, // 存储到输出buffer
//       //       0,                // buffer偏移+矩阵内部偏移
//       //       bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//       //       I,     // 每次加载I个行块
//       //       J,     // 每次加载J个列块
//       //       dim_J_stride);  
//     }
//   }


//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("- Generate compute(finish)\n");
//   printf("insn_count:%d\n",insn_idx);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }


// // 硬件执行gemm+bias
// // 该函数完成了两级分块的分块矩阵乘法，调用了硬件计算矩阵乘法的函数
// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int tile_I,    // 分块系数，代表分了多少个小块
//                       int tile_J,    // 分块系数，代表分了多少个小块
//                       int tile_K,    // 分块系数，代表分了多少个小块
//                       bool bias_use) // 是否使用偏置  
// {
//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, bias_use\n",
//          dim_I, dim_J, dim_K, tile_I, tile_J, tile_K, bias_use);

//   // 计算需要填充的数量，使用最小的块进行填充
//   // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
//   // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
//   const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
//   const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

//   // 计算填充0大小，后面可以使用硬件或者软件进行填充
//   const int padding_I = dim_I_padded - dim_I;
//   const int padding_J = dim_J_padded - dim_J;
//   const int padding_K = dim_K_padded - dim_K;

//   // 计算填充后的维度总共包含多少个MATRIX_WIDTH
//   const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
//   const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
//   const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

//   // 然后计算二级分块需要循环的次数
//   // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
//   // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
//   const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
//   const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
//   const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

//   // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
//   // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
//   // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
//   // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
//   const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
//   const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
//   const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = I0*J0*K0; // 计算加载A、B矩阵需要多少大分块指令
//   int insn_compute_size = I0*J0*K0; // 不使用权重复用
//   int insn_store_size = I0*J0*K0; // 输出矩阵大小
//   int insn_size = insn_load_size + insn_store_size + insn_compute_size+1 + 100;

//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引


//   // 断言判断加载的矩阵大小小于缓冲区大小

//   // 生成微操作指令,针对完整的大分块，对于不满足大分块的结果一样使用，根据下面I的值判断有效size
//   Uop * uop_buf = getGEMMUops(
//                       tile_I, // I
//                       tile_J, // J
//                       tile_K); // K

//   int uop_size = tile_I; //当前微操作缓冲区内有效的微操作数量

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//   //       OPCODE_LOAD,     // 加载指令
//   //       UOP_BUFFER_ID,// 加载到微操作缓冲区
//   //       0,   // buffer的0地址处加载
//   //       0,   // dram缓冲区的0地址处加载
//   //       1,   // 一行
//   //       uop_size,    // 加载微操作数量
//   //       0,   // 步进不使用
//   //       0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
//   //       0,
//   //       0,
//   //       0);  

//   // 外循环，计算大分块的循环
//   for (int i0 = 0; i0 < I0; i0++) {
//     // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
//     const int I = i0 < I0-1 ? tile_I : last_I;

//     for (int j0 = 0; j0 < J0; j0++) {
//       const int J = j0 < J0-1 ? tile_J : last_J;

//       // bias和output的参数一样
//       // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
//       const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
//       // printf("bias_dram_offset:%d,I:%d,J:%d\n",bias_dram_offset,I,J);

//       // 加载偏置biases(dim_I,dim_J)
//       // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//       //       OPCODE_LOAD,     // 加载指令
//       //       OUTPUT_BUFFER_ID, // 存储到输出buffer
//       //       0,     // buffer为0，因为每次计算都直接加载满
//       //       bias_dram_offset,     // dram中偏移一个块尝试
//       //       I,     // 每次加载I个行块
//       //       J,     // 每次加载J个列块
//       //       dim_J_stride, // 总矩阵的块的列数作为步进
//       //       0,     // pop_pre_raw  
//       //       (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
//       //       0,     // push_pre_war  ，如果有连续的load指令，bias会影响load模块，bias执行完，load才能加载然后gemm才能执行
//       //       1);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成

//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,     // buffer为0，因为每次计算都直接加载满
//             bias_dram_offset,     // dram中偏移一个块尝试
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成

//       for (int k0 = 0; k0 < K0; k0++) {
//         const int K = k0 < K0-1 ? tile_K : last_K;
//         // printf("I:%d,J:%d,K:%d\n",I,J,K);

//         // 计算指令参数

//           // // input
//           // const int input_buffer_offset = ;
//           // const int input_dram_offset = ;

//           // // weight
//           // const int weight_buffer_offset = ;
//           // const int weight_dram_offset = ;

//           // 

//         // 指令生成

//         // // 加载输入input(dim_I,dim_K)
//         // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         //       OPCODE_LOAD,     // 存储指令
//         //       INPUT_BUFFER_ID, // 存储到输出buffer
//         //       0,    // buffer偏移+矩阵内部偏移
//         //       0,      // 缓冲区偏移+矩阵内部偏移
//         //       I,     // 每次加载MATRIX_WIDTH行
//         //       K,     // 每次加载MATRIX_WIDTH列
//         //       dim_K_stride);     // output矩阵的列的分块数作为步进

//         // // 加载权重weight(dim_K,dim_J)
//         // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         //       OPCODE_LOAD,     // 存储指令
//         //       WEIGHT_BUFFER_ID, // 存储到输出buffer
//         //       0,    // buffer偏移+矩阵内部偏移
//         //       0,      // 缓冲区偏移+矩阵内部偏移
//         //       K,     // 每次加载MATRIX_WIDTH行
//         //       J,     // 每次加载MATRIX_WIDTH列
//         //       dim_J_stride);     // output矩阵的列的分块数作为步进


//         // // GEMM指令生成
//         // insn_buf[insn_idx++] = getGEMMInsn(
//         //                           I, // I
//         //                           J, // J
//         //                           K, // K
//         //                           bias_use); // 是否使用bias

//       }
//       // // 存储指令生成
//       // // 存储输出
//       // insn_buf[insn_idx++] = get2DLoadStoreInsn(
//       //       OPCODE_STORE,     // 存储指令
//       //       OUTPUT_BUFFER_ID, // 存储到输出buffer
//       //       0,                // buffer偏移+矩阵内部偏移
//       //       bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//       //       I,     // 每次加载I个行块
//       //       J,     // 每次加载J个列块
//       //       dim_J_stride,    // 总矩阵的块的列数作为步进
//       //       1,     // pop_pre_raw  , 依赖于bias加载
//       //       0,     // pop_next_war   
//       //       1,     // push_pre_war , 影响bias加载,会写入s2c_war_queue队列
//       //       0);    // push_next_raw


//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             0,                // buffer偏移+矩阵内部偏移
//             bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
//             I,     // 每次加载I个行块
//             J,     // 每次加载J个列块
//             dim_J_stride);  
//     }
//   }


//   // 结束所有大分块的计算后，发出结束指令
//   insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
//   printf("insn_count:%d\n",insn_idx);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }



// // 硬件执行gemm+bias
// // 该函数完成了片上的一级分块的计算，和微操作序列执行相比，该函数会花费更多的指令，造成浪费
// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int block,     // 分块系数，将矩阵分为多少块,目前作为加载分块计算
//                       bool bias_use) // 是否使用偏置  
// {
//   // 断言检查
//   assert(block % MATRIX_WIDTH == 0); // 分块系数是否能被MATRIX_WIDTH整除
//   assert(dim_I % block == 0); // I是否能被block整除
//   assert(dim_J % block == 0); // J是否能被block整除
//   assert(dim_K % block == 0); // K是否能被block整除

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, block=%d,bias_use\n",
//          dim_I, dim_J, dim_K, block,bias_use);

//   // 加载分块
//   const int dim_I_load_block = dim_I / block; // 行分多少个块
//   const int dim_J_load_block = dim_J / block; // 列分多少个块
//   const int dim_K_load_block = dim_K / block; // 列分多少个块

//   // 加载分块和计算分块的比值
//   const int load_compute_block_ratio = block / MATRIX_WIDTH; // 行分多少个块

//   // 计算分块
//   const int dim_I_block = dim_I / MATRIX_WIDTH; // 行分多少个块
//   const int dim_J_block = dim_J / MATRIX_WIDTH; // 列分多少个块
//   const int dim_K_block = dim_K / MATRIX_WIDTH; // 列分多少个块


//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block) +(dim_I_block * dim_J_block); // 计算加载A、B矩阵需要多少指令
//   int insn_compute_size = 2*dim_I_block*dim_K_block*dim_J_block; // 不使用权重复用
//   // int insn_compute_size = dim_I_block*dim_K_block*dim_J_block + dim_K_block*dim_J_block; // 使用权重复用
//   // int insn_compute_size = dim_J_block * dim_K_block + (dim_I_block - 1) * dim_J_block * dim_K_block + 1 ; // 使用权重复用和双缓冲
//   int insn_store_size = dim_I_block * dim_J_block; // 输出矩阵大小
//   int insn_size = insn_load_size + insn_store_size + insn_compute_size+1;
  

//   // 断言判断加载的矩阵大小小于缓冲区大小


//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 指令生成

//   // 加载指令生成
//   // 加载偏置biases(dim_I,dim_J)
//   if(bias_use==1 && bias!=NULL) // 如果使用偏置同时传入指针不为空
//   {
//   for (int i = 0; i < dim_I_load_block; i++) { // 迭代行块
//     for (int j = 0; j < dim_J_load_block; j++) { // 迭代列块
//       const int buffer_start = 0; // 加载到输出buffer的起始位置  
//       const int dram_start = 0; // 读取偏置缓冲区的起始位置      
//       const int A_block = i*dim_J_load_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = (dram_start + (A_block)*(block*block)); // 计算dram
//       const int x_size = block*block;//连续加载
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             OUTPUT_BUFFER_ID, // 加载到输出Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             1,    // 每次加载MATRIX_WIDTH行
//             x_size,    // 每次加载MATRIX_WIDTH列
//             dim_J);          // input矩阵总列数作为2D跨步DMA步进,如果只加载一行连续加载,这个就没用了
//       }
//     }
//   }

//   // 加载输入input(dim_I,dim_K)
//   for (int i = 0; i < dim_I_load_block; i++) { // 迭代行块
//     for (int k = 0; k < dim_K_load_block; k++) { // 迭代列块
//       const int buffer_start = 0; // 加载到输入buffer的起始位置  
//       const int dram_start = 0; // 读取输入缓冲区的起始位置      
//       const int A_block = i*dim_K_load_block+k; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = (dram_start + (A_block)*(block*block)); // 计算dram
//       const int x_size = block*block;//连续加载
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             INPUT_BUFFER_ID, // 加载到输入Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             1,    // 每次加载MATRIX_WIDTH行
//             x_size,    // 每次加载MATRIX_WIDTH列
//             dim_K);          // input矩阵总列数作为2D跨步DMA步进,如果只加载一行连续加载,这个就没用了
//       }
//     }

//   // 加载权重weight(dim_K,dim_J)
//   for (int k = 0; k < dim_K_load_block; k++) { // 迭代行块
//     for (int j = 0; j < dim_J_load_block; j++) { // 迭代列块
//       const int buffer_start = 0; // 加载到权重buffer的起始位置  
//       const int dram_start = 0; // 读取权重缓冲区的起始位置      
//       const int A_block = k*dim_J_load_block+j; // 第几个块
//       const int buffer_offset = (buffer_start +  (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = (dram_start + (A_block)*(block*block)); // 计算dram读取地址
//       const int x_size = block*block;//连续加载
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             WEIGHT_BUFFER_ID,// 加载到权重Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             1,    // 每次加载MATRIX_WIDTH行
//             x_size,    // 每次加载MATRIX_WIDTH列
//             dim_J);          // weight矩阵总列数作为2D跨步DMA步进
//       }
//     }

//   // 生成微操作指令
//   Uop * uop_buf = getGEMMUops(
//                       dim_I_block, // I
//                       dim_J_block, // J
//                       dim_K_block); // K

//   int uop_size = dim_I_block; //微操作缓冲区的大小是dim_I_block

//   // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
//   insn_buf[insn_idx++] = get2DLoadStoreInsn(
//         OPCODE_LOAD,     // 加载指令
//         UOP_BUFFER_ID,// 加载到微操作缓冲区
//         0,   // buffer的0地址处加载
//         0,   // dram缓冲区的0地址处加载
//         1,   // 每次加载MATRIX_WIDTH行
//         uop_size,    // 每次加载MATRIX_WIDTH列
//         0);  // 步进不使用


//   // GEMM指令生成
//   insn_buf[insn_idx++] = getGEMMInsn(
//                             dim_I_block, // I
//                             dim_J_block, // J
//                             dim_K_block); // K

//   for (int i = 0; i < dim_I_block; i++) {
// 	printf("input_idx:%d\n",(int)uop_buf[i].input_idx);
// 	printf("output_idx:%d\n",(int)uop_buf[i].output_idx);
//   }




//   // 存储指令生成
//   for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//       const int buffer_start = 0; // 读取输出buffer的起始位置  
//       const int dram_start = 0; // 输出缓冲区的起始位置      
//       const int A_block = i*dim_J_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*MATRIX_WIDTH); // 计算buffer读取地址(根据按行存储特点计算)
//       const int dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                         + j * MATRIX_WIDTH; // 计算dram存储地址(根据二维连续数组存储特点计算)
//       // 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             buffer_offset,    // buffer偏移+矩阵内部偏移
//             dram_offset,      // 缓冲区偏移+矩阵内部偏移
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH行
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH列
//             dim_J);           // output矩阵总列数作为2D跨步DMA步进
//     }
//   }

//   // 结束指令
//   insn_buf[insn_idx++] = getFinishInsn();
//   printf("insn_count:%d\n",insn_idx);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) uop_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }


// // 硬件执行gemm+bias
// // 该函数完成了片上的一级分块的计算，和微操作序列执行相比，该函数会花费更多的指令，造成浪费
// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int block,     // 分块系数，将矩阵分为多少块,目前作为加载分块计算
//                       bool bias_use) // 是否使用偏置  
// {
//   // 断言检查
//   assert(block % MATRIX_WIDTH == 0); // 分块系数是否能被MATRIX_WIDTH整除
//   assert(dim_I % block == 0); // I是否能被block整除
//   assert(dim_J % block == 0); // J是否能被block整除
//   assert(dim_K % block == 0); // K是否能被block整除

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, block=%d,bias_use\n",
//          dim_I, dim_J, dim_K, block,bias_use);

//   // 加载分块
//   const int dim_I_load_block = dim_I / block; // 行分多少个块
//   const int dim_J_load_block = dim_J / block; // 列分多少个块
//   const int dim_K_load_block = dim_K / block; // 列分多少个块

//   // 加载分块和计算分块的比值
//   const int load_compute_block_ratio = block / MATRIX_WIDTH; // 行分多少个块

//   // 计算分块
//   const int dim_I_block = dim_I / MATRIX_WIDTH; // 行分多少个块
//   const int dim_J_block = dim_J / MATRIX_WIDTH; // 列分多少个块
//   const int dim_K_block = dim_K / MATRIX_WIDTH; // 列分多少个块


//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block) +(dim_I_block * dim_J_block); // 计算加载A、B矩阵需要多少指令
//   int insn_compute_size = 2*dim_I_block*dim_K_block*dim_J_block; // 不使用权重复用
//   // int insn_compute_size = dim_I_block*dim_K_block*dim_J_block + dim_K_block*dim_J_block; // 使用权重复用
//   // int insn_compute_size = dim_J_block * dim_K_block + (dim_I_block - 1) * dim_J_block * dim_K_block + 1 ; // 使用权重复用和双缓冲
//   int insn_store_size = dim_I_block * dim_J_block; // 输出矩阵大小
//   int insn_size = insn_load_size + insn_store_size + insn_compute_size+1;
  

//   // 断言判断加载的矩阵大小小于缓冲区大小


//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 指令生成

//   // 加载指令生成
//   // 加载偏置biases(dim_I,dim_J)
//   if(bias_use==1 && bias!=NULL) // 如果使用偏置同时传入指针不为空
//   {
//   for (int i = 0; i < dim_I_load_block; i++) { // 迭代行块
//     for (int j = 0; j < dim_J_load_block; j++) { // 迭代列块
//       const int buffer_start = 0; // 加载到输出buffer的起始位置  
//       const int dram_start = 0; // 读取偏置缓冲区的起始位置      
//       const int A_block = i*dim_J_load_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = (dram_start + (A_block)*(block*block)); // 计算dram
//       const int x_size = block*block;//连续加载
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             OUTPUT_BUFFER_ID, // 加载到输出Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             1,    // 每次加载MATRIX_WIDTH行
//             x_size,    // 每次加载MATRIX_WIDTH列
//             dim_J);          // input矩阵总列数作为2D跨步DMA步进,如果只加载一行连续加载,这个就没用了
//       }
//     }
//   }

//   // 加载输入input(dim_I,dim_K)
//   for (int i = 0; i < dim_I_load_block; i++) { // 迭代行块
//     for (int k = 0; k < dim_K_load_block; k++) { // 迭代列块
//       const int buffer_start = 0; // 加载到输入buffer的起始位置  
//       const int dram_start = 0; // 读取输入缓冲区的起始位置      
//       const int A_block = i*dim_K_load_block+k; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = (dram_start + (A_block)*(block*block)); // 计算dram
//       const int x_size = block*block;//连续加载
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             INPUT_BUFFER_ID, // 加载到输入Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             1,    // 每次加载MATRIX_WIDTH行
//             x_size,    // 每次加载MATRIX_WIDTH列
//             dim_K);          // input矩阵总列数作为2D跨步DMA步进,如果只加载一行连续加载,这个就没用了
//       }
//     }

//   // 加载权重weight(dim_K,dim_J)
//   for (int k = 0; k < dim_K_load_block; k++) { // 迭代行块
//     for (int j = 0; j < dim_J_load_block; j++) { // 迭代列块
//       const int buffer_start = 0; // 加载到权重buffer的起始位置  
//       const int dram_start = 0; // 读取权重缓冲区的起始位置      
//       const int A_block = k*dim_J_load_block+j; // 第几个块
//       const int buffer_offset = (buffer_start +  (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = (dram_start + (A_block)*(block*block)); // 计算dram读取地址
//       const int x_size = block*block;//连续加载
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             WEIGHT_BUFFER_ID,// 加载到权重Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             1,    // 每次加载MATRIX_WIDTH行
//             x_size,    // 每次加载MATRIX_WIDTH列
//             dim_J);          // weight矩阵总列数作为2D跨步DMA步进
//       }
//     }

//   // 使用双缓冲，调整循环和加载指令进行权重复用
//   int compute_count = insn_idx;
//   bool pingpang = 0; // 用于切换权重寄存器,最先使用weight1
//   const int wb_start_addr = 0;
//   const int input_start_addr = 0;
//   const int output_start_addr = 0;
//   bool accumulate = 0;
//   bool accumulate_delay = 0;

//   // 偏移量
//   int output_offset = 0;
//   int input_offset = 0;
//   int weigth_offset = 0;

//   // 模拟仿射
//   int output_offset_i[dim_I_block]; // i循环能够计算的数据
//   int input_offset_i[dim_I_block];
//   for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块,计算不变量
//       output_offset_i[i] = i*dim_J_block*MATRIX_WIDTH; // 这是每个i循环都一样的东西
//       input_offset_i[i] = i*dim_K_block*MATRIX_WIDTH; 
//   }

//   // 
//   int output_idx = 0;
//   int input_idx = 0;
//   int weigth_idx = 0;

//   // 第一个权重加载
//   insn_buf[insn_idx++] = getWeightPreloadInsn(
//                           wb_start_addr, // 从Buffer中0行加载第一个权重块
//                           pingpang);     // 使用初始pingpang权重寄存器

//   for (int k = 0; k < dim_K_block; k++) { // 迭代公共维度块
//     accumulate = (bias_use==1)? 1 :(k == 0 ? 0 : 1); // 如果是第一个块，刷新累加器中的累加值,其他块进行累加,如果使用了biase，那么全部累加
//     output_offset = 0; // 在j循环外初始化
    
//     for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//       // 将剩下的权重加载完，可以进行双缓冲（最后计算还会剩下一个权重使用getComputeInsn计算）
//       if(k != 0 || j != 0){
//         // 替换为两个指令执行
//         insn_buf[insn_idx++] = getWeightPreloadComputeInsn(
//                                 input_idx, // 权重块加载偏移
//                                 weigth_idx, // 权重块加载偏移
//                                 output_idx, // 输出存储偏移
//                                 pingpang, // （这个指令中没有用，加载的是计算寄存器的相反寄存器）
//                                 pingpang, // 计算寄存器依然是当前寄存器
//                                 accumulate_delay); // 当前计算是否进行累加
//         pingpang = !pingpang; // 计算完成后，切换加载寄存器和计算寄存器
//       }    
//       for (int i = 0; i < dim_I_block -1 ; i++) { // 迭代输出行块
//           accumulate_delay = accumulate;// 延迟一位
//           output_idx=output_offset_i[i] + output_offset;
//           input_idx=input_offset_i[i] +  input_offset;
//           insn_buf[insn_idx++] =  getComputeInsn(
//                                   input_idx,  // 权重块加载偏移
//                                   output_idx, // 输出存储偏移
//                                   pingpang,      // i循环内不需要切换寄存器进行计算
//                                   accumulate);   // 当前计算是否在输出块进行累加
//         }
//         // 计算最后一行的索引,最后一行可以
//         output_idx=output_offset_i[dim_I_block-1] + output_offset;
//         input_idx=input_offset_i[dim_I_block-1] +  input_offset; // 这是当前最后一个块
//         output_offset += MATRIX_WIDTH; // 在前面计算块的基础上,循环J+1,使得可以提前预加载下一个权重块
//         weigth_idx = weigth_offset + output_offset; // 这是当前计算块的下一个权重块.这四句完成了计算当前块,预加载下一个块
//       }
//       weigth_offset += dim_K_block*MATRIX_WIDTH;
//       input_offset += MATRIX_WIDTH;
//     }

//   // 最后一个计算指令
//   insn_buf[insn_idx++] =  getComputeInsn(
//                           input_idx,  // 最后一个权重块的输入偏移
//                           output_idx, // 最后一个权重块的输出偏移
//                           pingpang,      // 最后一个计算使用当前寄存器
//                           accumulate);   // 最后一个计算是否在输出块进行累加

//     compute_count = insn_idx - compute_count;
//     printf("compute_count:%d\n",compute_count);

//   // 存储指令生成
//   for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//       const int buffer_start = 0; // 读取输出buffer的起始位置  
//       const int dram_start = 0; // 输出缓冲区的起始位置      
//       const int A_block = i*dim_J_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*MATRIX_WIDTH); // 计算buffer读取地址(根据按行存储特点计算)
//       const int dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                         + j * MATRIX_WIDTH; // 计算dram存储地址(根据二维连续数组存储特点计算)
//       // 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             buffer_offset,    // buffer偏移+矩阵内部偏移
//             dram_offset,      // 缓冲区偏移+矩阵内部偏移
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH行
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH列
//             dim_J);           // output矩阵总列数作为2D跨步DMA步进
//     }
//   }

//   // 结束指令
//   insn_buf[insn_idx++] = getFinishInsn();
//   printf("insn_count:%d\n",insn_idx);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) NULL,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }




// // 硬件执行gemm+bias
// // 该函数完成了片上的一级分块的计算，和微操作序列执行相比，该函数会花费更多的指令，造成浪费
// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int block,     // 分块系数，将矩阵分为多少块,目前作为加载分块计算
//                       bool bias_use) // 是否使用偏置  
// {
//   // 断言检查
//   assert(block % MATRIX_WIDTH == 0); // 分块系数是否能被MATRIX_WIDTH整除
//   assert(dim_I % block == 0); // I是否能被block整除
//   assert(dim_J % block == 0); // J是否能被block整除
//   assert(dim_K % block == 0); // K是否能被block整除

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, block=%d,bias_use\n",
//          dim_I, dim_J, dim_K, block,bias_use);

//   // 加载分块
//   const int dim_I_load_block = dim_I / block; // 行分多少个块
//   const int dim_J_load_block = dim_J / block; // 列分多少个块
//   const int dim_K_load_block = dim_K / block; // 列分多少个块

//   // 加载分块和计算分块的比值
//   const int load_compute_block_ratio = block / MATRIX_WIDTH; // 行分多少个块

//   // 计算分块
//   const int dim_I_block = dim_I / MATRIX_WIDTH; // 行分多少个块
//   const int dim_J_block = dim_J / MATRIX_WIDTH; // 列分多少个块
//   const int dim_K_block = dim_K / MATRIX_WIDTH; // 列分多少个块


//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block) +(dim_I_block * dim_J_block); // 计算加载A、B矩阵需要多少指令
//   int insn_compute_size = 2*dim_I_block*dim_K_block*dim_J_block; // 不使用权重复用
//   // int insn_compute_size = dim_I_block*dim_K_block*dim_J_block + dim_K_block*dim_J_block; // 使用权重复用
//   // int insn_compute_size = dim_J_block * dim_K_block + (dim_I_block - 1) * dim_J_block * dim_K_block + 1 ; // 使用权重复用和双缓冲
//   int insn_store_size = dim_I_block * dim_J_block; // 输出矩阵大小
//   int insn_size = insn_load_size + insn_store_size + insn_compute_size+1;
  

//   // 断言判断加载的矩阵大小小于缓冲区大小


//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 指令生成

//   // 加载指令生成
//   // 加载偏置biases(dim_I,dim_J)
//   if(bias_use==1 && bias!=NULL) // 如果使用偏置同时传入指针不为空
//   {
//   for (int i = 0; i < dim_I_load_block; i++) { // 迭代行块
//     for (int j = 0; j < dim_J_load_block; j++) { // 迭代列块
//       const int buffer_start = 0; // 加载到输出buffer的起始位置  
//       const int dram_start = 0; // 读取偏置缓冲区的起始位置      
//       const int A_block = i*dim_J_load_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = (dram_start + (A_block)*(block*block)); // 计算dram
//       const int x_size = block*block;//连续加载
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             OUTPUT_BUFFER_ID, // 加载到输出Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             1,    // 每次加载MATRIX_WIDTH行
//             x_size,    // 每次加载MATRIX_WIDTH列
//             dim_J);          // input矩阵总列数作为2D跨步DMA步进,如果只加载一行连续加载,这个就没用了
//       }
//     }
//   }

//   // 加载输入input(dim_I,dim_K)
//   for (int i = 0; i < dim_I_load_block; i++) { // 迭代行块
//     for (int k = 0; k < dim_K_load_block; k++) { // 迭代列块
//       const int buffer_start = 0; // 加载到输入buffer的起始位置  
//       const int dram_start = 0; // 读取输入缓冲区的起始位置      
//       const int A_block = i*dim_K_load_block+k; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = (dram_start + (A_block)*(block*block)); // 计算dram
//       const int x_size = block*block;//连续加载
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             INPUT_BUFFER_ID, // 加载到输入Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             1,    // 每次加载MATRIX_WIDTH行
//             x_size,    // 每次加载MATRIX_WIDTH列
//             dim_K);          // input矩阵总列数作为2D跨步DMA步进,如果只加载一行连续加载,这个就没用了
//       }
//     }

//   // 加载权重weight(dim_K,dim_J)
//   for (int k = 0; k < dim_K_load_block; k++) { // 迭代行块
//     for (int j = 0; j < dim_J_load_block; j++) { // 迭代列块
//       const int buffer_start = 0; // 加载到权重buffer的起始位置  
//       const int dram_start = 0; // 读取权重缓冲区的起始位置      
//       const int A_block = k*dim_J_load_block+j; // 第几个块
//       const int buffer_offset = (buffer_start +  (A_block)*(load_compute_block_ratio*load_compute_block_ratio)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = (dram_start + (A_block)*(block*block)); // 计算dram读取地址
//       const int x_size = block*block;//连续加载
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             WEIGHT_BUFFER_ID,// 加载到权重Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             1,    // 每次加载MATRIX_WIDTH行
//             x_size,    // 每次加载MATRIX_WIDTH列
//             dim_J);          // weight矩阵总列数作为2D跨步DMA步进
//       }
//     }

//   // 使用双缓冲，调整循环和加载指令进行权重复用
//   int compute_count = insn_idx;
//   bool pingpang = 0; // 用于切换权重寄存器,最先使用weight1
//   const int wb_start_addr = 0;
//   const int input_start_addr = 0;
//   const int output_start_addr = 0;
//   int output_offset = 0;
//   int input_offset = 0;
//   bool accumulate = 0;
//   bool accumulate_delay = 0;

//   for (int k = 0; k < dim_K_block; k++) { // 迭代公共维度块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//       const int weigth_offset = wb_start_addr + (k*dim_J_block + j)*MATRIX_WIDTH;   
//       accumulate = (bias_use==1)? 1 :(k == 0 ? 0 : 1); // 如果是第一个块，刷新累加器中的累加值,其他块进行累加,如果使用了biase，那么全部累加

//       // 第一次加载权重,也就是第一个权重块，加载到最初的pingpang处，该加载无法双缓冲
//       if(k == 0 && j == 0) 
//       {
//         insn_buf[insn_idx++] = getWeightPreloadInsn(
//                                       weigth_offset, // 从Buffer中0行加载权重块
//                                       pingpang);     // 使用初始pingpang权重寄存器
//       }
//       // 将剩下的权重加载完，可以进行双缓冲（最后计算还会剩下一个权重使用getComputeInsn计算）
//       else{
//         // 替换为两个指令执行
//         insn_buf[insn_idx++] = getWeightPreloadComputeInsn(
//                                        input_offset, // 权重块加载偏移
//                                        weigth_offset, // 权重块加载偏移
//                                        output_offset, // 输出存储偏移
//                                        pingpang, // （这个指令中没有用，加载的是计算寄存器的相反寄存器）
//                                        pingpang, // 计算寄存器依然是当前寄存器
//                                        accumulate_delay); // 当前计算是否进行累加
//         pingpang = !pingpang; // 计算完成后，切换加载寄存器和计算寄存器
//       }    
//       for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//           output_offset = output_start_addr + (i*dim_J_block + j)*MATRIX_WIDTH; // 计算输出地址，按照行块计算
//           input_offset = input_start_addr + (i*dim_K_block + k)*MATRIX_WIDTH;
//           accumulate_delay = accumulate;// 延迟一位
//           // 每次最后一个计算都由加载计算指令完成，因此最后一个i应该是不需要的，算出来的地址直接给上面的指令执行

//           if(i != dim_I_block -1) // 如果计算完成第一个权重，在计算最后一个权重时，换成使用getWeightPreloadComputeInsn计算
//           {
//             insn_buf[insn_idx++] =  getComputeInsn(
//                                     input_offset,  // 权重块加载偏移
//                                     output_offset, // 输出存储偏移
//                                     pingpang,      // i循环内不需要切换寄存器进行计算
//                                     accumulate);   // 当前计算是否在输出块进行累加
//           }
//           if(i == dim_I_block - 1 && j == dim_J_block - 1 && k == dim_K_block - 1) // 如果是最后一个权重块，使用当前寄存器进行计算
//           {
//             insn_buf[insn_idx++] =  getComputeInsn(
//                                     input_offset,  // 权重块加载偏移
//                                     output_offset, // 输出存储偏移
//                                     pingpang,      // i循环内不需要切换寄存器进行计算
//                                     accumulate);   // 当前计算是否在输出块进行累加
//           }
//         }
//       }
//     }
//     compute_count = insn_idx - compute_count;
//     printf("compute_count:%d\n",compute_count);

//   // 存储指令生成
//   for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//       const int buffer_start = 0; // 读取输出buffer的起始位置  
//       const int dram_start = 0; // 输出缓冲区的起始位置      
//       const int A_block = i*dim_J_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*MATRIX_WIDTH); // 计算buffer读取地址(根据按行存储特点计算)
//       const int dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                         + j * MATRIX_WIDTH; // 计算dram存储地址(根据二维连续数组存储特点计算)
//       // 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             buffer_offset,    // buffer偏移+矩阵内部偏移
//             dram_offset,      // 缓冲区偏移+矩阵内部偏移
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH行
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH列
//             dim_J);           // output矩阵总列数作为2D跨步DMA步进
//     }
//   }

//   // 结束指令
//   insn_buf[insn_idx++] = getFinishInsn();
//   printf("insn_count:%d\n",insn_idx);

//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_idx,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Uop_DataType *) NULL,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)bias, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }



// gemm指令，使用指令进行分块矩阵乘法，内部直接内置了SAA的运行函数，可以直接调用完成指令生成和计算

// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int block,     // 分块系数，将矩阵分为多少块 
//                       bool bias_use) // 是否使用偏置  
// {
//   // 断言检查
//   // assert(block % MATRIX_WIDTH == 0); // 分块系数是否能被MATRIX_WIDTH整除
//   // assert(dim_I % block == 0); // I是否能被block整除
//   // assert(dim_J % block == 0); // J是否能被block整除
//   // assert(dim_K % block == 0); // K是否能被block整除

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, block=%d,bias_use\n",
//          dim_I, dim_J, dim_K, block,bias_use);

//   // 计算分块
//   const int dim_I_block = dim_I / MATRIX_WIDTH; // 行分多少个块
//   const int dim_J_block = dim_J / MATRIX_WIDTH; // 列分多少个块
//   const int dim_K_block = dim_K / MATRIX_WIDTH; // 列分多少个块


//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block); // 计算加载A、B矩阵需要多少指令
//   int insn_compute_size = 2*dim_I_block*dim_K_block*dim_J_block; // 不使用权重复用
//   // int insn_compute_size = dim_I_block*dim_K_block*dim_J_block + dim_K_block*dim_J_block; // 使用权重复用
//   // int insn_compute_size = dim_I_block*dim_K_block*dim_J_block + dim_K_block*dim_J_block; // 使用权重复用和双缓冲
//   int insn_store_size = dim_I_block * dim_J_block; // 输出矩阵大小
//   int insn_size = insn_load_size + insn_store_size + insn_compute_size+1;
  

//   // 断言判断加载的矩阵大小小于缓冲区大小


//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 指令生成

//   // 加载指令生成
//   // 加载输入input(dim_I,dim_K)
//   for (int i = 0; i < dim_I_block; i++) { // 迭代行块
//     for (int k = 0; k < dim_K_block; k++) { // 迭代列块
//       const int buffer_start = 0; // 加载到输入buffer的起始位置  
//       const int dram_start = 0; // 读取输入缓冲区的起始位置      
//       const int A_block = i*dim_K_block+k; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = dram_start + i * dim_K_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                         + k * MATRIX_WIDTH; // 计算dram读取地址
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             INPUT_BUFFER_ID, // 加载到输入Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             MATRIX_WIDTH,    // 每次加载MATRIX_WIDTH行
//             MATRIX_WIDTH,    // 每次加载MATRIX_WIDTH列
//             dim_K);          // input矩阵总列数作为2D跨步DMA步进
//       }
//     }

//   // 加载权重weight(dim_K,dim_J)
//   for (int k = 0; k < dim_K_block; k++) { // 迭代行块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代列块
//       const int buffer_start = 0; // 加载到权重buffer的起始位置  
//       const int dram_start = 0; // 读取权重缓冲区的起始位置      
//       const int A_block = k*dim_J_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = dram_start + k * dim_J_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                         + j * MATRIX_WIDTH; // 计算dram读取地址
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             WEIGHT_BUFFER_ID,// 加载到权重Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             MATRIX_WIDTH,    // 每次加载MATRIX_WIDTH行
//             MATRIX_WIDTH,    // 每次加载MATRIX_WIDTH列
//             dim_J);          // weight矩阵总列数作为2D跨步DMA步进
//       }
//     }



//   // 计算指令生成

//   // // 不使用双缓冲和权重复用
//   // int compute_count = insn_idx;
//   // const int wb_start_addr = 0;
//   // const int input_start_addr = 0;
//   // const int output_start_addr = 0;
//   // for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//   //   for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//   //     const int output_offset = output_start_addr + (i*dim_J_block + j)*MATRIX_WIDTH; // 计算输出地址，按照行块计算

//   //     for (int k = 0; k < dim_K_block; k++) { // 迭代公共维度块
//   //       const int input_offset = input_start_addr + (i*dim_K_block + k)*MATRIX_WIDTH;
//   //       const int weigth_offset = wb_start_addr + (k*dim_J_block + j)*MATRIX_WIDTH;
//   //       bool accumulate = k == 0 ? 0 : 1; // 如果是第一个块，刷新累加器中的累加值,其他块进行累加
//   //       // 加载寄存器0
//   //       insn_buf[insn_idx++] = getWeightPreloadInsn(
//   //                                   weigth_offset, // 从Buffer中0行加载权重块
//   //                                   0);            // 使用第一个权重寄存器
//   //       //计算寄存器0
//   //       insn_buf[insn_idx++] =  getComputeInsn(
//   //                                 input_offset,  // 权重块加载偏移
//   //                                 output_offset, // 输出存储偏移
//   //                                 0,             // 使用哪个寄存器进行计算
//   //                                 accumulate);   // 当前计算是否在输出块进行累加
//   //     }
//   //   }
//   // }
//   // compute_count = insn_idx - compute_count;
//   // printf("compute_count:%d\n",compute_count);

//   // // 不使用双缓冲，调整循环和加载指令进行权重复用
//   // int compute_count = insn_idx;
//   // const int wb_start_addr = 0;
//   // const int input_start_addr = 0;
//   // const int output_start_addr = 0;
//   // for (int k = 0; k < dim_K_block; k++) { // 迭代公共维度块
//   //   for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//   //     const int weigth_offset = wb_start_addr + (k*dim_J_block + j)*MATRIX_WIDTH;
//   //     // 加载寄存器0
//   //     insn_buf[insn_idx++] = getWeightPreloadInsn(
//   //                                 weigth_offset, // 从Buffer中0行加载权重块
//   //                                 0);            // 使用第一个权重寄存器
//   //     for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//   //         const int output_offset = output_start_addr + (i*dim_J_block + j)*MATRIX_WIDTH; // 计算输出地址，按照行块计算
//   //         const int input_offset = input_start_addr + (i*dim_K_block + k)*MATRIX_WIDTH;
//   //         bool accumulate = k == 0 ? 0 : 1; // 如果是第一个块，刷新累加器中的累加值,其他块进行累加

//   //         //计算寄存器0
//   //         insn_buf[insn_idx++] =  getComputeInsn(
//   //                                   input_offset,  // 权重块加载偏移
//   //                                   output_offset, // 输出存储偏移
//   //                                   0,             // 使用哪个寄存器进行计算
//   //                                   accumulate);   // 当前计算是否在输出块进行累加
//   //       }
//   //     }
//   //   }
//   //   compute_count = insn_idx - compute_count;
//   //   printf("compute_count:%d\n",compute_count);


//   // 使用双缓冲，调整循环和加载指令进行权重复用
//   int compute_count = insn_idx;
//   bool pingpang = 0; // 用于切换权重寄存器,最先使用weight1
//   const int wb_start_addr = 0;
//   const int input_start_addr = 0;
//   const int output_start_addr = 0;
//   int output_offset = 0;
//   int input_offset = 0;
//   bool accumulate = 0;
//   bool accumulate_delay = 0;
//   for (int k = 0; k < dim_K_block; k++) { // 迭代公共维度块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//       const int weigth_offset = wb_start_addr + (k*dim_J_block + j)*MATRIX_WIDTH;   
//       accumulate = k == 0 ? 0 : 1; // 如果是第一个块，刷新累加器中的累加值,其他块进行累加

//       // 第一次加载权重,也就是第一个权重块，加载到最初的pingpang处，该加载无法双缓冲
//       if(k == 0 && j == 0) 
//       {
//         insn_buf[insn_idx++] = getWeightPreloadInsn(
//                                       weigth_offset, // 从Buffer中0行加载权重块
//                                       pingpang);     // 使用初始pingpang权重寄存器
//       }
//       // 将剩下的权重加载完，可以进行双缓冲（最后计算还会剩下一个权重使用getComputeInsn计算）
//       else{
//         insn_buf[insn_idx++] = getWeightPreloadComputeInsn(
//                                        input_offset, // 权重块加载偏移
//                                        weigth_offset, // 权重块加载偏移
//                                        output_offset, // 输出存储偏移
//                                        pingpang, // （这个指令中没有用，加载的是计算寄存器的相反寄存器）
//                                        pingpang, // 计算寄存器依然是当前寄存器
//                                        accumulate_delay); // 当前计算是否进行累加
//         pingpang = !pingpang; // 计算完成后，切换加载寄存器和计算寄存器
//       }    
//       for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//           output_offset = output_start_addr + (i*dim_J_block + j)*MATRIX_WIDTH; // 计算输出地址，按照行块计算
//           input_offset = input_start_addr + (i*dim_K_block + k)*MATRIX_WIDTH;
//           accumulate_delay = accumulate;
//           // 每次最后一个计算都由加载计算指令完成，因此最后一个i应该是不需要的，算出来的地址直接给上面的指令执行

//           if(i != dim_I_block -1) // 如果计算完成第一个权重，在计算最后一个权重时，换成使用getWeightPreloadComputeInsn计算
//           {
//             insn_buf[insn_idx++] =  getComputeInsn(
//                                     input_offset,  // 权重块加载偏移
//                                     output_offset, // 输出存储偏移
//                                     pingpang,      // i循环内不需要切换寄存器进行计算
//                                     accumulate);   // 当前计算是否在输出块进行累加
//           }
//           if(i == dim_I_block - 1 && j == dim_J_block - 1 && k == dim_K_block - 1) // 如果是最后一个权重块，使用当前寄存器进行计算
//           {
//             insn_buf[insn_idx++] =  getComputeInsn(
//                                     input_offset,  // 权重块加载偏移
//                                     output_offset, // 输出存储偏移
//                                     pingpang,      // i循环内不需要切换寄存器进行计算
//                                     accumulate);   // 当前计算是否在输出块进行累加
//           }
//         }
//       }
//     }
//     compute_count = insn_idx - compute_count;
//     printf("compute_count:%d\n",compute_count);


//   // // 使用双缓冲，调整循环和加载指令进行权重复用
//   // int compute_count = insn_idx;
//   // bool pingpang = 0; // 用于切换权重寄存器,最先使用weight1
//   // const int wb_start_addr = 0;
//   // const int input_start_addr = 0;
//   // const int output_start_addr = 0;
//   // int output_offset = 0;
//   // int input_offset = 0;
//   // bool accumulate = 0;
//   // for (int k = 0; k < dim_K_block; k++) { // 迭代公共维度块
//   //   for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//   //     const int weigth_offset = wb_start_addr + (k*dim_J_block + j)*MATRIX_WIDTH;   
//   //     accumulate = k == 0 ? 0 : 1; // 如果是第一个块，刷新累加器中的累加值,其他块进行累加

//   //     // 第一次加载权重,也就是第一个权重块，加载到最初的pingpang处，该加载无法双缓冲
//   //     if(k == 0 && j == 0) 
//   //     {
//   //       insn_buf[insn_idx++] = getWeightPreloadInsn(
//   //                                     weigth_offset, // 从Buffer中0行加载权重块
//   //                                     pingpang);     // 使用初始pingpang权重寄存器
//   //     }
//   //     // 将剩下的权重加载完，可以进行双缓冲（最后计算还会剩下一个权重使用getComputeInsn计算）
//   //     else{
//   //       // 替换为两个指令执行
//   //       insn_buf[insn_idx++] = getWeightPreloadInsn(
//   //                                     weigth_offset, // 从Buffer中0行加载权重块
//   //                                     !pingpang);     // 使用初始pingpang权重寄存器
//   //       insn_buf[insn_idx++] =  getComputeInsn(
//   //                               input_offset,  // 权重块加载偏移
//   //                               output_offset, // 输出存储偏移
//   //                               pingpang,      // i循环内不需要切换寄存器进行计算
//   //                               0);   // 当前计算是否在输出块进行累加
//   //       pingpang = !pingpang; // 计算完成后，切换加载寄存器和计算寄存器
//   //     }    
//   //     for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//   //         output_offset = output_start_addr + (i*dim_J_block + j)*MATRIX_WIDTH; // 计算输出地址，按照行块计算
//   //         input_offset = input_start_addr + (i*dim_K_block + k)*MATRIX_WIDTH;
          
//   //         // 每次最后一个计算都由加载计算指令完成，因此最后一个i应该是不需要的，算出来的地址直接给上面的指令执行

//   //         if(i != dim_I_block -1) // 如果计算完成第一个权重，在计算最后一个权重时，换成使用getWeightPreloadComputeInsn计算
//   //         {
//   //           insn_buf[insn_idx++] =  getComputeInsn(
//   //                                   input_offset,  // 权重块加载偏移
//   //                                   output_offset, // 输出存储偏移
//   //                                   pingpang,      // i循环内不需要切换寄存器进行计算
//   //                                   accumulate);   // 当前计算是否在输出块进行累加
//   //         }
//   //         if(i == dim_I_block - 1 && j == dim_J_block - 1 && k == dim_K_block - 1) // 如果是最后一个权重块，使用当前寄存器进行计算
//   //         {
//   //           insn_buf[insn_idx++] =  getComputeInsn(
//   //                                   input_offset,  // 权重块加载偏移
//   //                                   output_offset, // 输出存储偏移
//   //                                   pingpang,      // i循环内不需要切换寄存器进行计算
//   //                                   accumulate);   // 当前计算是否在输出块进行累加
//   //         }
//   //       }
//   //     }
//   //   }
//   //   compute_count = insn_idx - compute_count;
//   //   printf("compute_count:%d\n",compute_count);

//   // 存储指令生成
//   for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//       const int buffer_start = 0; // 读取输出buffer的起始位置  
//       const int dram_start = 0; // 输出缓冲区的起始位置      
//       const int A_block = i*dim_J_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*MATRIX_WIDTH); // 计算buffer读取地址(根据按行存储特点计算)
//       const int dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                         + j * MATRIX_WIDTH; // 计算dram存储地址(根据二维连续数组存储特点计算)
//       // 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             buffer_offset,    // buffer偏移+矩阵内部偏移
//             dram_offset,      // 缓冲区偏移+矩阵内部偏移
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH行
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH列
//             dim_J);           // output矩阵总列数作为2D跨步DMA步进
//     }
//   }

//   // 结束指令
//   insn_buf[insn_idx++] = getFinishInsn();

//   // // 查看生成指令的二进制
//   // for (int i = 0; i < insn_idx; i++) { // 迭代输出行块
//   // if(i>=insn_load_size && i<insn_load_size+compute_count)
//   // {
//   //     Instruct_DataType instruction_data;
//   //     memcpy(&instruction_data, &insn_buf[i], sizeof(MemIns));
//   //     printBinary(instruction_data,INSTRUCT_WIDTH);
//   // }
//   // }


//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_size,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }




// // gemm指令，使用指令进行分块矩阵乘法，内部直接内置了SAA的运行函数，可以直接调用完成指令生成和计算

// int blocked_gemm_test(size_t dim_I,  // input矩阵的I行
//                       size_t dim_J,  // weight矩阵的J列
//                       size_t dim_K,  // input矩阵的K列，weight矩阵的K行
//                       void * input,  // 输入
//                       void * weight, // 权重
//                       void * bias,   // 偏置
//                       void * output, // 输出
//                       int block,     // 分块系数，将矩阵分为多少块 
//                       bool bias_use) // 是否使用偏置  
// {
//   // 断言检查
//   // assert(block % MATRIX_WIDTH == 0); // 分块系数是否能被MATRIX_WIDTH整除
//   // assert(dim_I % block == 0); // I是否能被block整除
//   // assert(dim_J % block == 0); // J是否能被block整除
//   // assert(dim_K % block == 0); // K是否能被block整除

//   // 检查
//   printf("=====================================================================================\n");
//   printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, block=%d,bias_use\n",
//          dim_I, dim_J, dim_K, block,bias_use);

//   // 计算分块
//   const int dim_I_block = dim_I / MATRIX_WIDTH; // 行分多少个块
//   const int dim_J_block = dim_J / MATRIX_WIDTH; // 列分多少个块
//   const int dim_K_block = dim_K / MATRIX_WIDTH; // 列分多少个块


//   // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
//   int insn_load_size = (dim_I_block * dim_K_block) + (dim_J_block * dim_K_block); // 计算加载A、B矩阵需要多少指令
//   int insn_compute_size = 2*dim_I_block*dim_K_block*dim_J_block; // 不使用权重复用
//   // int insn_compute_size = dim_I_block*dim_K_block*dim_J_block + dim_K_block*dim_J_block; // 使用权重复用
//   // int insn_compute_size = dim_I_block*dim_K_block*dim_J_block + dim_K_block*dim_J_block; // 使用权重复用和双缓冲
//   int insn_store_size = dim_I_block * dim_J_block; // 输出矩阵大小
//   int insn_size = insn_load_size + insn_store_size + insn_compute_size+1;
  

//   // 断言判断加载的矩阵大小小于缓冲区大小


//   // 初始化指令缓冲区
//   GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
//   int insn_idx = 0; // 用于赋值指令的的索引

//   // 指令生成

//   // 加载指令生成
//   // 加载输入input(dim_I,dim_K)
//   for (int i = 0; i < dim_I_block; i++) { // 迭代行块
//     for (int k = 0; k < dim_K_block; k++) { // 迭代列块
//       const int buffer_start = 0; // 加载到输入buffer的起始位置  
//       const int dram_start = 0; // 读取输入缓冲区的起始位置      
//       const int A_block = i*dim_K_block+k; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = dram_start + i * dim_K_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                         + k * MATRIX_WIDTH; // 计算dram读取地址
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             INPUT_BUFFER_ID, // 加载到输入Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             MATRIX_WIDTH,    // 每次加载MATRIX_WIDTH行
//             MATRIX_WIDTH,    // 每次加载MATRIX_WIDTH列
//             dim_K);          // input矩阵总列数作为2D跨步DMA步进
//       }
//     }

//   // 加载权重weight(dim_K,dim_J)
//   for (int k = 0; k < dim_K_block; k++) { // 迭代行块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代列块
//       const int buffer_start = 0; // 加载到权重buffer的起始位置  
//       const int dram_start = 0; // 读取权重缓冲区的起始位置      
//       const int A_block = k*dim_J_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*MATRIX_WIDTH); // 计算buffer输入地址
//       const int dram_offset = dram_start + k * dim_J_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                         + j * MATRIX_WIDTH; // 计算dram读取地址
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_LOAD,     // 加载指令
//             WEIGHT_BUFFER_ID,// 加载到权重Buffer
//             buffer_offset,   // buffer偏移+矩阵内部偏移
//             dram_offset,     // 缓冲区偏移+矩阵内部偏移
//             MATRIX_WIDTH,    // 每次加载MATRIX_WIDTH行
//             MATRIX_WIDTH,    // 每次加载MATRIX_WIDTH列
//             dim_J);          // weight矩阵总列数作为2D跨步DMA步进
//       }
//     }

//   // 使用双缓冲，调整循环和加载指令进行权重复用
//   int compute_count = insn_idx;
//   bool pingpang = 0; // 用于切换权重寄存器,最先使用weight1
//   const int wb_start_addr = 0;
//   const int input_start_addr = 0;
//   const int output_start_addr = 0;
//   int output_offset = 0;
//   int input_offset = 0;
//   bool accumulate = 0;
//   for (int k = 0; k < dim_K_block; k++) { // 迭代公共维度块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//       const int weigth_offset = wb_start_addr + (k*dim_J_block + j)*MATRIX_WIDTH;   
//       accumulate = k == 0 ? 0 : 1; // 如果是第一个块，刷新累加器中的累加值,其他块进行累加

//       // 第一次加载权重,也就是第一个权重块，加载到最初的pingpang处，该加载无法双缓冲
//       if(k == 0 && j == 0) 
//       {
//         insn_buf[insn_idx++] = getWeightPreloadInsn(
//                                       weigth_offset, // 从Buffer中0行加载权重块
//                                       pingpang);     // 使用初始pingpang权重寄存器
//       }
//       // 将剩下的权重加载完，可以进行双缓冲（最后计算还会剩下一个权重使用getComputeInsn计算）
//       else{
//         insn_buf[insn_idx++] = getWeightPreloadComputeInsn(
//                                        input_offset, // 权重块加载偏移
//                                        weigth_offset, // 权重块加载偏移
//                                        output_offset, // 输出存储偏移
//                                        pingpang, // （这个指令中没有用，加载的是计算寄存器的相反寄存器）
//                                        pingpang, // 计算寄存器依然是当前寄存器
//                                        accumulate); // 当前计算是否进行累加
//         pingpang = !pingpang; // 计算完成后，切换加载寄存器和计算寄存器
//       }    
//       for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//           output_offset = output_start_addr + (i*dim_J_block + j)*MATRIX_WIDTH; // 计算输出地址，按照行块计算
//           input_offset = input_start_addr + (i*dim_K_block + k)*MATRIX_WIDTH;
          
//           // 每次最后一个计算都由加载计算指令完成，因此最后一个i应该是不需要的，算出来的地址直接给上面的指令执行

//           if(i != dim_I_block -1) // 如果计算完成第一个权重，在计算最后一个权重时，换成使用getWeightPreloadComputeInsn计算
//           {
//             insn_buf[insn_idx++] =  getComputeInsn(
//                                     input_offset,  // 权重块加载偏移
//                                     output_offset, // 输出存储偏移
//                                     pingpang,      // i循环内不需要切换寄存器进行计算
//                                     accumulate);   // 当前计算是否在输出块进行累加
//           }
//           if(i == dim_I_block - 1 && j == dim_J_block - 1 && k == dim_K_block - 1) // 如果是最后一个权重块，使用当前寄存器进行计算
//           {
//             insn_buf[insn_idx++] =  getComputeInsn(
//                                     input_offset,  // 权重块加载偏移
//                                     output_offset, // 输出存储偏移
//                                     pingpang,      // i循环内不需要切换寄存器进行计算
//                                     accumulate);   // 当前计算是否在输出块进行累加
//           }
//         }
//       }
//     }
//     compute_count = insn_idx - compute_count;
//     printf("compute_count:%d\n",compute_count);

//   // 存储指令生成
//   for (int i = 0; i < dim_I_block; i++) { // 迭代输出行块
//     for (int j = 0; j < dim_J_block; j++) { // 迭代输出列块
//       const int buffer_start = 0; // 读取输出buffer的起始位置  
//       const int dram_start = 0; // 输出缓冲区的起始位置      
//       const int A_block = i*dim_J_block+j; // 第几个块
//       const int buffer_offset = (buffer_start + (A_block)*MATRIX_WIDTH); // 计算buffer读取地址(根据按行存储特点计算)
//       const int dram_offset = dram_start + i * dim_J_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                         + j * MATRIX_WIDTH; // 计算dram存储地址(根据二维连续数组存储特点计算)
//       // 存储输出
//       insn_buf[insn_idx++] = get2DLoadStoreInsn(
//             OPCODE_STORE,     // 存储指令
//             OUTPUT_BUFFER_ID, // 存储到输出buffer
//             buffer_offset,    // buffer偏移+矩阵内部偏移
//             dram_offset,      // 缓冲区偏移+矩阵内部偏移
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH行
//             MATRIX_WIDTH,     // 每次加载MATRIX_WIDTH列
//             dim_J);           // output矩阵总列数作为2D跨步DMA步进
//     }
//   }

//   // 结束指令
//   insn_buf[insn_idx++] = getFinishInsn();


//   // 运行SAA硬件
//   uint32_t done;
//   // 计算时间
//   uint64_t t_fpga;
//   struct timespec start, stop;
//   clock_gettime(CLOCK_REALTIME, &start);
//   saa_top(
//       insn_size,
//       (volatile Instruct_DataType *)insn_buf,
//       (volatile Transfer_DataType *)input, 
//       (volatile Transfer_DataType *)weight, 
//       (volatile Transfer_DataType *)output,
//       done); 
//   clock_gettime(CLOCK_REALTIME, &stop);
//   t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

//   // 打印软件运行SAA的吞吐量
//   printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
//   printf("INFO - Throughput: %.6lfGOPs/s\n",
//          static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

//   return 0;
// }




// //总的指令缓冲区，可以容纳1000条指令
// #define INSTRUCTION_BUFFER_WEIGHT 1000

// // 统计指令信息的结构体
// struct InstructionStruct {
//     Instruct_DataType instruction_data[INSTRUCTION_BUFFER_WEIGHT]; // 指令数据缓冲区，用于传输指令给SAA
//     int total_count; // 总共生成了多少指令
// };

// // 生成加载指令的函数，根据当前矩阵的行列生成一批加载指令，暂时只能加载MATRIX_WIDTH倍数的矩阵
// void generate_load_instructions(
//     InstructionStruct* instruction_struct, // 传入结构体的地址
//     int total_rows, // 总行数
//     int total_cols, // 总列数
//     Buffer_Id_DataType buffer_id, // 当前矩阵加载到哪个缓冲区
//     Dram_Addr_DataType dram_start, // 读取位置相对于缓冲区的偏移，如果矩阵直接就从0存储，那这就是0，以总线为偏移基本单位
//     Buffer_Addr_DataType buffer_start) // 写入位置相对于buffer起始行的偏移，如果直接从0行存储，那就是0
// {
//     // 计算分多少个块，就生成多少个指令
//     const int row_block = total_rows / MATRIX_WIDTH; // 行分多少个块
//     const int col_block = total_cols / MATRIX_WIDTH; // 列分多少个块
//     int insn_count =  row_block * col_block;  // 总的分块数等于行*列
//     SAAInsn instruction[insn_count]; // 使用总的分块数生成指令数组

//      // 计算当前结构体指针的赋值位置

//     //循环块生成加载指令
//     for (int row = 0; row < row_block; ++row) {
//         for (int col = 0; col < col_block; ++col) {
//             const int block = row*col_block+col; // 第几个块
//             const int buffer_base = (buffer_start + (block)*MATRIX_WIDTH); // 计算buffer输入地址
//             const int dram_base = dram_start + row * col_block * MATRIX_WIDTH*MATRIX_WIDTH 
//                                              + col * MATRIX_WIDTH; // 计算dram读取地址
//             // 写入指令结构体
//             instruction[block].mem.opcode = OPCODE_LOAD;  // 加载指令
//             instruction[block].mem.buffer_id = buffer_id; // 存储在哪个缓冲区
//             instruction[block].mem.dram_base = dram_base; // 缓冲区偏移+矩阵内部偏移
//             instruction[block].mem.buffer_base = buffer_base; // buffer偏移+矩阵内部偏移
//             instruction[block].mem.y_size = MATRIX_WIDTH; // 每次加载MATRIX_WIDTH行
//             instruction[block].mem.x_size = MATRIX_WIDTH; // 每次加载MATRIX_WIDTH列
//             instruction[block].mem.x_stride = total_cols; // 假设每行数据在DRAM中是连续存储的，那么步长就是列宽

//             // 转换指令结构体为128位指令数据类型
//             std::memcpy(&instruction_struct->instruction_data[instruction_struct->total_count+block], 
//                         &instruction[block], sizeof(SAAInsn));
//         }
//     }
//     instruction_struct->total_count = instruction_struct->total_count + insn_count ; // 计算当前总指令数
// }










