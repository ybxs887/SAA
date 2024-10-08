
#ifndef SAA_SIM_H_
#define SAA_SIM_H_

#include "../src/SAA.h"
#include <cstdlib> 
#include <stdlib.h>

// 分配连续缓存
void * allocBuffer(size_t num_bytes) {
  return malloc(num_bytes);
}

// 释放连续缓存
void freeBuffer(void * buffer) {
  return free(buffer);
}


// 用于产生任意大小的初始化的2D矩阵（地址不连续）
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
//       array[i][j] = static_cast<T>(rand() / (float)1000);
//     }
//   }
//   return array;
// }

// 用于产生任意大小的初始化的2D矩阵（地址连续）
template <typename T>
T** allocInit2dArray(int rows, int cols) {
  // 分配所有数据的连续内存块
  T* data = static_cast<T*>(malloc(sizeof(T) * rows * cols));
  
  // 分配行指针数组
  T** array = static_cast<T**>(malloc(sizeof(T*) * rows));
  
  // 初始化行指针，使其指向正确的位置
  for (int i = 0; i < rows; ++i) {
    array[i] = data + i * cols;
  }
  
  // 初始化数据
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      array[i][j] = static_cast<T>(rand() / (float)1000-10);
    }
  }
  
  return array;
}


// 用于释放上面的2D矩阵内存
template <typename T>
void free2dArray(T **array, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    free(array[i]);
  }
  free(array);
}


//-----------打包解包只是单纯将数组变得连续和从连续数组中读取数据--------------//

/**
 * @brief 将数据分块从源二维数组打包到目标一维数组中。
 *
 * 该函数将源数据 `src` 中的数据块打包到目标数据 `dst` 中。打包过程中，会按照
 * `y_block` 和 `x_block` 定义的块大小，将 `SRC_T` 类型的数据转换为 `DST_T` 类型，
 * 并存储在目标数组中。这个过程涉及到按位操作，将多个源数据的位拼接到一个
 * `DST_T` 类型的变量中，直到填满为止，然后将该变量写入目标数组，继续处理下一个数据块。
 *
 * 代码按照块打包数据到连续缓存中，存好第一个块然后再存第二个块
 * 
 * @tparam DST_T 目标数据类型。
 * @tparam DST_T_WIDTH 目标数据类型的位宽。
 * @tparam SRC_T 源数据类型。
 * @tparam SRC_T_WIDTH 源数据类型的位宽。
 * @param dst 目标一维数组的指针。
 * @param src 源二维数组的指针。
 * @param y_size 源数据的行数。
 * @param x_size 源数据的列数。
 * @param y_block 每个块的行数。
 * @param x_block 每个块的列数。
 */
template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void packBuffer(DST_T *dst, SRC_T **src, int y_size, int x_size, int y_block, int x_block) {
  // 确保源数据和目标数据的宽度比例正确。
  assert((SRC_T_WIDTH * x_block * y_block) % DST_T_WIDTH == 0);
  assert(DST_T_WIDTH <= 64);

  int buffer_idx = 0; // 目标数组的索引。
  int ratio = DST_T_WIDTH / SRC_T_WIDTH; // 数据类型宽度比例。
  long long int mask = (1ULL << SRC_T_WIDTH) - 1; // 源数据类型的位掩码。
  DST_T tmp = 0; // 用于临时存储打包数据的变量。

  // 遍历源数据的每个块。
  for (int i = 0; i < y_size / y_block; i++) {
    for (int j = 0; j < x_size / x_block; j++) {
      for (int k = 0; k < y_block; k++) {
        for (int l = 0; l < x_block; l++) {
          int block_idx = l + k * x_block; // 当前块内元素的索引。
          // 将源数据的元素与掩码进行按位与操作，然后左移相应的位数，合并到tmp变量中。
          tmp |= (src[i * y_block + k][j * x_block + l] & mask) << ((block_idx % ratio) * SRC_T_WIDTH);
          // 当处理完一个数据类型的所有位后，将其写入目标数组。
          if (block_idx % ratio == ratio - 1) {
            dst[buffer_idx++] = tmp;
            tmp = 0; // 重置tmp变量，准备下一个打包周期。
          }
        }
      }
    }
  }
}

/**
 * @brief 将数据分块从源一维数组解包到目标二维数组中。
 *
 * 该函数将源数据 `src` 中的数据解包到目标数据 `dst` 中。解包过程与打包过程相反，
 * 它将 `DST_T` 类型的数据转换回 `SRC_T` 类型，并存储到目标二维数组中。
 *
 * @tparam DST_T 目标数据类型。
 * @tparam DST_T_WIDTH 目标数据类型的位宽。
 * @tparam SRC_T 源数据类型。
 * @tparam SRC_T_WIDTH 源数据类型的位宽。
 * @param dst 目标二维数组的指针。
 * @param src 源一维数组的指针。
 * @param y_size 目标数据的行数。
 * @param x_size 目标数据的列数。
 * @param y_block 每个块的行数。
 * @param x_block 每个块的列数。
 */
template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void unpackBuffer(DST_T **dst, SRC_T *src, int y_size, int x_size, int y_block, int x_block) {
  // 确保源数据和目标数据的宽度比例正确。
  assert((DST_T_WIDTH * x_block * y_block) % SRC_T_WIDTH == 0);

  int buffer_idx = 0; // 源数组的索引。
  long long int mask = (1ULL << DST_T_WIDTH) - 1; // 目标数据类型的位掩码。
  int ratio = SRC_T_WIDTH / DST_T_WIDTH; // 数据类型宽度比例。

  // 遍历目标数据的每个块。
  for (int i = 0; i < y_size / y_block; i++) {
    for (int j = 0; j < x_size / x_block; j++) {
      for (int k = 0; k < y_block; k++) {
        for (int l = 0; l < x_block; l++) {
          int block_idx = l + k * x_block; // 当前块内元素的索引。
          // 从源数组中读取数据，右移相应的位数，并与掩码进行按位与操作，得到目标数据。
          dst[i * y_block + k][j * x_block + l] = (src[buffer_idx] >> ((block_idx % ratio) * DST_T_WIDTH)) & mask;
          // 当处理完一个数据类型的所有位后，移动到源数组的下一个位置。
          if (block_idx % ratio == ratio - 1) {
            buffer_idx++;
          }
        }
      }
    }
  }
}

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



// 按块打包函数
template <typename T>
T* packData(T *src, int y_size, int x_size, int y_block, int x_block) {
  T* dst = static_cast<T*>(allocBuffer(y_size * x_size * sizeof(T)));
  int dst_idx = 0;

  for (int i = 0; i < y_size; i += y_block) {
    for (int j = 0; j < x_size; j += x_block) {
      for (int k = 0; k < y_block && i + k < y_size; ++k) {
        for (int l = 0; l < x_block && j + l < x_size; ++l) {
          dst[dst_idx++] = src[(i + k) * x_size + (j + l)];
        }
      }
    }
  }
  return dst;
}

// 按块解包函数
template <typename T>
T* unpackData(T *src, int y_size, int x_size, int y_block, int x_block) {
  T* dst = static_cast<T*>(allocBuffer(y_size * x_size * sizeof(T)));
  int src_idx = 0;

  for (int i = 0; i < y_size; i += y_block) {
    for (int j = 0; j < x_size; j += x_block) {
      for (int k = 0; k < y_block && i + k < y_size; ++k) {
        for (int l = 0; l < x_block && j + l < x_size; ++l) {
          dst[(i + k) * x_size + (j + l)] = src[src_idx++];
        }
      }
    }
  }
  return dst;
}


template <typename T>
void print_pack_buffer(void* packed_buffer, int rows,int cols)
{
  // 打印打包后的1D数组，转换回原始数据类型
  std::cout << "Packed Buffer (1D Array representing 2D Matrix):" << std::endl;
  for (size_t i = 0; i < rows * cols; ++i) {
    std::cout << *(reinterpret_cast<T*>(packed_buffer)+i)<< " ";
    if ((i + 1) % cols == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}



//  实现pad_matrix函数,完成自动填充功能,输入矩阵为一维,输出矩阵也是一维,对应numpy.pad
template<typename T>
T* pad_matrix(void* matrix, size_t original_rows, size_t original_cols, size_t padding_bottom, size_t padding_right) {
    
    // 计算填充后的矩阵大小
    size_t padded_rows = original_rows + padding_bottom; // 填充底部
    size_t padded_cols = original_cols + padding_right; // 填充右部

    // 分配新矩阵的内存
    T* padded_matrix = static_cast<T*>(allocBuffer(padded_rows * padded_cols * sizeof(T))); // 计算填充后的大小,申请连续空间

    // 初始化新矩阵的内存为0,然后进行复制,没有复制到的地方就是填充的0
    memset(padded_matrix, 0, padded_rows * padded_cols * sizeof(T));
    
    // 复制原始矩阵到新矩阵
    for (size_t i = 0; i < original_rows; ++i) {
        memcpy(padded_matrix + i * padded_cols,  // 以行为复制单位
              static_cast<const T*>(matrix) + i * original_cols, 
              original_cols * sizeof(T));
    }

    return padded_matrix; // 返回填充后的新的连续内存的指针
}

// 实现extract_matrix函数,完成自动提取功能,输入矩阵为一维,输出矩阵也是一维,对应numpy的切片提取
template<typename T>
T* extract_matrix(T* padded_matrix, size_t original_rows, size_t original_cols, size_t padded_rows, size_t padded_cols) {
    // 分配新矩阵的内存
    T* extracted_matrix = static_cast<T*>(allocBuffer(original_rows * original_cols * sizeof(T)));

    // 从填充后的矩阵中提取原始矩阵
    for (size_t i = 0; i < original_rows; ++i) {
        memcpy(extracted_matrix + i * original_cols,  // 目标位置
               padded_matrix + i * padded_cols,       // 源位置
               original_cols * sizeof(T));            // 复制的字节数
    }

    return extracted_matrix; // 返回提取后的新的连续内存的指针
}


//------------------------------------指令生成-------------------------------------//
//以下函数由于操作码都不大于32，因此都是用int传参

// 具有依赖关系的指令
// 生成2D加载、存储指令，加载存储矩阵块，需要指定opcode是加载还是存储
GenericIns get2DLoadStoreInsn(int opcode, 
                              int buffer_id, 
                              int buffer_offset, 
                              int dram_offset,
                              int y_size, 
                              int x_size, 
                              int x_stride,
                              bool pop_pre_raw,  
                              bool pop_next_war, 
                              bool push_pre_war,
                              bool push_next_raw) 
{
  // 独联体进行转换
  union SAAInsn converter;
  // 存储指令初始化
  MemIns insn = {};
  insn.opcode = opcode;  // 加载指令
  insn.buffer_id = buffer_id; // 存储在哪个缓冲区
  insn.dram_base = dram_offset; // 缓冲区偏移+矩阵内部偏移
  insn.buffer_base = buffer_offset; // buffer偏移+矩阵内部偏移
  insn.y_size = y_size; // 每次加载MATRIX_WIDTH行
  insn.x_size = x_size; // 每次加载MATRIX_WIDTH列
  insn.x_stride = x_stride; // 假设每行数据在DRAM中是连续存储的，那么步长就是列宽
  insn.pop_pre_raw   = pop_pre_raw  ; // 该指令对前一个模块的RAW依赖
  insn.pop_next_war  = pop_next_war ; // 该指令对后一个模块的WAR依赖
  insn.push_pre_war  = push_pre_war ; // 该指令对前一个模块的WAR影响
  insn.push_next_raw = push_next_raw; // 该指令对后一个模块的RAW影响

  converter.mem = insn;

  return converter.generic;
}

// 块矩阵乘法的指令
GenericIns getGEMMInsn(int uop_bgn, // 本次GEMM微操作缓冲区的起始位置
                       int uop_end, // 本次GEMM微操作缓冲区的结束位置，这二者之差就是dim_I_block也就是本次分块的I
                       int dim_J_block, // 硬件执行分块的J循环的次数
                       int dim_K_block, // 硬件执行的K循环的次数
                       int input_base, // 本次计算读取输入缓冲区的基地址,这是按照块计算的
                       int weight_base, // 本次计算读取权重缓冲区的基地址 
                       int bias_use,    // 硬件上是否使用bias，代表硬件分块计算是否和bias累加
                       int relu_use,    // 在反量化后是否应用relu
                       int scale_type,  // scale的类型,包括不scale/反量化/重量化
                       int scale,       // 如果进行scale,那么这是scale的整数定点表示(scale原来是小数)
                       bool pop_pre_raw,  
                       bool pop_next_war, 
                       bool push_pre_war,
                       bool push_next_raw) 
{
  // 独联体进行转换
  union SAAInsn converter;
  // 计算指令初始化
  ComIns insn = {};

  // GEMM字段
  insn.opcode = OPCODE_GEMM;  // GEMM指令
  insn.uop_bgn = uop_bgn;  // 微操作缓冲区开始的地方
  insn.uop_end = uop_end;  // 微操作缓冲区的大小
  insn.dim_K_block = dim_K_block;  // K循环的次数
  insn.dim_J_block = dim_J_block;  // J循环的次数
  insn.input_base = input_base * MATRIX_WIDTH;  // 本次计算读取输入缓冲区的基地址，转换为按照行计算的
  insn.weight_base = weight_base * MATRIX_WIDTH;  // 本次计算读取权重缓冲区的基地址 
  insn.bias_use = bias_use;  // 是否使用偏置
  insn.relu_use = relu_use; // 在反量化后是否应用relu
  insn.scale_type = scale_type;  // 是否进行缩放,是进行反量化还是重量化
  insn.scale = scale;  // 缩放系数(整数)
  // 依赖字段
  insn.pop_pre_raw = pop_pre_raw; // 执行前对前一个load的raw依赖
  insn.pop_next_war = pop_next_war; //  执行前对后一个store的war依赖
  insn.push_pre_war = push_pre_war; //  执行后是否对前一个load有war影响
  insn.push_next_raw = push_next_raw; // 执行后是否对后一个store的raw影响

  converter.com = insn;

  return converter.generic;
}

// 块矩阵乘法运算的微操作指令生成
// 生成了四种形状块的微操作，使得可以适应多种形式的硬件分块计算
Uop * getGEMMUops(int tile_I, // 硬件分块计算的最大分块大小I
                  int tile_J, // 硬件分块计算的最大分块大小J  
                  int tile_K, // 硬件分块计算的最大分块大小K
                  int last_I, // 此处不使用，直接按照tile_I生成，实际GEMM指令传入时bgn到end间会动态变化适应分块
                  int last_J, // 最后不满tile_J的分块
                  int last_K) // 最后不满tile_K的分块 
{
  //如果存在异形的分块计算，那么我们需要将异形的分块计算的大小考虑进来，一共有四种情况
  int uop_size = 4*tile_I ; // 因为压缩了微操作的数量,只计算I循环相关的两个偏移,权重偏移在片上利用仿射计算

  Uop *uop_buf = static_cast<Uop *>(malloc(sizeof(Uop) * uop_size));

  int uop_idx=0;
  //tile_I、tile_J、tile_K
  for (int i = 0; i < tile_I; i++) {
    uop_buf[uop_idx].input_idx = i*tile_K*MATRIX_WIDTH; // 以MATRIX_WIDTH为最小分块
    uop_buf[uop_idx].output_idx = i*tile_J*MATRIX_WIDTH;
    // uop_buf[i].weight_idx = 0; // 不计算权重的偏移
    uop_idx++;
  }

  //tile_I、tile_J、last_K
  for (int i = 0; i < tile_I; i++) {
    uop_buf[uop_idx].input_idx = i*last_K*MATRIX_WIDTH; 
    uop_buf[uop_idx].output_idx = i*tile_J*MATRIX_WIDTH;
    uop_idx++;
  }
  
  //tile_I、last_J、tile_K
  for (int i = 0; i < tile_I; i++) {
    uop_buf[uop_idx].input_idx = i*tile_K*MATRIX_WIDTH; 
    uop_buf[uop_idx].output_idx = i*last_J*MATRIX_WIDTH;
    uop_idx++;
  }

  //tile_I、last_J、last_K
  for (int i = 0; i < tile_I; i++) {
    uop_buf[uop_idx].input_idx = i*last_K*MATRIX_WIDTH; 
    uop_buf[uop_idx].output_idx = i*last_J*MATRIX_WIDTH;
    uop_idx++;
  }
  return uop_buf;
}

// 生成运算完成指令
GenericIns getFinishInsn(bool pop_pre_raw,bool pop_next_war) 
{
  // 独联体进行转换
  union SAAInsn converter;
  // 计算指令初始化
  ComIns insn = {};
  insn.opcode = OPCODE_DONE;  // 完成指令
  insn.pop_pre_raw = pop_pre_raw; // 对前一个load的raw依赖
  insn.pop_next_war = pop_next_war; // 对后一个store的war依赖
  insn.push_pre_war = 0;
  insn.push_next_raw = 0;
  converter.com = insn;
  return converter.generic;
}

// 生成LayerNorm指令
GenericIns getLayerNormInsn(int dim_J_block, // J分块的大小，用于循环列
                            int dim_I_block, // 当前I分块的大小
                            int dim_J, // 实际J的大小（未填充）
                            bool pop_pre_raw,  
                            bool pop_next_war, 
                            bool push_pre_war,
                            bool push_next_raw) 
{
  // 独联体进行转换
  union SAAInsn converter;
  // 计算指令初始化
  AnuIns insn = {};
  insn.opcode = OPCODE_ANU;  // ANU指令
  insn.anu_type = ANU_LAYERNORM; // 进行layernorm
  insn.iter_uop = dim_J_block; // J维度的分块大小
  insn.iter_I = dim_I_block; // I维度的分块大小
  insn.imm = dim_J; // 立即数用于输入归一化的列数目
  // 依赖字段
  insn.pop_pre_raw = pop_pre_raw; // 执行前对前一个load的raw依赖
  insn.pop_next_war = pop_next_war; //  执行前对后一个store的war依赖
  insn.push_pre_war = push_pre_war; //  执行后是否对前一个load有war影响
  insn.push_next_raw = push_next_raw; // 执行后是否对后一个store的raw影响

  converter.anu = insn;

  return converter.generic;
}

// 生成Softmax指令
GenericIns getSoftmaxInsn(int dim_J_block, // 当前J分块的大小
                          int dim_I_block, // 当前I分块的大小
                          int dim_J, // 实际J的大小（未填充）
                          bool pop_pre_raw,  
                          bool pop_next_war, 
                          bool push_pre_war,
                          bool push_next_raw) 
{
  // 独联体进行转换
  union SAAInsn converter;
  // 计算指令初始化
  AnuIns insn = {};
  insn.opcode = OPCODE_ANU;  // ANU指令
  insn.anu_type = ANU_SOFTMAX; // 进行softmax
  insn.iter_uop = dim_J_block; // J维度的分块大小
  insn.iter_I = dim_I_block; // I维度的分块大小
  insn.imm = dim_J; // 立即数用于输入归一化的列数目,是真实的列的数目，因此将输入分块大小乘以块大小
  // 依赖字段
  insn.pop_pre_raw = pop_pre_raw; // 执行前对前一个load的raw依赖
  insn.pop_next_war = pop_next_war; //  执行前对后一个store的war依赖
  insn.push_pre_war = push_pre_war; //  执行后是否对前一个load有war影响
  insn.push_next_raw = push_next_raw; // 执行后是否对后一个store的raw影响

  converter.anu = insn;

  return converter.generic;
}


//------------------------------------高级函数，调用指令进行计算-------------------------------------//

// 软件实现gemm+bias
template<typename T0,typename T1 ,typename T>
T** matrix_biase_dot(T0** input, T1** weight, T** bias,int row, int col, int col1,bool bias_use)
{
    // 创建结果矩阵
    T** result = init_matrix<T>(row, col1);

    // 计算矩阵乘法
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col1; j++) {
            result[i][j] = bias_use ? bias[i][j] : (T)0; // 判断是否使用bias
            for (int k = 0; k < col; k++) {
                result[i][j] += input[i][k] * weight[k][j];
            }
        }
    }

    return result;
}

// 软件实现gemm+bias,同时进行scale,注意进行了位复制
template<typename T0,typename T1 ,typename T,typename T2>
T2** matrix_scale_dot(T0** input, T1** weight, T** bias,
                     int row, int col, int col1,bool bias_use,int scale_type,float scale)
{
    // 创建结果矩阵
    T** result = init_matrix<T>(row, col1);
    int32_t scale_int = static_cast<int32_t>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
    T2 scale_fixed; // 定点的scale参数,有24位小数精度
    scale_fixed.range() = scale_int; // 位模式赋值

    // 计算矩阵乘法
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col1; j++) {
            result[i][j] = bias_use ? bias[i][j] : (T)0; // 判断是否使用bias
            for (int k = 0; k < col; k++) {
                result[i][j] += input[i][k] * weight[k][j];
            }
            // scale类型和result的int32类型进行计算,结果存储在int32类型中,打印时要按照scale类型打印
            result[i][j].range() =((T2)(scale_fixed * result[i][j])).range(); // 计算完得到一个i,j位置的结果,进行一次scale,使用位模式赋值,之后打印使用scale类型打印查看
        }
    }
    return (T2**)result;
}

// 在上面的基础上融合了relu操作
template<typename T0,typename T1 ,typename T,typename T2>
T2** matrix_relu_dot(T0** input, T1** weight, T** bias,
                     int row, int col, int col1,bool bias_use,bool relu_use,int scale_type,float scale)
{
    // 创建结果矩阵
    T** result = init_matrix<T>(row, col1);
    T2** result_t2 = reinterpret_cast<T2**>(result); //转换指针为T2指向同一个空间，可以按照T2处理result数据
    int32_t scale_int = static_cast<int32_t>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
    T2 scale_fixed; // 定点的scale参数,有24位小数精度
    scale_fixed.range() = scale_int; // 位模式赋值
    // 计算矩阵乘法
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col1; j++) {
            result[i][j] = bias_use ? bias[i][j] : (T)0; // 判断是否使用bias
            for (int k = 0; k < col; k++) {
                result[i][j] += input[i][k] * weight[k][j];
            }
            // scale类型和result的int32类型进行计算,结果存储在int32类型中,打印时要按照scale类型打印
            if(scale_type)
              result[i][j].range() =((T2)(scale_fixed * result[i][j])).range(); // 计算完得到一个i,j位置的结果,进行一次scale,使用位模式赋值,之后打印使用scale类型打印查看
            if (relu_use)
              result_t2[i][j] = std::max(result_t2[i][j], T2(0));// 融合relu操作，需要将其转换为scale类型计算relu
        }
    }
    return result_t2;
}


// 可以计算任意大小的矩阵乘法的函数,内部自行进行了填充和数据pack以及反填充和反pack
// 该函数自动对矩阵进行tile，以获得最佳分块系数，确保能够尽量填充片上缓存
// 并行方式:存在Bias那么load和bias并行,或者和store并行，在K维度上使用双缓冲，使用计算时间隐藏加载时间
// 可以融合relu、反量化、layernorm、softmax
int tiled_matmul_auto(size_t dim_I,  // input矩阵的I行
                      size_t dim_J,  // weight矩阵的J列
                      size_t dim_K,  // input矩阵的K列，weight矩阵的K行
                      void * input,  // 输入(dim_I,dim_K)
                      void * weight, // 权重(dim_K,dim_J)
                      void * bias,   // 偏置(dim_I,dim_J)
                      void * output, // 输出(dim_I,dim_J)
                      bool bias_use, // 是否使用偏置  
                      bool relu_use, // 是否应用relu
                      int  scale_type, // scale的类型
                      float scale, // scale的大小(浮点小数)
                      int act) // 是否使用激活，使用何种激活
{
  // 计算需要填充的数量，使用最小的块进行填充
  // 首先计算dim_I包含多少个完整的MATRIX_WIDTH，然后计算dim_I能否被MATRIX_WIDTH整除
  // 不能的话，说明少了一个MATRIX_WIDTH，因此表达式为真，在整除的基础上添加一个MATRIX_WIDTH，那么结果矩阵必然被整除
  const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
  const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
  const int dim_K_padded = (dim_K / MATRIX_WIDTH + (dim_K % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

  // 计算填充0大小，后面可以使用硬件或者软件进行填充
  const int padding_I = dim_I_padded - dim_I;
  const int padding_J = dim_J_padded - dim_J;
  const int padding_K = dim_K_padded - dim_K;

  // 计算填充后的维度总共包含多少个MATRIX_WIDTH
  const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
  const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   
  const int dim_K_stride = dim_K_padded / MATRIX_WIDTH;

  // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
  Input_DataType* input_pad = pad_matrix<Input_DataType>(input,dim_I,dim_K,padding_I,padding_K); // 填充输入
  Weight_DataType* weight_pad = pad_matrix<Weight_DataType>(weight,dim_K,dim_J,padding_K,padding_J); // 填充权重
  Output_DataType* bias_pad = pad_matrix<Output_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

  // 填充后再进行打包,此时使用填充后大小打包,会返回一个填充后的新矩阵,packdata的输入是一维数组,输出是一维数组,都是连续的
  Input_DataType* input_buffer = packData(input_pad, dim_I_padded, dim_K_padded, MATRIX_WIDTH, MATRIX_WIDTH);
  Weight_DataType* weight_buffer = packData(weight_pad, dim_K_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);
  Output_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

  // 声明输出缓冲区(输出的是没有去填充的输出,一维)
  Output_DataType* output_buffer = static_cast<Output_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

  // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
  // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
# define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
# define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

// 假设只对K维度进行双缓冲，因此I，J不变，K维度存储大小减少一半
#define db_max_tile_k (((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / 2) / max_tile_i_j) // 在不进行双缓冲的基础上，假设K维度减少一半的存储大小，计算得到最大K维度分块的大小

  bool virtual_threads = 1; // 如果不存在act，就使用k维度虚拟线程/双缓冲
  // virtual_threads = act ? 0 : 1;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲
  // virtual_threads = 0;// 如果存在act，就不使用双缓冲，如果不存在act，就使用双缓冲

  // 计算每个缓冲区的最大容量，在使用双缓冲的前提下
  const size_t max_input_buffer = virtual_threads ? INPUT_BUFFER_WIDTH / 2 :INPUT_BUFFER_WIDTH;
  const size_t max_weight_buffer = virtual_threads ? WEIGHT_BUFFER_WIDTH / 2 :WEIGHT_BUFFER_WIDTH;
  const size_t max_output_buffer = virtual_threads ? OUTPUT_BUFFER_WIDTH : OUTPUT_BUFFER_WIDTH;

  size_t tile_I, tile_J, tile_K;
  // 首先是初始分块系数的计算，如果有act，则首先选择act条件，如果没有就可以使用双缓冲条件
  if (act == ANU_LAYERNORM || act == ANU_SOFTMAX) // 如果融合了激活/归一化函数，且是LAYERNORM和SOFTMAX，首先满足act
  {
      tile_I = 1;
      tile_J = dim_J_stride; // J维度完全装入片上
      tile_K = 1;
  }
  else if (virtual_threads) // 如果没有act，才考虑使用虚拟线程/双缓冲初始条件，如果是针对K维度进行的双缓冲，那么K维度的最大分块减少一倍
  {
    tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
    tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
    tile_K = dim_K_stride < db_max_tile_k ? dim_K_stride : db_max_tile_k; 
  }
  else // 如果没有使用激活，就是正常的矩阵乘法，如果矩阵的维度小于初始最大的分块系数，那么分块系数就是该维度大小，否则就是最大分块系数
  {
    tile_I = dim_I_stride < max_tile_i_j ? dim_I_stride : max_tile_i_j;
    tile_J = dim_J_stride < max_tile_i_j ? dim_J_stride : max_tile_i_j;
    tile_K = dim_K_stride < max_tile_k ?   dim_K_stride : max_tile_k;
  }

  // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
  while (true) {
    bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）

    // 先增大K维度的分块系数，提高双缓冲效率
    if(tile_I * (tile_K+1) <= max_input_buffer &&
       (tile_K+1) * tile_J <= max_weight_buffer &&
       (tile_K+1) <= dim_K_stride) // 增大K不超过输入缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
    {
        tile_K++;
        increased = true;
    }

    // 增大J维度的分块系数
    if(tile_I * (tile_J+1) <= max_output_buffer &&
       tile_K * (tile_J+1) <= max_weight_buffer &&
       (tile_J+1) <= dim_J_stride) // 增大J不超过输出缓冲区和权重缓冲区的容量,同时该分块系数不超过该维度大小
    {
        tile_J++;
        increased = true;
    }

    // 增大I维度的分块系数
    if((tile_I+1) * tile_K <= max_input_buffer &&
       (tile_I+1) * tile_J <= max_output_buffer &&
       (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
    {
        tile_I++;
        increased = true;
    }

    if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
      break;
  }

  // 如果得到的分块系数计算还是大于权重缓冲区的一半，那么不使用双缓冲,这一步保证了双缓冲让步于完整完成计算
  if((tile_K * tile_J)>max_weight_buffer) virtual_threads = 0;// 如果使用双缓冲还是大于权重缓冲区大小，那么只能暂停使用双缓冲
  if((tile_I * tile_K)>max_input_buffer) virtual_threads = 0;// 如果使用双缓冲还是大于输入缓冲区大小，那么只能暂停使用双缓冲

  // 在上面的基础上，还是判断分块系数不使用双缓冲能不能加载上
  assert((tile_I * tile_J)<=OUTPUT_BUFFER_WIDTH);
  assert((tile_I * tile_K)<=INPUT_BUFFER_WIDTH);
  assert((tile_K * tile_J)<=WEIGHT_BUFFER_WIDTH); // 输出缓冲区要计算

  // 然后计算二级分块需要循环的次数
  // 填充后的矩阵必然能被MATRIX_WIDTH整除，此处计算填充后矩阵能被分成多少大块
  // 和上面一样首先计算填充后的矩阵包含多少个完整的大块，然后如果矩阵不能被整除，说明要多计算一次大分块
  const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);
  const int J0 = dim_J_padded / (tile_J*MATRIX_WIDTH) + (dim_J_padded % (tile_J*MATRIX_WIDTH) != 0);
  const int K0 = dim_K_padded / (tile_K*MATRIX_WIDTH) + (dim_K_padded % (tile_K*MATRIX_WIDTH) != 0);

  // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
  // 使用原来的分块系数可能会造成大量空间浪费，因此需要根据多出来的小块数目决定最后一次大分块计算的大小
  // 这样，每个维度填充计算的无效大小最多是MATRIX_WIDTH-1，而不会是tile_I*MATRIX_WIDTH - 1
  // 如果可以被大分块整除，那么最后一个块的分块系数就是原来的大分块，如果不能，直接计算多余多少MATRIX_WIDTH作为分块系数
  const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;
  const int last_J = dim_J_padded % (tile_J*MATRIX_WIDTH) == 0 ? tile_J : (dim_J_padded/MATRIX_WIDTH) % tile_J;
  const int last_K = dim_K_padded % (tile_K*MATRIX_WIDTH) == 0 ? tile_K : (dim_K_padded/MATRIX_WIDTH) % tile_K;

  // 检查
  printf("==========================================GEMM Start===========================================\n");
  printf("INFO - Blocked GEMM test: dim_I=%d, dim_J=%d, dim_K=%d, tile_I=%d, tile_J=%d, tile_K=%d, double_buffer_use=%d, bias_use=%d, relu_use=%d, scale=%f, activate=%d\n",
         dim_I, dim_J, dim_K, tile_I, tile_J, tile_K,virtual_threads,bias_use,relu_use,scale,act);
  printf("UOP_BUFFER_WIDTH:%d\n",UOP_BUFFER_WIDTH);
  printf("INPUT_BUFFER_WIDTH:%d\n",INPUT_BUFFER_WIDTH);
  printf("WEIGHT_BUFFER_WIDTH:%d\n",WEIGHT_BUFFER_WIDTH);
  printf("OUTPUT_BUFFER_WIDTH:%d\n",OUTPUT_BUFFER_WIDTH);
  printf("INPUT_BUFFER_DEPTH:%d\n",INPUT_BUFFER_DEPTH);
  printf("WEIGHT_BUFFER_DEPTH:%d\n",WEIGHT_BUFFER_DEPTH);
  printf("OUTPUT_BUFFER_DEPTH:%d\n",OUTPUT_BUFFER_DEPTH);

  // 计算需要生成的指令数量
  int insn_bias_size = bias_use ? I0*J0 : 0; //加载bias,需要判断是否使用bias
  int insn_output_size = I0*J0; //加载bias，存储output,需要判断是否使用bias
  int insn_act_size = act ? I0*J0 : 0; //计算一个输出块在store前进行anu操作,如果使用则存在该指令
  int insn_input_weight_size = I0*J0*K0*2; //加载input和weight
  int insn_gemm_size = I0*J0*K0; // 计算gemm的数量
  int insn_uop_size = 1; // 加载uop的大小
  int insn_finish_size = 1; // 结束指令
  int insn_size = insn_bias_size+ insn_output_size + insn_input_weight_size + insn_gemm_size + insn_uop_size + insn_finish_size + insn_act_size;

  // 断言判断生成的指令数量是否超过了指令队列大小(UOP队列我们保证不会超过大小,因为是根据片上缓冲大小决定的)
  assert((insn_bias_size+insn_gemm_size+insn_finish_size+insn_uop_size+insn_act_size) <= STREAM_IN_DEPTH); // 计算队列
  assert((insn_input_weight_size) <= STREAM_IN_DEPTH); // 加载队列
  assert((insn_output_size) <= STREAM_IN_DEPTH); // 存储队列

  // 初始化指令缓冲区
  GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
  int insn_idx = 0; // 用于赋值指令的的索引

  // 生成微操作指令,针对完整的大分块，对于不满足大分块都计算其UOP微操作，得到四种类型分块的计算，有效部分使用I衡量
  Uop * uop_buf = getGEMMUops(
                      tile_I, // I
                      tile_J, // J
                      tile_K, // K
                      last_I,
                      last_J,
                      last_K); 

  int uop_size = 4*tile_I; //当前微操作缓冲区内需要加载的有效的微操作数量

  // 先加载微操作指令生成,然后再进行GEMM,因为compute模块按顺序执行的,加载了微操作,GEMM才能用
  insn_buf[insn_idx++] = get2DLoadStoreInsn(
        OPCODE_LOAD,     // 加载指令
        UOP_BUFFER_ID,// 加载到微操作缓冲区
        0,   // buffer的0地址处加载
        0,   // dram缓冲区的0地址处加载
        1,   // 一行
        uop_size,    // 加载微操作数量
        0,   // 步进不使用
        0,    // compute加载微操作不需要依赖，不涉及对三个缓冲区的读写
        0,
        0,
        0);  
  printf("- Generate compute(uop)\n");

  bool pingpang = 0; // 用于判断此次加载到哪个buffer中
  // 外循环，计算大分块的循环
  for (int i0 = 0; i0 < I0; i0++) {
    // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
    const int I = i0 < I0-1 ? tile_I : last_I;

    for (int j0 = 0; j0 < J0; j0++) {
      const int J = j0 < J0-1 ? tile_J : last_J;

      // bias和output的参数一样
      // 第i0行，经过了dim_J_stride * tile_I个块，加上j0列大块偏移j0 * tile_J个小块作为计算
      const int bias_dram_offset = i0 * dim_J_stride * tile_I + j0 * tile_J; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
      // 加载偏置biases(dim_I,dim_J)
      if(bias_use==1 && bias!=NULL)
      {
        insn_buf[insn_idx++] = get2DLoadStoreInsn(
              OPCODE_LOAD,     // 加载指令
              OUTPUT_BUFFER_ID, // 存储到输出buffer
              0,     // buffer为0，因为每次计算都直接加载满
              bias_dram_offset,     // dram中偏移一个块尝试
              I,     // 每次加载I个行块
              J,     // 每次加载J个列块
              dim_J_stride, // 总矩阵的块的列数作为步进
              0,     // pop_pre_raw  
              (i0 > 0 || j0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
              0,     // push_pre_war  ，如果是双缓冲使用bias，bias不对load产生影响
              0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
        printf("- Generate compute(bias)\n");
      }
      for (int k0 = 0; k0 < K0; k0++) { // 对K维度划分虚拟线程，一次计算完两个k0块，因此增加2
          const int K = (k0) < K0-1 ? tile_K : last_K; // 根据k0+l判断当前块，得到计算大小，判断是否是最后一个块的计算，从而选择此次计算的大小

          // 输入地址，按照块计算,根据l当前的值判断加载到双Buffer中的哪一个
          int input_base = pingpang * (tile_I * tile_K);// 本次计算读取输入缓冲区的基地址,这是按照块计算的
          int weight_base = pingpang * (tile_K * tile_J); // 本次计算读取权重缓冲区的基地址,这是按照块计算的 
          pingpang = virtual_threads ? (!pingpang) : 0 ; // 反转pingpang,如果使用双缓冲

          // 加载输入input(dim_I,dim_K)
          // 第一个load input依赖于后面compute是否完成,第一次启动依赖于bias，后面依赖于gemm，如果不存在Bias的话，第一次启动就不需要依赖
          bool input_pop_next_war = (i0 > 0 || j0 > 0 || (virtual_threads ? (k0 > 1) : (k0 > 0))); // 输入依赖，第一个k0块的加载不需要依赖，也就是刚开始的两个load都不依赖，后面都有依赖
          const int input_dram_offset = i0 * dim_K_stride * tile_I + k0 * tile_K; // 在input的K方向递增
          insn_buf[insn_idx++] = get2DLoadStoreInsn(
                OPCODE_LOAD,     // 存储指令
                INPUT_BUFFER_ID, // 存储到输出buffer
                input_base,    // input buffer偏移+矩阵内部偏移
                input_dram_offset,      // 缓冲区偏移+矩阵内部偏移
                I,     // 每次加载MATRIX_WIDTH行
                K,     // 每次加载MATRIX_WIDTH列
                dim_K_stride,  // output矩阵的列的分块数作为步进
                0,     // pop_pre_raw  
                input_pop_next_war, // pop_next_war   
                0,     // push_pre_war 
                0);    // push_next_raw  由于接下来是执行load weight，影响使用load weight生成
          printf("- Generate load(input)\n");

          // 加载权重weight(dim_K,dim_J)
          const int weight_dram_offset = k0 * dim_J_stride * tile_K + j0 * tile_J; // 在K方向递增
          insn_buf[insn_idx++] = get2DLoadStoreInsn(
                OPCODE_LOAD,     // 存储指令
                WEIGHT_BUFFER_ID, // 存储到输出buffer
                weight_base,    // weight buffer偏移+矩阵内部偏移
                weight_dram_offset,      // 缓冲区偏移+矩阵内部偏移
                K,     // 每次加载MATRIX_WIDTH行
                J,     // 每次加载MATRIX_WIDTH列
                dim_J_stride, // output矩阵的列的分块数作为步进
                0,     // pop_pre_raw  
                0,     // pop_next_war   load input完成后直接执行，因此没有依赖
                0,     // push_pre_war 
                1);    // push_next_raw  load weight的完成影响后面gemm的执行
          printf("- Generate load(weight)\n");

          // GEMM指令生成
          bool accumulate = bias_use ? 1 : (k0 == 0 ? 0 : 1); // 如果使用bias则一直累加，因为切换i,j时bias会刷新缓冲区，不使用则k0=0时刷新，而后累加
          int scale_int = static_cast<int>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
          int scale_type1 = (k0==K0-1) ? scale_type: 0; // 只有K循环执行到最后才进行scale，保证只对输出块进行scale
          // 判断当前执行的I、J、K，通过这种方式得到uop_bgn，可以适应不同大小的分块计算
          int uop_bgn = 0; 
          if( J==tile_J && K==tile_K ) // 分块计算的形状
            uop_bgn = 0;
          else if( J==tile_J && K==last_K )
            uop_bgn = tile_I;
          else if( J==last_J && K==tile_K )
            uop_bgn = 2*tile_I;
          else if( J==last_J && K==last_K )
            uop_bgn = 3*tile_I;
          // 依赖
          // gemm会一直对load产生影响，如果是最后一个GEMM就不用影响了,因为不会再回去加载input了，如果是双缓冲，那么最后两个GEMM都不影响load
          bool gemm_push_pre_war =(j0!=J0-1) || (i0 != I0-1) || ((virtual_threads ? (k0<K0-2) : (k0!=K0-1))? 1 : 0); // 最后两个gemm都不对load产生影响
          bool gemm_push_next_raw = act ? 0 : ((k0==K0-1) ? 1 : 0); // 在k循环执行到最后一个，对store产生影响，如果使用act，该影响被act取代
          // 当不存在bias时，K循环第一个GEMM会受到store的影响，但是第一个K循环不受影响，因为前面还没有store,如果存在bias，那么store的影响就给了bias了，那么就不用gemm受影响了
          bool gemm_pop_next_war = bias_use ? 0 : ((i0 > 0 || j0 > 0) && (k0==0) ); 
          insn_buf[insn_idx++] = getGEMMInsn(
                                    uop_bgn,
                                    uop_bgn + I, // I uop结束位置，是uop起始位置向后偏移I个
                                    J, // J
                                    K, // K
                                    input_base, // 本次计算读取输入缓冲区的基地址,按块计算
                                    weight_base, // 本次计算读取权重缓冲区的基地址 
                                    accumulate, // 是否直接进行累加，如果是那么直接在此基础上累加，如果不是则会更新值
                                    relu_use,
                                    scale_type1, // scale的类型，只有在k循环快结束时有效
                                    scale_int, // scale的大小(整数)
                                    1,    // pop_pre_raw  需要等待load input和weight执行完毕
                                    gemm_pop_next_war,    // pop_next_war 
                                    gemm_push_pre_war,    // push_pre_war 
                                    gemm_push_next_raw);   // push_next_raw  
          printf("- Generate compute(gemm)\n");
      }
      // 在store i,j前插入ANU操作
      if(act == ANU_LAYERNORM) // 生成LayerNorm指令
      {
        insn_buf[insn_idx++] = getLayerNormInsn(dim_J_stride, // 写入当前填充矩阵的分块数
                                                I,  // 传入DIM_I用于计算
                                                dim_J,// 实际未填充的dim_J列数
                                                0,     // pop_pre_raw  
                                                0,  // pop_next_war
                                                0,     // push_pre_war  
                                                1);    // push_next_raw ，对store产生影响
      }
      else if(act == ANU_SOFTMAX) // 生成softmax指令
      {
        insn_buf[insn_idx++] = getSoftmaxInsn(dim_J_stride, // 写入当前填充矩阵的分块数
                                              I,
                                              dim_J, //实际未填充的dim_J列数
                                              0,     // pop_pre_raw  
                                              0,  // pop_next_war
                                              0,     // push_pre_war  
                                              1);    // push_next_raw 
      }
      // 存储指令生成, 存储输出
      // 影响bias的加载，最后一次影响finish命令，因为bias先执行一个，因此store只影响三个bias加载，然后影响最后一个finish命令，如果不存在bias，那么不对前一个产生影响但是最后还是要影响finish
      bool store_push_pre_war = (bias_use ? 1 : (i0 == I0-1 && j0 == J0-1));
      insn_buf[insn_idx++] = get2DLoadStoreInsn(
            OPCODE_STORE,     // 存储指令
            OUTPUT_BUFFER_ID, // 存储到输出buffer
            0,                // buffer偏移+矩阵内部偏移
            bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
            I,     // 每次加载I个行块
            J,     // 每次加载J个列块
            dim_J_stride,    // 总矩阵的块的列数作为步进
            1,     // pop_pre_raw  , 依赖于GWMM的执行完毕
            0,     // pop_next_war   
            1,  // push_pre_war , 
            0);    // push_next_raw
      printf("- Generate store\n");
    }
  }

  // 结束所有大分块的计算后，发出结束指令
  insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
  printf("- Generate compute(finish)\n");
  printf("insn_idx:%d\n",insn_idx);
  printf("insn_size:%d\n",insn_size);

  // 运行SAA硬件
  uint32_t done;
  // 计算时间
  uint64_t t_fpga;
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
  saa_top(
      insn_idx,
      (volatile Instruct_DataType *)insn_buf,
      (volatile Uop_DataType *) uop_buf,
      (volatile Transfer_DataType *)input_buffer, 
      (volatile Transfer_DataType *)weight_buffer, 
      (volatile Transfer_DataType *)biases_buffer, 
      (volatile Transfer_DataType *)output_buffer,
      done); 
  clock_gettime(CLOCK_REALTIME, &stop);
  t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

  // 打印软件运行SAA的吞吐量
  printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);
  printf("INFO - Throughput: %.6lfGOPs/s\n",
         static_cast<float>(dim_I) * dim_J * dim_K * 2 / t_fpga);

  // 解包输出提取有效部分并返回输出给output指针
  Output_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
  Output_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
  memcpy(output, output_extract, dim_I * dim_J * sizeof(Output_DataType));// 将结果copy给输出指针
  
  printf("==========================================GEMM End===========================================\n");
  return 0;
}


// 软件计算layernorm,遍历两次
template<typename T>
T** layer_norm(T** x, int batch_size, int features, float eps) {
    // 分配内存以存储每批的均值和方差
    float* means = new float[batch_size];
    float* variances = new float[batch_size];

    // 计算每个样本的均值和方差
    for (int i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        float sum_of_squares = 0.0f;
        
        // 第一次遍历：计算均值和平方的均值
        for (int j = 0; j < features; ++j) {
            sum += (float)x[i][j];
            sum_of_squares += static_cast<float>(x[i][j]) * static_cast<float>(x[i][j]);
        }
        
        // 保存每个样本的均值
        float mean = sum ;
        // 计算方差
        float variance = sum_of_squares;
        // printf("\nsum:%f\n",mean);
        // printf("var:%f\n",variance);

        mean = mean / features;
        variance = variance / features - (mean * mean);
        means[i] = mean;
        variances[i] = variance;
    }

    // 计算所有样本的均值和方差
    float mean_of_means = 0.0f;
    float mean_of_variances = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        mean_of_means += means[i];
        mean_of_variances += variances[i];
    }
    mean_of_means /= batch_size;
    mean_of_variances /= batch_size;

    // 第二次遍历：归一化
    // 创建结果矩阵
    T** result = init_matrix<T>(batch_size, features);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < features; ++j) {
            float mean_diff = (float)x[i][j] - means[i];
            float normalized_val = mean_diff / (sqrtf(variances[i] + eps));
            result[i][j] = (T)normalized_val;
        }
    }
    
    // printf("\nsum:\n");
    // print_vec(means,batch_size);
    // printf("var:\n");
    // print_vec(variances,batch_size);

    // 释放分配的内存
    delete[] means;
    delete[] variances;

    return result;
}

// 软件计算softmax,遍历三次,第一次找最大值,第二次进行exp并求exp的和,第三次应用softmax
template<typename T>
T** softmax(T** x, int batch_size, int features) {
    // 创建结果矩阵
    T** result = init_matrix<T>(batch_size, features);

    // 计算softmax，对每个样本的特征进行操作
    for (int i = 0; i < batch_size; ++i) {
        // 先找到最大值，用于数值稳定性
        T max_val = x[i][0];
        for (int j = 1; j < features; ++j) {
            if (x[i][j] > max_val) {
                max_val = x[i][j];
            }
        }
        // printf("\nmax_val:%f\n",(float)max_val);
        // 计算每个特征的指数，并累计求和
        float sum_exp = 0.0f;
        for (int j = 0; j < features; ++j) {
            result[i][j] = expf(x[i][j] - max_val); // 防止溢出
            sum_exp += (float)result[i][j];
        }

        // printf("\nsum_exp:%f\n",sum_exp);

        // 归一化，使得每个样本的特征值加起来为1
        for (int j = 0; j < features; ++j) {
            result[i][j] /= (T)sum_exp;
        }
    }

    return result;
}

// 在上面的基础上融合了softmax和layernorm操作
template<typename T0,typename T1 ,typename T,typename T2>
T2** matrix_act_dot(T0** input, T1** weight, T** bias,
                     int row, int col, int col1,bool bias_use,bool relu_use,int scale_type,float scale,int act)
{
    // 创建结果矩阵
    T** result = init_matrix<T>(row, col1);
    T2** result_t2 = reinterpret_cast<T2**>(result); //转换指针为T2指向同一个空间，可以按照T2处理result数据
    T2 **result_t3; // 声明输出的result_t3指针
    float eps = 1e-5f; // layernorm的一个小偏差值
    int32_t scale_int = static_cast<int32_t>(scale * (1LL << 24));;  // 转换float类型的scale系数为整数类型,放大倍数根据我们片上定点的小数位数相关
    T2 scale_fixed; // 定点的scale参数,有24位小数精度
    scale_fixed.range() = scale_int; // 位模式赋值
    // 计算矩阵乘法
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col1; j++) {
            result[i][j] = bias_use ? bias[i][j] : (T)0; // 判断是否使用bias
            for (int k = 0; k < col; k++) {
                result[i][j] += input[i][k] * weight[k][j];
            }
            // scale类型和result的int32类型进行计算,结果存储在int32类型中,打印时要按照scale类型打印
            result[i][j].range() =((T2)(scale_fixed * result[i][j])).range(); // 计算完得到一个i,j位置的结果,进行一次scale,使用位模式赋值,之后打印使用scale类型打印查看
            if (relu_use) // 如果使用relu操作
              result_t2[i][j] = std::max(result_t2[i][j], T2(0));// 融合relu操作，需要将其转换为scale类型计算relu
        }
    }
    // 计算完成后，应用act
    if (act == ANU_LAYERNORM)
      result_t3 = layer_norm(result_t2, row, col1, eps);
    else if(act == ANU_SOFTMAX)
      result_t3 = softmax(result_t2, row, col1);
    else // 不进行
      result_t3 = result_t2;
    return result_t3;
}

// 硬件执行ANU计算矩阵的layernorm和softmax
// 该函数使用指令控制微操作循环，使得指令分块计算数量减少 ，减少了生成的指令数量
// 目前只能测试计算MATRIX_WIDTH行的数据,因为硬件一次只能执行这么多,如果要多次执行,需要管理依赖关系
// 这要求bias是float转换为scale的定点数加载进来
int anu_test(int opcode,   // 进行什么ANU操作
            size_t dim_I,  // input矩阵的I行
            size_t dim_J,  // input矩阵的J列
            void * bias,   // 偏置，作为非线性运算的输入矩阵
            void * output) // 输出
{

  // 计算填充，使得其行列满足MATRIX_WIDTH的整数倍的条件
  const int dim_I_padded = (dim_I / MATRIX_WIDTH + (dim_I % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;
  const int dim_J_padded = (dim_J / MATRIX_WIDTH + (dim_J % MATRIX_WIDTH != 0)) * MATRIX_WIDTH;

  // 计算填充0大小，后面可以使用硬件或者软件进行填充
  const int padding_I = dim_I_padded - dim_I;
  const int padding_J = dim_J_padded - dim_J;

  // 计算填充后的维度总共包含多少个MATRIX_WIDTH
  const int dim_I_stride = dim_I_padded / MATRIX_WIDTH;
  const int dim_J_stride = dim_J_padded / MATRIX_WIDTH;   

  // 对矩阵进行填充(input,weight,bias),这将会返回一个新的矩阵(numpy.pad原理),要求输入矩阵是连续的
  Scale_DataType* bias_pad = pad_matrix<Scale_DataType>(bias,dim_I,dim_J,padding_I,padding_J); // 填充偏置

  // 填充后进行打包
  Scale_DataType* biases_buffer = packData(bias_pad,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH);

  // 声明输出缓冲区(输出的是没有去填充的输出,一维)
  Scale_DataType* output_buffer = static_cast<Scale_DataType*>(allocBuffer(dim_I_padded * dim_J_padded * sizeof(Output_DataType)));

  // 根据缓冲区大小自动计算分块系数，原则上尽量填充满缓冲区（）
  // 计算得到每个维度的最大分块系数，首先先计算I、J维度的初始最大分块系数，通过输出缓冲区大小得到，假设I，J维度分块系数一直，因此直接将输出缓冲区能够存储多少块进行根号
# define max_tile_i_j ((size_t)sqrt(OUTPUT_BUFFER_WIDTH)) // 先计算输出缓冲区得到I、J维度分块系数（二者相等）
# define max_tile_k ((INPUT_BUFFER_WIDTH > WEIGHT_BUFFER_WIDTH ? WEIGHT_BUFFER_WIDTH : INPUT_BUFFER_WIDTH) / max_tile_i_j) // 在上面的I、J分块的基础上，再计算K维度的分块系数，判断权重缓冲区和输入缓冲区哪个更小，然后根据小的决定初始分块系数

  size_t tile_I, tile_J, tile_K;
  // 首先是初始分块系数的计算
  tile_I = 1; // I维度只计算一个分块
  tile_J = dim_J_stride; // J维度完全装入片上

  // 然后在此基础上，增加tile_I的分块系数
  // 在初始分块系数的前提下，继续增加分块系数，尽可能的填充缓冲区以减少数据搬移次数减少指令数
  while (true) {
    bool increased = false; // 每次循环开始，将增加标志位复位，该标志位用来表示是否有分块系数增加（无论I、J、K）
    // 增大I维度的分块系数
    if((tile_I+1) * tile_J <= OUTPUT_BUFFER_WIDTH &&
       (tile_I+1) <= dim_I_stride) // 增大I不超过输入缓冲区和输出缓冲区的容量,同时该分块系数不超过该维度大小
    {
        tile_I++;
        increased = true;
    }

    if (!increased) // 没有增加代表已经当前分块系数已经占满了缓冲区了，因此可以退出循环
      break;
  }

  // 然后计算二级分块需要循环的次数,一级分块在片上，一次只能算I维度一个分块，这里由于J维度全部装进片上因此J维度不用分块
  const int I0 = dim_I_padded / (tile_I*MATRIX_WIDTH) + (dim_I_padded % (tile_I*MATRIX_WIDTH) != 0);

  // 如果不能整除，上面最后一次大分块计算，我们不能使用原来的分块系数
  const int last_I = dim_I_padded % (tile_I*MATRIX_WIDTH) == 0 ? tile_I : (dim_I_padded/MATRIX_WIDTH) % tile_I;


  // 检查
  printf("==========================================ANU Start===========================================\n");
  printf("INFO -  ANU test: opcode=%d, dim_I=%d, dim_J=%d, tile_I=%d, tile_J=%d\n",
         opcode, dim_I, dim_J ,tile_I ,tile_J);

  // 计算需要生成的指令数量(不使用偏置,暂时一个块一个块的加载,不复用权重，进行输出累加，一次只能计算一个块)
  int insn_load_size = I0; // 加载输入，一次加载一个大分块
  int insn_anu_size = I0; // 一次计算一个大分块
  int insn_store_size = I0; // 存储输出，一次存储一个大分块
  int insn_finish_size = 1; // 结束指令
  int insn_size = insn_load_size + insn_store_size + insn_anu_size+insn_finish_size;

  // 初始化指令缓冲区
  GenericIns *insn_buf = static_cast<GenericIns *>(allocBuffer(sizeof(GenericIns) * insn_size)); // 使用allocbuffer生成连续缓冲区
  int insn_idx = 0; // 用于赋值指令的的索引

  // 外循环，计算大分块的循环
  for (int i0 = 0; i0 < I0; i0++) {
    // 计算一级分块的循环次数，需要传入硬件，最后一次大分块循环，使用最后一个分块系数计算
    const int I = i0 < I0-1 ? tile_I : last_I;

    // 2D跨步加载，最小加载单位为块
    const int bias_dram_offset = i0 * dim_J_stride * tile_I; // 根据当前是第几个大块计算这个大块的偏移，单位是按照小块计算
    insn_buf[insn_idx++] = get2DLoadStoreInsn(
          OPCODE_LOAD,     // 加载指令
          OUTPUT_BUFFER_ID, // 存储到输出buffer
          0,     // buffer为0，因为每次计算都直接加载满
          bias_dram_offset,     // dram中偏移一个块尝试
          I,     // 每次加载I个行块
          tile_J,     // 每次加载J个列块
          dim_J_stride, // 总矩阵的块的列数作为步进
          0,     // pop_pre_raw  
          (i0 > 0),  // pop_next_war  ，如果不是第一个大bias分块，那么需要考虑store有没有完成
          0,     // push_pre_war  ，不依赖于load
          0);    // push_next_raw ，加载完一次后，store一次，通知store表示我加载完成
    printf("- Generate compute(input load)\n");

    // 计算指令生成
    const int read_src_base = 0;  //每次计算的是同一个物理地址起始
    const int write_dst_base = 0; // 写入buffer块的基地址,读取写入同一片地址     
    const int src_offset =  MATRIX_WIDTH;      // 片上微操作每次的读取块偏移,就是MATRIX_WIDTH行
    const int dst_offset =  MATRIX_WIDTH;      // 片上微操作每次的写入块偏移

    // 生成LayerNorm指令
    if(opcode == ANU_LAYERNORM) 
    {
      insn_buf[insn_idx++] = getLayerNormInsn(dim_J_stride, // 写入当前填充矩阵的分块数
                                              I,  // 传入DIM_I用于计算
                                              dim_J,// 实际未填充的dim_J列数
                                              0,     // pop_pre_raw  
                                              0,  // pop_next_war
                                              0,     // push_pre_war  
                                              1);    // push_next_raw ，对store产生影响
    }
    else if(opcode == ANU_SOFTMAX)
    {
      insn_buf[insn_idx++] = getSoftmaxInsn(dim_J_stride, // 写入当前填充矩阵的分块数
                                            I,
                                            dim_J, //实际未填充的dim_J列数
                                            0,     // pop_pre_raw  
                                            0,  // pop_next_war
                                            0,     // push_pre_war  
                                            1);    // push_next_raw 
    }

    // 存储指令行生成
    bool store_push_pre_war = 0;
    insn_buf[insn_idx++] = get2DLoadStoreInsn(
          OPCODE_STORE,     // 存储指令
          OUTPUT_BUFFER_ID, // 存储到输出buffer
          0,                // buffer偏移+矩阵内部偏移
          bias_dram_offset, // 缓冲区偏移+矩阵内部偏移
          I,     // 每次加载I个行块
          tile_J,     // 每次加载J个列块
          dim_J_stride,    // 总矩阵的块的列数作为步进
          1,     // pop_pre_raw  , 依赖于anu操作完成
          0,     // pop_next_war   
          1,  // push_pre_war , 影响bias的加载，最后影响finish
          0);    // push_next_raw
    printf("- Generate store\n");
  }

  // 结束指令
  insn_buf[insn_idx++] = getFinishInsn(0,1); // 等待存储完成后再执行finish
  printf("- Generate compute(finish)\n");
  printf("insn_idx:%d\n",insn_idx);
  printf("insn_size:%d\n",insn_size);

  // 运行SAA硬件
  uint32_t done;
  // 计算时间
  uint64_t t_fpga;
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
  saa_top(
      insn_idx,
      (volatile Instruct_DataType *)insn_buf,
      (volatile Uop_DataType *) NULL,
      (volatile Transfer_DataType *)NULL, 
      (volatile Transfer_DataType *)NULL, 
      (volatile Transfer_DataType *)biases_buffer, 
      (volatile Transfer_DataType *)output_buffer,
      done); 
  clock_gettime(CLOCK_REALTIME, &stop);
  t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

  // 打印软件运行SAA的吞吐量
  printf("INFO - Synchronization time: %.6lfms\n", static_cast<float>(t_fpga) / 1E6);

  // 解包输出提取有效部分并返回输出给output指针
  Scale_DataType* output_pad = unpackData(output_buffer,dim_I_padded, dim_J_padded, MATRIX_WIDTH, MATRIX_WIDTH); // 按照填充维度解包
  Scale_DataType* output_extract = extract_matrix(output_pad,dim_I,dim_J,dim_I_padded,dim_J_padded);; // 提取有效部分
  memcpy(output, output_extract, dim_I * dim_J * sizeof(Scale_DataType));// 将结果copy给输出指针

  return 0;
}


#endif
