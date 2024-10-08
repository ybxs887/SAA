//本文件定义了SAA运行时也就是SAA硬件交互函数
#ifndef SAA_RUNTIME_HPP
#define SAA_RUNTIME_HPP

#include "SAA.h"
#include "SAA_driver.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>
#include <malloc.h>
#include <cassert>

//------------------------------------指令生成-------------------------------------//
// 注意windows不需要处理缓存一致性问题，如果使用arm linux就需要处理，使用flush和invalid控制缓存
// 当使用zynq之类的DMA传输，为了使用cache加速以及保证缓存一致性，我们需要使用flush和invalid控制缓存
/*! \brief 始终缓存缓冲区（否则，从CPU写回DRAM）*/
static const bool kAlwaysCache = true;

// 声明saa的命名空间
namespace saa {
//------------------------------------内存操作-------------------------------------//

/**
 * @class DeviceAllocStat
 * @brief 用于跟踪和管理设备内存分配状态的类。
 * 该类提供了添加、检查和删除内存分配记录的方法。
 */
class DeviceAllocStat {
 public:
    // 添加一个新的内存分配记录
    void AddAlloc(const void* ptr) {
        std::lock_guard<std::mutex> lock(mtx_); // 使用互斥锁保证线程安全访问allocated_
        allocated_.insert(ptr); // 插入集合
    }

    //检查指定的内存是否已被分配。如果内存已被分配，则返回true；否则返回false。
    bool CheckAlloc(const void* ptr) {
        std::lock_guard<std::mutex> lock(mtx_);
        return allocated_.count(ptr); //查找指针是否在集合中，在就是ture
    }

    // 删除一个内存分配记录。
    void DelAlloc(const void* ptr) {
        std::lock_guard<std::mutex> lock(mtx_);
        allocated_.erase(ptr);
    }

    private:
    std::set<const void*> allocated_; // 已分配内存的集合，用于存储所有分配的内存指针。
    std::mutex mtx_; // 互斥锁，用于同步对已分配内存集合的访问。
};

// 使用一个静态变量记录全局的内存分配
static std::shared_ptr<DeviceAllocStat> alloc_stat; // 全局内存分配状态跟踪器

/*!
* \brief 内存缓冲区类，管理系统内存（DRAM）中和硬件加速器内存（SRAM）有关的数据缓冲区
*        实际上该类需要能够为加速器从DRAM中申请连续空间，其主要是通过调用在不同平台的连续内存申请函数(在driver内定义)
*        完成申请操作。主要功能是对缓冲区的连续内存分配、复制、释放方法、以及查看使用情况等
*/
class DataBuffer {

public:
    /*! \brief 构造函数，初始化 `alloc_stat_` 成员被赋值全局的内存分配alloc_stat。*/
    DataBuffer() { alloc_stat_ = alloc_stat; }

    /*! \return 数据的虚拟地址*/
    void* virt_addr() const { return data_; }

    /*! \return 数据的物理地址*/
    saa_phy_addr_t phy_addr() const { return phy_addr_; }

    /*!
    *\brief 使数据缓冲区中给定位置的缓存无效，迫使CPU下次访问时从DDR中重新加载数据。
    *\param offset数据的偏移量。
    *\param size数据的大小。
    */

    /*!
    *\brief 将缓存中的数据写回到DDR中，并清空缓存，确保缓存不会包含过时的数据。
    *\param offset数据的偏移量。
    *\param size数据的大小。
    */


    /*!
    *\brief 执行从主机内存到SAAMemAlloc分配的缓冲区的复制操作。
    *\param dst FPGA可访问的DDR中的缓冲区。必须使用SAAMemAlloc()分配.
    *\param src 主机内存中的源缓冲区. 
    *\param size 复制的字节大小
    */
    void MemCopyFromHost(void* dst, const void* src, size_t size){SAAMemCopyFromHost(dst, src, size);}

    /*!
    *\brief 执行从SAAMemAlloc分配的缓冲区到主机内存的复制操作。
    *\param dst 主机内存中的指定缓冲区。
    *\param src FPGA可访问内存中的源缓冲区。必须使用SAAMemAlloc()进行分配。
    *\param size 复制的字节大小。
    */
    void MemCopyToHost(void* dst, const void* src, size_t size) { SAAMemCopyToHost(dst, src, size); }

    /*!
    *\brief 分配一个给定大小的缓冲区。
    *\param size 缓冲区的字节大小。
    */
    static DataBuffer* Alloc(size_t size) 
    {
        void* data = SAAMemAlloc(size, kAlwaysCache); // 使用SAAMemAlloc分配一块连续的虚拟地址，使用cache缓存
        assert(data != nullptr);                      // 断言检查是否是空指针
        DataBuffer* buffer = new DataBuffer();        // 建立一个新的databuffer对象
        buffer->data_ = data;                         // 将分配的虚拟地址赋值给databuffer对象
        buffer->phy_addr_ = SAAMemGetPhyAddr(data);   // 从虚拟地址获取物理地址(暂时不使用)
        alloc_stat->AddAlloc(buffer);                 // 全局记录添加内存分配记录
        return buffer;                                // 返回databuffer对象
    }

    /*!
    *\brief 释放数据缓冲区。
    *\param buffer 要释放的缓冲区databuffer类。
    */
    static void Free(DataBuffer* buffer) 
    {
        alloc_stat->DelAlloc(buffer); // 删除内存分配记录
        SAAMemFree(buffer->data_);    // 释放虚拟地址指针
        delete buffer;
    }
    
    /*!
    *\brief 从缓冲区ptr指针创建一个 `DataBuffer` 对象。
    *\param buffer 缓冲区指针。
    *\return 对应的 `DataBuffer` 对象。
    */
    static DataBuffer* FromHandle(const void* buffer) 
    {
        if (alloc_stat->CheckAlloc(buffer)) // 检查buffe指针是否在分配内存集合中，在的话说明该内存可以被DataBuffer使用
            return const_cast<DataBuffer*>(reinterpret_cast<const DataBuffer*>(buffer)); // 将buffer指针转换为DataBuffer类
        else 
            return nullptr;
    }

private:
    void* data_;                                  // 数据的指针（虚拟地址）
    saa_phy_addr_t phy_addr_;                     // 数据的物理地址（在PYNQ以及Linux平台才有用）
    std::shared_ptr<DeviceAllocStat> alloc_stat_; // 类的内存分配状态跟踪器，当作alloc_stat副本用
};

//------------------------------------basequeue分配指令内存-------------------------------------//
#define ALLOC_ALIGNMENT 64 //按照64字节对齐，使得其符合地址是64倍数，可能是为了和cache line对齐
// AlignmentAllocator 类模板，用于创建特定对齐要求的内存块，作为std::vector的分配器使用，存储指令队列
template <typename T, std::size_t N = ALLOC_ALIGNMENT>
class AlignmentAllocator : public std::allocator<T> {
 public:
    // 值类型别名，表示分配器分配的类型
    typedef T value_type;
    // 大小类型别名，表示大小的类型
    typedef std::size_t size_type;
    // 差值类型别名，表示两个指针之间的差值类型
    typedef std::ptrdiff_t difference_type;

    // 指针类型别名
    typedef T* pointer;
    // 常量指针类型别名
    typedef const T* const_pointer;

    // 引用类型别名
    typedef T& reference;
    // 常量引用类型别名
    typedef const T& const_reference;

    // 默认构造函数
    inline AlignmentAllocator() throw() {}

    // 模板拷贝构造函数，用于从同类型分配器构造
    template <typename T2>
    inline AlignmentAllocator(const AlignmentAllocator<T2, N>&) throw() {}

    // 析构函数
    inline ~AlignmentAllocator() throw() {}

    // 获取对象地址的方法
    inline pointer address(reference r) { return &r; }

    // 获取常量对象地址的方法
    inline const_pointer address(const_reference r) const { return &r; }

    // 分配内存的方法，确保内存按 N 字节对齐
    inline pointer allocate(size_type n) {
    // _aligned_malloc直接返回分配的内存指针
    pointer mem = (pointer)_aligned_malloc(n * sizeof(value_type), N);
    ICHECK(mem != nullptr) << "InternalError: failed to allocate aligned memory.";
    return mem;
    }

    // 释放内存的方法
    inline void deallocate(pointer p, size_type) {  _aligned_free(p); }

    // 在分配的内存上构造对象的方法
    inline void construct(pointer p, const value_type& wert) { new (p) value_type(wert); }

    // 销毁对象并释放内存的方法
    inline void destroy(pointer p) { p->~value_type(); }

    // 返回可以分配的最大对象数量的方法
    inline size_type max_size() const throw() { return size_type(-1) / sizeof(value_type); }

    // 模板重载，用于支持模板参数的重新绑定
    template <typename T2>
    struct rebind {
        typedef AlignmentAllocator<T2, N> other;
    };

    // 不等于操作符，用于比较两个分配器是否不相等
    bool operator!=(const AlignmentAllocator<T, N>& other) const { return !(*this == other); }

    // 等于操作符，用于比较两个分配器是否相等
    // 对于无状态的分配器，总是返回 true
    bool operator==(const AlignmentAllocator<T, N>& other) const { return true; }
};


/*!
* \brief 指令队列基类，作为所有特定队列类型的基类，完成了对指令队列内存的管理分配
*        可以生成队列的dram缓冲区（vector队列，使用分配器对齐内存）
*        使用SAAMemAlloc生成fpga的缓冲区
*        可以返回fpga缓冲区的物理地址
* \param T 队列存储类型的基本类型，elem_bytes_用于记录T的字节大小
*/
template <class T>
class BaseQueue {
 public:
    /*! \brief 构析函数回收内存，虚函数先回收派生类再回收基类 */
    virtual ~BaseQueue() {   
        if (fpga_buff_ != nullptr) {
        SAAMemFree(fpga_buff_);
        }
    }

    /*! \return DRAM缓冲区的内容。 */
    char* dram_buffer() const { return dram_buffer_; }

    /*! \return DRAM的物理地址。 */
    saa_phy_addr_t dram_phy_addr() const {
        return fpga_buff_phy_;
    }

    /*! \return 是否有待处理的信息。 */
    bool pending() const { return sram_begin_ != sram_end_; }

    /*! 
    * \brief 重置缓冲区的指针。
    *  将SRAM指针设置为当前结束位置。
    */
    virtual void Reset() {
        dram_buffer_.clear();
        // 重置为0，因为我们总是从fpga_buff基址开始复制数据
        // 每次DeviceRun都会进行内存复制
        sram_end_ = 0;
        sram_begin_ = sram_end_;
    }

    /*!
    * \brief 初始化缓冲区的空间。
    * 该函数用于配置缓冲区的参数，并预先分配所需的内存。
    * \param elem_bytes 每个元素的字节大小。
    * \param max_bytes 最大分配的字节数。
    * \param coherent 是否需要缓存一致性。
    * \param always_cache 是否总是缓存缓冲区。
    */
    void InitSpace(uint32_t elem_bytes, uint32_t max_bytes, bool coherent, bool always_cache) {
        coherent_ = coherent;
        always_cache_ = always_cache;
        elem_bytes_ = elem_bytes;
        // 提前分配缓冲区
        fpga_buff_ = static_cast<char*>(SAAMemAlloc(max_bytes, coherent_ || always_cache_)); // 转换为字节寻址;
        // assert(fpga_buff_ != nullptr);
        fpga_buff_phy_ = SAAMemGetPhyAddr(fpga_buff_); // 获取物理地址（windows中无用）
    }

 protected:
    // 缓存一致性访问（仅共享内存）
    bool coherent_{false};
    // 使缓冲区可缓存
    bool always_cache_{false};
    // 元素字节
    uint32_t elem_bytes_{0};
    // 当前SRAM读取的开始位置（FIFO模式）
    uint32_t sram_begin_{0};
    // 当前SRAM写入的结束位置（FIFO模式）
    uint32_t sram_end_{0};
    // DRAM中的缓冲区
    std::vector<T, AlignmentAllocator<T, ALLOC_ALIGNMENT>> dram_buffer_;
    // FPGA可访问的缓冲区
    void* fpga_buff_{NULL};
    // FPGA缓冲区的物理地址
    saa_phy_addr_t fpga_buff_phy_{0};
};

//------------------------------------insnqueue管理指令-------------------------------------//

// 定义枚举类型，表示指令流水线中的不同阶段
enum PipelineStage : int { kNoneStage = 0, kLoadStage = 1, kComputeStage = 2, kStoreStage = 3 };

/**
 * @brief 指令队列类，用于存储 GenericIns 类型的指令
 *        该队列继承自BaseQueue，可以使用BaseQueue管理指令队列的内存空间
 * @tparam kMaxBytes 队列最大字节数
 * @tparam kCoherent 是否需要缓存一致性
 * @tparam kAlwaysCache 是否总是缓存缓冲区
 */
template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class InsnQueue : public BaseQueue<GenericIns> {
public:
 
    /*! \brief 调用BaseQueue父类方法初始化BaseQueue的fpga的内存空间. */
    void InitSpace() {
        BaseQueue::InitSpace(kElemBytes, kMaxBytes, kCoherent, kAlwaysCache);
    }

    /*! \return 获取指向drma缓冲区（vector）的第一个指令的数据指针. */
    GenericIns* data() { return dram_buffer_.data(); }

    /*! \return 获取drma缓冲区（vector）存储的指令数量. */
    uint32_t count() { return dram_buffer_.size(); }

    /**
     * @brief 插入一个加载指令的依赖弹出操作（依赖关系在这里暂时不控制）
     *
     * 该函数用于在指令流水线中建立不同阶段之间的依赖关系。具体来说，
     * 它处理“弹出”操作，该操作会将指令从一个流水线阶段移除并推送到下一个阶段。
     * 依赖关系从“from”阶段推送到“to”阶段。
     *
     * @param from 要推送依赖关系的流水线阶段。
     * @param to 接受依赖关系的流水线阶段。
     */


    // 创建一个新的GEMM阶段指令
    ComIns* CreateGemInsn() { return reinterpret_cast<ComIns*>(Create(kComputeStage)); }
    // 创建一个新的加载阶段指令
    MemIns* CreateMemInsn() { return reinterpret_cast<MemIns*>(Create(kLoadStage));}
    // 创建一个新的存储阶段指令
    MemIns* CreateStoreInsn() { return reinterpret_cast<MemIns*>(Create(kStoreStage)); }


    
protected:
    /*! \return 向缓冲区添加新指令*/
    GenericIns* NextInsn() {
        GenericIns insn = {};
        dram_buffer_.push_back(insn);
        return &dram_buffer_.back();
    }

    //Create函数，通过输入的枚举指令类型，创建对应的指令
    GenericIns* Create(PipelineStage stage) {
        GenericIns* gptr = NextInsn();
        MemIns* mptr = reinterpret_cast<MemIns*>(gptr);
        return gptr;
    }

private:
    // 队列中元素的大小，就是初始化BaseQueue时的T的大小
    static constexpr int kElemBytes = sizeof(GenericIns);    
    // 常量表达式，编译器直接计算得到结果，计算得到初始化的BaseQueue中fpga_buff_最多能够存储多少个kElemBytes
    static constexpr int kMaxElems = kMaxBytes / kElemBytes; 
};

}





//-----------------------------调用saa命名空间的内存操作----------------------------//

/*! \brief SAA命令句柄 */
typedef void* SAACommandHandle;

/*!
*\brief DataBuffer类分配数据缓冲区。
*\param size缓冲区大小。
*\return 指向已分配缓冲区的指针。
*/
void* SAABufferAlloc(size_t size) { return saa::DataBuffer::Alloc(size); }

/*!
*\brief 释放数据缓冲区，将指针转换为DataBuffer类然后Free释放掉该类指向的内存。
*\param buffer要释放的数据缓冲区。
*/
void SAABufferFree(void* buffer) { saa::DataBuffer::Free(saa::DataBuffer::FromHandle(buffer)); }

/*!
*\brief 将数据缓冲区从一个位置复制到另一个位置。需要能够识别是从FPGA到内存还是内存到FPGA。
*\param from 源缓冲区基地址。
*\param from_offset 源缓冲区的偏移量。
*\param to 目标缓冲区基地址。
*\param to_offset 目标缓冲区的偏移量。
*\param size 副本的大小。
*\param kind_mask 内存复制类型。
* 注意，如果是实际的FPGA和CPU的内存交互，一定要根据是写到FPGA的内存还是FPGA的内存写入正常的内存区分，使用cache
*/









//总的指令缓冲区，可以容纳1000条指令
#define INSTRUCTION_BUFFER_WEIGHT 1000

// 统计指令信息的结构体
struct InstructionStruct {
    Instruct_DataType instruction_data[INSTRUCTION_BUFFER_WEIGHT]; // 指令数据缓冲区，用于传输指令给SAA
    int total_count; // 总共生成了多少指令
};

// 指令


// 生成加载指令的函数，根据当前矩阵的行列生成一批加载指令，暂时只能加载MATRIX_WIDTH倍数的矩阵
void generate_load_instructions(
    InstructionStruct* instruction_struct, // 传入结构体的地址
    int total_rows, // 总行数
    int total_cols, // 总列数
    Buffer_Id_DataType buffer_id, // 当前矩阵加载到哪个缓冲区
    Dram_Addr_DataType dram_start, // 读取位置相对于缓冲区的偏移，如果矩阵直接就从0存储，那这就是0，以总线为偏移基本单位
    Buffer_Addr_DataType buffer_start) // 写入位置相对于buffer起始行的偏移，如果直接从0行存储，那就是0
{
    // 计算分多少个块，就生成多少个指令
    const int row_block = total_rows / MATRIX_WIDTH; // 行分多少个块
    const int col_block = total_cols / MATRIX_WIDTH; // 列分多少个块
    int insn_count =  row_block * col_block;  // 总的分块数等于行*列
    SAAInsn instruction[insn_count]; // 使用总的分块数生成指令数组

     // 计算当前结构体指针的赋值位置

    //循环块生成加载指令
    for (int row = 0; row < row_block; ++row) {
        for (int col = 0; col < col_block; ++col) {
            const int block = row*col_block+col; // 第几个块
            const int buffer_base = (buffer_start + (block)*MATRIX_WIDTH); // 计算buffer输入地址
            const int dram_base = dram_start + row * col_block * MATRIX_WIDTH*MATRIX_WIDTH 
                                             + col * MATRIX_WIDTH; // 计算dram读取地址
            // 写入指令结构体
            instruction[block].mem.opcode = OPCODE_LOAD;  // 加载指令
            instruction[block].mem.buffer_id = buffer_id; // 存储在哪个缓冲区
            instruction[block].mem.dram_base = dram_base; // 缓冲区偏移+矩阵内部偏移
            instruction[block].mem.buffer_base = buffer_base; // buffer偏移+矩阵内部偏移
            instruction[block].mem.y_size = MATRIX_WIDTH; // 每次加载MATRIX_WIDTH行
            instruction[block].mem.x_size = MATRIX_WIDTH; // 每次加载MATRIX_WIDTH列
            instruction[block].mem.x_stride = total_cols; // 假设每行数据在DRAM中是连续存储的，那么步长就是列宽

            // 转换指令结构体为128位指令数据类型
            std::memcpy(&instruction_struct->instruction_data[instruction_struct->total_count+block], 
                        &instruction[block], sizeof(SAAInsn));
        }
    }
    instruction_struct->total_count = instruction_struct->total_count + insn_count ; // 计算当前总指令数
}

#endif





