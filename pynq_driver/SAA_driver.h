//本文件定义了SAA的驱动函数也就是SAA对平台的基本驱动
#ifndef SAA_DRIVER_H
#define SAA_DRIVER_H


//--------------------SAA内存管理--------------------//
#include <stdint.h> // 包含对uint32_t等类型的支持
#include <windows.h>
#include <assert.h>
#include <cstring>

// 定义物理地址类型
typedef uint64_t saa_phy_addr_t;


// 定义缓存标志
#define SAA_CACHED 1
#define SAA_NOT_CACHED 0

// 定义页面大小（根据Windows系统实际情况调整）
#define SAA_PAGE_SIZE 4096 // Windows页面大小通常是4KB

// 分配一块物理上连续的内存区域，供FPGA使用
void* SAAMemAlloc(size_t size, int cached) {
    // 确保请求的内存大小不超过预定义的最大传输大小
    // 使用Windows API VirtualAlloc来分配内存
    void* addr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, (cached == SAA_CACHED) ? PAGE_READWRITE : PAGE_WRITECOMBINE);
    return addr;
}

// 释放之前分配的内存区域
void SAAMemFree(void* buf) {
    // 使用Windows API VirtualFree来释放内存
    VirtualFree(buf, 0, MEM_RELEASE);
}

// 从主机内存复制数据到FPGA可访问的内存
void SAAMemCopyFromHost(void* dst, const void* src, size_t size) {
    // 使用memcpy进行内存复制
    memcpy(dst, src, size);
}

// 从FPGA可访问的内存复制数据到主机内存
void SAAMemCopyToHost(void* dst, const void* src, size_t size) {
    // 使用memcpy进行内存复制
    memcpy(dst, src, size);
}

// 获取之前分配的内存区域的物理地址
saa_phy_addr_t SAAMemGetPhyAddr(void* buf) {
    // 在Windows上，通常不需要获取物理地址，因为VirtualAlloc分配的内存已经是连续的
    // 如果需要获取物理地址，可能需要使用更底层的硬件接口或第三方库
    return static_cast<saa_phy_addr_t>(0); // 这里简单地返回0作为物理地址
}

// 刷新CPU缓存，使FPGA可以读取最新的内存数据
void SAAFlushCache(void* vir_addr, saa_phy_addr_t phy_addr, int size) {
    // 在Windows上，通常不需要手动刷新缓存，因为系统会自动处理
    // 如果使用的是特定硬件，可能需要调用硬件供应商提供的API来刷新缓存
}

// 使CPU缓存的数据失效，强制CPU下次读取时从内存中重新获取数据
void SAAInvalidateCache(void* vir_addr, saa_phy_addr_t phy_addr, int size) {
    // 同上，如果硬件需要，可能需要调用硬件供应商提供的API来使缓存失效
}


//----------------SAA硬件寄存器映射------------------//



//----------------SAA硬件寄存器驱动------------------//




#endif