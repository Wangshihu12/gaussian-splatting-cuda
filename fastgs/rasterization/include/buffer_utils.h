#pragma once

#include "helper_math.h"
#include "rasterization_config.h"
#include <cstdint>
#include <cub/cub.cuh>

namespace fast_gs::rasterization {

    /**
     * [功能描述]：3x3矩阵结构体，用于表示完整的3D变换矩阵
     * 包含9个浮点数值，表示3x3矩阵的所有元素
     */
    struct mat3x3 {
        float m11, m12, m13;  // 第一行：m11, m12, m13
        float m21, m22, m23;  // 第二行：m21, m22, m23
        float m31, m32, m33;  // 第三行：m31, m32, m33
    };

    /**
     * [功能描述]：3x3上三角矩阵结构体，只存储上三角部分的6个元素
     * 使用8字节对齐优化内存访问性能
     * 只存储：m11, m12, m13, m22, m23, m33
     */
    struct __align__(8) mat3x3_triu {
        float m11, m12, m13, m22, m23, m33;  // 上三角矩阵的6个非零元素
    };

    /**
     * [功能描述]：从blob中分配指定类型和数量的内存，并更新blob指针
     * @param blob 输入输出：内存blob的指针引用
     * @param ptr 输出：分配的内存指针
     * @param count 需要分配的元素数量
     * @param alignment 内存对齐要求
     */
    template <typename T>
    static void obtain(char*& blob, T*& ptr, std::size_t count, std::size_t alignment) {
        // 计算对齐后的偏移地址
        std::size_t offset = reinterpret_cast<std::uintptr_t>(blob) + alignment - 1 & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);  // 将偏移地址转换为目标类型指针
        blob = reinterpret_cast<char*>(ptr + count);  // 更新blob指针到下一个可用位置
    }

    /**
     * [功能描述]：计算指定结构体类型所需的blob大小
     * @param P 主要参数（通常是样条数量）
     * @param args 其他参数
     * @return 所需的blob大小（字节数）
     */
    template <typename T, typename... Args>
    size_t required(size_t P, Args... args) {
        char* size = nullptr;
        T::from_blob(size, P, args...);  // 调用from_blob计算所需大小
        return ((size_t)size) + 128;     // 返回计算的大小加上128字节的安全余量
    }

    /**
     * [功能描述]：每个样条的缓冲区结构体，包含样条处理所需的所有数据
     */
    struct PerPrimitiveBuffers {
        size_t cub_workspace_size;                    // CUB库工作空间大小
        char* cub_workspace;                          // CUB库工作空间指针
        cub::DoubleBuffer<uint> depth_keys;           // 深度键的双缓冲（用于排序）
        cub::DoubleBuffer<uint> primitive_indices;    // 样条索引的双缓冲（用于排序）
        uint* n_touched_tiles;                        // 每个样条触及的瓦片数量
        uint* offset;                                 // 每个样条的偏移量
        ushort4* screen_bounds;                       // 屏幕边界（4个short值）
        float2* mean2d;                               // 2D投影中心点
        float4* conic_opacity;                        // 圆锥体参数和不透明度
        float3* color;                                // 样条颜色
        uint* n_visible_primitives;                   // 可见样条数量
        uint* n_instances;                            // 实例数量

        /**
         * [功能描述]：从blob中创建PerPrimitiveBuffers结构体
         * @param blob 输入输出：内存blob的指针引用
         * @param n_primitives 样条总数
         * @return 初始化完成的PerPrimitiveBuffers结构体
         */
        static PerPrimitiveBuffers from_blob(char*& blob, size_t n_primitives) {
            PerPrimitiveBuffers buffers;
            
            // 分配深度键的双缓冲内存
            uint* depth_keys_current;
            obtain(blob, depth_keys_current, n_primitives, 128);
            uint* depth_keys_alternate;
            obtain(blob, depth_keys_alternate, n_primitives, 128);
            buffers.depth_keys = cub::DoubleBuffer<uint>(depth_keys_current, depth_keys_alternate);
            
            // 分配样条索引的双缓冲内存
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_primitives, 128);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_primitives, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            
            // 分配其他缓冲区内存
            obtain(blob, buffers.n_touched_tiles, n_primitives, 128);
            obtain(blob, buffers.offset, n_primitives, 128);
            obtain(blob, buffers.screen_bounds, n_primitives, 128);
            obtain(blob, buffers.mean2d, n_primitives, 128);
            obtain(blob, buffers.conic_opacity, n_primitives, 128);
            obtain(blob, buffers.color, n_primitives, 128);
            
            // 计算CUB库的ExclusiveSum操作所需的工作空间大小
            cub::DeviceScan::ExclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.offset, buffers.offset,
                n_primitives);
            
            // 计算CUB库的基数排序所需的工作空间大小
            size_t sorting_workspace_size;
            cub::DeviceRadixSort::SortPairs(
                nullptr, sorting_workspace_size,
                buffers.depth_keys, buffers.primitive_indices,
                n_primitives);
            
            // 取两个工作空间大小的最大值
            buffers.cub_workspace_size = max(buffers.cub_workspace_size, sorting_workspace_size);
            
            // 分配CUB工作空间和计数器内存
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            obtain(blob, buffers.n_visible_primitives, 1, 128);
            obtain(blob, buffers.n_instances, 1, 128);
            
            return buffers;
        }
    };

    /**
     * [功能描述]：每个实例的缓冲区结构体，用于管理实例级别的数据
     */
    struct PerInstanceBuffers {
        size_t cub_workspace_size;                    // CUB库工作空间大小
        char* cub_workspace;                          // CUB库工作空间指针
        cub::DoubleBuffer<ushort> keys;               // 键值的双缓冲（用于排序）
        cub::DoubleBuffer<uint> primitive_indices;    // 样条索引的双缓冲（用于排序）

        /**
         * [功能描述]：从blob中创建PerInstanceBuffers结构体
         * @param blob 输入输出：内存blob的指针引用
         * @param n_instances 实例总数
         * @return 初始化完成的PerInstanceBuffers结构体
         */
        static PerInstanceBuffers from_blob(char*& blob, size_t n_instances) {
            PerInstanceBuffers buffers;
            
            // 分配键值的双缓冲内存
            ushort* keys_current;
            obtain(blob, keys_current, n_instances, 128);
            ushort* keys_alternate;
            obtain(blob, keys_alternate, n_instances, 128);
            buffers.keys = cub::DoubleBuffer<ushort>(keys_current, keys_alternate);
            
            // 分配样条索引的双缓冲内存
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_instances, 128);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_instances, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            
            // 计算CUB库的基数排序所需的工作空间大小
            cub::DeviceRadixSort::SortPairs(
                nullptr, buffers.cub_workspace_size,
                buffers.keys, buffers.primitive_indices,
                n_instances);
            
            // 分配CUB工作空间内存
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            
            return buffers;
        }
    };

    /**
     * [功能描述]：每个瓦片的缓冲区结构体，用于管理瓦片级别的数据
     */
    struct PerTileBuffers {
        size_t cub_workspace_size;                    // CUB库工作空间大小
        char* cub_workspace;                          // CUB库工作空间指针
        uint2* instance_ranges;                       // 每个瓦片的实例范围（起始和结束索引）
        uint* n_buckets;                              // 每个瓦片的桶数量
        uint* bucket_offsets;                         // 每个瓦片的桶偏移量
        uint* max_n_contributions;                    // 每个瓦片的最大贡献数
        uint* n_contributions;                        // 每个像素的贡献数

        /**
         * [功能描述]：从blob中创建PerTileBuffers结构体
         * @param blob 输入输出：内存blob的指针引用
         * @param n_tiles 瓦片总数
         * @return 初始化完成的PerTileBuffers结构体
         */
        static PerTileBuffers from_blob(char*& blob, size_t n_tiles) {
            PerTileBuffers buffers;
            
            // 分配瓦片相关的缓冲区内存
            obtain(blob, buffers.instance_ranges, n_tiles, 128);
            obtain(blob, buffers.n_buckets, n_tiles, 128);
            obtain(blob, buffers.bucket_offsets, n_tiles, 128);
            obtain(blob, buffers.max_n_contributions, n_tiles, 128);
            // 贡献数数组的大小是瓦片数乘以混合块大小
            obtain(blob, buffers.n_contributions, n_tiles * config::block_size_blend, 128);
            
            // 计算CUB库的InclusiveSum操作所需的工作空间大小
            cub::DeviceScan::InclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.n_buckets, buffers.bucket_offsets,
                n_tiles);
            
            // 分配CUB工作空间内存
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            
            return buffers;
        }
    };

    /**
     * [功能描述]：每个桶的缓冲区结构体，用于管理桶级别的数据
     */
    struct PerBucketBuffers {
        uint* tile_index;                             // 桶对应的瓦片索引
        float4* color_transmittance;                  // 颜色和透射率数据

        /**
         * [功能描述]：从blob中创建PerBucketBuffers结构体
         * @param blob 输入输出：内存blob的指针引用
         * @param n_buckets 桶的总数
         * @return 初始化完成的PerBucketBuffers结构体
         */
        static PerBucketBuffers from_blob(char*& blob, size_t n_buckets) {
            PerBucketBuffers buffers;
            
            // 分配桶相关的缓冲区内存，大小是桶数乘以混合块大小
            obtain(blob, buffers.tile_index, n_buckets * config::block_size_blend, 128);
            obtain(blob, buffers.color_transmittance, n_buckets * config::block_size_blend, 128);
            
            return buffers;
        }
    };

} // namespace fast_gs::rasterization
