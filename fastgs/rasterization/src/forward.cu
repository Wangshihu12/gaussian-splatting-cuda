#include "buffer_utils.h"
#include "forward.h"
#include "helper_math.h"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <functional>

// 排序分别针对深度和瓦片进行，如 https://github.com/m-schuetz/Splatshop 中提出的方法
/**
 * [功能描述]：快速高斯溅射光栅化的核心前向传播函数。
 * 这个函数实现了完整的高斯溅射光栅化流程，包括预处理、深度排序、实例创建、
 * 瓦片处理和最终混合等关键步骤。整个流程经过精心优化，实现了高效的3D到2D投影渲染。
 * 
 * 返回值是一个包含5个元素的元组，包含光栅化统计信息和索引选择器。
 * 
 * @param per_primitive_buffers_func [参数说明]：图元缓冲区调整函数，用于动态分配图元级别的内存。
 * @param per_tile_buffers_func [参数说明]：瓦片缓冲区调整函数，用于动态分配瓦片级别的内存。
 * @param per_instance_buffers_func [参数说明]：实例缓冲区调整函数，用于动态分配实例级别的内存。
 * @param per_bucket_buffers_func [参数说明]：桶缓冲区调整函数，用于动态分配桶级别的内存。
 * @param means [参数说明]：高斯中心位置数组，形状为[N, 3]，N是高斯数量。
 * @param scales_raw [参数说明]：原始缩放参数数组，形状为[N, 3]。
 * @param rotations_raw [参数说明]：原始旋转参数数组，形状为[N, 4]，使用四元数表示。
 * @param opacities_raw [参数说明]：原始不透明度数组，形状为[N, 1]。
 * @param sh_coefficients_0 [参数说明]：球谐函数0阶系数数组，形状为[N, 1, 3]。
 * @param sh_coefficients_rest [参数说明]：球谐函数高阶系数数组，形状为[N, K, 3]，K是基函数数量。
 * @param w2c [参数说明]：世界坐标系到相机坐标系的变换矩阵，形状为[4, 4]。
 * @param cam_position [参数说明]：相机在世界坐标系中的位置，形状为[3]。
 * @param image [参数说明]：输出图像缓冲区，形状为[3, H, W]。
 * @param alpha [参数说明]：输出透明度缓冲区，形状为[1, H, W]。
 * @param n_primitives [参数说明]：高斯图元的总数量。
 * @param active_sh_bases [参数说明]：当前活跃的球谐函数基函数数量。
 * @param total_bases_sh_rest [参数说明]：高阶球谐函数的基函数总数。
 * @param width [参数说明]：输出图像的宽度。
 * @param height [参数说明]：输出图像的高度。
 * @param fx [参数说明]：x方向焦距。
 * @param fy [参数说明]：y方向焦距。
 * @param cx [参数说明]：x方向主点坐标。
 * @param cy [参数说明]：y方向主点坐标。
 * @param near_ [参数说明]：近裁剪平面距离（避免Windows宏冲突）。
 * @param far_ [参数说明]：远裁剪平面距离。
 * @return [返回值说明]：返回包含5个元素的元组，详见函数末尾的返回语句。
 */
std::tuple<int, int, int, int, int> fast_gs::rasterization::forward(
    std::function<char*(size_t)> per_primitive_buffers_func,    // 图元缓冲区调整函数
    std::function<char*(size_t)> per_tile_buffers_func,         // 瓦片缓冲区调整函数
    std::function<char*(size_t)> per_instance_buffers_func,     // 实例缓冲区调整函数
    std::function<char*(size_t)> per_bucket_buffers_func,       // 桶缓冲区调整函数
    const float3* means,                                        // 高斯中心位置数组
    const float3* scales_raw,                                   // 原始缩放参数数组
    const float4* rotations_raw,                                // 原始旋转参数数组
    const float* opacities_raw,                                 // 原始不透明度数组
    const float3* sh_coefficients_0,                            // 球谐函数0阶系数数组
    const float3* sh_coefficients_rest,                         // 球谐函数高阶系数数组
    const float4* w2c,                                          // 世界到相机的变换矩阵
    const float3* cam_position,                                 // 相机位置
    float* image,                                                // 输出图像缓冲区
    float* alpha,                                                // 输出透明度缓冲区
    const int n_primitives,                                     // 高斯图元总数
    const int active_sh_bases,                                   // 活跃的球谐函数基函数数量
    const int total_bases_sh_rest,                               // 高阶球谐函数的基函数总数
    const int width,                                             // 图像宽度
    const int height,                                            // 图像高度
    const float fx,                                              // x方向焦距
    const float fy,                                              // y方向焦距
    const float cx,                                              // x方向主点坐标
    const float cy,                                              // y方向主点坐标
    const float near_,                                           // 近裁剪平面（避免Windows宏冲突）
    const float far_) {                                          // 远裁剪平面
    
    // 步骤1：计算CUDA网格和块配置
    // 根据图像尺寸和瓦片大小计算CUDA内核的网格和块配置
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);  // 瓦片网格配置
    const dim3 block(config::tile_width, config::tile_height, 1);                                           // 瓦片块配置
    const int n_tiles = grid.x * grid.y;                                                                    // 总瓦片数量

    // 步骤2：分配和初始化瓦片缓冲区
    // 为每个瓦片分配必要的缓冲区内存，包括实例范围、桶偏移等信息
    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));  // 分配瓦片缓冲区内存
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);  // 从内存块创建瓦片缓冲区对象

    // 步骤3：异步初始化瓦片缓冲区（性能优化）
    // 使用CUDA流进行异步内存清零，提高性能
    static cudaStream_t memset_stream = 0;  // 静态CUDA流，用于异步操作
    if constexpr (!config::debug) {
        // 非调试模式：使用异步流进行内存清零
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);  // 创建CUDA流
            memset_stream_initialized = true;
        }
        // 异步清零实例范围数组，提高性能
        cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
    } else
        // 调试模式：同步清零，便于调试
        cudaMemset(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);

    // 步骤4：分配和初始化图元缓冲区
    // 为每个图元分配必要的缓冲区内存，包括深度键、索引、屏幕边界等
    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives));  // 分配图元缓冲区内存
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);  // 从内存块创建图元缓冲区对象

    // 初始化图元统计计数器
    cudaMemset(per_primitive_buffers.n_visible_primitives, 0, sizeof(uint));  // 清零可见图元计数
    cudaMemset(per_primitive_buffers.n_instances, 0, sizeof(uint));           // 清零实例计数

    // 步骤5：预处理阶段 - 计算图元的2D投影和可见性
    // 这是光栅化的第一步，将3D高斯投影到2D屏幕空间
    kernels::forward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        means,                                                                    // 高斯中心位置
        scales_raw,                                                               // 原始缩放参数
        rotations_raw,                                                            // 原始旋转参数
        opacities_raw,                                                            // 原始不透明度
        sh_coefficients_0,                                                        // 球谐函数0阶系数
        sh_coefficients_rest,                                                     // 球谐函数高阶系数
        w2c,                                                                      // 世界到相机的变换矩阵
        cam_position,                                                             // 相机位置
        per_primitive_buffers.depth_keys.Current(),                               // 深度键输出缓冲区
        per_primitive_buffers.primitive_indices.Current(),                        // 图元索引输出缓冲区
        per_primitive_buffers.n_touched_tiles,                                    // 触摸瓦片数量输出缓冲区
        per_primitive_buffers.screen_bounds,                                      // 屏幕边界输出缓冲区
        per_primitive_buffers.mean2d,                                             // 2D投影位置输出缓冲区
        per_primitive_buffers.conic_opacity,                                      // 圆锥体不透明度输出缓冲区
        per_primitive_buffers.color,                                              // 颜色输出缓冲区
        per_primitive_buffers.n_visible_primitives,                               // 可见图元计数输出缓冲区
        per_primitive_buffers.n_instances,                                        // 实例计数输出缓冲区
        n_primitives,                                                             // 图元总数
        grid.x,                                                                   // 瓦片网格宽度
        grid.y,                                                                   // 瓦片网格高度
        active_sh_bases,                                                          // 活跃的球谐函数基函数数量
        total_bases_sh_rest,                                                      // 高阶球谐函数的基函数总数
        static_cast<float>(width),                                                // 图像宽度（转换为float）
        static_cast<float>(height),                                               // 图像高度（转换为float）
        fx,                                                                       // x方向焦距
        fy,                                                                       // y方向焦距
        cx,                                                                       // x方向主点坐标
        cy,                                                                       // y方向主点坐标
        near_,                                                                    // 近裁剪平面
        far_);                                                                    // 远裁剪平面
    CHECK_CUDA(config::debug, "preprocess")  // 检查CUDA错误

    // 步骤6：获取预处理结果统计信息
    // 将设备内存中的统计信息复制到主机内存，用于后续处理
    int n_visible_primitives;  // 可见图元数量
    // 从GPU复制可见图元数量
    // 源地址：per_primitive_buffers.n_visible_primitives（GPU设备内存）
    // 目标地址：&n_visible_primitives（CPU主机内存）
    // 数据大小：sizeof(uint)（4字节）
    // 传输方向：cudaMemcpyDeviceToHost（从GPU到CPU）
    cudaMemcpy(&n_visible_primitives, per_primitive_buffers.n_visible_primitives, sizeof(uint), cudaMemcpyDeviceToHost);
    int n_instances;  // 实例数量
    cudaMemcpy(&n_instances, per_primitive_buffers.n_instances, sizeof(uint), cudaMemcpyDeviceToHost);

    // 步骤7：深度排序 - 使用CUB库进行基数排序
    // 根据深度键对可见图元进行排序，确保正确的深度顺序
    cub::DeviceRadixSort::SortPairs(
        per_primitive_buffers.cub_workspace,                                      // CUB工作空间
        per_primitive_buffers.cub_workspace_size,                                // 工作空间大小
        per_primitive_buffers.depth_keys,                                        // 深度键输入/输出
        per_primitive_buffers.primitive_indices,                                 // 图元索引输入/输出
        n_visible_primitives);                                                   // 排序元素数量
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Depth)")  // 检查CUDA错误

    // 步骤8：应用深度排序 - 重新排列相关数组
    // 根据排序后的图元索引重新排列触摸瓦片数量、屏幕边界等数组
    kernels::forward::apply_depth_ordering_cu<<<div_round_up(n_visible_primitives, config::block_size_apply_depth_ordering), config::block_size_apply_depth_ordering>>>(
        per_primitive_buffers.primitive_indices.Current(),                        // 排序后的图元索引
        per_primitive_buffers.n_touched_tiles,                                    // 触摸瓦片数量数组
        per_primitive_buffers.offset,                                            // 偏移量输出数组
        n_visible_primitives);                                                   // 可见图元数量
    CHECK_CUDA(config::debug, "apply_depth_ordering")  // 检查CUDA错误

    // 步骤9：计算图元偏移量 - 使用CUB库进行前缀和
    // 计算每个图元在实例数组中的起始位置
    cub::DeviceScan::ExclusiveSum(
        per_primitive_buffers.cub_workspace,                                      // CUB工作空间
        per_primitive_buffers.cub_workspace_size,                                // 工作空间大小
        per_primitive_buffers.offset,                                            // 输入偏移量数组
        per_primitive_buffers.offset,                                            // 输出偏移量数组
        n_visible_primitives);                                                   // 数组长度
    CHECK_CUDA(config::debug, "cub::DeviceScan::ExclusiveSum (Primitive Offsets)")  // 检查CUDA错误

    // 步骤10：分配和初始化实例缓冲区
    // 为每个实例分配必要的缓冲区内存，包括键、图元索引等
    char* per_instance_buffers_blob = per_instance_buffers_func(required<PerInstanceBuffers>(n_instances));  // 分配实例缓冲区内存
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances);  // 从内存块创建实例缓冲区对象

    // 步骤11：创建实例 - 为每个图元在每个触摸的瓦片中创建实例
    // 这是光栅化的关键步骤，将图元转换为瓦片级别的实例
    kernels::forward::create_instances_cu<<<div_round_up(n_visible_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
        per_primitive_buffers.primitive_indices.Current(),                        // 排序后的图元索引
        per_primitive_buffers.offset,                                            // 图元偏移量
        per_primitive_buffers.screen_bounds,                                      // 屏幕边界
        per_primitive_buffers.mean2d,                                             // 2D投影位置
        per_primitive_buffers.conic_opacity,                                      // 圆锥体不透明度
        per_instance_buffers.keys.Current(),                                      // 实例键输出缓冲区
        per_instance_buffers.primitive_indices.Current(),                         // 实例图元索引输出缓冲区
        grid.x,                                                                   // 瓦片网格宽度
        n_visible_primitives);                                                   // 可见图元数量
    CHECK_CUDA(config::debug, "create_instances")  // 检查CUDA错误

    // 步骤12：瓦片排序 - 使用CUB库对实例进行瓦片排序
    // 根据瓦片键对实例进行排序，确保同一瓦片内的实例连续存储
    cub::DeviceRadixSort::SortPairs(
        per_instance_buffers.cub_workspace,                                       // CUB工作空间
        per_instance_buffers.cub_workspace_size,                                 // 工作空间大小
        per_instance_buffers.keys,                                               // 实例键输入/输出
        per_instance_buffers.primitive_indices,                                  // 实例图元索引输入/输出
        n_instances);                                                            // 实例数量
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Tile)")  // 检查CUDA错误

    // 步骤13：同步异步流（非调试模式）
    // 确保所有异步操作完成后再继续
    if constexpr (!config::debug)
        cudaStreamSynchronize(memset_stream);  // 同步异步流

    // 步骤14：提取实例范围（如果有实例）
    // 计算每个瓦片中实例的起始和结束索引
    if (n_instances > 0) {
        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            per_instance_buffers.keys.Current(),                                  // 排序后的实例键
            per_tile_buffers.instance_ranges,                                     // 实例范围输出缓冲区
            n_instances);                                                         // 实例数量
        CHECK_CUDA(config::debug, "extract_instance_ranges")  // 检查CUDA错误
    }

    // 步骤15：提取桶计数 - 计算每个瓦片中的桶数量
    // 桶是瓦片内的子区域，用于组织贡献到同一像素的图元
    kernels::forward::extract_bucket_counts<<<div_round_up(n_tiles, config::block_size_extract_bucket_counts), config::block_size_extract_bucket_counts>>>(
        per_tile_buffers.instance_ranges,                                        // 实例范围
        per_tile_buffers.n_buckets,                                              // 桶计数输出缓冲区
        n_tiles);                                                                // 瓦片数量
    CHECK_CUDA(config::debug, "extract_bucket_counts")  // 检查CUDA错误

    // 步骤16：计算桶偏移量 - 使用CUB库进行前缀和
    // 计算每个瓦片中桶的起始位置
    cub::DeviceScan::InclusiveSum(
        per_tile_buffers.cub_workspace,                                          // CUB工作空间
        per_tile_buffers.cub_workspace_size,                                     // 工作空间大小
        per_tile_buffers.n_buckets,                                              // 输入桶计数数组
        per_tile_buffers.bucket_offsets,                                         // 输出桶偏移量数组
        n_tiles);                                                                // 数组长度
    CHECK_CUDA(config::debug, "cub::DeviceScan::InclusiveSum (Bucket Counts)")  // 检查CUDA错误

    // 步骤17：获取总桶数量
    // 从最后一个瓦片的桶偏移量中获取总桶数量
    int n_buckets;
    cudaMemcpy(&n_buckets, per_tile_buffers.bucket_offsets + n_tiles - 1, sizeof(uint), cudaMemcpyDeviceToHost);

    // 步骤18：分配和初始化桶缓冲区
    // 为每个桶分配必要的缓冲区内存，包括瓦片索引、颜色透射率等
    char* per_bucket_buffers_blob = per_bucket_buffers_func(required<PerBucketBuffers>(n_buckets));  // 分配桶缓冲区内存
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(per_bucket_buffers_blob, n_buckets);  // 从内存块创建桶缓冲区对象

    // 步骤19：最终混合阶段 - 将图元贡献混合到输出图像
    // 这是光栅化的最后一步，将排序后的图元按深度顺序混合到最终图像
    kernels::forward::blend_cu<<<grid, block>>>(                                // 使用瓦片网格和块配置
        per_tile_buffers.instance_ranges,                                        // 实例范围
        per_tile_buffers.bucket_offsets,                                         // 桶偏移量
        per_instance_buffers.primitive_indices.Current(),                         // 实例图元索引
        per_primitive_buffers.mean2d,                                             // 2D投影位置
        per_primitive_buffers.conic_opacity,                                      // 圆锥体不透明度
        per_primitive_buffers.color,                                              // 颜色
        image,                                                                    // 输出图像缓冲区
        alpha,                                                                    // 输出透明度缓冲区
        per_tile_buffers.max_n_contributions,                                     // 最大贡献数量
        per_tile_buffers.n_contributions,                                         // 贡献数量
        per_bucket_buffers.tile_index,                                            // 桶瓦片索引
        per_bucket_buffers.color_transmittance,                                   // 桶颜色透射率
        width,                                                                    // 图像宽度
        height,                                                                   // 图像高度
        grid.x);                                                                  // 瓦片网格宽度
    CHECK_CUDA(config::debug, "blend")  // 检查CUDA错误

    // 步骤20：返回光栅化统计信息和索引选择器
    // 返回包含5个元素的元组，用于后续处理和调试
    return {
        n_visible_primitives,                                                     // 可见图元数量
        n_instances,                                                              // 实例数量
        n_buckets,                                                                // 桶数量
        per_primitive_buffers.primitive_indices.selector,                         // 图元索引选择器
        per_instance_buffers.primitive_indices.selector                           // 实例索引选择器
    };
}
