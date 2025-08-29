#include "backward.h"
#include "buffer_utils.h"
#include "helper_math.h"
#include "kernels_backward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <functional>

/**
 * [功能描述]：高斯样条渲染的反向传播主函数，计算渲染损失相对于所有参数的梯度
 * @param grad_image 图像颜色的梯度数组
 * @param grad_alpha 透明度图的梯度数组
 * @param image 前向传播生成的图像数组
 * @param alpha 前向传播生成的透明度图数组
 * @param means 3D高斯样条中心点坐标数组
 * @param scales_raw 原始缩放参数数组（对数形式）
 * @param rotations_raw 原始旋转四元数数组
 * @param sh_coefficients_rest 球谐函数系数数组（除第0阶外）
 * @param w2c 世界坐标系到相机坐标系的变换矩阵
 * @param cam_position 相机位置
 * @param per_primitive_buffers_blob 每个样条的缓冲区数据块
 * @param per_tile_buffers_blob 每个瓦片的缓冲区数据块
 * @param per_instance_buffers_blob 每个实例的缓冲区数据块
 * @param per_bucket_buffers_blob 每个桶的缓冲区数据块
 * @param grad_means 输出：3D中心点的梯度数组
 * @param grad_scales_raw 输出：原始缩放参数的梯度数组
 * @param grad_rotations_raw 输出：原始旋转四元数的梯度数组
 * @param grad_opacities_raw 输出：原始不透明度的梯度数组
 * @param grad_sh_coefficients_0 输出：第0阶球谐系数的梯度数组
 * @param grad_sh_coefficients_rest 输出：高阶球谐系数的梯度数组
 * @param grad_mean2d_helper 输出：2D投影中心点的辅助梯度数组
 * @param grad_conic_helper 输出：圆锥体参数的辅助梯度数组
 * @param grad_w2c 输出：世界到相机变换矩阵的梯度数组
 * @param densification_info 密度控制信息数组
 * @param n_primitives 样条总数
 * @param n_visible_primitives 可见样条数量
 * @param n_instances 实例总数
 * @param n_buckets 桶的总数
 * @param primitive_primitive_indices_selector 样条索引选择器
 * @param instance_primitive_indices_selector 实例索引选择器
 * @param active_sh_bases 活跃的球谐函数基数量
 * @param total_bases_sh_rest 高阶球谐函数基总数
 * @param width 图像宽度
 * @param height 图像高度
 * @param fx 相机内参fx
 * @param fy 相机内参fy
 * @param cx 相机内参cx
 * @param cy 相机内参cy
 */
void fast_gs::rasterization::backward(
    const float* grad_image,
    const float* grad_alpha,
    const float* image,
    const float* alpha,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    char* per_instance_buffers_blob,
    char* per_bucket_buffers_blob,
    float3* grad_means,
    float3* grad_scales_raw,
    float4* grad_rotations_raw,
    float* grad_opacities_raw,
    float3* grad_sh_coefficients_0,
    float3* grad_sh_coefficients_rest,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float4* grad_w2c,
    float* densification_info,
    const int n_primitives,
    const int n_visible_primitives,
    const int n_instances,
    const int n_buckets,
    const int primitive_primitive_indices_selector,
    const int instance_primitive_indices_selector,
    const int active_sh_bases,
    const int total_bases_sh_rest,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy) {
    
    // 计算瓦片网格的尺寸
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = grid.x * grid.y;  // 计算总瓦片数

    // 从数据块中解析各种缓冲区结构
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances);
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(per_bucket_buffers_blob, n_buckets);
    
    // 设置索引选择器，用于选择正确的索引数组
    per_primitive_buffers.primitive_indices.selector = primitive_primitive_indices_selector;
    per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    // 第一步：调用混合反向传播内核，计算混合操作中的梯度
    // 使用n_buckets个线程块，每个线程块32个线程
    kernels::backward::blend_backward_cu<<<n_buckets, 32>>>(
        per_tile_buffers.instance_ranges,           // 每个瓦片的实例范围
        per_tile_buffers.bucket_offsets,            // 每个瓦片的桶偏移
        per_instance_buffers.primitive_indices.Current(), // 当前实例的样条索引
        per_primitive_buffers.mean2d,               // 样条的2D投影中心点
        per_primitive_buffers.conic_opacity,        // 样条的圆锥体参数和不透明度
        per_primitive_buffers.color,                // 样条的颜色
        grad_image,                                 // 图像颜色梯度
        grad_alpha,                                 // 透明度梯度
        image,                                      // 前向传播生成的图像
        alpha,                                      // 前向传播生成的透明度图
        per_tile_buffers.max_n_contributions,       // 每个瓦片的最大贡献数
        per_tile_buffers.n_contributions,           // 每个像素的贡献数
        per_bucket_buffers.tile_index,              // 桶到瓦片的索引映射
        per_bucket_buffers.color_transmittance,     // 桶中的颜色和透射率
        grad_mean2d_helper,                         // 2D中心点辅助梯度输出
        grad_conic_helper,                          // 圆锥体参数辅助梯度输出
        grad_opacities_raw,                         // 不透明度梯度输出
        grad_sh_coefficients_0,                     // 第0阶球谐系数梯度（临时存储中间梯度）
        n_buckets,                                  // 桶的总数
        n_primitives,                               // 样条总数
        width,                                      // 图像宽度
        height,                                     // 图像高度
        grid.x);                                    // 瓦片网格宽度
    CHECK_CUDA(config::debug, "blend_backward")     // 检查CUDA错误（仅在调试模式下）

    // 第二步：调用预处理反向传播内核，计算样条参数的梯度
    // 使用向上取整的线程块数量，每个线程块config::block_size_preprocess_backward个线程
    kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        means,                                      // 3D高斯样条中心点
        scales_raw,                                 // 原始缩放参数
        rotations_raw,                              // 原始旋转四元数
        sh_coefficients_rest,                       // 高阶球谐函数系数
        w2c,                                        // 世界到相机变换矩阵
        cam_position,                               // 相机位置
        per_primitive_buffers.n_touched_tiles,      // 每个样条触及的瓦片数
        grad_mean2d_helper,                         // 从第一步获得的2D中心点梯度
        grad_conic_helper,                          // 从第一步获得的圆锥体参数梯度
        grad_means,                                 // 输出：3D中心点梯度
        grad_scales_raw,                            // 输出：原始缩放参数梯度
        grad_rotations_raw,                         // 输出：原始旋转四元数梯度
        grad_sh_coefficients_0,                     // 输出：第0阶球谐系数梯度
        grad_sh_coefficients_rest,                  // 输出：高阶球谐系数梯度
        grad_w2c,                                   // 输出：世界到相机变换矩阵梯度
        densification_info,                         // 密度控制信息
        n_primitives,                               // 样条总数
        active_sh_bases,                            // 活跃的球谐函数基数量
        total_bases_sh_rest,                        // 高阶球谐函数基总数
        static_cast<float>(width),                  // 图像宽度（转换为float）
        static_cast<float>(height),                 // 图像高度（转换为float）
        fx,                                         // 相机内参fx
        fy,                                         // 相机内参fy
        cx,                                         // 相机内参cx
        cy);                                        // 相机内参cy
    CHECK_CUDA(config::debug, "preprocess_backward") // 检查CUDA错误（仅在调试模式下）
}
