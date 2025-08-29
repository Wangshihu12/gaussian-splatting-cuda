#pragma once

#include "buffer_utils.h"
#include "helper_math.h"
#include "kernel_utils.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cooperative_groups.h>
#include <cstdint>
namespace cg = cooperative_groups;

namespace fast_gs::rasterization::kernels::forward {

    /**
     * [功能描述]：快速高斯溅射光栅化的预处理CUDA内核函数。
     * 这个内核负责将3D高斯图元投影到2D屏幕空间，计算可见性、2D协方差矩阵、
     * 屏幕边界和颜色等关键信息，为后续的光栅化阶段做准备。
     * 
     * 主要功能包括：
     * 1. 3D到2D投影变换
     * 2. 深度剔除和可见性判断
     * 3. 3D协方差矩阵计算
     * 4. EWA溅射的2D协方差计算
     * 5. 屏幕边界计算
     * 6. 球谐函数颜色转换
     * 
     * @param means [参数说明]：3D高斯中心位置数组，形状为[N, 3]。
     * @param raw_scales [参数说明]：原始缩放参数数组，形状为[N, 3]。
     * @param raw_rotations [参数说明]：原始旋转参数数组（四元数），形状为[N, 4]。
     * @param raw_opacities [参数说明]：原始不透明度数组，形状为[N, 1]。
     * @param sh_coefficients_0 [参数说明]：球谐函数0阶系数数组，形状为[N, 1, 3]。
     * @param sh_coefficients_rest [参数说明]：球谐函数高阶系数数组，形状为[N, K, 3]。
     * @param w2c [参数说明]：世界坐标系到相机坐标系的变换矩阵，形状为[4, 4]。
     * @param cam_position [参数说明]：相机在世界坐标系中的位置，形状为[3]。
     * @param primitive_depth_keys [参数说明]：图元深度键输出数组，用于深度排序。
     * @param primitive_indices [参数说明]：图元索引输出数组，与深度键对应。
     * @param primitive_n_touched_tiles [参数说明]：图元触摸瓦片数量输出数组。
     * @param primitive_screen_bounds [参数说明]：图元屏幕边界输出数组。
     * @param primitive_mean2d [参数说明]：图元2D投影位置输出数组。
     * @param primitive_conic_opacity [参数说明]：图元圆锥体参数和不透明度输出数组。
     * @param primitive_color [参数说明]：图元颜色输出数组。
     * @param n_visible_primitives [参数说明]：可见图元计数器的原子指针。
     * @param n_instances [参数说明]：实例计数器的原子指针。
     * @param n_primitives [参数说明]：图元总数。
     * @param grid_width [参数说明]：瓦片网格宽度。
     * @param grid_height [参数说明]：瓦片网格高度。
     * @param active_sh_bases [参数说明]：当前活跃的球谐函数基函数数量。
     * @param total_bases_sh_rest [参数说明]：高阶球谐函数的基函数总数。
     * @param w [参数说明]：图像宽度。
     * @param h [参数说明]：图像高度。
     * @param fx [参数说明]：x方向焦距。
     * @param fy [参数说明]：y方向焦距。
     * @param cx [参数说明]：x方向主点坐标。
     * @param cy [参数说明]：y方向主点坐标。
     * @param near_ [参数说明]：近裁剪平面距离（避免Windows宏冲突）。
     * @param far_ [参数说明]：远裁剪平面距离。
     */
    __global__ void preprocess_cu(
        const float3* means,                    // 3D高斯中心位置数组
        const float3* raw_scales,               // 原始缩放参数数组
        const float4* raw_rotations,            // 原始旋转参数数组（四元数）
        const float* raw_opacities,             // 原始不透明度数组
        const float3* sh_coefficients_0,        // 球谐函数0阶系数数组
        const float3* sh_coefficients_rest,     // 球谐函数高阶系数数组
        const float4* w2c,                      // 世界到相机的变换矩阵
        const float3* cam_position,             // 相机位置
        uint* primitive_depth_keys,             // 图元深度键输出数组
        uint* primitive_indices,                // 图元索引输出数组
        uint* primitive_n_touched_tiles,        // 图元触摸瓦片数量输出数组
        ushort4* primitive_screen_bounds,       // 图元屏幕边界输出数组
        float2* primitive_mean2d,               // 图元2D投影位置输出数组
        float4* primitive_conic_opacity,        // 图元圆锥体参数和不透明度输出数组
        float3* primitive_color,                // 图元颜色输出数组
        uint* n_visible_primitives,             // 可见图元计数器
        uint* n_instances,                      // 实例计数器
        const uint n_primitives,                // 图元总数
        const uint grid_width,                  // 瓦片网格宽度
        const uint grid_height,                 // 瓦片网格高度
        const uint active_sh_bases,             // 活跃的球谐函数基函数数量
        const uint total_bases_sh_rest,         // 高阶球谐函数的基函数总数
        const float w,                          // 图像宽度
        const float h,                          // 图像高度
        const float fx,                         // x方向焦距
        const float fy,                         // y方向焦距
        const float cx,                         // x方向主点坐标
        const float cy,                         // y方向主点坐标
        const float near_,                      // 近裁剪平面（避免Windows宏冲突）
        const float far_) {                     // 远裁剪平面
        
        // 步骤1：线程索引计算和边界检查
        // 使用协作组获取当前线程在网格中的全局索引
        auto primitive_idx = cg::this_grid().thread_rank();
        bool active = true;  // 线程活跃状态标志
        
        // 边界检查：确保线程索引在有效范围内
        if (primitive_idx >= n_primitives) {
            active = false;                    // 超出范围的线程标记为非活跃
            primitive_idx = n_primitives - 1;  // 设置为最后一个有效索引（避免越界）
        }

        // 初始化触摸瓦片数量（仅对活跃线程）
        if (active)
            primitive_n_touched_tiles[primitive_idx] = 0;

        // 步骤2：加载3D高斯中心位置
        // 从全局内存加载当前图元的3D中心位置
        const float3 mean3d = means[primitive_idx];

        // 步骤3：深度剔除（Z-culling）
        // 使用变换矩阵的第三行进行深度计算
        const float4 w2c_r3 = w2c[2];  // 变换矩阵第三行 [0, 0, 1, 0]
        // 计算图元在相机坐标系中的深度：depth = z = w2c_r3 · [x, y, z, 1]
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        
        // 深度测试：剔除在近远裁剪平面外的图元
        if (depth < near_ || depth > far_)
            active = false;

        // 步骤4：Warp级别的早期退出优化
        // 如果整个warp（32个线程）都不活跃，直接返回，避免不必要的计算
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        // 步骤5：不透明度计算和阈值测试
        // 加载原始不透明度值
        const float raw_opacity = raw_opacities[primitive_idx];
        // 应用sigmoid激活函数：opacity = 1 / (1 + exp(-raw_opacity))
        const float opacity = 1.0f / (1.0f + expf(-raw_opacity));
        // 不透明度阈值测试：剔除过小的不透明度值
        if (opacity < config::min_alpha_threshold)
            active = false;

        // 步骤6：3D协方差矩阵计算
        // 从原始缩放参数计算方差：variance = exp(2 * raw_scale)
        const float3 raw_scale = raw_scales[primitive_idx];
        const float3 variance = make_float3(expf(2.0f * raw_scale.x), expf(2.0f * raw_scale.y), expf(2.0f * raw_scale.z));
        
        // 加载四元数旋转参数
        auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
        
        // 计算四元数分量的平方值
        const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
        const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
        
        // 四元数归一化检查：避免数值不稳定
        if (q_norm_sq < 1e-8f)
            active = false;
        
        // 四元数归一化：将四元数分量除以模长
        const float qxx = 2.0f * qxx_raw / q_norm_sq, qyy = 2.0f * qyy_raw / q_norm_sq, qzz = 2.0f * qzz_raw / q_norm_sq;
        const float qxy = 2.0f * qx * qy / q_norm_sq, qxz = 2.0f * qx * qz / q_norm_sq, qyz = 2.0f * qy * qz / q_norm_sq;
        const float qrx = 2.0f * qr * qx / q_norm_sq, qry = 2.0f * qr * qy / q_norm_sq, qrz = 2.0f * qr * qz / q_norm_sq;
        
        // 从四元数构建3x3旋转矩阵（Rodrigues公式）
        const mat3x3 rotation = {
            1.0f - (qyy + qzz), qxy - qrz, qry + qxz,      // 第一行
            qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,      // 第二行
            qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)       // 第三行
        };
        
        // 计算缩放后的旋转矩阵：rotation_scaled = rotation * diag(variance)
        const mat3x3 rotation_scaled = {
            rotation.m11 * variance.x, rotation.m12 * variance.y, rotation.m13 * variance.z,  // 第一行
            rotation.m21 * variance.x, rotation.m22 * variance.y, rotation.m23 * variance.z,  // 第二行
            rotation.m31 * variance.x, rotation.m32 * variance.y, rotation.m33 * variance.z   // 第三行
        };
        
        // 计算3D协方差矩阵的上三角部分：cov3d = rotation_scaled * rotation^T
        const mat3x3_triu cov3d{
            rotation_scaled.m11 * rotation.m11 + rotation_scaled.m12 * rotation.m12 + rotation_scaled.m13 * rotation.m13,  // m11
            rotation_scaled.m11 * rotation.m21 + rotation_scaled.m12 * rotation.m22 + rotation_scaled.m13 * rotation.m23,  // m12
            rotation_scaled.m11 * rotation.m31 + rotation_scaled.m12 * rotation.m32 + rotation_scaled.m13 * rotation.m33,  // m13
            rotation_scaled.m21 * rotation.m21 + rotation_scaled.m22 * rotation.m22 + rotation_scaled.m23 * rotation.m23,  // m22
            rotation_scaled.m21 * rotation.m31 + rotation_scaled.m22 * rotation.m32 + rotation_scaled.m23 * rotation.m33,  // m23
            rotation_scaled.m31 * rotation.m31 + rotation_scaled.m32 * rotation.m32 + rotation_scaled.m33 * rotation.m33,  // m33
        };

        // 步骤7：3D到2D投影变换
        // 使用变换矩阵的第一行和第二行计算归一化图像坐标
        const float4 w2c_r1 = w2c[0];  // 变换矩阵第一行
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;  // x坐标
        const float4 w2c_r2 = w2c[1];  // 变换矩阵第二行
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;  // y坐标

        // 步骤8：EWA溅射（Elliptical Weighted Average）的2D协方差计算
        // 定义裁剪边界（扩展15%以避免边界效应）
        const float clip_left = (-0.15f * w - cx) / fx;      // 左边界
        const float clip_right = (1.15f * w - cx) / fx;      // 右边界
        const float clip_top = (-0.15f * h - cy) / fy;       // 上边界
        const float clip_bottom = (1.15f * h - cy) / fy;     // 下边界
        
        // 将坐标限制在裁剪范围内
        const float tx = clamp(x, clip_left, clip_right);
        const float ty = clamp(y, clip_top, clip_bottom);
        
        // 计算雅可比矩阵元素（投影变换的导数）
        const float j11 = fx / depth;                        // ∂x/∂X
        const float j13 = -j11 * tx;                         // ∂x/∂Z
        const float j22 = fy / depth;                        // ∂y/∂Y
        const float j23 = -j22 * ty;                         // ∂y/∂Z
        
        // 计算雅可比矩阵与变换矩阵的乘积
        const float3 jw_r1 = make_float3(
            j11 * w2c_r1.x + j13 * w2c_r3.x,                // 第一行第一列
            j11 * w2c_r1.y + j13 * w2c_r3.y,                // 第一行第二列
            j11 * w2c_r1.z + j13 * w2c_r3.z);               // 第一行第三列
        const float3 jw_r2 = make_float3(
            j22 * w2c_r2.x + j23 * w2c_r3.x,                // 第二行第一列
            j22 * w2c_r2.y + j23 * w2c_r3.y,                // 第二行第二列
            j22 * w2c_r2.z + j23 * w2c_r3.z);               // 第二行第三列
        
        // 计算雅可比矩阵与3D协方差矩阵的乘积
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,  // 第一行
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,  // 第二行
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33   // 第三行
        );
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,  // 第一行
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,  // 第二行
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33   // 第三行
        );
        
        // 计算2D协方差矩阵：cov2d = J * cov3d * J^T
        float3 cov2d = make_float3(
            dot(jwc_r1, jw_r1),    // cov2d.xx = jwc_r1 · jw_r1
            dot(jwc_r1, jw_r2),    // cov2d.xy = jwc_r1 · jw_r2
            dot(jwc_r2, jw_r2));   // cov2d.yy = jwc_r2 · jw_r2
        
        // 添加膨胀因子，避免协方差矩阵过于奇异
        cov2d.x += config::dilation;  // 膨胀x方向方差
        cov2d.z += config::dilation;  // 膨胀y方向方差
        
        // 计算协方差矩阵的行列式
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant < 1e-8f)  // 行列式过小，协方差矩阵接近奇异
            active = false;
        
        // 计算圆锥体参数（协方差矩阵的逆矩阵的上三角部分）
        const float3 conic = make_float3(
            cov2d.z / determinant,      // conic.xx = cov2d.yy / det
            -cov2d.y / determinant,     // conic.xy = -cov2d.xy / det
            cov2d.x / determinant);     // conic.yy = cov2d.xx / det

        // 步骤9：计算2D投影位置（屏幕空间坐标）
        // 将归一化坐标转换为像素坐标
        const float2 mean2d = make_float2(
            x * fx + cx,  // 屏幕x坐标
            y * fy + cy); // 屏幕y坐标

        // 步骤10：计算屏幕边界
        // 计算不透明度的功率阈值
        const float power_threshold = logf(opacity * config::min_alpha_threshold_rcp);
        const float power_threshold_factor = sqrtf(2.0f * power_threshold);
        
        // 计算高斯在x和y方向上的扩展范围
        float extent_x = fmaxf(power_threshold_factor * sqrtf(cov2d.x) - 0.5f, 0.0f);
        float extent_y = fmaxf(power_threshold_factor * sqrtf(cov2d.z) - 0.5f, 0.0f);
        
        // 计算屏幕边界（瓦片坐标）
        const uint4 screen_bounds = make_uint4(
            min(grid_width, static_cast<uint>(max(0, __float2int_rd((mean2d.x - extent_x) / static_cast<float>(config::tile_width))))),   // x_min：左边界
            min(grid_width, static_cast<uint>(max(0, __float2int_ru((mean2d.x + extent_x) / static_cast<float>(config::tile_width))))),   // x_max：右边界
            min(grid_height, static_cast<uint>(max(0, __float2int_rd((mean2d.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min：下边界
            min(grid_height, static_cast<uint>(max(0, __float2int_ru((mean2d.y + extent_y) / static_cast<float>(config::tile_height)))))  // y_max：上边界
        );
        
        // 计算最大触摸瓦片数量（边界框内的瓦片总数）
        const uint n_touched_tiles_max = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
        if (n_touched_tiles_max == 0)  // 没有触摸任何瓦片
            active = false;

        // 步骤11：Warp级别的第二次早期退出优化
        // 再次检查整个warp的活跃状态
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        // 步骤12：计算精确的触摸瓦片数量
        // 调用辅助函数计算图元实际触摸的瓦片数量
        const uint n_touched_tiles = compute_exact_n_touched_tiles(
            mean2d, conic, screen_bounds,
            power_threshold, n_touched_tiles_max, active);

        // 步骤13：最终有效性检查
        // 如果触摸瓦片数量为0或图元不活跃，直接返回
        if (n_touched_tiles == 0 || !active)
            return;

        // 步骤14：存储计算结果
        // 将计算得到的各种参数存储到输出缓冲区
        
        // 存储触摸瓦片数量
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        
        // 存储屏幕边界（转换为ushort以节省内存）
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),   // x_min
            static_cast<ushort>(screen_bounds.y),   // x_max
            static_cast<ushort>(screen_bounds.z),   // y_min
            static_cast<ushort>(screen_bounds.w));  // y_max
        
        // 存储2D投影位置
        primitive_mean2d[primitive_idx] = mean2d;
        
        // 存储圆锥体参数和不透明度
        primitive_conic_opacity[primitive_idx] = make_float4(conic, opacity);
        
        // 存储球谐函数计算的颜色
        primitive_color[primitive_idx] = convert_sh_to_color(
            sh_coefficients_0, sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, total_bases_sh_rest);

        // 步骤15：原子操作更新计数器
        // 使用原子操作安全地更新可见图元计数
        const uint offset = atomicAdd(n_visible_primitives, 1);
        
        // 将深度值转换为无符号整数作为排序键
        const uint depth_key = __float_as_uint(depth);
        
        // 存储深度键和图元索引（用于后续的深度排序）
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        
        // 原子更新实例计数（每个触摸的瓦片创建一个实例）
        atomicAdd(n_instances, n_touched_tiles);
    }

    /**
     * [功能描述]：快速高斯溅射光栅化的深度排序应用CUDA内核函数。
     * 这个内核负责根据深度排序后的图元索引，重新排列触摸瓦片数量数组，
     * 确保所有相关数组的数据顺序与深度排序保持一致。
     * 
     * 主要功能：
     * 1. 根据排序后的图元索引重新排列触摸瓦片数量
     * 2. 为后续的实例创建阶段提供正确的偏移量信息
     * 3. 维护数据一致性，确保深度排序的有效性
     * 
     * 在光栅化流程中的作用：
     * 在预处理阶段，图元按深度排序后，其索引顺序发生了变化。
     * 但是触摸瓦片数量数组仍然保持原始的图元顺序。
     * 这个内核将触摸瓦片数量重新排列，使其与排序后的图元索引对应。
     * 
     * @param primitive_indices_sorted [参数说明]：深度排序后的图元索引数组，形状为[n_visible_primitives]。
     *                                         这个数组已经按照深度从近到远排序。
     * @param primitive_n_touched_tiles [参数说明]：原始顺序的触摸瓦片数量数组，形状为[n_primitives]。
     *                                          包含每个图元触摸的瓦片数量，仍按原始图元顺序排列。
     * @param primitive_offset [参数说明]：重新排列后的触摸瓦片数量输出数组，形状为[n_visible_primitives]。
     *                                 输出数组将按照深度排序后的图元顺序排列。
     * @param n_visible_primitives [参数说明]：可见图元的数量，也是需要处理的元素数量。
     *                                      这个值决定了内核的执行范围。
     */
    __global__ void apply_depth_ordering_cu(
        const uint* primitive_indices_sorted,    // 深度排序后的图元索引数组
        const uint* primitive_n_touched_tiles,   // 原始顺序的触摸瓦片数量数组
        uint* primitive_offset,                  // 重新排列后的触摸瓦片数量输出数组
        const uint n_visible_primitives) {       // 可见图元数量
        
        // 步骤1：获取当前线程的全局索引
        // 使用协作组API获取线程在网格中的全局索引
        auto idx = cg::this_grid().thread_rank();
        
        // 步骤2：边界检查
        // 确保线程索引在有效范围内，避免越界访问
        if (idx >= n_visible_primitives)
            return;  // 超出范围的线程直接返回，不执行任何操作
        
        // 步骤3：获取排序后的图元索引
        // 从深度排序后的索引数组中获取当前线程对应的图元索引
        const uint primitive_idx = primitive_indices_sorted[idx];
        
        // 步骤4：重新排列触摸瓦片数量
        // 根据排序后的图元索引，从原始数组中获取对应的触摸瓦片数量
        // 并将结果存储到输出数组的对应位置
        primitive_offset[idx] = primitive_n_touched_tiles[primitive_idx];
    }

    // 基于 https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    /**
     * [功能描述]：快速高斯溅射光栅化的实例创建CUDA内核函数。
     * 这个内核负责为每个可见图元创建实例，将图元转换为瓦片级别的实例表示。
     * 实现了两种处理模式：顺序处理和协作处理，以优化不同规模的图元处理性能。
     * 
     * 主要功能：
     * 1. 为每个图元在每个触摸的瓦片中创建实例
     * 2. 计算实例的瓦片键和图元索引
     * 3. 实现顺序和协作两种处理模式
     * 4. 使用共享内存优化数据访问
     * 5. 支持warp级别的协作计算
     * 
     * 算法原理：
     * 每个图元可能影响多个瓦片，在每个瓦片中创建一个实例。
     * 对于触摸瓦片数量较少的图元，使用顺序处理；
     * 对于触摸瓦片数量较多的图元，使用warp协作处理以提高效率。
     * 
     * @param primitive_indices_sorted [参数说明]：深度排序后的图元索引数组，形状为[n_visible_primitives]。
     * @param primitive_offsets [参数说明]：图元在实例数组中的偏移量，形状为[n_visible_primitives]。
     * @param primitive_screen_bounds [参数说明]：图元的屏幕边界（瓦片坐标），形状为[n_visible_primitives]。
     * @param primitive_mean2d [参数说明]：图元的2D投影位置，形状为[n_visible_primitives, 2]。
     * @param primitive_conic_opacity [参数说明]：图元的圆锥体参数和不透明度，形状为[n_visible_primitives, 4]。
     * @param instance_keys [参数说明]：实例瓦片键输出数组，用于后续的瓦片排序。
     * @param instance_primitive_indices [参数说明]：实例图元索引输出数组，与瓦片键对应。
     * @param grid_width [参数说明]：瓦片网格的宽度，用于计算瓦片键。
     * @param n_visible_primitives [参数说明]：可见图元的数量，决定内核的执行范围。
     */
    __global__ void create_instances_cu(
        const uint* primitive_indices_sorted,    // 深度排序后的图元索引数组
        const uint* primitive_offsets,           // 图元在实例数组中的偏移量
        const ushort4* primitive_screen_bounds,  // 图元的屏幕边界（瓦片坐标）
        const float2* primitive_mean2d,          // 图元的2D投影位置
        const float4* primitive_conic_opacity,   // 图元的圆锥体参数和不透明度
        ushort* instance_keys,                   // 实例瓦片键输出数组
        uint* instance_primitive_indices,        // 实例图元索引输出数组
        const uint grid_width,                   // 瓦片网格宽度
        const uint n_visible_primitives) {       // 可见图元数量
        
        // 步骤1：协作组和线程索引初始化
        // 获取当前线程块和warp的协作组对象
        auto block = cg::this_thread_block();                    // 当前线程块
        auto warp = cg::tiled_partition<32u>(block);            // 32线程的warp分区
        uint idx = cg::this_grid().thread_rank();                // 全局线程索引

        // 步骤2：线程活跃状态检查和早期退出
        bool active = true;  // 线程活跃状态标志
        if (idx >= n_visible_primitives) {
            active = false;                    // 超出范围的线程标记为非活跃
            idx = n_visible_primitives - 1;    // 设置为最后一个有效索引（避免越界）
        }

        // Warp级别的早期退出优化：如果整个warp都不活跃，直接返回
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        // 步骤3：获取图元索引和基本信息
        const uint primitive_idx = primitive_indices_sorted[idx];  // 当前线程对应的图元索引

        // 步骤4：计算图元的瓦片覆盖信息
        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];  // 图元的屏幕边界
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);  // 边界宽度（瓦片数）
        const uint tile_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;  // 总触摸瓦片数

        // 步骤5：共享内存数据收集
        // 将图元数据收集到共享内存中，提高后续访问效率
        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];      // 收集的屏幕边界
        __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];      // 收集的2D位置（偏移0.5）
        __shared__ float4 collected_conic_opacity[config::block_size_create_instances];       // 收集的圆锥体参数
        
        // 将当前图元的数据存储到共享内存的对应位置
        collected_screen_bounds[block.thread_rank()] = screen_bounds;
        collected_mean2d_shifted[block.thread_rank()] = primitive_mean2d[primitive_idx] - 0.5f;  // 偏移0.5像素
        collected_conic_opacity[block.thread_rank()] = primitive_conic_opacity[primitive_idx];

        // 步骤6：获取写入偏移量
        uint current_write_offset = primitive_offsets[idx];  // 当前图元在实例数组中的起始位置

        // 步骤7：顺序处理模式（适用于触摸瓦片数量较少的图元）
        if (active) {
            // 从共享内存加载图元数据
            const float2 mean2d_shifted = collected_mean2d_shifted[block.thread_rank()];  // 偏移后的2D位置
            const float4 conic_opacity = collected_conic_opacity[block.thread_rank()];     // 圆锥体参数和不透明度
            const float3 conic = make_float3(conic_opacity);                              // 提取圆锥体参数
            const float power_threshold = logf(conic_opacity.w * config::min_alpha_threshold_rcp);  // 功率阈值

            // 顺序处理触摸瓦片，但限制在阈值范围内
            for (uint instance_idx = 0; instance_idx < tile_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
                // 计算当前瓦片的坐标
                const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);  // 瓦片y坐标
                const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);  // 瓦片x坐标
                
                // 检查图元是否对当前瓦片有贡献
                if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                    // 计算瓦片键：tile_key = tile_y * grid_width + tile_x
                    const ushort tile_key = static_cast<ushort>(tile_y * grid_width + tile_x);
                    
                    // 存储实例信息
                    instance_keys[current_write_offset] = tile_key;                    // 瓦片键
                    instance_primitive_indices[current_write_offset] = primitive_idx;  // 图元索引
                    current_write_offset++;                                           // 更新写入位置
                }
            }
        }

        // 步骤8：协作处理模式（适用于触摸瓦片数量较多的图元）
        // 计算warp内的线程位置信息
        const uint lane_idx = cg::this_thread_block().thread_rank() % 32u;           // 在warp中的位置（0-31）
        const uint warp_idx = cg::this_thread_block().thread_rank() / 32u;           // warp在块中的索引
        const uint lane_mask_allprev_excl = 0xffffffffu >> (32u - lane_idx);        // 当前线程之前所有线程的掩码
        
        // 判断是否需要协作计算：图元活跃且触摸瓦片数量超过顺序阈值
        const int compute_cooperatively = active && tile_count > config::n_sequential_threshold;
        const uint remaining_threads = __ballot_sync(0xffffffffu, compute_cooperatively);  // 需要协作的线程掩码
        
        // 如果没有线程需要协作，直接返回
        if (remaining_threads == 0)
            return;

        // 步骤9：协作计算准备
        const uint n_remaining_threads = __popc(remaining_threads);  // 需要协作的线程数量
        
        // 遍历需要协作的线程，每个线程处理一个图元的剩余瓦片
        for (int n = 0; n < n_remaining_threads && n < 32; n++) {
            // 找到当前要处理的线程
            int current_lane = __fns(remaining_threads, 0, n + 1);  // 找到第n+1个需要协作的线程
            
            // 通过warp shuffle获取该线程的图元信息
            uint primitive_idx_coop = __shfl_sync(0xffffffffu, primitive_idx, current_lane);           // 图元索引
            uint current_write_offset_coop = __shfl_sync(0xffffffffu, current_write_offset, current_lane);  // 写入偏移量

            // 从共享内存获取该图元的数据
            const ushort4 screen_bounds_coop = collected_screen_bounds[warp.meta_group_rank() * 32 + current_lane];  // 屏幕边界
            const uint screen_bounds_width_coop = static_cast<uint>(screen_bounds_coop.y - screen_bounds_coop.x);    // 边界宽度
            const uint tile_count_coop = screen_bounds_width_coop * static_cast<uint>(screen_bounds_coop.w - screen_bounds_coop.z);  // 瓦片总数

            const float2 mean2d_shifted_coop = collected_mean2d_shifted[warp.meta_group_rank() * 32 + current_lane];  // 2D位置
            const float4 conic_opacity_coop = collected_conic_opacity[warp.meta_group_rank() * 32 + current_lane];     // 圆锥体参数
            const float3 conic_coop = make_float3(conic_opacity_coop);                                                // 圆锥体参数
            const float power_threshold_coop = logf(conic_opacity_coop.w * config::min_alpha_threshold_rcp);          // 功率阈值

            // 步骤10：协作处理剩余瓦片
            const uint remaining_tile_count = tile_count_coop - config::n_sequential_threshold;  // 剩余的瓦片数量
            const int n_iterations = div_round_up(remaining_tile_count, 32u);                   // 需要的迭代次数
            
            // 每次迭代处理32个瓦片（warp大小）
            for (int i = 0; i < n_iterations; i++) {
                // 计算当前线程负责的瓦片索引
                const int instance_idx = i * 32 + lane_idx + config::n_sequential_threshold;  // 瓦片实例索引
                const int active_current = instance_idx < tile_count_coop;                    // 当前瓦片是否有效
                
                // 计算瓦片坐标
                const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);  // 瓦片y坐标
                const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);  // 瓦片x坐标
                
                // 检查图元是否对当前瓦片有贡献
                const uint write = active_current && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                
                // 使用warp ballot计算写入信息
                const uint write_ballot = __ballot_sync(0xffffffffu, write);                    // 所有线程的写入掩码
                const uint n_writes = __popc(write_ballot);                                     // 总写入数量
                const uint write_offset_current = __popc(write_ballot & lane_mask_allprev_excl); // 当前线程的写入偏移量
                const uint write_offset = current_write_offset_coop + write_offset_current;      // 最终写入位置
                
                // 如果需要写入，则存储实例信息
                if (write) {
                    const ushort tile_key = static_cast<ushort>(tile_y * grid_width + tile_x);  // 计算瓦片键
                    instance_keys[write_offset] = tile_key;                                     // 存储瓦片键
                    instance_primitive_indices[write_offset] = primitive_idx_coop;              // 存储图元索引
                }
                
                // 更新写入偏移量
                current_write_offset_coop += n_writes;
            }

            // 步骤11：Warp同步
            __syncwarp();  // 确保warp内所有线程完成当前迭代
        }
    }

    /**
     * [功能描述]：快速高斯溅射光栅化的实例范围提取CUDA内核函数。
     * 这个内核负责从排序后的实例数组中提取每个瓦片的实例范围信息，
     * 为后续的瓦片处理和桶组织提供必要的索引边界。
     * 
     * 主要功能：
     * 1. 分析排序后的实例数组，识别瓦片边界
     * 2. 计算每个瓦片中实例的起始和结束索引
     * 3. 为后续的瓦片级别处理提供索引范围信息
     * 
     * 算法原理：
     * 由于实例数组已经按瓦片键排序，相同瓦片的实例连续存储。
     * 通过比较相邻实例的瓦片键，可以识别瓦片边界。
     * 对于每个瓦片，记录其第一个实例的索引（x）和最后一个实例的索引（y）。
     * 
     * 数据结构：
     * tile_instance_ranges[i] = {start_idx, end_idx} 表示瓦片i的实例范围
     * - start_idx: 瓦片i中第一个实例的索引
     * - end_idx: 瓦片i中最后一个实例的索引（不包含）
     * 
     * @param instance_keys [参数说明]：实例瓦片键数组，已经按瓦片键排序，形状为[n_instances]。
     *                               每个元素表示对应实例所属的瓦片索引。
     * @param tile_instance_ranges [参数说明]：瓦片实例范围输出数组，形状为[n_tiles]。
     *                                      每个元素是一个uint2，包含瓦片的实例起始和结束索引。
     * @param n_instances [参数说明]：实例的总数量，决定内核的执行范围。
     */
    __global__ void extract_instance_ranges_cu(
        const ushort* instance_keys,           // 实例瓦片键数组（已排序）
        uint2* tile_instance_ranges,           // 瓦片实例范围输出数组
        const uint n_instances) {              // 实例总数
        
        // 步骤1：获取当前线程的全局索引
        // 使用协作组API获取线程在网格中的全局索引
        auto instance_idx = cg::this_grid().thread_rank();
        
        // 步骤2：边界检查
        // 确保线程索引在有效范围内，避免越界访问
        if (instance_idx >= n_instances)
            return;  // 超出范围的线程直接返回，不执行任何操作
        
        // 步骤3：获取当前实例的瓦片索引
        // 从排序后的实例键数组中获取当前实例所属的瓦片
        const ushort instance_tile_idx = instance_keys[instance_idx];
        
        // 步骤4：处理第一个实例（索引0）
        // 第一个实例总是开始一个新瓦片的实例序列
        if (instance_idx == 0) {
            tile_instance_ranges[instance_tile_idx].x = 0;  // 设置瓦片的起始索引为0
        } else {
            // 步骤5：处理非第一个实例
            // 获取前一个实例的瓦片索引
            const ushort previous_instance_tile_idx = instance_keys[instance_idx - 1];
            
            // 步骤6：检查瓦片边界
            // 如果当前实例与前一个实例属于不同瓦片，说明发生了瓦片切换
            if (instance_tile_idx != previous_instance_tile_idx) {
                // 完成前一个瓦片的实例范围
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;  // 设置前一个瓦片的结束索引
                
                // 开始新瓦片的实例范围
                tile_instance_ranges[instance_tile_idx].x = instance_idx;  // 设置新瓦片的起始索引
            }
            // 注意：如果瓦片相同，不需要做任何操作，继续累积实例
        }
        
        // 步骤7：处理最后一个实例
        // 最后一个实例总是结束当前瓦片的实例序列
        if (instance_idx == n_instances - 1) {
            tile_instance_ranges[instance_tile_idx].y = n_instances;  // 设置瓦片的结束索引为实例总数
        }
    }

    /**
     * [功能描述]：快速高斯溅射光栅化的桶计数提取CUDA内核函数。
     * 这个内核负责计算每个瓦片中需要的桶数量，为后续的桶组织和最终混合阶段提供必要的配置信息。
     * 
     * 主要功能：
     * 1. 分析每个瓦片的实例数量，计算所需的桶数量
     * 2. 为桶的内存分配和偏移量计算提供基础数据
     * 3. 支持瓦片级别的并行处理，优化性能
     * 
     * 桶的概念：
     * 桶是瓦片内的子区域，用于组织贡献到同一像素的图元实例。
     * 每个桶可以容纳最多32个实例（warp大小），这样可以实现像素级别的并行处理。
     * 通过将瓦片内的实例分组到不同的桶中，可以优化内存访问和计算效率。
     * 
     * 算法原理：
     * 对于每个瓦片，计算其包含的实例数量，然后除以32（向上取整）得到所需的桶数量。
     * 这种设计确保了每个桶内的实例数量不会超过warp大小，支持高效的并行处理。
     * 
     * 在光栅化流程中的作用：
     * 这个内核是桶组织系统的第一步，为后续的桶偏移量计算和实例分配到桶提供基础信息。
     * 通过合理的桶数量计算，可以平衡内存使用和计算效率。
     * 
     * @param tile_instance_ranges [参数说明]：瓦片实例范围数组，形状为[n_tiles]。
     *                                      每个元素是一个uint2，包含瓦片的实例起始和结束索引。
     *                                      这个数组由extract_instance_ranges_cu内核生成。
     * @param tile_n_buckets [参数说明]：瓦片桶计数输出数组，形状为[n_tiles]。
     *                                  每个元素表示对应瓦片需要的桶数量。
     *                                  这个数组将用于后续的桶偏移量计算。
     * @param n_tiles [参数说明]：瓦片的总数量，决定内核的执行范围。
     *                            这个值决定了需要处理多少个瓦片。
     */
    __global__ void extract_bucket_counts(
        uint2* tile_instance_ranges,    // 瓦片实例范围数组
        uint* tile_n_buckets,           // 瓦片桶计数输出数组
        const uint n_tiles) {           // 瓦片总数
        
        // 步骤1：获取当前线程的全局索引
        // 使用协作组API获取线程在网格中的全局索引
        // 每个线程负责处理一个瓦片的桶计数计算
        auto tile_idx = cg::this_grid().thread_rank();
        
        // 步骤2：边界检查
        // 确保线程索引在有效范围内，避免越界访问
        // 这是CUDA内核的标准安全检查模式
        if (tile_idx >= n_tiles)
            return;  // 超出范围的线程直接返回，不执行任何操作
        
        // 步骤3：获取瓦片的实例范围信息
        // 从tile_instance_ranges数组中读取当前瓦片的实例范围
        // instance_range.x: 瓦片中第一个实例的索引
        // instance_range.y: 瓦片中最后一个实例的索引（不包含）
        const uint2 instance_range = tile_instance_ranges[tile_idx];
        
        // 步骤4：计算瓦片需要的桶数量
        // 使用公式：n_buckets = ceil((end_idx - start_idx) / 32)
        // 其中32是warp大小，也是每个桶的最大容量
        // div_round_up函数实现向上取整的除法运算
        const uint n_buckets = div_round_up(instance_range.y - instance_range.x, 32u);
        
        // 步骤5：存储桶计数结果
        // 将计算得到的桶数量存储到输出数组中
        // 这个值将被后续的桶偏移量计算使用
        tile_n_buckets[tile_idx] = n_buckets;
    }

    /**
     * [功能描述]：快速高斯溅射光栅化的最终混合CUDA内核函数。
     * 这是整个光栅化流程的最后一步，负责将排序后的图元实例按深度顺序混合到最终图像。
     * 实现了基于瓦片和桶的并行渲染，支持透明度混合、高斯权重计算和贡献统计。
     * 
     * 主要功能：
     * 1. 瓦片级别的并行像素渲染
     * 2. 基于圆锥体参数的高斯权重计算
     * 3. 透明度混合和透射率累积
     * 4. 桶级别的数据组织和存储
     * 5. 贡献数量统计和最大值归约
     * 
     * 算法原理：
     * 每个CUDA块处理一个瓦片，块内线程协作处理瓦片内的像素。
     * 图元按深度顺序处理，使用透明度混合公式计算最终颜色。
     * 通过桶组织优化内存访问，支持高效的并行处理。
     * 
     * 透明度混合公式：
     * color_final = Σ(transmittance_i * alpha_i * color_i)
     * transmittance_i = Π(1 - alpha_j) for j < i
     * 
     * @param tile_instance_ranges [参数说明]：瓦片实例范围数组，形状为[n_tiles]。
     *                                      每个元素是uint2，包含瓦片的实例起始和结束索引。
     * @param tile_bucket_offsets [参数说明]：瓦片桶偏移量数组，形状为[n_tiles]。
     *                                       每个元素表示瓦片在桶数组中的起始位置。
     * @param instance_primitive_indices [参数说明]：实例图元索引数组，形状为[n_instances]。
     *                                           将实例映射到对应的图元。
     * @param primitive_mean2d [参数说明]：图元2D投影位置数组，形状为[n_primitives, 2]。
     *                                    包含每个图元在屏幕空间中的位置。
     * @param primitive_conic_opacity [参数说明]：图元圆锥体参数和不透明度数组，形状为[n_primitives, 4]。
     *                                         包含圆锥体参数(xx, xy, yy)和不透明度。
     * @param primitive_color [参数说明]：图元颜色数组，形状为[n_primitives, 3]。
     *                                  包含每个图元的RGB颜色值。
     * @param image [参数说明]：输出图像缓冲区，形状为[3, height, width]。
     *                         存储最终的RGB渲染结果。
     * @param alpha_map [参数说明]：输出透明度缓冲区，形状为[height, width]。
     *                             存储最终的透明度值。
     * @param tile_max_n_contributions [参数说明]：瓦片最大贡献数量输出数组，形状为[n_tiles]。
     *                                          记录每个瓦片的最大贡献数量。
     * @param tile_n_contributions [参数说明]：瓦片贡献数量输出数组，形状为[height, width]。
     *                                      记录每个像素的贡献数量。
     * @param bucket_tile_index [参数说明]：桶瓦片索引输出数组，用于桶组织。
     * @param bucket_color_transmittance [参数说明]：桶颜色透射率输出数组，存储桶级别的中间结果。
     * @param width [参数说明]：输出图像的宽度。
     * @param height [参数说明]：输出图像的高度。
     * @param grid_width [参数说明]：瓦片网格的宽度，用于计算瓦片索引。
     */
    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* tile_instance_ranges,        // 瓦片实例范围数组
        const uint* tile_bucket_offsets,          // 瓦片桶偏移量数组
        const uint* instance_primitive_indices,   // 实例图元索引数组
        const float2* primitive_mean2d,           // 图元2D投影位置数组
        const float4* primitive_conic_opacity,    // 图元圆锥体参数和不透明度数组
        const float3* primitive_color,            // 图元颜色数组
        float* image,                             // 输出图像缓冲区
        float* alpha_map,                         // 输出透明度缓冲区
        uint* tile_max_n_contributions,           // 瓦片最大贡献数量输出数组
        uint* tile_n_contributions,               // 瓦片贡献数量输出数组
        uint* bucket_tile_index,                  // 桶瓦片索引输出数组
        float4* bucket_color_transmittance,       // 桶颜色透射率输出数组
        const uint width,                         // 图像宽度
        const uint height,                        // 图像高度
        const uint grid_width) {                  // 瓦片网格宽度
        
        // 步骤1：协作组和索引初始化
        // 获取当前线程块和线程的索引信息
        auto block = cg::this_thread_block();                    // 当前线程块
        const dim3 group_index = block.group_index();            // 块在网格中的索引
        const dim3 thread_index = block.thread_index();          // 线程在块中的索引
        const uint thread_rank = block.thread_rank();            // 线程在块中的排名
        
        // 步骤2：像素坐标计算
        // 根据块索引和线程索引计算像素坐标
        const uint2 pixel_coords = make_uint2(
            group_index.x * config::tile_width + thread_index.x,   // 像素x坐标
            group_index.y * config::tile_height + thread_index.y   // 像素y坐标
        );
        
        // 检查像素是否在图像边界内
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        
        // 将像素坐标转换为浮点数，并添加0.5像素偏移（像素中心）
        const float2 pixel = make_float2(
            __uint2float_rn(pixel_coords.x), 
            __uint2float_rn(pixel_coords.y)
        ) + 0.5f;

        // 步骤3：瓦片信息获取
        // 计算当前块对应的瓦片索引
        const uint tile_idx = group_index.y * grid_width + group_index.x;
        
        // 获取瓦片的实例范围
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = tile_range.y - tile_range.x;  // 瓦片内的总实例数

        // 步骤4：桶偏移量和索引设置
        // 计算瓦片在桶数组中的起始偏移量
        uint bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
        
        // 重新计算桶数量（比从数组读取更快）
        const int n_buckets = div_round_up(n_points_total, 32);
        
        // 设置桶的瓦片索引
        for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank; 
             n_buckets_remaining > 0; 
             n_buckets_remaining -= config::block_size_blend, current_bucket_idx += config::block_size_blend) {
            if (current_bucket_idx < n_buckets)
                bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
        }

        // 步骤5：共享内存设置
        // 为图元数据分配共享内存，提高访问效率
        __shared__ float2 collected_mean2d[config::block_size_blend];      // 收集的2D位置
        __shared__ float4 collected_conic_opacity[config::block_size_blend]; // 收集的圆锥体参数和不透明度
        __shared__ float3 collected_color[config::block_size_blend];        // 收集的颜色
        
        // 步骤6：局部变量初始化
        float3 color_pixel = make_float3(0.0f);    // 像素的累积颜色
        float transmittance = 1.0f;                // 当前透射率
        uint n_possible_contributions = 0;         // 可能的贡献数量
        uint n_contributions = 0;                  // 实际贡献数量
        bool done = !inside;                       // 完成标志（像素在边界外时设为true）

        // 步骤7：协作加载和处理
        // 协作加载图元数据并处理实例
        for (int n_points_remaining = n_points_total, current_fetch_idx = tile_range.x + thread_rank; 
             n_points_remaining > 0; 
             n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            
            // 检查是否所有线程都已完成
            if (__syncthreads_count(done) == config::block_size_blend)
                break;
            
            // 如果当前线程的实例索引有效，加载图元数据
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];  // 获取图元索引
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];         // 加载2D位置
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx]; // 加载圆锥体参数
                const float3 color = fmaxf(primitive_color[primitive_idx], 0.0f);       // 加载颜色并确保非负
                collected_color[thread_rank] = color;
            }
            
            // 同步所有线程，确保数据加载完成
            block.sync();
            
            // 步骤8：批处理实例
            // 处理当前批次中的实例
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            
            for (int j = 0; !done && j < current_batch_size; ++j) {
                // 每32个实例存储一次桶数据
                if (j % 32 == 0) {
                    const float4 current_color_transmittance = make_float4(color_pixel, transmittance);
                    bucket_color_transmittance[bucket_offset * config::block_size_blend + thread_rank] = current_color_transmittance;
                    bucket_offset++;
                }
                
                // 增加可能的贡献计数
                n_possible_contributions++;
                
                // 步骤9：高斯权重计算
                // 从共享内存获取当前实例的数据
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);  // 圆锥体参数 (xx, xy, yy)
                const float2 delta = collected_mean2d[j] - pixel; // 像素到图元中心的距离
                const float opacity = conic_opacity.w;            // 不透明度
                
                // 计算高斯权重：sigma_over_2 = 0.5 * (xx*dx² + yy*dy² + 2*xy*dx*dy)
                const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                
                // 如果sigma_over_2为负，跳过此实例（数值不稳定）
                if (sigma_over_2 < 0.0f)
                    continue;
                
                // 计算高斯函数值：exp(-sigma_over_2)
                const float gaussian = expf(-sigma_over_2);
                
                // 步骤10：透明度混合
                // 计算最终的alpha值：alpha = opacity * gaussian
                const float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
                
                // 如果alpha太小，跳过此实例
                if (alpha < config::min_alpha_threshold)
                    continue;
                
                // 计算下一个透射率：transmittance_next = transmittance * (1 - alpha)
                const float next_transmittance = transmittance * (1.0f - alpha);
                
                // 如果透射率太小，停止处理（像素已完全不透明）
                if (next_transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
                
                // 步骤11：颜色累积
                // 累积颜色：color += transmittance * alpha * instance_color
                color_pixel += transmittance * alpha * collected_color[j];
                
                // 更新透射率
                transmittance = next_transmittance;
                
                // 更新实际贡献数量
                n_contributions = n_possible_contributions;
            }
        }
        
        // 步骤12：结果存储
        // 如果像素在图像边界内，存储渲染结果
        if (inside) {
            // 计算像素在图像中的索引
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const int n_pixels = width * height;
            
            // 存储RGB颜色到图像缓冲区
            image[pixel_idx] = color_pixel.x;                    // 红色分量
            image[pixel_idx + n_pixels] = color_pixel.y;         // 绿色分量
            image[pixel_idx + n_pixels * 2] = color_pixel.z;    // 蓝色分量
            
            // 存储透明度：alpha = 1 - transmittance
            alpha_map[pixel_idx] = 1.0f - transmittance;
            
            // 存储贡献数量
            tile_n_contributions[pixel_idx] = n_contributions;
        }

        // 步骤13：贡献数量最大值归约
        // 使用CUB库进行块内归约，找到最大贡献数量
        typedef cub::BlockReduce<uint, config::tile_width, cub::BLOCK_REDUCE_WARP_REDUCTIONS, config::tile_height> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        // 执行最大值归约
        n_contributions = BlockReduce(temp_storage).Reduce(n_contributions, cub::Max());
        
        // 块内第一个线程存储结果
        if (thread_rank == 0)
            tile_max_n_contributions[tile_idx] = n_contributions;
    }

} // namespace fast_gs::rasterization::kernels::forward
