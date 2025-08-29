#pragma once

#include "buffer_utils.h"
#include "helper_math.h"
#include "kernel_utils.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cooperative_groups.h>
#include <cstdint>
namespace cg = cooperative_groups;

namespace fast_gs::rasterization::kernels::backward {

    /**
     * [功能描述]：高斯样条反向传播预处理内核函数，计算高斯样条参数相对于渲染损失的梯度
     * @param means 3D高斯样条中心点坐标数组
     * @param raw_scales 原始缩放参数数组（对数形式）
     * @param raw_rotations 原始旋转四元数数组
     * @param sh_coefficients_rest 球谐函数系数数组（除第0阶外）
     * @param w2c 世界坐标系到相机坐标系的变换矩阵
     * @param cam_position 相机位置
     * @param primitive_n_touched_tiles 每个样条触及的瓦片数量
     * @param grad_mean2d 2D投影点的梯度
     * @param grad_conic 圆锥体参数的梯度
     * @param grad_means 输出：3D中心点的梯度
     * @param grad_raw_scales 输出：原始缩放参数的梯度
     * @param grad_raw_rotations 输出：原始旋转四元数的梯度
     * @param grad_sh_coefficients_0 输出：第0阶球谐系数的梯度
     * @param grad_sh_coefficients_rest 输出：高阶球谐系数的梯度
     * @param grad_w2c 输出：世界到相机变换矩阵的梯度
     * @param densification_info 密度控制信息数组
     * @param n_primitives 样条总数
     * @param active_sh_bases 活跃的球谐函数基数量
     * @param total_bases_sh_rest 高阶球谐函数基总数
     * @param w 图像宽度
     * @param h 图像高度
     * @param fx 相机内参fx
     * @param fy 相机内参fy
     * @param cx 相机内参cx
     * @param cy 相机内参cy
     */
    __global__ void preprocess_backward_cu(
        const float3* means,
        const float3* raw_scales,
        const float4* raw_rotations,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        const uint* primitive_n_touched_tiles,
        const float2* grad_mean2d,
        const float* grad_conic,
        float3* grad_means,
        float3* grad_raw_scales,
        float4* grad_raw_rotations,
        float3* grad_sh_coefficients_0,
        float3* grad_sh_coefficients_rest,
        float4* grad_w2c,
        float* densification_info,
        const uint n_primitives,
        const uint active_sh_bases,
        const uint total_bases_sh_rest,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy) {
        
        // 获取当前线程对应的样条索引
        auto primitive_idx = cg::this_grid().thread_rank();
        // 如果样条索引超出范围或该样条没有触及任何瓦片，则直接返回
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0)
            return;

        // 加载3D高斯样条中心点坐标
        const float3 mean3d = means[primitive_idx];

        // 球谐函数反向传播：计算颜色对3D中心点的梯度
        const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
            sh_coefficients_rest, grad_sh_coefficients_0, grad_sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, total_bases_sh_rest);

        // 从世界坐标系变换到相机坐标系
        const float4 w2c_r3 = w2c[2];  // 变换矩阵第3行
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;  // 计算深度
        const float4 w2c_r1 = w2c[0];  // 变换矩阵第1行
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;  // 计算x坐标
        const float4 w2c_r2 = w2c[1];  // 变换矩阵第2行
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;  // 计算y坐标

        // 从原始缩放和旋转参数计算3D协方差矩阵
        const float3 raw_scale = raw_scales[primitive_idx];
        const float3 variance = make_float3(expf(2.0f * raw_scale.x), expf(2.0f * raw_scale.y), expf(2.0f * raw_scale.z));  // 计算方差
        auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];  // 解构四元数
        const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;  // 计算四元数各分量的平方
        const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;  // 计算四元数模长的平方
        // 归一化四元数分量
        const float qxx = 2.0f * qxx_raw / q_norm_sq, qyy = 2.0f * qyy_raw / q_norm_sq, qzz = 2.0f * qzz_raw / q_norm_sq;
        const float qxy = 2.0f * qx * qy / q_norm_sq, qxz = 2.0f * qx * qz / q_norm_sq, qyz = 2.0f * qy * qz / q_norm_sq;
        const float qrx = 2.0f * qr * qx / q_norm_sq, qry = 2.0f * qr * qy / q_norm_sq, qrz = 2.0f * qr * qz / q_norm_sq;
        
        // 构建旋转矩阵（从四元数转换）
        const mat3x3 rotation = {
            1.0f - (qyy + qzz), qxy - qrz, qry + qxz,
            qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,
            qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)};
        
        // 应用缩放后的旋转矩阵
        const mat3x3 rotation_scaled = {
            rotation.m11 * variance.x, rotation.m12 * variance.y, rotation.m13 * variance.z,
            rotation.m21 * variance.x, rotation.m22 * variance.y, rotation.m23 * variance.z,
            rotation.m31 * variance.x, rotation.m32 * variance.y, rotation.m33 * variance.z};
        
        // 计算3D协方差矩阵（上三角形式）
        const mat3x3_triu cov3d{
            rotation_scaled.m11 * rotation.m11 + rotation_scaled.m12 * rotation.m12 + rotation_scaled.m13 * rotation.m13,
            rotation_scaled.m11 * rotation.m21 + rotation_scaled.m12 * rotation.m22 + rotation_scaled.m13 * rotation.m23,
            rotation_scaled.m11 * rotation.m31 + rotation_scaled.m12 * rotation.m32 + rotation_scaled.m13 * rotation.m33,
            rotation_scaled.m21 * rotation.m21 + rotation_scaled.m22 * rotation.m22 + rotation_scaled.m23 * rotation.m23,
            rotation_scaled.m21 * rotation.m31 + rotation_scaled.m22 * rotation.m32 + rotation_scaled.m23 * rotation.m33,
            rotation_scaled.m31 * rotation.m31 + rotation_scaled.m32 * rotation.m32 + rotation_scaled.m33 * rotation.m33,
        };

        // EWA样条梯度计算辅助变量
        const float clip_left = (-0.15f * w - cx) / fx;    // 左裁剪边界
        const float clip_right = (1.15f * w - cx) / fx;    // 右裁剪边界
        const float clip_top = (-0.15f * h - cy) / fy;     // 上裁剪边界
        const float clip_bottom = (1.15f * h - cy) / fy;   // 下裁剪边界
        const float tx = clamp(x, clip_left, clip_right);  // 限制x坐标在裁剪范围内
        const float ty = clamp(y, clip_top, clip_bottom);  // 限制y坐标在裁剪范围内
        
        // 计算雅可比矩阵的非零元素
        const float j11 = fx / depth;      // 雅可比矩阵(1,1)元素
        const float j13 = -j11 * tx;       // 雅可比矩阵(1,3)元素
        const float j22 = fy / depth;      // 雅可比矩阵(2,2)元素
        const float j23 = -j22 * ty;       // 雅可比矩阵(2,3)元素
        
        // 计算雅可比矩阵与旋转矩阵的乘积
        const float3 jw_r1 = make_float3(
            j11 * w2c_r1.x + j13 * w2c_r3.x,
            j11 * w2c_r1.y + j13 * w2c_r3.y,
            j11 * w2c_r1.z + j13 * w2c_r3.z);
        const float3 jw_r2 = make_float3(
            j22 * w2c_r2.x + j23 * w2c_r3.x,
            j22 * w2c_r2.y + j23 * w2c_r3.y,
            j22 * w2c_r2.z + j23 * w2c_r3.z);
        
        // 计算雅可比矩阵、旋转矩阵和协方差矩阵的乘积
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33);
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33);

        // 计算2D协方差矩阵的梯度
        const float a = dot(jwc_r1, jw_r1) + config::dilation, b = dot(jwc_r1, jw_r2), c = dot(jwc_r2, jw_r2) + config::dilation;
        const float aa = a * a, bb = b * b, cc = c * c;
        const float ac = a * c, ab = a * b, bc = b * c;
        const float determinant = ac - bb;  // 行列式
        const float determinant_rcp = 1.0f / determinant;  // 行列式的倒数
        const float determinant_rcp_sq = determinant_rcp * determinant_rcp;  // 行列式倒数的平方
        
        // 加载圆锥体参数的梯度
        const float3 dL_dconic = make_float3(
            grad_conic[primitive_idx],
            grad_conic[n_primitives + primitive_idx],
            grad_conic[2 * n_primitives + primitive_idx]);
        
        // 计算2D协方差矩阵的梯度
        const float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                                                          2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                                                          bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
                                                          2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

        // 计算3D协方差矩阵的梯度
        const mat3x3_triu dL_dcov3d = {
            (jw_r1.x * jw_r1.x) * dL_dcov2d.x + 2.0f * (jw_r1.x * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.x) * dL_dcov2d.z,
            (jw_r1.x * jw_r1.y) * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.y) * dL_dcov2d.z,
            (jw_r1.x * jw_r1.z) * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.z) * dL_dcov2d.z,
            (jw_r1.y * jw_r1.y) * dL_dcov2d.x + 2.0f * (jw_r1.y * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.y) * dL_dcov2d.z,
            (jw_r1.y * jw_r1.z) * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.z) * dL_dcov2d.z,
            (jw_r1.z * jw_r1.z) * dL_dcov2d.x + 2.0f * (jw_r1.z * jw_r2.z) * dL_dcov2d.y + (jw_r2.z * jw_r2.z) * dL_dcov2d.z,
        };

        // 计算J*W的梯度
        const float3 dL_djw_r1 = 2.0f * make_float3(
                                            jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
                                            jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
                                            jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y);
        const float3 dL_djw_r2 = 2.0f * make_float3(
                                            jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
                                            jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
                                            jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z);

        // 计算雅可比矩阵非零元素的梯度
        const float dL_dj11 = w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z;
        const float dL_dj22 = w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z;
        const float dL_dj13 = w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z;
        const float dL_dj23 = w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z;

        // 从雅可比矩阵和2D投影点计算相机空间中3D中心点的梯度
        // TODO: 原始3DGS在这里考虑了tx/ty的裁剪，但似乎这不是必要的
        float djwr1_dz_helper = dL_dj11 - 2.0f * tx * dL_dj13;
        float djwr2_dz_helper = dL_dj22 - 2.0f * ty * dL_dj23;
        const float2 dL_dmean2d = grad_mean2d[primitive_idx];  // 加载2D投影点的梯度
        const float3 dL_dmean3d_cam = make_float3(
            j11 * (dL_dmean2d.x - dL_dj13 / depth),
            j22 * (dL_dmean2d.y - dL_dj23 / depth),
            -j11 * (x * dL_dmean2d.x + djwr1_dz_helper / depth) - j22 * (y * dL_dmean2d.y + djwr2_dz_helper / depth));

        // 如果梯度输出指针不为空，则计算世界到相机变换矩阵的梯度
        if (grad_w2c != nullptr) {
            // 使用原子操作累加变换矩阵的梯度
            atomicAdd(&grad_w2c[0].w, dL_dmean3d_cam.x);
            atomicAdd(&grad_w2c[1].w, dL_dmean3d_cam.y);
            atomicAdd(&grad_w2c[2].w, dL_dmean3d_cam.z);
            atomicAdd(&grad_w2c[0].x, dL_dmean3d_cam.x * mean3d.x);
            atomicAdd(&grad_w2c[0].y, dL_dmean3d_cam.x * mean3d.y);
            atomicAdd(&grad_w2c[0].z, dL_dmean3d_cam.x * mean3d.z);
            atomicAdd(&grad_w2c[1].x, dL_dmean3d_cam.y * mean3d.x);
            atomicAdd(&grad_w2c[1].y, dL_dmean3d_cam.y * mean3d.y);
            atomicAdd(&grad_w2c[1].z, dL_dmean3d_cam.y * mean3d.z);
            atomicAdd(&grad_w2c[2].x, dL_dmean3d_cam.z * mean3d.x);
            atomicAdd(&grad_w2c[2].y, dL_dmean3d_cam.z * mean3d.y);
            atomicAdd(&grad_w2c[2].z, dL_dmean3d_cam.z * mean3d.z);
        }

        // 从样条计算3D中心点的梯度
        const float3 dL_dmean3d_from_splatting = make_float3(
            w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
            w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
            w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z);

        // 写入总的3D中心点梯度（样条梯度 + 颜色梯度）
        const float3 dL_dmean3d = dL_dmean3d_from_splatting + dL_dmean3d_from_color;
        grad_means[primitive_idx] = dL_dmean3d;

        // 计算原始缩放参数的梯度
        const float dL_dvariance_x = rotation.m11 * rotation.m11 * dL_dcov3d.m11 + rotation.m21 * rotation.m21 * dL_dcov3d.m22 + rotation.m31 * rotation.m31 * dL_dcov3d.m33 +
                                     2.0f * (rotation.m11 * rotation.m21 * dL_dcov3d.m12 + rotation.m11 * rotation.m31 * dL_dcov3d.m13 + rotation.m21 * rotation.m31 * dL_dcov3d.m23);
        const float dL_dvariance_y = rotation.m12 * rotation.m12 * dL_dcov3d.m11 + rotation.m22 * rotation.m22 * dL_dcov3d.m22 + rotation.m32 * rotation.m32 * dL_dcov3d.m33 +
                                     2.0f * (rotation.m12 * rotation.m22 * dL_dcov3d.m12 + rotation.m12 * rotation.m32 * dL_dcov3d.m13 + rotation.m22 * rotation.m32 * dL_dcov3d.m23);
        const float dL_dvariance_z = rotation.m13 * rotation.m13 * dL_dcov3d.m11 + rotation.m23 * rotation.m23 * dL_dcov3d.m22 + rotation.m33 * rotation.m33 * dL_dcov3d.m33 +
                                     2.0f * (rotation.m13 * rotation.m23 * dL_dcov3d.m12 + rotation.m13 * rotation.m33 * dL_dcov3d.m13 + rotation.m23 * rotation.m33 * dL_dcov3d.m23);
        
        // 计算原始缩放参数的梯度（考虑方差到原始缩放的变换）
        const float3 dL_draw_scale = make_float3(
            2.0f * variance.x * dL_dvariance_x,
            2.0f * variance.y * dL_dvariance_y,
            2.0f * variance.z * dL_dvariance_z);
        grad_raw_scales[primitive_idx] = dL_draw_scale;

        // 计算原始旋转四元数的梯度
        const mat3x3 dL_drotation = {
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m11 + rotation_scaled.m21 * dL_dcov3d.m12 + rotation_scaled.m31 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m11 + rotation_scaled.m22 * dL_dcov3d.m12 + rotation_scaled.m32 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m11 + rotation_scaled.m23 * dL_dcov3d.m12 + rotation_scaled.m33 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m12 + rotation_scaled.m21 * dL_dcov3d.m22 + rotation_scaled.m31 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m12 + rotation_scaled.m22 * dL_dcov3d.m22 + rotation_scaled.m32 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m12 + rotation_scaled.m23 * dL_dcov3d.m22 + rotation_scaled.m33 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m13 + rotation_scaled.m21 * dL_dcov3d.m23 + rotation_scaled.m31 * dL_dcov3d.m33),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m13 + rotation_scaled.m22 * dL_dcov3d.m23 + rotation_scaled.m32 * dL_dcov3d.m33),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m13 + rotation_scaled.m23 * dL_dcov3d.m23 + rotation_scaled.m33 * dL_dcov3d.m33)};
        
        // 从旋转矩阵梯度计算四元数分量的梯度
        const float dL_dqxx = -dL_drotation.m22 - dL_drotation.m33;
        const float dL_dqyy = -dL_drotation.m11 - dL_drotation.m33;
        const float dL_dqzz = -dL_drotation.m11 - dL_drotation.m22;
        const float dL_dqxy = dL_drotation.m12 + dL_drotation.m21;
        const float dL_dqxz = dL_drotation.m13 + dL_drotation.m31;
        const float dL_dqyz = dL_drotation.m23 + dL_drotation.m32;
        const float dL_dqrx = dL_drotation.m32 - dL_drotation.m23;
        const float dL_dqry = dL_drotation.m13 - dL_drotation.m31;
        const float dL_dqrz = dL_drotation.m21 - dL_drotation.m12;
        
        // 计算四元数归一化的辅助变量
        const float dL_dq_norm_helper = qxx * dL_dqxx + qyy * dL_dqyy + qzz * dL_dqzz + qxy * dL_dqxy + qxz * dL_dqxz + qyz * dL_dqyz + qrx * dL_dqrx + qry * dL_dqry + qrz * dL_dqrz;
        
        // 计算原始旋转四元数的梯度
        const float4 dL_draw_rotation = 2.0f * make_float4(qx * dL_dqrx + qy * dL_dqry + qz * dL_dqrz - qr * dL_dq_norm_helper, 2.0f * qx * dL_dqxx + qy * dL_dqxy + qz * dL_dqxz + qr * dL_dqrx - qx * dL_dq_norm_helper, 2.0f * qy * dL_dqyy + qx * dL_dqxy + qz * dL_dqyz + qr * dL_dqry - qy * dL_dq_norm_helper, 2.0f * qz * dL_dqzz + qx * dL_dqxz + qy * dL_dqyz + qr * dL_dqrz - qz * dL_dq_norm_helper) / q_norm_sq;
        grad_raw_rotations[primitive_idx] = dL_draw_rotation;

        // TODO: 仅原始3DGS中自适应密度控制需要
        if (densification_info != nullptr) {
            densification_info[primitive_idx] += 1.0f;  // 增加密度计数
            densification_info[n_primitives + primitive_idx] += length(dL_dmean2d * make_float2(0.5f * w, 0.5f * h));  // 计算梯度幅值
        }
    }

    // 基于 https://github.com/humansensinglab/taming-3dgs/blob/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L404
    /**
     * [功能描述]：高斯样条混合反向传播内核函数，计算混合操作中各个参数的梯度
     * @param tile_instance_ranges 每个瓦片的实例范围数组
     * @param tile_bucket_offsets 每个瓦片的桶偏移数组
     * @param instance_primitive_indices 实例到样条索引的映射数组
     * @param primitive_mean2d 样条的2D投影中心点数组
     * @param primitive_conic_opacity 样条的圆锥体参数和不透明度数组
     * @param primitive_color 样条的颜色数组
     * @param grad_image 图像颜色的梯度数组
     * @param grad_alpha_map 透明度图的梯度数组
     * @param image 前向传播生成的图像数组
     * @param alpha_map 前向传播生成的透明度图数组
     * @param tile_max_n_contributions 每个瓦片的最大贡献数数组
     * @param tile_n_contributions 每个像素的贡献数数组
     * @param bucket_tile_index 桶到瓦片索引的映射数组
     * @param bucket_color_transmittance 桶中颜色和透射率数组
     * @param grad_mean2d 输出：2D投影中心点的梯度数组
     * @param grad_conic 输出：圆锥体参数的梯度数组
     * @param grad_raw_opacity 输出：原始不透明度的梯度数组
     * @param grad_color 输出：颜色的梯度数组
     * @param n_buckets 桶的总数
     * @param n_primitives 样条的总数
     * @param width 图像宽度
     * @param height 图像高度
     * @param grid_width 瓦片网格的宽度
     */
    __global__ void blend_backward_cu(
        const uint2* tile_instance_ranges,
        const uint* tile_bucket_offsets,
        const uint* instance_primitive_indices,
        const float2* primitive_mean2d,
        const float4* primitive_conic_opacity,
        const float3* primitive_color,
        const float* grad_image,
        const float* grad_alpha_map,
        const float* image,
        const float* alpha_map,
        const uint* tile_max_n_contributions,
        const uint* tile_n_contributions,
        const uint* bucket_tile_index,
        const float4* bucket_color_transmittance,
        float2* grad_mean2d,
        float* grad_conic,
        float* grad_raw_opacity,
        float3* grad_color,
        const uint n_buckets,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width) {
        
        // 获取当前线程块和桶索引
        auto block = cg::this_thread_block();
        const uint bucket_idx = block.group_index().x;
        if (bucket_idx >= n_buckets)
            return;
        
        // 创建32线程的warp分区
        auto warp = cg::tiled_partition<32>(block);
        const uint lane_idx = warp.thread_rank();

        // 获取当前桶对应的瓦片信息
        const uint tile_idx = bucket_tile_index[bucket_idx];
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        const uint tile_first_bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
        
        // 检查桶索引是否超出瓦片的最大贡献数限制
        if (tile_bucket_idx * 32 >= tile_max_n_contributions[tile_idx])
            return;

        // 计算当前线程对应的样条索引
        const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
        const int instance_idx = tile_instance_range.x + tile_primitive_idx;
        const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

        // 加载高斯样条数据
        uint primitive_idx = 0;
        float2 mean2d = {0.0f, 0.0f};      // 2D投影中心点
        float3 conic = {0.0f, 0.0f, 0.0f}; // 圆锥体参数
        float opacity = 0.0f;               // 不透明度
        float3 color = {0.0f, 0.0f, 0.0f}; // 颜色
        float3 color_grad_factor = {0.0f, 0.0f, 0.0f}; // 颜色梯度因子
        
        if (valid_primitive) {
            primitive_idx = instance_primitive_indices[instance_idx];
            mean2d = primitive_mean2d[primitive_idx];
            const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
            conic = make_float3(conic_opacity);  // 提取圆锥体参数
            opacity = conic_opacity.w;            // 提取不透明度
            const float3 color_unclamped = primitive_color[primitive_idx];
            color = fmaxf(color_unclamped, 0.0f); // 限制颜色为非负值
            
            // 计算颜色梯度因子（只对非负颜色值计算梯度）
            if (color_unclamped.x >= 0.0f)
                color_grad_factor.x = 1.0f;
            if (color_unclamped.y >= 0.0f)
                color_grad_factor.y = 1.0f;
            if (color_unclamped.z >= 0.0f)
                color_grad_factor.z = 1.0f;
        }

        // 辅助变量
        const uint n_pixels = width * height;

        // 梯度累积变量
        float2 dL_dmean2d_accum = {0.0f, 0.0f};        // 2D中心点梯度累积
        float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};   // 圆锥体参数梯度累积
        float dL_draw_opacity_partial_accum = 0.0f;      // 原始不透明度部分梯度累积
        float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};   // 颜色梯度累积

        // 瓦片元数据
        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

        // 局部变量声明
        uint last_contributor;           // 最后一个贡献者
        float3 color_pixel_after;        // 像素处理后的颜色
        float transmittance;             // 透射率
        float3 grad_color_pixel;         // 像素颜色梯度
        float grad_alpha_common;         // 通用透明度梯度

        // 指向当前桶的颜色透射率数据
        bucket_color_transmittance += bucket_idx * config::block_size_blend;
        
        // 共享内存数组，用于收集数据
        __shared__ uint collected_last_contributor[32];                    // 收集的最后一个贡献者
        __shared__ float4 collected_color_pixel_after_transmittance[32];  // 收集的颜色像素和透射率
        __shared__ float4 collected_grad_info_pixel[32];                  // 收集的梯度信息像素

        // 遍历瓦片中的所有像素
#pragma unroll
        for (int i = 0; i < config::block_size_blend + 31; ++i) {
            // 每32个迭代加载一次数据到共享内存
            if (i % 32 == 0) {
                const uint local_idx = i + lane_idx;
                const float4 color_transmittance = bucket_color_transmittance[local_idx];
                const uint2 pixel_coords = {start_pixel_coords.x + local_idx % config::tile_width, start_pixel_coords.y + local_idx / config::tile_width};
                const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                
                // 从前向传播获取最终值（背景混合前）和相应的梯度
                float3 color_pixel, grad_color_pixel;
                float alpha_pixel, grad_alpha_pixel;
                if (pixel_coords.x < width && pixel_coords.y < height) {
                    // 加载像素颜色和梯度
                    color_pixel = make_float3(
                        image[pixel_idx],
                        image[n_pixels + pixel_idx],
                        image[2 * n_pixels + pixel_idx]);
                    grad_color_pixel = make_float3(
                        grad_image[pixel_idx],
                        grad_image[n_pixels + pixel_idx],
                        grad_image[2 * n_pixels + pixel_idx]);
                    alpha_pixel = alpha_map[pixel_idx];
                    grad_alpha_pixel = grad_alpha_map[pixel_idx];
                }
                
                // 收集数据到共享内存
                collected_color_pixel_after_transmittance[lane_idx] = make_float4(
                    color_pixel - make_float3(color_transmittance),
                    color_transmittance.w);
                collected_grad_info_pixel[lane_idx] = make_float4(
                    grad_color_pixel,
                    grad_alpha_pixel * (1.0f - alpha_pixel));
                collected_last_contributor[lane_idx] = tile_n_contributions[pixel_idx];
                __syncwarp(); // 同步warp内的所有线程
            }

            // 从上一个迭代获取数据（用于warp内的数据共享）
            if (i > 0) {
                last_contributor = warp.shfl_up(last_contributor, 1);
                color_pixel_after.x = warp.shfl_up(color_pixel_after.x, 1);
                color_pixel_after.y = warp.shfl_up(color_pixel_after.y, 1);
                color_pixel_after.z = warp.shfl_up(color_pixel_after.z, 1);
                transmittance = warp.shfl_up(transmittance, 1);
                grad_color_pixel.x = warp.shfl_up(grad_color_pixel.x, 1);
                grad_color_pixel.y = warp.shfl_up(grad_color_pixel.y, 1);
                grad_color_pixel.z = warp.shfl_up(grad_color_pixel.z, 1);
                grad_alpha_common = warp.shfl_up(grad_alpha_common, 1);
            }

            // 确定当前线程应该处理哪个像素索引
            const int idx = i - static_cast<int>(lane_idx);
            const uint2 pixel_coords = {start_pixel_coords.x + idx % config::tile_width, start_pixel_coords.y + idx / config::tile_width};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;

            // 领导者线程从共享内存加载值到寄存器
            if (valid_primitive && valid_pixel && lane_idx == 0 && idx < config::block_size_blend) {
                const int current_shmem_index = i % 32;
                last_contributor = collected_last_contributor[current_shmem_index];
                const float4 color_pixel_after_transmittance = collected_color_pixel_after_transmittance[current_shmem_index];
                color_pixel_after = make_float3(color_pixel_after_transmittance);
                transmittance = color_pixel_after_transmittance.w;
                const float4 grad_info_pixel = collected_grad_info_pixel[current_shmem_index];
                grad_color_pixel = make_float3(grad_info_pixel);
                grad_alpha_common = grad_info_pixel.w;
            }

            // 检查是否应该跳过当前计算
            const bool skip = !valid_primitive || !valid_pixel || idx < 0 || idx >= config::block_size_blend || tile_primitive_idx >= last_contributor;
            if (skip)
                continue;

            // 计算像素坐标和到样条中心的距离
            const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
            const float2 delta = mean2d - pixel;
            
            // 计算高斯函数的指数部分
            const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            if (sigma_over_2 < 0.0f)
                continue;
            
            // 计算高斯函数值和最终的alpha值
            const float gaussian = expf(-sigma_over_2);
            const float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
            if (alpha < config::min_alpha_threshold)
                continue;
            const float one_minus_alpha = 1.0f - alpha;

            // 计算混合权重
            const float blending_weight = transmittance * alpha;

            // 颜色梯度计算
            const float3 dL_dcolor = blending_weight * grad_color_pixel * color_grad_factor;
            dL_dcolor_accum += dL_dcolor;

            // 更新像素颜色（减去当前样条的贡献）
            color_pixel_after -= blending_weight * color;

            // 透明度梯度计算
            const float one_minus_alpha_rcp = 1.0f / one_minus_alpha;
            const float dL_dalpha_from_color = dot(transmittance * color - color_pixel_after * one_minus_alpha_rcp, grad_color_pixel);
            const float dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp;
            const float dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_alpha;
            
            // 未激活的不透明度梯度
            const float dL_draw_opacity_partial = alpha * dL_dalpha;
            dL_draw_opacity_partial_accum += dL_draw_opacity_partial;

            // 圆锥体参数和2D中心点的梯度计算
            const float gaussian_grad_helper = -alpha * dL_dalpha;
            const float3 dL_dconic = 0.5f * gaussian_grad_helper * make_float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
            dL_dconic_accum += dL_dconic;
            const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
                                                                 conic.x * delta.x + conic.y * delta.y,
                                                                 conic.y * delta.x + conic.z * delta.y);
            dL_dmean2d_accum += dL_dmean2d;

            // 更新透射率
            transmittance *= one_minus_alpha;
        }

        // 最后使用原子操作累加梯度
        if (valid_primitive) {
            atomicAdd(&grad_mean2d[primitive_idx].x, dL_dmean2d_accum.x);
            atomicAdd(&grad_mean2d[primitive_idx].y, dL_dmean2d_accum.y);
            atomicAdd(&grad_conic[primitive_idx], dL_dconic_accum.x);
            atomicAdd(&grad_conic[n_primitives + primitive_idx], dL_dconic_accum.y);
            atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], dL_dconic_accum.z);
            
            // 计算原始不透明度的梯度（考虑激活函数）
            const float dL_draw_opacity = dL_draw_opacity_partial_accum * (1.0f - opacity);
            atomicAdd(&grad_raw_opacity[primitive_idx], dL_draw_opacity);
            
            // 累加颜色梯度
            atomicAdd(&grad_color[primitive_idx].x, dL_dcolor_accum.x);
            atomicAdd(&grad_color[primitive_idx].y, dL_dcolor_accum.y);
            atomicAdd(&grad_color[primitive_idx].z, dL_dcolor_accum.z);
        }
    }

} // namespace fast_gs::rasterization::kernels::backward