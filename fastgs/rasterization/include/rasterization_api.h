#pragma once

#include <torch/extension.h>
#include <tuple>

/**
 * @brief FastGS光栅化命名空间，提供高效的高斯散射体渲染接口
 * @details 该命名空间包含FastGS（Fast Gaussian Splatting）渲染系统的核心API，
 *          提供前向渲染和反向传播功能，支持实时高斯散射体渲染和训练。
 */
namespace fast_gs::rasterization {

    /**
     * @struct FastGSSettings
     * @brief FastGS渲染设置结构体，包含相机和渲染参数
     * @details 该结构体定义了高斯散射体渲染所需的所有配置参数，
     *          包括相机位置、球谐系数数量、图像尺寸、相机内参等。
     */
    struct FastGSSettings {
        torch::Tensor cam_position;    ///< 相机位置 [3]，世界坐标系中的相机位置
        int active_sh_bases;           ///< 激活的球谐基函数数量，控制光照复杂度
        int width;                     ///< 输出图像宽度（像素）
        int height;                    ///< 输出图像高度（像素）
        float focal_x;                 ///< X轴焦距（像素），相机内参
        float focal_y;                 ///< Y轴焦距（像素），相机内参
        float center_x;                ///< 主点X坐标（像素），相机内参
        float center_y;                ///< 主点Y坐标（像素），相机内参
        float near_plane;              ///< 近裁剪平面距离，避免z-fighting
        float far_plane;               ///< 远裁剪平面距离，设置很大的值
    };

    /**
     * @brief 前向渲染包装函数，执行高斯散射体的光栅化渲染
     * @param means 高斯体位置坐标 [N, 3]，N为高斯体数量
     * @param scales_raw 缩放参数 [N, 3]（对数形式），控制高斯体大小
     * @param rotations_raw 旋转四元数 [N, 4]，控制高斯体方向
     * @param opacities_raw 不透明度 [N, 1]（对数形式），控制可见性
     * @param sh_coefficients_0 0阶球谐系数 [N, 1, 3]，基础颜色
     * @param sh_coefficients_rest 高阶球谐系数 [N, K, 3]，复杂光照
     * @param w2c 世界到相机变换矩阵 [4, 4]，包含旋转和平移
     * @param cam_position 相机位置 [3]，世界坐标系
     * @param active_sh_bases 激活的球谐基函数数量
     * @param width 输出图像宽度
     * @param height 输出图像高度
     * @param focal_x X轴焦距
     * @param focal_y Y轴焦距
     * @param center_x 主点X坐标
     * @param center_y 主点Y坐标
     * @param near_plane 近裁剪平面距离
     * @param far_plane 远裁剪平面距离
     * @return 返回渲染结果和相关缓冲区：
     *         - 渲染图像 [H, W, 3]
     *         - Alpha通道 [H, W, 1]
     *         - 每图元缓冲区（用于反向传播）
     *         - 每瓦片缓冲区（用于反向传播）
     *         - 每实例缓冲区（用于反向传播）
     *         - 每桶缓冲区（用于反向传播）
     *         - 可见图元数量
     *         - 实例数量
     *         - 桶数量
     *         - 图元索引选择器
     *         - 实例索引选择器
     * @details 该函数执行高斯散射体的前向渲染，将3D高斯体投影到2D图像平面，
     *          生成RGB图像和Alpha通道。同时准备反向传播所需的缓冲区。
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int>
    forward_wrapper(
        const torch::Tensor& means,                    // 高斯体位置 [N, 3]
        const torch::Tensor& scales_raw,               // 缩放参数 [N, 3]（对数形式）
        const torch::Tensor& rotations_raw,            // 旋转四元数 [N, 4]
        const torch::Tensor& opacities_raw,            // 不透明度 [N, 1]（对数形式）
        const torch::Tensor& sh_coefficients_0,        // 0阶球谐系数 [N, 1, 3]
        const torch::Tensor& sh_coefficients_rest,     // 高阶球谐系数 [N, K, 3]
        const torch::Tensor& w2c,                      // 世界到相机变换矩阵 [4, 4]
        const torch::Tensor& cam_position,             // 相机位置 [3]
        const int active_sh_bases,                     // 激活的球谐基函数数量
        const int width,                               // 图像宽度
        const int height,                              // 图像高度
        const float focal_x,                           // X轴焦距
        const float focal_y,                           // Y轴焦距
        const float center_x,                           // 主点X坐标
        const float center_y,                           // 主点Y坐标
        const float near_plane,                         // 近裁剪平面距离
        const float far_plane);                        // 远裁剪平面距离

    /**
     * @brief 反向传播包装函数，计算高斯散射体参数的梯度
     * @param densification_info 密度化信息 [N]，存储梯度幅值
     * @param grad_image 图像梯度 [H, W, 3]，来自损失函数的梯度
     * @param grad_alpha Alpha梯度 [H, W, 1]，来自损失函数的梯度
     * @param image 渲染图像 [H, W, 3]，前向传播的输出
     * @param alpha Alpha通道 [H, W, 1]，前向传播的输出
     * @param means 高斯体位置 [N, 3]，前向传播的输入
     * @param scales_raw 缩放参数 [N, 3]（对数形式），前向传播的输入
     * @param rotations_raw 旋转四元数 [N, 4]，前向传播的输入
     * @param sh_coefficients_rest 高阶球谐系数 [N, K, 3]，前向传播的输入
     * @param per_primitive_buffers 每图元缓冲区，前向传播的输出
     * @param per_tile_buffers 每瓦片缓冲区，前向传播的输出
     * @param per_instance_buffers 每实例缓冲区，前向传播的输出
     * @param per_bucket_buffers 每桶缓冲区，前向传播的输出
     * @param w2c 世界到相机变换矩阵 [4, 4]，前向传播的输入
     * @param cam_position 相机位置 [3]，前向传播的输入
     * @param active_sh_bases 激活的球谐基函数数量
     * @param width 图像宽度
     * @param height 图像高度
     * @param focal_x X轴焦距
     * @param focal_y Y轴焦距
     * @param center_x 主点X坐标
     * @param center_y 主点Y坐标
     * @param near_plane 近裁剪平面距离
     * @param far_plane 远裁剪平面距离
     * @param n_visible_primitives 可见图元数量，前向传播的输出
     * @param n_instances 实例数量，前向传播的输出
     * @param n_buckets 桶数量，前向传播的输出
     * @param primitive_primitive_indices_selector 图元索引选择器
     * @param instance_primitive_indices_selector 实例索引选择器
     * @return 返回各参数的梯度：
     *         - 位置梯度 [N, 3]
     *         - 缩放梯度 [N, 3]
     *         - 旋转梯度 [N, 4]
     *         - 不透明度梯度 [N, 1]
     *         - 0阶球谐系数梯度 [N, 1, 3]
     *         - 高阶球谐系数梯度 [N, K, 3]
     *         - 更新后的密度化信息 [N]
     * @details 该函数执行反向传播，计算高斯散射体各参数对损失函数的梯度。
     *          使用前向传播的缓冲区和输出，高效地计算梯度并更新密度化信息。
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward_wrapper(
        torch::Tensor& densification_info,             // 密度化信息 [N]（输入输出）
        const torch::Tensor& grad_image,               // 图像梯度 [H, W, 3]
        const torch::Tensor& grad_alpha,                // Alpha梯度 [H, W, 1]
        const torch::Tensor& image,                    // 渲染图像 [H, W, 3]
        const torch::Tensor& alpha,                    // Alpha通道 [H, W, 1]
        const torch::Tensor& means,                     // 高斯体位置 [N, 3]
        const torch::Tensor& scales_raw,               // 缩放参数 [N, 3]（对数形式）
        const torch::Tensor& rotations_raw,            // 旋转四元数 [N, 4]
        const torch::Tensor& sh_coefficients_rest,     // 高阶球谐系数 [N, K, 3]
        const torch::Tensor& per_primitive_buffers,     // 每图元缓冲区
        const torch::Tensor& per_tile_buffers,         // 每瓦片缓冲区
        const torch::Tensor& per_instance_buffers,     // 每实例缓冲区
        const torch::Tensor& per_bucket_buffers,      // 每桶缓冲区
        const torch::Tensor& w2c,                      // 世界到相机变换矩阵 [4, 4]
        const torch::Tensor& cam_position,             // 相机位置 [3]
        const int active_sh_bases,                     // 激活的球谐基函数数量
        const int width,                               // 图像宽度
        const int height,                              // 图像高度
        const float focal_x,                           // X轴焦距
        const float focal_y,                           // Y轴焦距
        const float center_x,                           // 主点X坐标
        const float center_y,                           // 主点Y坐标
        const float near_plane,                         // 近裁剪平面距离
        const float far_plane,                         // 远裁剪平面距离
        const int n_visible_primitives,                // 可见图元数量
        const int n_instances,                         // 实例数量
        const int n_buckets,                           // 桶数量
        const int primitive_primitive_indices_selector, // 图元索引选择器
        const int instance_primitive_indices_selector); // 实例索引选择器

} // namespace fast_gs::rasterization