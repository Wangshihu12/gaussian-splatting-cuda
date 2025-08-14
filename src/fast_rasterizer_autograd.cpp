#include "core/fast_rasterizer_autograd.hpp"

/**
 * [文件描述]：快速高斯散点光栅化自动微分实现文件
 * 功能：实现FastGSRasterize类的前向和反向传播函数
 * 用途：为高斯散点渲染提供完整的PyTorch自动微分支持
 */

namespace gs {

    /**
     * [功能描述]：FastGS光栅化前向传播函数实现
     * @param ctx：自动微分上下文，用于保存反向传播需要的变量和数据
     * @param means：高斯中心位置 [N, 3]，N为高斯数量
     * @param scales_raw：原始缩放参数 [N, 3]，需要经过激活函数处理
     * @param rotations_raw：原始旋转四元数 [N, 4]，需要归一化处理
     * @param opacities_raw：原始不透明度 [N, 1]，需要sigmoid激活
     * @param sh_coefficients_0：球谐0阶系数 [N, 1, 3]，直流颜色分量
     * @param sh_coefficients_rest：球谐高阶系数 [C, B-1, 3]，方向性颜色
     * @param densification_info：密集化信息 [2, N] 或空张量，用于动态高斯管理
     * @param settings：光栅化配置结构体，包含相机和渲染参数
     * @return 张量列表，包含渲染的图像和alpha通道
     */
    torch::autograd::tensor_list FastGSRasterize::forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& means,                               // [N, 3] 高斯中心位置
        const torch::Tensor& scales_raw,                          // [N, 3] 原始缩放参数
        const torch::Tensor& rotations_raw,                       // [N, 4] 原始旋转四元数
        const torch::Tensor& opacities_raw,                       // [N, 1] 原始不透明度
        const torch::Tensor& sh_coefficients_0,                   // [N, 1, 3] 球谐0阶系数
        const torch::Tensor& sh_coefficients_rest,                // [C, B-1, 3] 球谐高阶系数
        torch::Tensor& densification_info,                        // [2, N] 密集化信息或空张量
        const fast_gs::rasterization::FastGSSettings& settings) { // 光栅化配置

        // =============================================================================
        // 调用底层C++/CUDA前向传播包装函数
        // =============================================================================
        // 该函数执行完整的光栅化管道：投影、排序、混合等
        auto outputs = fast_gs::rasterization::forward_wrapper(
            means,                          // 高斯位置
            scales_raw,                     // 缩放参数
            rotations_raw,                  // 旋转参数
            opacities_raw,                  // 不透明度
            sh_coefficients_0,              // 球谐0阶系数
            sh_coefficients_rest,           // 球谐高阶系数
            settings.w2c,                   // 世界到相机变换矩阵
            settings.cam_position,          // 相机位置
            settings.active_sh_bases,       // 激活的球谐基数量
            settings.width,                 // 图像宽度
            settings.height,                // 图像高度
            settings.focal_x,               // X轴焦距
            settings.focal_y,               // Y轴焦距
            settings.center_x,              // 主点X坐标
            settings.center_y,              // 主点Y坐标
            settings.near_plane,            // 近裁剪平面
            settings.far_plane);            // 远裁剪平面

        // =============================================================================
        // 解构前向传播的输出结果
        // =============================================================================
        // 主要输出
        auto image = std::get<0>(outputs);                    // 渲染的RGB图像 [3, H, W]
        auto alpha = std::get<1>(outputs);                    // Alpha通道 [1, H, W]
        
        // 中间缓存数据（用于反向传播）
        auto per_primitive_buffers = std::get<2>(outputs);    // 每个高斯的缓存数据
        auto per_tile_buffers = std::get<3>(outputs);         // 每个图像瓦片的缓存数据
        auto per_instance_buffers = std::get<4>(outputs);     // 每个实例的缓存数据
        auto per_bucket_buffers = std::get<5>(outputs);       // 每个深度桶的缓存数据
        
        // 统计信息（用于反向传播配置）
        int n_visible_primitives = std::get<6>(outputs);                    // 可见高斯数量
        int n_instances = std::get<7>(outputs);                            // 实例总数
        int n_buckets = std::get<8>(outputs);                              // 深度桶数量
        int primitive_primitive_indices_selector = std::get<9>(outputs);    // 高斯索引选择器
        int instance_primitive_indices_selector = std::get<10>(outputs);    // 实例索引选择器

        // =============================================================================
        // 保存反向传播需要的张量变量
        // =============================================================================
        // 这些张量在反向传播时需要用到，PyTorch会自动管理其生命周期
        ctx->save_for_backward({image,                    // 渲染图像
                                alpha,                    // Alpha通道
                                means,                    // 高斯位置（输入）
                                scales_raw,               // 缩放参数（输入）
                                rotations_raw,            // 旋转参数（输入）
                                sh_coefficients_rest,     // 球谐高阶系数（输入）
                                per_primitive_buffers,    // 高斯缓存
                                per_tile_buffers,         // 瓦片缓存
                                per_instance_buffers,     // 实例缓存
                                per_bucket_buffers,       // 深度桶缓存
                                densification_info});     // 密集化信息

        // 标记不需要梯度的张量，提高反向传播效率
        ctx->mark_non_differentiable({per_primitive_buffers,    // 缓存数据不需要梯度
                                      per_tile_buffers,         // 缓存数据不需要梯度
                                      per_instance_buffers,     // 缓存数据不需要梯度
                                      per_bucket_buffers,       // 缓存数据不需要梯度
                                      densification_info});     // 密集化信息不需要梯度

        // =============================================================================
        // 保存反向传播需要的标量和配置数据
        // =============================================================================
        // 使用saved_data字典保存非张量类型的数据
        ctx->saved_data["w2c"] = settings.w2c;                                      // 变换矩阵
        ctx->saved_data["cam_position"] = settings.cam_position;                    // 相机位置
        ctx->saved_data["active_sh_bases"] = settings.active_sh_bases;              // 球谐基数量
        ctx->saved_data["width"] = settings.width;                                  // 图像宽度
        ctx->saved_data["height"] = settings.height;                                // 图像高度
        ctx->saved_data["focal_x"] = settings.focal_x;                              // X轴焦距
        ctx->saved_data["focal_y"] = settings.focal_y;                              // Y轴焦距
        ctx->saved_data["center_x"] = settings.center_x;                            // 主点X坐标
        ctx->saved_data["center_y"] = settings.center_y;                            // 主点Y坐标
        ctx->saved_data["near_plane"] = settings.near_plane;                        // 近裁剪平面
        ctx->saved_data["far_plane"] = settings.far_plane;                          // 远裁剪平面
        ctx->saved_data["n_visible_primitives"] = n_visible_primitives;             // 可见高斯数量
        ctx->saved_data["n_instances"] = n_instances;                               // 实例数量
        ctx->saved_data["n_buckets"] = n_buckets;                                   // 深度桶数量
        ctx->saved_data["primitive_primitive_indices_selector"] = primitive_primitive_indices_selector; // 高斯索引选择器
        ctx->saved_data["instance_primitive_indices_selector"] = instance_primitive_indices_selector;   // 实例索引选择器

        // 返回前向传播的主要输出：图像和Alpha通道
        return {image, alpha};
    }

    /**
     * [功能描述]：FastGS光栅化反向传播函数实现
     * @param ctx：自动微分上下文，包含前向传播保存的变量和数据
     * @param grad_outputs：输出张量的梯度，来自损失函数的反向传播
     * @return 输入张量的梯度列表，顺序对应forward函数的输入参数
     * 
     * 该函数计算渲染损失对所有高斯参数的梯度，使得整个渲染过程
     * 可以参与端到端的梯度下降优化
     */
    torch::autograd::tensor_list FastGSRasterize::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {

        // =============================================================================
        // 提取输出梯度
        // =============================================================================
        auto grad_image = grad_outputs[0];  // 图像梯度 [3, H, W]
        auto grad_alpha = grad_outputs[1];  // Alpha通道梯度 [1, H, W]

        // =============================================================================
        // 恢复前向传播保存的张量变量
        // =============================================================================
        auto saved = ctx->get_saved_variables();
        const torch::Tensor& image = saved[0];                    // 前向传播的图像输出
        const torch::Tensor& alpha = saved[1];                    // 前向传播的Alpha输出
        const torch::Tensor& means = saved[2];                    // 高斯位置（输入）
        const torch::Tensor& scales_raw = saved[3];               // 缩放参数（输入）
        const torch::Tensor& rotations_raw = saved[4];            // 旋转参数（输入）
        const torch::Tensor& sh_coefficients_rest = saved[5];     // 球谐高阶系数（输入）
        const torch::Tensor& per_primitive_buffers = saved[6];    // 高斯缓存数据
        const torch::Tensor& per_tile_buffers = saved[7];         // 瓦片缓存数据
        const torch::Tensor& per_instance_buffers = saved[8];     // 实例缓存数据
        const torch::Tensor& per_bucket_buffers = saved[9];       // 深度桶缓存数据
        torch::Tensor densification_info = saved[10];             // 密集化信息
        // FIXME注释：由于libtorch的奇异行为，这可能不是原始张量，但在MCMC中不需要

        // =============================================================================
        // 调用底层C++/CUDA反向传播包装函数
        // =============================================================================
        // 该函数使用前向传播的中间结果计算所有参数的梯度
        auto outputs = fast_gs::rasterization::backward_wrapper(
            densification_info,                                                      // 密集化信息
            grad_image,                                                              // 图像梯度（输入）
            grad_alpha,                                                              // Alpha梯度（输入）
            image,                                                                   // 前向传播图像
            alpha,                                                                   // 前向传播Alpha
            means,                                                                   // 高斯位置
            scales_raw,                                                              // 缩放参数
            rotations_raw,                                                           // 旋转参数
            sh_coefficients_rest,                                                    // 球谐高阶系数
            per_primitive_buffers,                                                   // 高斯缓存
            per_tile_buffers,                                                        // 瓦片缓存
            per_instance_buffers,                                                    // 实例缓存
            per_bucket_buffers,                                                      // 深度桶缓存
            // 从上下文恢复的配置参数
            ctx->saved_data["w2c"].toTensor(),                                       // 变换矩阵
            ctx->saved_data["cam_position"].toTensor(),                              // 相机位置
            ctx->saved_data["active_sh_bases"].toInt(),                              // 球谐基数量
            ctx->saved_data["width"].toInt(),                                        // 图像宽度
            ctx->saved_data["height"].toInt(),                                       // 图像高度
            static_cast<float>(ctx->saved_data["focal_x"].toDouble()),               // X轴焦距
            static_cast<float>(ctx->saved_data["focal_y"].toDouble()),               // Y轴焦距
            static_cast<float>(ctx->saved_data["center_x"].toDouble()),              // 主点X坐标
            static_cast<float>(ctx->saved_data["center_y"].toDouble()),              // 主点Y坐标
            static_cast<float>(ctx->saved_data["near_plane"].toDouble()),            // 近裁剪平面
            static_cast<float>(ctx->saved_data["far_plane"].toDouble()),             // 远裁剪平面
            ctx->saved_data["n_visible_primitives"].toInt(),                         // 可见高斯数量
            ctx->saved_data["n_instances"].toInt(),                                  // 实例数量
            ctx->saved_data["n_buckets"].toInt(),                                    // 深度桶数量
            ctx->saved_data["primitive_primitive_indices_selector"].toInt(),         // 高斯索引选择器
            ctx->saved_data["instance_primitive_indices_selector"].toInt());         // 实例索引选择器

        // =============================================================================
        // 解构反向传播的输出梯度
        // =============================================================================
        auto grad_means = std::get<0>(outputs);                    // 位置参数的梯度 [N, 3]
        auto grad_scales_raw = std::get<1>(outputs);               // 缩放参数的梯度 [N, 3]
        auto grad_rotations_raw = std::get<2>(outputs);            // 旋转参数的梯度 [N, 4]
        auto grad_opacities_raw = std::get<3>(outputs);            // 不透明度的梯度 [N, 1]
        auto grad_sh_coefficients_0 = std::get<4>(outputs);        // 球谐0阶系数的梯度 [N, 1, 3]
        auto grad_sh_coefficients_rest = std::get<5>(outputs);     // 球谐高阶系数的梯度 [C, B-1, 3]

        // =============================================================================
        // 返回所有输入参数的梯度
        // =============================================================================
        // 梯度的顺序必须与forward函数的参数顺序完全一致
        return {
            grad_means,                 // 高斯位置的梯度
            grad_scales_raw,            // 缩放参数的梯度
            grad_rotations_raw,         // 旋转参数的梯度
            grad_opacities_raw,         // 不透明度的梯度
            grad_sh_coefficients_0,     // 球谐0阶系数的梯度
            grad_sh_coefficients_rest,  // 球谐高阶系数的梯度
            torch::Tensor(),            // densification_info无梯度（返回空张量）
            torch::Tensor(),            // settings无梯度（返回空张量）
        };
    }

} // namespace gs - 高斯散点项目命名空间结束