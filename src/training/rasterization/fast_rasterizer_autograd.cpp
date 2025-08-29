#include "fast_rasterizer_autograd.hpp"

namespace gs::training {
    /**
     * [功能描述]：FastGSRasterize的前向传播函数，实现快速高斯溅射光栅化的前向计算。
     * 这是PyTorch自动微分系统的核心函数，负责光栅化计算和梯度信息的保存。
     * @param ctx [参数说明]：PyTorch自动微分上下文，用于管理前向和反向传播的状态。
     * @param means [参数说明]：高斯中心位置张量，形状为[N, 3]，N是高斯数量。
     * @param scales_raw [参数说明]：原始缩放参数张量，形状为[N, 3]。
     * @param rotations_raw [参数说明]：原始旋转参数张量，形状为[N, 4]，使用四元数表示。
     * @param opacities_raw [参数说明]：原始不透明度张量，形状为[N, 1]。
     * @param sh_coefficients_0 [参数说明]：球谐函数0阶系数，形状为[N, 1, 3]。
     * @param sh_coefficients_rest [参数说明]：球谐函数高阶系数，形状为[C, B-1, 3]，C是相机数量，B是球谐函数阶数。
     * @param w2c [参数说明]：世界坐标系到相机坐标系的变换矩阵，形状为[C, 4, 4]。
     * @param densification_info [参数说明]：密集化信息张量，形状为[2, N]或空张量，用于动态调整高斯数量。
     * @param settings [参数说明]：光栅化设置，包含相机参数和渲染配置。
     * @return [返回值说明]：返回包含渲染图像和透明度通道的张量列表。
     */
    torch::autograd::tensor_list FastGSRasterize::forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& means,                // [N, 3]
        const torch::Tensor& scales_raw,           // [N, 3]
        const torch::Tensor& rotations_raw,        // [N, 4]
        const torch::Tensor& opacities_raw,        // [N, 1]
        const torch::Tensor& sh_coefficients_0,    // [N, 1, 3]
        const torch::Tensor& sh_coefficients_rest, // [C, B-1, 3]
        const torch::Tensor& w2c,                  // [C, 4, 4]
        torch::Tensor& densification_info,         // [2, N] or empty tensor
        const fast_gs::rasterization::FastGSSettings& settings) {
        
        // 步骤1：调用底层光栅化包装函数进行前向计算
        auto outputs = fast_gs::rasterization::forward_wrapper(
            means,                    // 高斯中心位置
            scales_raw,               // 缩放参数
            rotations_raw,            // 旋转参数
            opacities_raw,            // 不透明度
            sh_coefficients_0,        // 球谐函数0阶系数
            sh_coefficients_rest,     // 球谐函数高阶系数
            w2c,                      // 世界到相机的变换矩阵
            settings.cam_position,    // 相机位置
            settings.active_sh_bases, // 活跃的球谐函数基函数数量
            settings.width,           // 图像宽度
            settings.height,          // 图像高度
            settings.focal_x,         // x方向焦距
            settings.focal_y,         // y方向焦距
            settings.center_x,        // x方向主点坐标
            settings.center_y,        // y方向主点坐标
            settings.near_plane,      // 近裁剪平面
            settings.far_plane);      // 远裁剪平面

        // 步骤2：从输出元组中提取各个组件
        auto image = std::get<0>(outputs);                    // 渲染的图像 [H, W, 3]
        auto alpha = std::get<1>(outputs);                    // 透明度通道 [H, W, 1]
        auto per_primitive_buffers = std::get<2>(outputs);    // 每个图元的缓冲区信息
        auto per_tile_buffers = std::get<3>(outputs);         // 每个瓦片的缓冲区信息
        auto per_instance_buffers = std::get<4>(outputs);     // 每个实例的缓冲区信息
        auto per_bucket_buffers = std::get<5>(outputs);       // 每个桶的缓冲区信息
        int n_visible_primitives = std::get<6>(outputs);      // 可见图元数量
        int n_instances = std::get<7>(outputs);               // 实例数量
        int n_buckets = std::get<8>(outputs);                 // 桶数量
        int primitive_primitive_indices_selector = std::get<9>(outputs);      // 图元索引选择器
        int instance_primitive_indices_selector = std::get<10>(outputs);     // 实例索引选择器

        // 步骤3：标记不可微分的张量，这些张量不会参与梯度计算
        ctx->mark_non_differentiable({per_primitive_buffers,
                                      per_tile_buffers,
                                      per_instance_buffers,
                                      per_bucket_buffers,
                                      densification_info});

        // 步骤4：保存前向传播中的张量，供反向传播使用
        ctx->save_for_backward({image,                    // 渲染图像
                                alpha,                    // 透明度通道
                                means,                    // 高斯中心位置
                                scales_raw,               // 缩放参数
                                rotations_raw,            // 旋转参数
                                sh_coefficients_rest,     // 球谐函数高阶系数
                                per_primitive_buffers,    // 图元缓冲区
                                per_tile_buffers,         // 瓦片缓冲区
                                per_instance_buffers,     // 实例缓冲区
                                per_bucket_buffers,       // 桶缓冲区
                                w2c,                      // 世界到相机的变换矩阵
                                densification_info});     // 密集化信息

        // 步骤5：保存非张量数据到上下文中，供反向传播使用
        ctx->saved_data["cam_position"] = settings.cam_position;                    // 相机位置
        ctx->saved_data["active_sh_bases"] = settings.active_sh_bases;             // 活跃球谐函数基函数数量
        ctx->saved_data["width"] = settings.width;                                 // 图像宽度
        ctx->saved_data["height"] = settings.height;                               // 图像高度
        ctx->saved_data["focal_x"] = settings.focal_x;                             // x方向焦距
        ctx->saved_data["focal_y"] = settings.focal_y;                             // y方向焦距
        ctx->saved_data["center_x"] = settings.center_x;                           // x方向主点坐标
        ctx->saved_data["center_y"] = settings.center_y;                           // y方向主点坐标
        ctx->saved_data["near_plane"] = settings.near_plane;                       // 近裁剪平面
        ctx->saved_data["far_plane"] = settings.far_plane;                         // 远裁剪平面
        ctx->saved_data["n_visible_primitives"] = n_visible_primitives;            // 可见图元数量
        ctx->saved_data["n_instances"] = n_instances;                              // 实例数量
        ctx->saved_data["n_buckets"] = n_buckets;                                  // 桶数量
        ctx->saved_data["primitive_primitive_indices_selector"] = primitive_primitive_indices_selector;  // 图元索引选择器
        ctx->saved_data["instance_primitive_indices_selector"] = instance_primitive_indices_selector;   // 实例索引选择器

        // 返回渲染结果：图像和透明度通道
        return {image, alpha};
    }

    /**
     * [功能描述]：FastGSRasterize的反向传播函数，计算各个输入参数的梯度。
     * 这是PyTorch自动微分系统的核心函数，负责梯度计算和传播。
     * @param ctx [参数说明]：PyTorch自动微分上下文，包含前向传播中保存的信息。
     * @param grad_outputs [参数说明]：输出梯度张量列表，包含图像和透明度通道的梯度。
     * @return [返回值说明]：返回各个输入参数的梯度张量列表。
     */
    torch::autograd::tensor_list FastGSRasterize::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        
        // 步骤1：提取输出梯度
        auto grad_image = grad_outputs[0];    // 图像梯度
        auto grad_alpha = grad_outputs[1];    // 透明度梯度

        // 步骤2：从上下文中恢复前向传播中保存的张量
        auto saved = ctx->get_saved_variables();
        const torch::Tensor& image = saved[0];                    // 渲染图像
        const torch::Tensor& alpha = saved[1];                    // 透明度通道
        const torch::Tensor& means = saved[2];                    // 高斯中心位置
        const torch::Tensor& scales_raw = saved[3];               // 缩放参数
        const torch::Tensor& rotations_raw = saved[4];            // 旋转参数
        const torch::Tensor& sh_coefficients_rest = saved[5];     // 球谐函数高阶系数
        const torch::Tensor& per_primitive_buffers = saved[6];    // 图元缓冲区
        const torch::Tensor& per_tile_buffers = saved[7];         // 瓦片缓冲区
        const torch::Tensor& per_instance_buffers = saved[8];     // 实例缓冲区
        const torch::Tensor& per_bucket_buffers = saved[9];       // 桶缓冲区
        const torch::Tensor& w2c = saved[10];                     // 世界到相机的变换矩阵
        torch::Tensor& densification_info = saved[11];            // 密集化信息

        // 步骤3：调用底层反向传播包装函数计算梯度
        auto outputs = fast_gs::rasterization::backward_wrapper(
            densification_info,       // 密集化信息
            grad_image,               // 图像梯度
            grad_alpha,               // 透明度梯度
            image,                    // 前向传播的图像
            alpha,                    // 前向传播的透明度
            means,                    // 高斯中心位置
            scales_raw,               // 缩放参数
            rotations_raw,            // 旋转参数
            sh_coefficients_rest,     // 球谐函数高阶系数
            per_primitive_buffers,    // 图元缓冲区
            per_tile_buffers,         // 瓦片缓冲区
            per_instance_buffers,     // 实例缓冲区
            per_bucket_buffers,       // 桶缓冲区
            w2c,                      // 世界到相机的变换矩阵
            ctx->saved_data["cam_position"].toTensor(),           // 相机位置
            ctx->saved_data["active_sh_bases"].toInt(),           // 活跃球谐函数基函数数量
            ctx->saved_data["width"].toInt(),                     // 图像宽度
            ctx->saved_data["height"].toInt(),                    // 图像高度
            static_cast<float>(ctx->saved_data["focal_x"].toDouble()),      // x方向焦距
            static_cast<float>(ctx->saved_data["focal_y"].toDouble()),      // y方向焦距
            static_cast<float>(ctx->saved_data["center_x"].toDouble()),     // x方向主点坐标
            static_cast<float>(ctx->saved_data["center_y"].toDouble()),     // y方向主点坐标
            static_cast<float>(ctx->saved_data["near_plane"].toDouble()),   // 近裁剪平面
            static_cast<float>(ctx->saved_data["far_plane"].toDouble()),    // 远裁剪平面
            ctx->saved_data["n_visible_primitives"].toInt(),               // 可见图元数量
            ctx->saved_data["n_instances"].toInt(),                        // 实例数量
            ctx->saved_data["n_buckets"].toInt(),                          // 桶数量
            ctx->saved_data["primitive_primitive_indices_selector"].toInt(), // 图元索引选择器
            ctx->saved_data["instance_primitive_indices_selector"].toInt()); // 实例索引选择器

        // 步骤4：从输出元组中提取各个参数的梯度
        auto grad_means = std::get<0>(outputs);              // 高斯中心位置的梯度
        auto grad_scales_raw = std::get<1>(outputs);         // 缩放参数的梯度
        auto grad_rotations_raw = std::get<2>(outputs);      // 旋转参数的梯度
        auto grad_opacities_raw = std::get<3>(outputs);      // 不透明度的梯度
        auto grad_sh_coefficients_0 = std::get<4>(outputs);  // 球谐函数0阶系数的梯度
        auto grad_sh_coefficients_rest = std::get<5>(outputs); // 球谐函数高阶系数的梯度
        auto grad_w2c = std::get<6>(outputs);                // 世界到相机变换矩阵的梯度

        // 步骤5：返回所有参数的梯度，对于不需要梯度的参数返回空张量
        return {
            grad_means,               // 高斯中心位置梯度
            grad_scales_raw,          // 缩放参数梯度
            grad_rotations_raw,       // 旋转参数梯度
            grad_opacities_raw,       // 不透明度梯度
            grad_sh_coefficients_0,   // 球谐函数0阶系数梯度
            grad_sh_coefficients_rest, // 球谐函数高阶系数梯度
            grad_w2c,                 // 世界到相机变换矩阵梯度
            torch::Tensor(),          // densification_info（无梯度）
            torch::Tensor(),          // settings（无梯度）
        };
    }
} // namespace gs::training
