#pragma once

#include "rasterization_api.h"  // 光栅化API头文件，提供快速高斯溅射光栅化的底层接口
#include <torch/torch.h>         // PyTorch核心库，提供自动微分和神经网络功能

namespace gs::training {
    /**
     * [功能描述]：快速高斯溅射光栅化的自动微分函数类。
     * 这个类继承自PyTorch的autograd::Function，实现了快速高斯溅射光栅化的前向和反向传播。
     * 它允许PyTorch自动微分系统自动计算梯度，使得高斯溅射模型可以进行端到端的训练。
     * 
     * 主要特点：
     * 1. 支持批量处理多个相机视角
     * 2. 集成球谐函数用于颜色表示
     * 3. 支持密集化信息用于动态调整高斯数量
     * 4. 提供完整的梯度计算支持
     */
    class FastGSRasterize : public torch::autograd::Function<FastGSRasterize> {
    public:
        /**
         * [功能描述]：前向传播函数，执行快速高斯溅射光栅化的前向计算。
         * 这个函数将3D高斯模型投影到2D图像平面，生成渲染图像和透明度通道。
         * 同时，它会保存必要的信息用于后续的反向传播计算。
         * 
         * @param ctx [参数说明]：PyTorch自动微分上下文，用于管理前向和反向传播的状态信息。
         * @param means [参数说明]：高斯中心位置张量，形状为[N, 3]，N是高斯数量，3表示xyz坐标。
         * @param scales_raw [参数说明]：原始缩放参数张量，形状为[N, 3]，表示每个高斯在xyz方向上的缩放。
         * @param rotations_raw [参数说明]：原始旋转参数张量，形状为[N, 4]，使用四元数表示旋转。
         * @param opacities_raw [参数说明]：原始不透明度张量，形状为[N, 1]，表示每个高斯的透明度。
         * @param sh_coefficients_0 [参数说明]：球谐函数0阶系数，形状为[N, 1, 3]，表示每个高斯的基础颜色。
         * @param sh_coefficients_rest [参数说明]：球谐函数高阶系数，形状为[C, B-1, 3]。
         *                                     C是相机数量，B是球谐函数阶数，3表示RGB通道。
         * @param w2c [参数说明]：世界坐标系到相机坐标系的变换矩阵，形状为[C, 4, 4]。
         * @param densification_info [参数说明]：密集化信息张量，形状为[2, N]或空张量。
         *                                用于动态调整高斯数量，支持训练过程中的自适应优化。
         * @param settings [参数说明]：光栅化设置，包含图像尺寸、裁剪平面、相机类型等配置参数。
         * @return [返回值说明]：返回张量列表，包含渲染的图像和透明度通道。
         */
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            const torch::Tensor& means,                              // [N, 3]
            const torch::Tensor& scales_raw,                         // [N, 3]
            const torch::Tensor& rotations_raw,                      // [N, 4]
            const torch::Tensor& opacities_raw,                      // [N, 1]
            const torch::Tensor& sh_coefficients_0,                  // [N, 1, 3]
            const torch::Tensor& sh_coefficients_rest,               // [C, B-1, 3]
            const torch::Tensor& w2c,                                // [C, 4, 4]
            torch::Tensor& densification_info,                       // [2, N] or empty tensor
            const fast_gs::rasterization::FastGSSettings& settings); // rasterizer settings

        /**
         * [功能描述]：反向传播函数，计算各个输入参数的梯度。
         * 这个函数使用链式法则计算损失函数对各个输入参数的梯度，
         * 使得高斯溅射模型可以通过梯度下降进行优化。
         * 
         * @param ctx [参数说明]：PyTorch自动微分上下文，包含前向传播中保存的信息。
         * @param grad_outputs [参数说明]：输出梯度张量列表，包含图像和透明度通道的梯度。
         * @return [返回值说明]：返回各个输入参数的梯度张量列表，顺序与forward函数的参数顺序一致。
         *                     包括：means梯度、scales_raw梯度、rotations_raw梯度、opacities_raw梯度、
         *                     球谐函数系数梯度、w2c梯度等。
         */
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };
} // namespace gs::training
