#pragma once

#include "rasterization_api.h"
#include <torch/torch.h>

/**
 * [文件描述]：快速高斯散点光栅化自动微分头文件
 * 功能：定义FastGS光栅化的PyTorch自动微分函数类
 * 用途：为高斯散点渲染提供可微分的GPU加速实现，支持训练时的梯度反向传播
 */

namespace gs {

    /**
     * [类描述]：快速高斯散点光栅化自动微分函数类
     * 
     * 继承自torch::autograd::Function，实现自定义的PyTorch算子
     * 该类封装了高斯散点的前向渲染和反向传播过程，使得整个渲染管道
     * 可以无缝集成到PyTorch的自动微分系统中，支持端到端的训练
     * 
     * 主要功能：
     * - 前向传播：将3D高斯散点渲染为2D图像
     * - 反向传播：计算渲染损失对所有高斯参数的梯度
     * - GPU加速：使用CUDA内核实现高性能渲染
     * - 内存优化：高效的GPU内存管理和数据传输
     */
    class FastGSRasterize : public torch::autograd::Function<FastGSRasterize> {
    public:
        /**
         * [功能描述]：前向传播函数，执行高斯散点到图像的渲染过程
         * @param ctx：自动微分上下文，用于保存反向传播需要的中间结果
         * @param means：高斯中心位置张量 [N, 3]，N为高斯数量，每个高斯的3D坐标
         * @param scales_raw：原始缩放参数张量 [N, 3]，控制高斯在各轴向的尺寸
         * @param rotations_raw：原始旋转四元数张量 [N, 4]，控制高斯的3D朝向
         * @param opacities_raw：原始不透明度张量 [N, 1]，控制高斯的透明程度
         * @param sh_coefficients_0：球谐函数0阶系数 [N, 1, 3]，直流颜色分量（RGB）
         * @param sh_coefficients_rest：球谐函数高阶系数 [C, B-1, 3]，方向性颜色变化
         *                             其中C为高斯数量，B为球谐基函数总数
         * @param densification_info：密集化信息张量 [2, N] 或空张量，用于动态高斯管理
         * @param settings：光栅化设置结构体，包含相机参数、渲染配置等
         * @return 渲染输出张量列表，包含RGB图像和Alpha通道等
         * 
         * 前向传播流程：
         * 1. 变换高斯参数（缩放、旋转、不透明度激活）
         * 2. 世界坐标到相机坐标变换
         * 3. 3D到2D投影和视锥体裁剪
         * 4. 球谐函数颜色计算
         * 5. 深度排序和Alpha混合
         * 6. 最终图像合成
         */
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            const torch::Tensor& means,                              // [N, 3] 高斯中心位置
            const torch::Tensor& scales_raw,                         // [N, 3] 原始缩放参数
            const torch::Tensor& rotations_raw,                      // [N, 4] 原始旋转四元数
            const torch::Tensor& opacities_raw,                      // [N, 1] 原始不透明度
            const torch::Tensor& sh_coefficients_0,                  // [N, 1, 3] 球谐0阶系数
            const torch::Tensor& sh_coefficients_rest,               // [C, B-1, 3] 球谐高阶系数
            torch::Tensor& densification_info,                       // [2, N] 密集化信息或空张量
            const fast_gs::rasterization::FastGSSettings& settings); // 光栅化配置

        /**
         * [功能描述]：反向传播函数，计算输出损失对所有输入参数的梯度
         * @param ctx：自动微分上下文，包含前向传播保存的中间变量和状态
         * @param grad_outputs：输出张量的梯度列表，来自后续层的反向传播
         * @return 输入张量的梯度列表，顺序对应forward函数的输入参数
         * 
         * 反向传播流程：
         * 1. 从上下文恢复前向传播的中间结果
         * 2. 根据输出梯度计算像素级别的贡献
         * 3. 反向追踪每个高斯对像素的影响
         * 4. 计算颜色、几何参数的梯度
         * 5. 聚合所有像素的梯度贡献
         * 6. 返回对应输入参数的梯度张量
         * 
         * 梯度计算涵盖：
         * - 位置梯度：dL/d_means
         * - 缩放梯度：dL/d_scales
         * - 旋转梯度：dL/d_rotations  
         * - 不透明度梯度：dL/d_opacities
         * - 球谐系数梯度：dL/d_sh_coefficients
         */
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };

} // namespace gs - 高斯散点项目命名空间结束