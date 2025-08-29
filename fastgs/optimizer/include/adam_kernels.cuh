#pragma once

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace fast_gs::optimizer::kernels::adam {

    // 基于 https://github.com/pytorch/pytorch/blob/9d32aa9789fc0ef0cad01a788157ecc2121db810/torch/csrc/api/src/optim/adam.cpp#L72-L142
    /**
     * [功能描述]：Adam优化器的CUDA内核函数，并行更新参数、一阶矩和二阶矩估计
     * @param param 输入输出：需要更新的参数数组
     * @param exp_avg 输入输出：一阶矩估计（动量）数组
     * @param exp_avg_sq 输入输出：二阶矩估计（方差）数组
     * @param param_grad 输入：参数的梯度数组
     * @param n_elements 需要处理的元素总数
     * @param lr 学习率
     * @param beta1 一阶矩估计的指数衰减率
     * @param beta2 二阶矩估计的指数衰减率
     * @param eps 数值稳定性常数，防止除零错误
     * @param bias_correction1_rcp 一阶矩偏差校正的倒数
     * @param bias_correction2_sqrt_rcp 二阶矩偏差校正平方根的倒数
     */
    __global__ void adam_step_cu(
        float* param,
        float* exp_avg,
        float* exp_avg_sq,
        const float* param_grad,
        const int n_elements,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp) {
        
        // 获取当前线程在网格中的全局索引
        auto idx = cg::this_grid().thread_rank();
        
        // 边界检查：如果线程索引超出元素数量，则直接返回
        if (idx >= n_elements)
            return;
        
        // 获取当前线程对应的梯度值
        const float grad = param_grad[idx];
        
        // 计算一阶矩估计（动量）
        // moment1 = β1 * exp_avg[idx] + (1 - β1) * grad
        // 这是梯度的指数移动平均，用于估计梯度的方向
        const float moment1 = beta1 * exp_avg[idx] + (1.0f - beta1) * grad;
        
        // 计算二阶矩估计（方差）
        // moment2 = β2 * exp_avg_sq[idx] + (1 - β2) * grad²
        // 这是梯度平方的指数移动平均，用于估计梯度的幅度
        const float moment2 = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * grad * grad;
        
        // 计算分母，包含偏差校正和数值稳定性
        // denom = sqrt(moment2) * bias_correction2_sqrt_rcp + eps
        // 这里使用偏差校正来修正初始偏差，eps防止除零
        const float denom = sqrtf(moment2) * bias_correction2_sqrt_rcp + eps;
        
        // 计算步长，包含学习率和一阶矩偏差校正
        // step_size = lr * bias_correction1_rcp
        const float step_size = lr * bias_correction1_rcp;
        
        // 更新参数值
        // param[idx] -= step_size * moment1 / denom
        // 这是Adam算法的核心更新公式
        param[idx] -= step_size * moment1 / denom;
        
        // 更新一阶矩估计（动量）
        exp_avg[idx] = moment1;
        
        // 更新二阶矩估计（方差）
        exp_avg_sq[idx] = moment2;
    }

} // namespace fast_gs::optimizer::kernels::adam
