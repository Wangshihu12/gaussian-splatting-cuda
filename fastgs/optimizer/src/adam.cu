#include "adam.h"
#include "adam_kernels.cuh"
#include "optimizer_config.h"
#include "utils.h"

/**
 * [功能描述]：Adam优化器的CUDA主机函数，配置并启动CUDA内核来执行Adam优化步骤
 * @param param 输入输出：需要更新的参数数组指针
 * @param exp_avg 输入输出：一阶矩估计（动量）数组指针
 * @param exp_avg_sq 输入输出：二阶矩估计（方差）数组指针
 * @param param_grad 输入：参数的梯度数组指针
 * @param n_elements 需要处理的元素总数
 * @param lr 学习率
 * @param beta1 一阶矩估计的指数衰减率
 * @param beta2 二阶矩估计的指数衰减率
 * @param eps 数值稳定性常数，防止除零错误
 * @param bias_correction1_rcp 一阶矩偏差校正的倒数
 * @param bias_correction2_sqrt_rcp 二阶矩偏差校正平方根的倒数
 */
void fast_gs::optimizer::adam_step(
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
    
    // 启动CUDA内核执行Adam优化步骤
    // 使用三重尖括号语法 <<<grid_size, block_size>>>
    kernels::adam::adam_step_cu<<<div_round_up(n_elements, config::block_size_adam_step), config::block_size_adam_step>>>(
        param,                              // 参数数组指针
        exp_avg,                            // 一阶矩估计数组指针
        exp_avg_sq,                         // 二阶矩估计数组指针
        param_grad,                         // 梯度数组指针
        n_elements,                         // 元素总数
        lr,                                 // 学习率
        beta1,                              // 一阶矩衰减率
        beta2,                              // 二阶矩衰减率
        eps,                                // 数值稳定性常数
        bias_correction1_rcp,               // 一阶矩偏差校正倒数
        bias_correction2_sqrt_rcp);         // 二阶矩偏差校正平方根倒数
    
    // 检查CUDA操作是否成功（仅在调试模式下）
    // 如果启用调试模式，会检查内核执行是否出现错误
    CHECK_CUDA(config::debug, "adam step")
}
