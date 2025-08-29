#include "adam.h"
#include "adam_api.h"

/**
 * [功能描述]：Adam优化器的PyTorch包装函数，将PyTorch张量转换为原始指针并调用CUDA内核
 * @param param 输入输出：需要更新的参数张量
 * @param exp_avg 输入输出：一阶矩估计（动量）张量
 * @param exp_avg_sq 输入输出：二阶矩估计（方差）张量
 * @param param_grad 输入：参数的梯度张量
 * @param lr 学习率
 * @param beta1 一阶矩估计的指数衰减率
 * @param beta2 二阶矩估计的指数衰减率
 * @param eps 数值稳定性常数，防止除零错误
 * @param bias_correction1_rcp 一阶矩偏差校正的倒数
 * @param bias_correction2_sqrt_rcp 二阶矩偏差校正平方根的倒数
 */
void fast_gs::optimizer::adam_step_wrapper(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& param_grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1_rcp,
    const float bias_correction2_sqrt_rcp) {
    
    // 获取参数张量中元素的总数
    const int n_elements = param.numel();

    // 调用CUDA内核函数执行Adam优化步骤
    // 将PyTorch张量转换为原始浮点指针，以便CUDA内核处理
    adam_step(
        param.data_ptr<float>(),           // 参数张量的原始浮点指针
        exp_avg.data_ptr<float>(),         // 一阶矩估计张量的原始浮点指针
        exp_avg_sq.data_ptr<float>(),      // 二阶矩估计张量的原始浮点指针
        param_grad.data_ptr<float>(),      // 梯度张量的原始浮点指针
        n_elements,                        // 元素总数
        lr,                                // 学习率
        beta1,                             // 一阶矩衰减率
        beta2,                             // 二阶矩衰减率
        eps,                               // 数值稳定性常数
        bias_correction1_rcp,              // 一阶矩偏差校正倒数
        bias_correction2_sqrt_rcp);        // 二阶矩偏差校正平方根倒数
}
