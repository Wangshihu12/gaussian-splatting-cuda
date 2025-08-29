#include "fused_adam.hpp"
#include "adam_api.h"

// TODO: 这只是为了奖金的一个噱头。我认为它不应该集成到主代码库中。
// TODO: 移除SH步骤跳步也意味着自定义的zero_grad()方法不再需要。
// TODO: 所有跳步条件都假设迭代计数从1开始（目前确实如此）。
// 在迭代1000到25000之间，我们可以跳过高阶SH系数的每第二个步骤。
// 这确实将奖金基准测试推到了20分钟以下，但我真的不喜欢它的实际影响。
// 它还会导致质量指标和鲁棒性的*非常*小的下降。因此，我默认禁用它。
#define SKIP_SH_STEPS false

namespace gs::training {
    /**
     * [功能描述]：FusedAdam的闭包版本step函数实现。
     * 这个函数是为了满足PyTorch优化器接口要求而实现的，但FusedAdam不支持闭包。
     * 直接抛出错误，因为FusedAdam使用迭代次数版本的step函数。
     * 
     * @param closure [参数说明]：损失计算闭包（未使用）。
     * @return [返回值说明]：返回空张量（实际上永远不会执行到这里）。
     */
    torch::Tensor FusedAdam::step(LossClosure closure) {
        TORCH_CHECK(false, "FusedAdam does not support closures.");  // 检查失败，FusedAdam不支持闭包
        return {};  // 返回空张量（永远不会执行）
    }

    /**
     * [功能描述]：FusedAdam的核心优化步骤函数实现。
     * 这个函数实现了完整的Adam优化算法，包括状态管理、偏差校正和CUDA内核调用。
     * 支持参数组级别的配置，并实现了SH系数的特殊优化策略。
     * 
     * @param iteration [参数说明]：当前迭代次数，用于控制SH系数的更新策略。
     */
    void FusedAdam::step(int iteration) {
        // 禁用梯度计算，因为这是优化器内部操作
        torch::NoGradGuard no_grad;

        // 步骤1：获取全局优化器选项
        const auto& global_options = options();

        int i = 0; // 技巧：计数器，用于跟踪我们正在处理哪个高斯参数
        // 遍历所有参数组
        for (auto& group : param_groups()) {
            ++i;  // 递增参数组计数器

            // 步骤2：为每个参数组检查是否有特定选项
            double lr = global_options.lr();           // 学习率
            double eps = global_options.eps();         // epsilon值
            auto [beta1, beta2] = global_options.betas();  // beta1和beta2参数

            // 如果参数组有自己的选项，使用那些选项
            if (group.has_options()) {
                if (auto* group_opts = dynamic_cast<const Options*>(&group.options())) {
                    lr = group_opts->lr();                    // 使用组特定的学习率
                    eps = group_opts->eps();                  // 使用组特定的epsilon
                    std::tie(beta1, beta2) = group_opts->betas();  // 使用组特定的beta参数
                }
            }

            // 步骤3：遍历参数组中的每个参数
            for (auto& param : group.params()) {
                // 检查参数是否有梯度，如果没有则跳过
                if (!param.grad().defined()) {
                    continue;
                }

                // 步骤4：延迟状态初始化
                auto state_ptr = state_.find(param.unsafeGetTensorImpl());
                if (state_ptr == state_.end()) {
                    // 创建新的Adam状态
                    auto new_state = std::make_unique<AdamParamState>();
                    new_state->step_count = 0;  // 初始化步数计数
                    // 创建与参数形状相同的零张量，保持内存格式
                    new_state->exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                    new_state->exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);

                    // 将新状态添加到状态映射中
                    state_[param.unsafeGetTensorImpl()] = std::move(new_state);
                    state_ptr = state_.find(param.unsafeGetTensorImpl());
                }

                // 获取参数状态引用
                auto& state = static_cast<AdamParamState&>(*state_ptr->second);

                // 步骤5：递增步数计数
                state.step_count++;

                // 步骤6：SH系数的特殊优化策略
                // 高阶SH系数在前1000次迭代中不使用，所以这是一个免费的加速
                if (i == 3 && iteration <= 1000)
                    continue;  // 跳过前1000次迭代中的SH系数更新

                // 步骤7：可选的SH步骤跳步机制（默认禁用）
                if constexpr (SKIP_SH_STEPS) {
                    // 在训练期间跳过每第二个步骤，除了最后5000次迭代
                    if (i == 3 && (iteration % 2 != 0 && iteration <= 25000))
                        continue;  // 跳过奇数迭代（除了最后5000次）
                }

                // 步骤8：计算偏差校正因子
                // bias_correction1 = 1 / (1 - beta1^step_count)
                auto bias_correction1_rcp = 1.0 / (1.0 - std::pow(beta1, state.step_count));
                // bias_correction2_sqrt = 1 / sqrt(1 - beta2^step_count)
                auto bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(beta2, state.step_count));

                // 步骤9：调用fastgs的融合CUDA内核
                fast_gs::optimizer::adam_step_wrapper(
                    param,                           // 参数张量
                    state.exp_avg,                   // 一阶矩估计（动量）
                    state.exp_avg_sq,                // 二阶矩估计
                    param.grad(),                    // 参数梯度
                    static_cast<float>(lr),          // 学习率
                    static_cast<float>(beta1),       // beta1参数
                    static_cast<float>(beta2),       // beta2参数
                    static_cast<float>(eps),         // epsilon值
                    static_cast<float>(bias_correction1_rcp),      // 偏差校正1的倒数
                    static_cast<float>(bias_correction2_sqrt_rcp)); // 偏差校正2平方根的倒数
            }
        }
    }

    /**
     * [功能描述]：自定义的梯度清零函数，基于PyTorch官方实现。
     * 当启用SH步骤跳步时，这个函数会跳过被跳过的参数组的梯度清零，
     * 允许梯度累积。当禁用跳步时，回退到标准的PyTorch实现。
     * 
     * 参考实现：https://github.com/pytorch/pytorch/blob/ee343ce60ceb449da09d229db25fa9d425d85a4b/torch/csrc/api/src/optim/optimizer.cpp#L122
     * 
     * @param set_to_none [参数说明]：是否将梯度设置为None（true）或零张量（false）。
     * @param iteration [参数说明]：当前迭代次数，用于判断是否应该跳过梯度清零。
     */
    void FusedAdam::zero_grad(bool set_to_none, int iteration) {
        if constexpr (SKIP_SH_STEPS) {
            // 当启用SH步骤跳步时的自定义梯度清零逻辑
            int i = 0; // 技巧：计数器，用于跟踪我们正在处理哪个高斯参数
            for (auto& group : param_groups()) {
                ++i;  // 递增参数组计数器
                for (auto& p : group.params()) {
                    // 如果优化器步骤被跳过，我们希望保持梯度累积
                    if (i == 3 && (iteration % 2 != 0 && iteration <= 25000))
                        continue;  // 跳过奇数迭代中的SH系数梯度清零
                    
                    // 检查参数是否有可变的梯度
                    if (p.mutable_grad().defined()) {
                        p.mutable_grad().detach_();  // 分离梯度
                        if (set_to_none)
                            p.mutable_grad().reset();  // 设置为None
                        else
                            p.mutable_grad().zero_();  // 设置为零张量
                    }
                }
            }
        } else {
            // 当禁用SH步骤跳步时，使用标准的PyTorch实现
            Optimizer::zero_grad(set_to_none);
        }
    }
} // namespace gs::training
