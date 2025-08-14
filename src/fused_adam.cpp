#include "core/fused_adam.hpp"
#include "adam_api.h"
#include <torch/torch.h>

/**
 * [文件描述]：融合Adam优化器实现文件
 * 功能：实现高斯散点训练的自定义Adam优化器，支持CUDA融合内核和球谐系数优化策略
 * 用途：为高斯散点参数优化提供高性能的梯度下降算法
 */

// TODO: 这只是为了奖金竞赛的临时实现，不应该集成到主代码库中
// TODO: 移除球谐系数跳步优化也意味着不再需要自定义的zero_grad()方法
// TODO: 所有跳步条件都假设迭代计数从1开始（目前确实如此）
// 在迭代1000到25000之间，我们可以跳过高阶球谐系数的每隔一步优化
// 这确实可以将奖金基准测试推到20分钟以下，但我不太喜欢这样做的实际影响
// 它还会导致质量指标和鲁棒性的轻微下降，因此默认情况下我禁用它
#define SKIP_SH_STEPS false

namespace gs {

    /**
     * [功能描述]：带闭包的优化步骤函数（不支持）
     * @param closure：损失闭包函数，用于重新计算损失
     * @return torch::Tensor：优化后的损失值（此实现中抛出异常）
     * 
     * 注意：融合Adam优化器不支持闭包操作，调用此函数将抛出异常
     */
    torch::Tensor FusedAdam::step(LossClosure closure) {
        TORCH_CHECK(false, "FusedAdam does not support closures.");  // 抛出不支持闭包的错误
        return {};
    }

    /**
     * [功能描述]：执行融合Adam优化步骤的主函数
     * @param iteration：当前训练迭代次数，用于控制球谐系数的优化策略
     * 
     * 该函数执行以下操作：
     * 1. 遍历所有参数组（高斯散点的不同参数类型）
     * 2. 为每个参数组应用特定的学习率和优化设置
     * 3. 初始化或更新Adam优化器状态（动量和二阶动量）
     * 4. 根据迭代次数应用球谐系数优化策略
     * 5. 调用CUDA融合内核执行参数更新
     */
    void FusedAdam::step(int iteration) {
        torch::NoGradGuard no_grad;  // 禁用梯度计算，因为这是优化器更新步骤

        // =============================================================================
        // 步骤1：获取全局优化器选项
        // =============================================================================
        const auto& global_options = options();

        // =============================================================================
        // 步骤2：遍历所有参数组进行优化
        // =============================================================================
        int i = 0;  // HACK: 计数器，用于跟踪当前处理的高斯参数类型
        for (auto& group : param_groups()) {
            ++i;  // 参数组索引递增

            // =========================================================================
            // 步骤2.1：提取优化超参数
            // =========================================================================
            // 从全局选项中获取默认值
            double lr = global_options.lr();        // 学习率
            double eps = global_options.eps();      // 数值稳定性参数
            auto [beta1, beta2] = global_options.betas();  // Adam动量参数

            // 如果参数组有自己的选项，则使用组特定的设置
            if (group.has_options()) {
                if (auto* group_opts = dynamic_cast<const Options*>(&group.options())) {
                    lr = group_opts->lr();
                    eps = group_opts->eps();
                    std::tie(beta1, beta2) = group_opts->betas();
                }
            }

            // =========================================================================
            // 步骤2.2：遍历参数组中的每个参数
            // =========================================================================
            for (auto& param : group.params()) {
                // 跳过没有梯度的参数
                if (!param.grad().defined()) {
                    continue;
                }

                // =====================================================================
                // 步骤2.3：懒初始化Adam状态
                // =====================================================================
                auto state_ptr = state_.find(param.unsafeGetTensorImpl());
                if (state_ptr == state_.end()) {
                    // 创建新的Adam参数状态
                    auto new_state = std::make_unique<AdamParamState>();
                    new_state->step_count = 0;  // 步数计数器
                    new_state->exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);     // 一阶动量
                    new_state->exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve); // 二阶动量

                    state_[param.unsafeGetTensorImpl()] = std::move(new_state);
                    state_ptr = state_.find(param.unsafeGetTensorImpl());
                }

                auto& state = static_cast<AdamParamState&>(*state_ptr->second);

                // =====================================================================
                // 步骤2.4：更新步数计数
                // =====================================================================
                state.step_count++;

                // =====================================================================
                // 步骤2.5：球谐系数优化策略
                // =====================================================================
                // 在前1000次迭代中不使用高阶球谐系数，这是一个免费的加速
                if (i == 3 && iteration <= 1000)
                    continue;

                // 编译时条件：如果启用了跳步优化
                if constexpr (SKIP_SH_STEPS) {
                    // 在训练期间跳过每隔一步，除了最后5000次迭代
                    if (i == 3 && (iteration % 2 != 0 && iteration <= 25000))
                        continue;
                }

                // =====================================================================
                // 步骤2.6：计算偏差修正因子
                // =====================================================================
                auto bias_correction1_rcp = 1.0 / (1.0 - std::pow(beta1, state.step_count));        // 一阶动量偏差修正
                auto bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(beta2, state.step_count)); // 二阶动量偏差修正

                // =====================================================================
                // 步骤2.7：调用融合CUDA内核执行参数更新
                // =====================================================================
                fast_gs::optimizer::adam_step_wrapper(
                    param,                                                      // 参数张量
                    state.exp_avg,                                             // 一阶动量估计
                    state.exp_avg_sq,                                          // 二阶动量估计
                    param.grad(),                                              // 参数梯度
                    static_cast<float>(lr),                                    // 学习率
                    static_cast<float>(beta1),                                 // 一阶动量衰减率
                    static_cast<float>(beta2),                                 // 二阶动量衰减率
                    static_cast<float>(eps),                                   // 数值稳定性参数
                    static_cast<float>(bias_correction1_rcp),                  // 一阶动量偏差修正
                    static_cast<float>(bias_correction2_sqrt_rcp));            // 二阶动量偏差修正
            }
        }
    }

    /**
     * [功能描述]：清零参数梯度的自定义实现
     * @param set_to_none：如果为true则将梯度设置为nullptr，否则设置为零张量
     * @param iteration：当前训练迭代次数，用于控制球谐系数的梯度处理策略
     * 
     * 该函数的自定义实现考虑了球谐系数的跳步优化：
     * - 如果某个优化步骤被跳过，则保持梯度累积而不清零
     * - 这确保了跳步策略的正确性，避免梯度信息丢失
     * 
     * 基于PyTorch官方实现修改：
     * https://github.com/pytorch/pytorch/blob/ee343ce60ceb449da09d229db25fa9d425d85a4b/torch/csrc/api/src/optim/optimizer.cpp#L122
     */
    void FusedAdam::zero_grad(bool set_to_none, int iteration) {
        // 编译时条件：如果启用了跳步优化，则使用自定义梯度清零逻辑
        if constexpr (SKIP_SH_STEPS) {
            int i = 0;  // HACK: 计数器，用于跟踪当前处理的高斯参数类型
            for (auto& group : param_groups()) {
                ++i;  // 参数组索引递增
                for (auto& p : group.params()) {
                    // 如果优化器步骤被跳过，我们希望保持梯度累积
                    if (i == 3 && (iteration % 2 != 0 && iteration <= 25000))
                        continue;  // 跳过梯度清零，保持累积
                    
                    // 清零参数梯度
                    if (p.mutable_grad().defined()) {
                        p.mutable_grad().detach_();  // 从计算图中分离梯度
                        if (set_to_none)
                            p.mutable_grad().reset();   // 设置梯度为nullptr
                        else
                            p.mutable_grad().zero_();   // 设置梯度为零张量
                    }
                }
            }
        } else {
            // 如果未启用跳步优化，则使用标准的梯度清零逻辑
            Optimizer::zero_grad(set_to_none);
        }
    }

} // namespace gs