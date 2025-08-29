#include "scheduler.hpp"
#include "fused_adam.hpp"

namespace gs::training {
    /**
     * [功能描述]：指数学习率调度器的step函数实现。
     * 这个函数根据设置的衰减因子gamma更新优化器的学习率，
     * 支持两种模式：更新单个参数组或更新所有参数组。
     * 使用指数衰减公式：new_lr = current_lr * gamma
     */
    void ExponentialLR::step() {
        // 步骤1：检查参数组索引，决定更新模式
        if (param_group_index_ >= 0) {
            // 模式1：更新指定的单个参数组
            // 获取指定索引的参数组引用
            auto& group = optimizer_.param_groups()[param_group_index_];

            // 将参数组选项转换为FusedAdam::Options类型
            // 这里假设使用的是FusedAdam优化器，需要类型转换来访问学习率
            auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
            
            // 获取当前学习率
            double current_lr = fused_adam_options->lr();
            
            // 应用指数衰减：新学习率 = 当前学习率 × 衰减因子
            fused_adam_options->lr(current_lr * gamma_);
            
        } else {
            // 模式2：更新所有参数组的学习率
            // 遍历优化器中的所有参数组
            for (auto& group : optimizer_.param_groups()) {
                // 将每个参数组的选项转换为FusedAdam::Options类型
                auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
                
                // 获取当前参数组的学习率
                double current_lr = fused_adam_options->lr();
                
                // 对所有参数组应用相同的指数衰减
                fused_adam_options->lr(current_lr * gamma_);
            }
        }
    }
} // namespace gs::training
