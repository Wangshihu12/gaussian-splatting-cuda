#include "core/strategy.hpp"

/**
 * [文件描述]：训练策略工具函数实现文件
 * 功能：提供高斯散点训练中的通用工具函数，包括参数初始化、优化器创建、参数更新等
 * 用途：为不同的训练策略（MCMC、DefaultStrategy等）提供共享的基础功能
 */

namespace strategy {

    /**
     * [功能描述]：初始化高斯散点的所有可训练参数
     * @param splat_data：高斯散点数据对象的引用，包含所有高斯参数
     * 
     * 该函数将所有高斯参数：
     * - 转移到CUDA设备上
     * - 启用梯度计算（requires_grad=true）
     * - 为后续的训练过程做好准备
     */
    void initialize_gaussians(gs::SplatData& splat_data) {
        const auto dev = torch::kCUDA;  // 指定CUDA设备
        
        // 将所有高斯参数转移到GPU并启用梯度计算
        splat_data.means() = splat_data.means().to(dev).set_requires_grad(true);           // 3D位置参数
        splat_data.scaling_raw() = splat_data.scaling_raw().to(dev).set_requires_grad(true);  // 缩放参数（原始值）
        splat_data.rotation_raw() = splat_data.rotation_raw().to(dev).set_requires_grad(true); // 旋转参数（原始四元数）
        splat_data.opacity_raw() = splat_data.opacity_raw().to(dev).set_requires_grad(true);   // 不透明度参数（原始值）
        splat_data.sh0() = splat_data.sh0().to(dev).set_requires_grad(true);              // 球谐函数0阶系数（直流分量）
        splat_data.shN() = splat_data.shN().to(dev).set_requires_grad(true);              // 球谐函数高阶系数
    }

    /**
     * [功能描述]：为高斯散点参数创建Adam优化器
     * @param splat_data：高斯散点数据对象引用
     * @param params：优化参数配置，包含各类参数的学习率设置
     * @return 配置好的Adam优化器的智能指针
     * 
     * 该函数为不同类型的参数设置不同的学习率：
     * - 位置参数：根据场景尺度调整学习率
     * - 球谐函数参数：基础学习率和降低的高阶学习率
     * - 几何参数：独立的学习率设置
     */
    std::unique_ptr<torch::optim::Optimizer> create_optimizer(
        gs::SplatData& splat_data,
        const gs::param::OptimizationParameters& params) {
        
        using torch::optim::AdamOptions;
        std::vector<torch::optim::OptimizerParamGroup> groups;  // 参数组列表

        // =============================================================================
        // 创建不同参数组，每组使用不同的学习率
        // =============================================================================
        
        // 位置参数组：学习率根据场景尺度调整
        // 场景越大，位置更新的步长需要相应增大
        groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.means()},
                                                              std::make_unique<AdamOptions>(params.means_lr * splat_data.get_scene_scale())));
        
        // 球谐函数0阶系数组（直流分量，主要控制颜色）
        groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.sh0()},
                                                              std::make_unique<AdamOptions>(params.shs_lr)));
        
        // 球谐函数高阶系数组（控制颜色的方向性变化）
        // 使用较低的学习率（1/20）因为高阶系数通常需要更精细的调整
        groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.shN()},
                                                              std::make_unique<AdamOptions>(params.shs_lr / 20.f)));
        
        // 缩放参数组（控制高斯的大小）
        groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.scaling_raw()},
                                                              std::make_unique<AdamOptions>(params.scaling_lr)));
        
        // 旋转参数组（控制高斯的朝向）
        groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.rotation_raw()},
                                                              std::make_unique<AdamOptions>(params.rotation_lr)));
        
        // 不透明度参数组（控制高斯的透明度）
        groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.opacity_raw()},
                                                              std::make_unique<AdamOptions>(params.opacity_lr)));

        // =============================================================================
        // 设置数值稳定性参数
        // =============================================================================
        
        // 为所有参数组设置eps=1e-15，提高数值稳定性，防止除零错误
        for (auto& g : groups)
            static_cast<AdamOptions&>(g.options()).eps(1e-15);

        // 创建并返回Adam优化器，使用0.0作为默认学习率（由各参数组覆盖）
        return std::make_unique<torch::optim::Adam>(groups, AdamOptions(0.f).eps(1e-15));
    }

    /**
     * [功能描述]：创建指数衰减学习率调度器
     * @param params：优化参数配置，包含总迭代次数
     * @param optimizer：要调度的优化器指针
     * @param param_group_index：要调度的参数组索引
     * @return 指数学习率调度器的智能指针
     * 
     * 学习率衰减策略：
     * - 在训练结束时，学习率衰减到初始值的1%
     * - 使用指数衰减确保平滑的学习率变化
     */
    std::unique_ptr<ExponentialLR> create_scheduler(
        const gs::param::OptimizationParameters& params,
        torch::optim::Optimizer* optimizer,
        int param_group_index) {
        
        // 计算指数衰减因子
        // Python等价代码: gamma = 0.01^(1/max_steps)
        // 这意味着在max_steps步后，学习率将变为 0.01 * initial_lr
        const double gamma = std::pow(0.01, 1.0 / params.iterations);
        
        return std::make_unique<ExponentialLR>(*optimizer, gamma, param_group_index);
    }

    /**
     * [功能描述]：更新参数和优化器状态的通用函数
     * @param param_fn：参数更新函数，定义如何变换参数
     * @param optimizer_fn：优化器状态更新函数，定义如何更新优化器状态
     * @param optimizer：优化器的智能指针引用
     * @param splat_data：高斯散点数据对象引用
     * @param param_idxs：要更新的参数组索引列表
     * 
     * 该函数处理复杂的参数更新场景，如：
     * - 高斯分裂后的参数扩展
     * - 高斯剪枝后的参数收缩
     * - 保持优化器状态的一致性
     */
    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<torch::optim::Optimizer>& optimizer,
        gs::SplatData& splat_data,
        std::vector<size_t> param_idxs) {
        
        // =============================================================================
        // 步骤1：准备参数指针数组和新参数存储
        // =============================================================================
        
        // 创建指向所有可训练参数的指针数组
        std::array<torch::Tensor*, 6> params = {
            &splat_data.means(),        // 0: 位置参数
            &splat_data.sh0(),          // 1: 球谐0阶系数
            &splat_data.shN(),          // 2: 球谐高阶系数
            &splat_data.scaling_raw(),  // 3: 缩放参数
            &splat_data.rotation_raw(), // 4: 旋转参数
            &splat_data.opacity_raw()   // 5: 不透明度参数
        };

        std::array<torch::Tensor, 6> new_params;  // 存储更新后的参数

        // =============================================================================
        // 步骤2：收集旧参数的优化器状态
        // =============================================================================
        
        // 保存旧参数的键值和优化器状态，用于后续的状态迁移
        std::vector<void*> old_param_keys;
        std::array<std::unique_ptr<torch::optim::OptimizerParamState>, 6> saved_states;

        // 遍历要更新的参数组
        for (auto i : param_idxs) {
            auto param = params[i];
            
            // 使用参数更新函数生成新参数
            auto new_param = param_fn(i, *param);
            new_params[i] = new_param;

            // 获取优化器中旧参数的键值
            auto& old_param = optimizer->param_groups()[i].params()[0];
            void* old_param_key = old_param.unsafeGetTensorImpl();
            old_param_keys.push_back(old_param_key);

            // 检查并保存优化器状态
            auto state_it = optimizer->state().find(old_param_key);
            if (state_it != optimizer->state().end()) {
                // 克隆状态以防修改过程中出现问题
                // 支持Adam优化器状态的处理
                if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(state_it->second.get())) {
                    auto new_state = optimizer_fn(*adam_state, new_param);
                    saved_states[i] = std::move(new_state);
                } else {
                    saved_states[i] = nullptr;  // 不支持的优化器类型
                }
            } else {
                saved_states[i] = nullptr;  // 没有现有状态
            }
        }

        // =============================================================================
        // 步骤3：清理旧的优化器状态
        // =============================================================================
        
        // 从优化器状态映射中移除所有旧参数的状态
        for (auto key : old_param_keys) {
            optimizer->state().erase(key);
        }

        // =============================================================================
        // 步骤4：更新参数和恢复优化器状态
        // =============================================================================
        
        // 更新优化器中的参数引用并恢复状态
        for (auto i : param_idxs) {
            // 更新优化器参数组中的参数引用
            optimizer->param_groups()[i].params()[0] = new_params[i];

            // 如果有保存的状态，则恢复到新参数
            if (saved_states[i]) {
                void* new_param_key = new_params[i].unsafeGetTensorImpl();
                optimizer->state()[new_param_key] = std::move(saved_states[i]);
            }
        }

        // =============================================================================
        // 步骤5：更新SplatData中的参数
        // =============================================================================
        
        // 根据参数索引更新对应的SplatData成员
        for (auto i : param_idxs) {
            if (i == 0) {
                splat_data.means() = new_params[i];        // 位置参数
            } else if (i == 1) {
                splat_data.sh0() = new_params[i];          // 球谐0阶系数
            } else if (i == 2) {
                splat_data.shN() = new_params[i];          // 球谐高阶系数
            } else if (i == 3) {
                splat_data.scaling_raw() = new_params[i];  // 缩放参数
            } else if (i == 4) {
                splat_data.rotation_raw() = new_params[i]; // 旋转参数
            } else if (i == 5) {
                splat_data.opacity_raw() = new_params[i];  // 不透明度参数
            }
        }
    }

} // namespace strategy
