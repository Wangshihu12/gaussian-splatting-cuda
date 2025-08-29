#include "mcmc.hpp"                    // MCMC策略类的头文件声明
#include "Ops.h"                        // 高斯溅射操作函数
#include "core/parameters.hpp"          // 核心参数定义
#include "optimizers/fused_adam.hpp"    // 融合Adam优化器
#include "rasterization/rasterizer.hpp" // 光栅化渲染器
#include <iostream>                     // 标准输入输出流
#include <random>                       // 随机数生成

#ifdef _WIN32
#include <c10/cuda/CUDACachingAllocator.h> // Windows平台需要，用于emptyCache操作
#endif

namespace gs::training {
    /**
     * [功能描述]：指数学习率调度器的step函数实现。
     * 这个函数根据衰减因子gamma更新优化器的学习率，实现学习率的指数衰减。
     * 支持更新单个参数组或所有参数组的学习率。
     */
    void MCMC::ExponentialLR::step() {
        if (param_group_index_ >= 0) {
            // 更新指定的参数组
            auto& group = optimizer_.param_groups()[param_group_index_];

            // 尝试首先转换为我们的自定义选项类型
            // 这里假设使用的是FusedAdam优化器
            auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
            double current_lr = fused_adam_options->lr();                    // 获取当前学习率
            fused_adam_options->lr(current_lr * gamma_);                    // 应用衰减因子
        } else {
            // 更新所有参数组的学习率
            for (auto& group : optimizer_.param_groups()) {
                auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
                double current_lr = fused_adam_options->lr();                // 获取当前学习率
                fused_adam_options->lr(current_lr * gamma_);                // 应用衰减因子
            }
        }
    }

    /**
     * [功能描述]：MCMC类的构造函数实现。
     * 使用移动语义初始化高斯溅射数据，避免不必要的数据拷贝。
     * @param splat_data [参数说明]：高斯溅射数据，通过移动语义转移所有权。
     */
    MCMC::MCMC(gs::SplatData&& splat_data)
        : _splat_data(std::move(splat_data)) {  // 使用移动语义转移数据所有权
    }

    /**
     * [功能描述]：多项分布采样函数实现。
     * 从权重分布中进行随机采样，支持有放回和无放回采样。
     * 对于大型数组（超过2^24个元素），实现了手动采样算法。
     * 
     * @param weights [参数说明]：权重张量，表示每个元素的采样概率。
     * @param n [参数说明]：要采样的元素数量。
     * @param replacement [参数说明]：是否允许重复采样，true表示有放回。
     * @return [返回值说明]：采样结果的索引张量。
     */
    torch::Tensor MCMC::multinomial_sample(const torch::Tensor& weights, int n, bool replacement) {
        const int64_t num_elements = weights.size(0);  // 获取权重张量的元素数量

        // PyTorch的multinomial函数有2^24个元素的限制
        if (num_elements <= (1 << 24)) {
            // 如果元素数量在限制范围内，直接使用PyTorch的multinomial函数
            return torch::multinomial(weights, n, replacement);
        }
        
        // 对于更大的数组，需要手动实现采样算法
        // 步骤1：归一化权重，确保权重和为1
        auto weights_normalized = weights / weights.sum();
        // 步骤2：将权重张量移动到CPU，因为手动采样在CPU上更高效
        auto weights_cpu = weights_normalized.cpu();

        // 步骤3：准备存储采样索引的向量
        std::vector<int64_t> sampled_indices;
        sampled_indices.reserve(n);  // 预分配内存空间

        // 步骤4：创建累积分布函数（CDF）
        auto cumsum = weights_cpu.cumsum(0);  // 计算累积和
        auto cumsum_data = cumsum.accessor<float, 1>();  // 获取数据访问器

        // 步骤5：初始化随机数生成器
        std::random_device rd;                    // 随机设备
        std::mt19937 gen(rd());                   // Mersenne Twister随机数生成器
        std::uniform_real_distribution<float> dis(0.0, 1.0);  // 均匀分布

        // 步骤6：执行n次采样
        for (int i = 0; i < n; ++i) {
            float u = dis(gen);  // 生成[0,1]范围内的随机数
            
            // 使用二分查找在累积分布中找到对应的索引
            int64_t idx = 0;
            int64_t left = 0, right = num_elements - 1;
            while (left <= right) {
                int64_t mid = (left + right) / 2;  // 计算中点
                if (cumsum_data[mid] < u) {
                    // 如果中点值小于随机数，在右半部分继续查找
                    left = mid + 1;
                } else {
                    // 否则，记录当前索引并在左半部分继续查找
                    idx = mid;
                    right = mid - 1;
                }
            }
            sampled_indices.push_back(idx);  // 将找到的索引添加到结果中
        }

        // 步骤7：将采样结果转换为PyTorch张量并移动到原始设备
        auto result = torch::tensor(sampled_indices, torch::kLong);
        return result.to(weights.device());
    }

    /**
     * [功能描述]：为重新定位操作更新优化器状态。
     * 当高斯点被重新定位时，需要相应地重置优化器中的动量状态，
     * 确保新位置的高斯点从"干净"的状态开始优化。
     * 
     * @param optimizer [参数说明]：要更新的优化器指针。
     * @param sampled_indices [参数说明]：采样的索引张量，表示新位置的高斯点。
     * @param dead_indices [参数说明]：被移除的高斯点索引张量。
     * @param param_position [参数说明]：参数在优化器中的位置。
     */
    void MCMC::update_optimizer_for_relocate(torch::optim::Optimizer* optimizer,
                                             const torch::Tensor& sampled_indices,
                                             const torch::Tensor& dead_indices,
                                             int param_position) {
        // 步骤1：获取指定位置的参数
        auto& param = optimizer->param_groups()[param_position].params()[0];
        void* param_key = param.unsafeGetTensorImpl();  // 获取参数的内部实现指针作为键

        // 步骤2：检查优化器状态是否存在
        auto state_it = optimizer->state().find(param_key);
        if (state_it == optimizer->state().end()) {
            // 如果状态不存在，说明优化器还没有调用过step()函数
            // 在这种情况下，没有需要重置的状态，可以安全返回
            return;
        }

        // 步骤3：获取优化器状态并处理两种Adam类型
        auto& param_state = *state_it->second;
        // 将状态转换为FusedAdam的AdamParamState类型
        auto* fused_adam_state = static_cast<FusedAdam::AdamParamState*>(&param_state);
        
        // 步骤4：重置新位置高斯点的动量状态
        // 将一阶矩估计（exp_avg）重置为0
        fused_adam_state->exp_avg.index_put_({sampled_indices}, 0);
        // 将二阶矩估计（exp_avg_sq）重置为0
        fused_adam_state->exp_avg_sq.index_put_({sampled_indices}, 0);

        if (fused_adam_state->max_exp_avg_sq.defined()) {
            fused_adam_state->max_exp_avg_sq.index_put_({sampled_indices}, 0);
        }
    }

    /**
     * [功能描述]：重新定位高斯点函数，实现MCMC策略中的密集化操作。
     * 这个函数识别"死亡"的高斯点（不透明度过低或旋转参数过小），
     * 从"存活"的高斯点中采样，并将采样的参数重新分配到死亡位置，
     * 实现高斯点的动态重新分布和模型结构的优化。
     * 
     * @return [返回值说明]：返回重新定位的高斯点数量。
     */
    int MCMC::relocate_gs() {
        // 步骤1：获取不透明度参数并处理不同的张量形状
        torch::NoGradGuard no_grad;  // 禁用梯度计算，因为这是参数重新分配操作
        auto opacities = _splat_data.get_opacity();  // 获取当前的不透明度值
        if (opacities.dim() == 2 && opacities.size(1) == 1) {
            // 如果形状是[N, 1]，压缩为[N]以统一处理
            opacities = opacities.squeeze(-1);
        }

        // 步骤2：识别"死亡"的高斯点
        auto rotation_raw = _splat_data.rotation_raw();  // 获取原始旋转参数
        // 创建死亡掩码：不透明度低于阈值 或 旋转参数的平方和过小
        auto dead_mask = opacities <= _params->min_opacity | (rotation_raw * rotation_raw).sum(-1) < 1e-8f;
        auto dead_indices = dead_mask.nonzero().squeeze(-1);  // 获取死亡高斯点的索引
        int n_dead = dead_indices.numel();  // 计算死亡高斯点的数量

        // 如果没有死亡的高斯点，直接返回
        if (n_dead == 0)
            return 0;

        // 步骤3：识别"存活"的高斯点
        auto alive_mask = ~dead_mask;  // 存活掩码是死亡掩码的反转
        auto alive_indices = alive_mask.nonzero().squeeze(-1);  // 获取存活高斯点的索引

        // 如果没有存活的高斯点，无法进行重新定位
        if (alive_indices.numel() == 0)
            return 0;

        // 步骤4：基于不透明度从存活高斯点中采样
        auto probs = opacities.index_select(0, alive_indices);  // 选择存活高斯点的不透明度作为采样概率
        auto sampled_idxs_local = multinomial_sample(probs, n_dead, true);  // 进行多项分布采样
        auto sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);  // 将局部索引转换为全局索引

        // 步骤5：获取采样高斯点的参数
        auto sampled_opacities = opacities.index_select(0, sampled_idxs);  // 采样的不透明度
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);  // 采样的缩放参数

        // 步骤6：计算每个采样索引的出现次数（用于CUDA重新定位函数）
        auto ratios = torch::ones_like(opacities, torch::kInt32);  // 创建全1张量
        // 在采样索引位置累加1，统计每个索引被选中的次数
        ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kInt32));
        // 选择采样索引对应的比率值
        ratios = ratios.index_select(0, sampled_idxs).contiguous();

        // 重要：按照Python实现的方式进行裁剪
        const int n_max = static_cast<int>(_binoms.size(0));  // 获取二项式系数的最大数量
        ratios = torch::clamp_max_(ratios, n_max);  // 将比率限制在最大值范围内

        // 步骤7：调用gsplat的CUDA重新定位函数
        auto relocation_result = gsplat::relocation(
            sampled_opacities,    // 采样的不透明度
            sampled_scales,       // 采样的缩放参数
            ratios,               // 采样比率
            _binoms,              // 二项式系数
            n_max);               // 最大数量

        // 步骤8：提取重新定位后的参数
        auto new_opacities = std::get<0>(relocation_result);  // 新的不透明度值
        auto new_scales = std::get<1>(relocation_result);     // 新的缩放值

        // 步骤9：裁剪新的不透明度值到有效范围
        new_opacities = torch::clamp_(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

        // 步骤10：更新采样索引位置的参数
        // 处理不透明度形状的兼容性
        if (_splat_data.opacity_raw().dim() == 2) {
            // 如果原始形状是[N, 1]，需要添加维度
            _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                                 torch::logit(new_opacities).unsqueeze(-1));
        } else {
            // 如果原始形状是[N]，直接赋值
            _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
        }
        // 更新缩放参数（使用对数空间）
        _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

        // 步骤11：将采样的参数复制到死亡索引位置
        // 复制位置参数
        _splat_data.means().index_put_({dead_indices}, _splat_data.means().index_select(0, sampled_idxs));
        // 复制球谐函数0阶系数（基础颜色）
        _splat_data.sh0().index_put_({dead_indices}, _splat_data.sh0().index_select(0, sampled_idxs));
        // 复制球谐函数高阶系数
        _splat_data.shN().index_put_({dead_indices}, _splat_data.shN().index_select(0, sampled_idxs));
        // 复制缩放参数
        _splat_data.scaling_raw().index_put_({dead_indices}, _splat_data.scaling_raw().index_select(0, sampled_idxs));
        // 复制旋转参数
        _splat_data.rotation_raw().index_put_({dead_indices}, _splat_data.rotation_raw().index_select(0, sampled_idxs));
        // 复制不透明度参数
        _splat_data.opacity_raw().index_put_({dead_indices}, _splat_data.opacity_raw().index_select(0, sampled_idxs));

        // 步骤12：更新优化器状态
        // 遍历所有6个参数组，更新优化器状态
        for (int i = 0; i < 6; ++i) {
            update_optimizer_for_relocate(_optimizer.get(), sampled_idxs, dead_indices, i);
        }

        return n_dead;  // 返回重新定位的高斯点数量
    }

    /**
     * [功能描述]：添加新高斯点函数，实现MCMC策略中的模型扩展操作。
     * 这个函数通过复制现有高斯点并添加噪声来创建新的高斯点，
     * 实现模型的动态增长和密集化，同时保持优化器状态的同步更新。
     * 
     * @return [返回值说明]：返回新添加的高斯点数量。
     */
    int MCMC::add_new_gs() {
        // 步骤1：初始检查和设置
        torch::NoGradGuard no_grad;  // 禁用梯度计算，因为这是参数扩展操作
        if (!_optimizer) {
            // 检查优化器是否已初始化，如果未初始化则发出警告并返回
            std::cerr << "Warning: add_new_gs called but optimizer not initialized" << std::endl;
            return 0;
        }

        // 步骤2：计算目标高斯点数量
        const int current_n = _splat_data.size();                    // 当前高斯点数量
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));  // 目标数量：当前数量的1.05倍，但不超过最大容量
        const int n_new = std::max(0, n_target - current_n);        // 需要新增的高斯点数量

        // 如果不需要添加新的高斯点，直接返回
        if (n_new == 0)
            return 0;

        // 步骤3：获取不透明度参数并处理形状兼容性
        auto opacities = _splat_data.get_opacity();  // 获取当前的不透明度值
        if (opacities.dim() == 2 && opacities.size(1) == 1) {
            // 如果形状是[N, 1]，压缩为[N]以统一处理
            opacities = opacities.squeeze(-1);
        }

        // 步骤4：基于不透明度进行采样
        auto probs = opacities.flatten();                           // 将不透明度展平为一维张量
        auto sampled_idxs = multinomial_sample(probs, n_new, true); // 基于不透明度进行多项分布采样

        // 步骤5：获取采样高斯点的参数
        auto sampled_opacities = opacities.index_select(0, sampled_idxs);  // 采样的不透明度
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);  // 采样的缩放参数

        // 步骤6：计算采样比率
        auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);  // 创建全零张量
        ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32)); // 在采样位置累加1
        ratios = ratios.index_select(0, sampled_idxs) + 1;  // 选择采样位置的比率并加1

        // 重要：按照Python实现的方式进行裁剪和类型转换
        const int n_max = static_cast<int>(_binoms.size(0));  // 获取二项式系数的最大数量
        ratios = torch::clamp(ratios, 1, n_max);              // 将比率限制在[1, n_max]范围内
        ratios = ratios.to(torch::kInt32).contiguous();        // 转换为整数类型并确保内存连续

        // 步骤7：调用gsplat的CUDA重新定位函数
        auto relocation_result = gsplat::relocation(
            sampled_opacities,    // 采样的不透明度
            sampled_scales,       // 采样的缩放参数
            ratios,               // 采样比率
            _binoms,              // 二项式系数
            n_max);               // 最大数量

        // 步骤8：提取重新定位后的参数
        auto new_opacities = std::get<0>(relocation_result);  // 新的不透明度值
        auto new_scales = std::get<1>(relocation_result);     // 新的缩放值

        // 步骤9：裁剪新的不透明度值到有效范围
        new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

        // 步骤10：更新现有高斯点的参数（在拼接之前）
        if (_splat_data.opacity_raw().dim() == 2) {
            // 如果原始形状是[N, 1]，需要添加维度
            _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                                 torch::logit(new_opacities).unsqueeze(-1));
        } else {
            // 如果原始形状是[N]，直接赋值
            _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
        }
        // 更新缩放参数（使用对数空间）
        _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

        // 步骤11：准备要拼接的新高斯点参数
        auto new_means = _splat_data.means().index_select(0, sampled_idxs);        // 新高斯点的位置
        auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);            // 新高斯点的球谐函数0阶系数
        auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);            // 新高斯点的球谐函数高阶系数
        auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs); // 新高斯点的缩放参数
        auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs); // 新高斯点的旋转参数
        auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);   // 新高斯点的不透明度参数

        // 步骤12：拼接所有参数
        // 将现有参数与新参数在第一个维度上拼接，并启用梯度计算
        auto concat_means = torch::cat({_splat_data.means(), new_means}, 0).set_requires_grad(true);
        auto concat_sh0 = torch::cat({_splat_data.sh0(), new_sh0}, 0).set_requires_grad(true);
        auto concat_shN = torch::cat({_splat_data.shN(), new_shN}, 0).set_requires_grad(true);
        auto concat_scaling = torch::cat({_splat_data.scaling_raw(), new_scaling}, 0).set_requires_grad(true);
        auto concat_rotation = torch::cat({_splat_data.rotation_raw(), new_rotation}, 0).set_requires_grad(true);
        auto concat_opacity = torch::cat({_splat_data.opacity_raw(), new_opacity}, 0).set_requires_grad(true);

        // 步骤13：安全的优化器状态更新
        // 首先将新参数存储在临时数组中
        std::array new_params = {
            &concat_means, &concat_sh0, &concat_shN,
            &concat_scaling, &concat_rotation, &concat_opacity};

        // 收集旧参数的键和状态
        std::vector<void*> old_param_keys;                                    // 旧参数的键
        std::vector<std::unique_ptr<torch::optim::OptimizerParamState>> saved_states;  // 保存的状态

        // 步骤14：为每个参数组处理优化器状态
        for (int i = 0; i < 6; ++i) {
            auto& old_param = _optimizer->param_groups()[i].params()[0];  // 获取旧参数
            void* old_param_key = old_param.unsafeGetTensorImpl();        // 获取旧参数的键
            old_param_keys.push_back(old_param_key);

            // 检查状态是否存在
            auto state_it = _optimizer->state().find(old_param_key);
            if (state_it == _optimizer->state().end()) {
                // 如果状态不存在，保存空指针
                saved_states.push_back(nullptr);
            }

            // 获取FusedAdam状态
            auto* fused_adam_state = static_cast<FusedAdam::AdamParamState*>(state_it->second.get());
            
            // 根据参数类型确定新形状
            torch::IntArrayRef new_shape;
            if (i == 0)
                new_shape = new_means.sizes();      // 位置参数
            else if (i == 1)
                new_shape = new_sh0.sizes();        // 球谐函数0阶系数
            else if (i == 2)
                new_shape = new_shN.sizes();        // 球谐函数高阶系数
            else if (i == 3)
                new_shape = new_scaling.sizes();    // 缩放参数
            else if (i == 4)
                new_shape = new_rotation.sizes();   // 旋转参数
            else
                new_shape = new_opacity.sizes();    // 不透明度参数

            // 创建要添加的零张量，用于扩展优化器状态
            auto zeros_to_add = torch::zeros(new_shape, fused_adam_state->exp_avg.options());
            // 拼接现有的一阶矩估计和新的零张量
            auto new_exp_avg = torch::cat({fused_adam_state->exp_avg, zeros_to_add}, 0);
            // 拼接现有的二阶矩估计和新的零张量
            auto new_exp_avg_sq = torch::cat({fused_adam_state->exp_avg_sq, zeros_to_add}, 0);

            // 创建新的优化器状态
            auto new_state = std::make_unique<FusedAdam::AdamParamState>();
            new_state->step_count = fused_adam_state->step_count;  // 保持步数计数
            new_state->exp_avg = new_exp_avg;                      // 新的一阶矩估计
            new_state->exp_avg_sq = new_exp_avg_sq;               // 新的二阶矩估计
            
            // 如果存在最大二阶矩估计，也需要扩展
            if (fused_adam_state->max_exp_avg_sq.defined()) {
                auto new_max_exp_avg_sq = torch::cat({fused_adam_state->max_exp_avg_sq, zeros_to_add}, 0);
                new_state->max_exp_avg_sq = new_max_exp_avg_sq;
            }

            saved_states.push_back(std::move(new_state));  // 保存新状态
        }

        // 步骤15：移除所有旧状态
        for (auto key : old_param_keys) {
            _optimizer->state().erase(key);
        }

        // 步骤16：更新参数并添加新状态
        for (int i = 0; i < 6; ++i) {
            // 更新优化器中的参数引用
            _optimizer->param_groups()[i].params()[0] = *new_params[i];

            // 如果存在保存的状态，添加到优化器中
            if (saved_states[i]) {
                void* new_param_key = new_params[i]->unsafeGetTensorImpl();
                _optimizer->state()[new_param_key] = std::move(saved_states[i]);
            }
        }

        // 步骤17：最后更新模型的参数
        _splat_data.means() = concat_means;           // 更新位置参数
        _splat_data.sh0() = concat_sh0;               // 更新球谐函数0阶系数
        _splat_data.shN() = concat_shN;               // 更新球谐函数高阶系数
        _splat_data.scaling_raw() = concat_scaling;   // 更新缩放参数
        _splat_data.rotation_raw() = concat_rotation; // 更新旋转参数
        _splat_data.opacity_raw() = concat_opacity;   // 更新不透明度参数

        return n_new;  // 返回新添加的高斯点数量
    }

    void MCMC::inject_noise() {
        torch::NoGradGuard no_grad;

        // Get current learning rate from optimizer (after scheduler has updated it)
        auto& group = _optimizer->param_groups()[0];
        auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
        const float current_lr = static_cast<float>(fused_adam_options->lr()) * _noise_lr;

        // Generate noise
        auto noise = torch::randn_like(_splat_data.means());

        gsplat::add_noise(
            _splat_data.opacity_raw(),
            _splat_data.scaling_raw(),
            _splat_data.rotation_raw(),
            noise,
            _splat_data.means(),
            current_lr);
    }

    /**
     * [功能描述]：后向传播后的处理函数，在每个训练步骤后执行。
     * 这个函数是MCMC策略的核心，负责在每次反向传播完成后进行模型优化、
     * 结构调整和内存管理，确保训练过程的稳定性和模型质量的持续改进。
     * 
     * @param iter [参数说明]：当前迭代次数，用于控制各种操作的执行频率。
     * @param render_output [参数说明]：渲染输出结果，包含图像和透明度等信息（当前未使用）。
     */
    void MCMC::post_backward(int iter, RenderOutput& render_output) {
        // 步骤1：球谐函数阶数递增
        // 禁用梯度计算，因为这是模型结构调整操作
        torch::NoGradGuard no_grad;
        
        // 每隔指定迭代次数递增球谐函数阶数
        // 球谐函数阶数控制颜色表示的复杂度，从低阶开始逐步增加
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data.increment_sh_degree();  // 递增球谐函数阶数
        }

        // 步骤2：细化阶段的高斯点优化
        // 检查当前迭代是否处于细化阶段
        if (is_refining(iter)) {
            // 重新定位"死亡"的高斯点
            // 识别并重新分配不透明度过低或旋转参数过小的高斯点
            relocate_gs();

            // 添加新的高斯点
            // 通过复制现有高斯点并添加噪声来扩展模型容量
            add_new_gs();
        }

        // 步骤3：位置噪声注入
        // 向高斯点的位置参数注入随机噪声
        // 这是MCMC策略的重要组成部分，帮助模型跳出局部最优解
        inject_noise();

        // 步骤4：Windows平台特殊处理
#ifdef _WIN32
        // Windows平台不支持CUDACachingAllocator的expandable_segments功能
        // 因此需要定期清理CUDA缓存以避免内存碎片问题
        if (iter % 10 == 0)  // 每10次迭代执行一次
            c10::cuda::CUDACachingAllocator::emptyCache();  // 清空CUDA缓存
#endif
    }

    /**
     * [功能描述]：执行单个训练步骤，实现MCMC策略的核心训练循环。
     * 这个函数负责在每次迭代中执行参数更新、梯度清零和学习率调度等关键操作，
     * 确保模型参数按照优化策略正确更新，同时维护训练过程的稳定性。
     * 
     * @param iter [参数说明]：当前迭代次数，用于控制训练流程和优化器状态管理。
     */
    void MCMC::step(int iter) {
        // 步骤1：训练迭代检查
        // 确保当前迭代在预定的训练迭代范围内
        if (iter < _params->iterations) {
            
            // 步骤2：优化器类型转换和参数更新
            // 将通用优化器指针转换为FusedAdam类型
            // FusedAdam是专门为高斯溅射优化的融合Adam优化器
            auto* fused_adam = dynamic_cast<FusedAdam*>(_optimizer.get());
            
            // 执行优化器的step操作，更新模型参数
            // 这一步使用计算出的梯度来更新所有可训练参数
            fused_adam->step(iter);
            
            // 步骤3：梯度清零
            // 清零所有参数的梯度，为下一次前向传播做准备
            // 第二个参数iter用于优化器的内部状态管理
            fused_adam->zero_grad(true, iter);
            
            // 步骤4：学习率调度
            // 更新学习率调度器，实现学习率的动态调整
            // 这通常包括学习率的衰减、周期性调整等策略
            _scheduler->step();
        }
    }

    /**
     * [功能描述]：初始化MCMC策略，设置训练环境、设备配置、优化器和学习率调度器。
     * 这个函数是MCMC训练策略的起点，负责将所有模型参数转移到CUDA设备、
     * 计算二项式系数、配置FusedAdam优化器参数组和设置指数学习率调度器。
     * 
     * @param optimParams [参数说明]：优化参数配置，包含各种学习率、迭代次数等训练设置。
     */
    void MCMC::initialize(const gs::param::OptimizationParameters& optimParams) {
        // 步骤1：保存优化参数
        // 将传入的优化参数保存到成员变量中，供后续训练使用
        _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

        // 步骤2：设备配置和参数转移
        const auto dev = torch::kCUDA;  // 设置目标设备为CUDA
        
        // 将所有高斯溅射参数转移到CUDA设备并启用梯度计算
        _splat_data.means() = _splat_data.means().to(dev).set_requires_grad(true);           // 位置参数
        _splat_data.scaling_raw() = _splat_data.scaling_raw().to(dev).set_requires_grad(true); // 缩放参数
        _splat_data.rotation_raw() = _splat_data.rotation_raw().to(dev).set_requires_grad(true); // 旋转参数
        _splat_data.opacity_raw() = _splat_data.opacity_raw().to(dev).set_requires_grad(true);   // 不透明度参数
        _splat_data.sh0() = _splat_data.sh0().to(dev).set_requires_grad(true);                 // 球谐函数0阶系数
        _splat_data.shN() = _splat_data.shN().to(dev).set_requires_grad(true);                 // 球谐函数高阶系数
        
        // 初始化密集化信息张量（当前为空，用于后续的动态调整）
        _splat_data._densification_info = torch::empty({0});

        // 步骤3：二项式系数计算和初始化
        const int n_max = 51;  // 设置二项式系数的最大维度
        // 创建二维零张量存储二项式系数
        _binoms = torch::zeros({n_max, n_max}, torch::kFloat32);
        auto binoms_accessor = _binoms.accessor<float, 2>();  // 获取张量访问器
        
        // 计算所有需要的二项式系数 C(n,k)
        for (int n = 0; n < n_max; ++n) {
            for (int k = 0; k <= n; ++k) {
                // 计算二项式系数 C(n,k) = n! / (k! * (n-k)!)
                float binom = 1.0f;
                for (int i = 0; i < k; ++i) {
                    // 使用递推公式：C(n,k) = C(n,k-1) * (n-k+1) / k
                    binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
                }
                binoms_accessor[n][k] = binom;  // 存储计算出的二项式系数
            }
        }
        // 将二项式系数张量转移到CUDA设备
        _binoms = _binoms.to(dev);

        // 步骤4：优化器配置
        using Options = FusedAdam::Options;  // 使用FusedAdam的选项类型
        std::vector<torch::optim::OptimizerParamGroup> groups;  // 参数组向量

        // 创建参数组的辅助函数
        // 为每个参数组设置学习率和优化器选项
        auto add_param_group = [&groups](const torch::Tensor& param, double lr) {
            auto options = std::make_unique<Options>(lr);  // 创建选项对象，设置学习率
            options->eps(1e-15).betas(std::make_tuple(0.9, 0.999));  // 设置epsilon和beta参数
            // 创建参数组，包含参数张量和选项
            groups.emplace_back(
                std::vector<torch::Tensor>{param},
                std::unique_ptr<torch::optim::OptimizerOptions>(std::move(options)));
        };

        // 为不同类型的参数创建参数组，设置不同的学习率
        add_param_group(_splat_data.means(), _params->means_lr * _splat_data.get_scene_scale());  // 位置参数：学习率乘以场景缩放
        add_param_group(_splat_data.sh0(), _params->shs_lr);                                      // 球谐函数0阶系数：标准学习率
        add_param_group(_splat_data.shN(), _params->shs_lr / 20.f);                              // 球谐函数高阶系数：降低的学习率（1/20）
        add_param_group(_splat_data.scaling_raw(), _params->scaling_lr);                          // 缩放参数：专用学习率
        add_param_group(_splat_data.rotation_raw(), _params->rotation_lr);                        // 旋转参数：专用学习率
        add_param_group(_splat_data.opacity_raw(), _params->opacity_lr);                          // 不透明度参数：专用学习率

        // 步骤5：全局优化器选项和优化器创建
        auto global_options = std::make_unique<Options>(0.f);  // 全局选项，学习率设为0（不使用）
        global_options->eps(1e-15);  // 设置全局epsilon值
        // 创建FusedAdam优化器，传入参数组和全局选项
        _optimizer = std::make_unique<FusedAdam>(std::move(groups), std::move(global_options));

        // 步骤6：学习率调度器配置
        // 计算指数衰减因子：gamma = (0.01)^(1/iterations)
        // 这确保在训练结束时学习率衰减到初始值的1%
        const double gamma = std::pow(0.01, 1.0 / _params->iterations);
        // 创建指数学习率调度器，传入优化器和衰减因子
        // 第三个参数0表示应用到所有参数组
        _scheduler = std::make_unique<ExponentialLR>(*_optimizer, gamma, 0);
    }

    bool MCMC::is_refining(int iter) const {
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0);
    }
} // namespace gs::training
