#include "core/default_strategy.hpp"
#include "Ops.h"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include "core/strategy.hpp"
#include <c10/cuda/CUDACachingAllocator.h>

/**
 * [功能描述]：DefaultStrategy类构造函数
 * @param splat_data：高斯散点数据对象，使用移动语义转移所有权
 * 
 * 该构造函数：
 * - 接收并存储高斯散点模型数据
 * - 使用移动语义避免昂贵的数据拷贝
 * - 为后续的训练策略初始化做准备
 */
DefaultStrategy::DefaultStrategy(gs::SplatData&& splat_data)
    : _splat_data(std::move(splat_data)) {  // 移动构造，转移splat_data的所有权
    // 构造函数体为空，所有初始化都通过成员初始化列表完成
}

/**
 * [功能描述]：初始化默认训练策略的所有组件
 * @param optimParams：优化参数配置，包含学习率、迭代次数等训练超参数
 * 
 * 初始化流程：
 * 1. 保存优化参数配置
 * 2. 初始化高斯参数（转移到GPU并启用梯度）
 * 3. 创建Adam优化器，为不同参数组设置不同学习率
 * 4. 创建指数衰减学习率调度器
 */
void DefaultStrategy::initialize(const gs::param::OptimizationParameters& optimParams) {
    // =============================================================================
    // 步骤1：保存优化参数配置
    // =============================================================================
    // 创建参数的深拷贝并存储为智能指针，确保配置在整个训练过程中保持不变
    _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

    // =============================================================================
    // 步骤2：初始化高斯散点参数
    // =============================================================================
    // 调用策略工具函数，将所有高斯参数：
    // - 转移到CUDA设备上进行GPU加速计算
    // - 启用梯度计算（requires_grad=true）为训练做准备
    strategy::initialize_gaussians(_splat_data);

    // =============================================================================
    // 步骤3：创建Adam优化器
    // =============================================================================
    // 使用策略工具函数创建分组优化器：
    // - 位置参数：学习率根据场景尺度调整
    // - 球谐系数：基础学习率和降低的高阶学习率
    // - 几何参数（缩放、旋转、不透明度）：各自独立的学习率
    _optimizer = strategy::create_optimizer(_splat_data, *_params);

    // =============================================================================
    // 步骤4：创建指数衰减学习率调度器
    // =============================================================================
    // 为位置参数组（索引0）创建指数衰减调度器：
    // - 在训练结束时学习率衰减到初始值的1%
    // - 使用平滑的指数衰减确保训练稳定性
    _scheduler = strategy::create_scheduler(*_params, _optimizer.get(), 0);
}

/**
 * [功能描述]：反向传播前的预处理函数
 * @param render_output：渲染输出对象，包含图像、Alpha通道和中间计算结果
 * 
 * 该函数在损失计算和反向传播之前调用，用于：
 * - 保留特定中间变量的梯度信息
 * - 为特殊的训练需求做准备（如密集化、剪枝等）
 * - 根据训练阶段动态调整梯度计算策略
 */
void DefaultStrategy::pre_backward(gs::RenderOutput& render_output) {
    // =============================================================================
    // 条件梯度保留：2D投影位置梯度
    // =============================================================================
    // 检查当前是否需要计算2D投影位置的梯度
    if (_key_for_gradient == "means2d") {
        // 保留means2d张量的梯度信息
        // means2d是3D高斯中心在2D图像平面上的投影位置
        // 这些梯度信息用于：
        // - 高斯分裂决策：梯度大的区域需要更多细节
        // - 高斯剪枝决策：梯度小的高斯可能不重要
        // - 自适应密集化：根据梯度分布动态调整高斯数量
        render_output.means2d.retain_grad();
    }
    // 注：retain_grad()确保中间张量的梯度在反向传播后不被释放
    // 这样可以在后续的优化步骤中访问这些梯度信息
}

/**
 * [功能描述]：更新训练状态，收集用于自适应密集化的统计信息
 * @param render_output：渲染输出对象，包含梯度信息和渲染统计数据
 * 
 * 该函数执行以下操作：
 * 1. 提取并处理梯度信息（主要是2D投影位置的梯度）
 * 2. 对梯度进行尺度归一化处理
 * 3. 初始化或更新累积统计张量
 * 4. 更新每个高斯的梯度累积、计数和半径信息
 * 
 * 这些统计信息用于后续的高斯分裂、剪枝和密集化决策
 */
void DefaultStrategy::update_state(gs::RenderOutput& render_output) {
    // =============================================================================
    // 步骤1：提取和验证梯度信息
    // =============================================================================
    torch::Tensor grads;
    if (_key_for_gradient == "means2d") {
        // 获取2D投影位置的梯度
        // means2d是3D高斯中心在图像平面上的投影位置 [N, 2]
        grads = _absgrad
                    ? render_output.means2d.grad().abs().clone()  // 使用梯度的绝对值
                    : render_output.means2d.grad().clone();       // 使用原始梯度值
        
        // 梯度数值稳定性检查
        // 检测并报告NaN或无穷大值，这些值会导致训练不稳定
        if (!torch::isfinite(grads).all().item<bool>()) {
            throw std::runtime_error("Gradient contains NaN or Inf values.");
        }
    } else {
        // 当前DefaultStrategy仅支持means2d梯度的更新
        throw std::runtime_error("Only means2d is supported for gradient updates in DefaultStrategy.");
    }

    // =============================================================================
    // 步骤2：梯度尺度归一化处理
    // =============================================================================
    
    // 计算相机数量（支持批量渲染）
    const size_t num_cameras = render_output.image.dim() == 4 ? render_output.image.size(0) : 1;
    
    // 计算尺度因子，将梯度归一化到图像尺寸
    // 除以2.0是因为图像坐标通常以中心为原点，范围是[-width/2, width/2]
    const float scale_x = render_output.width / 2.0f * num_cameras;   // X轴尺度因子
    const float scale_y = render_output.height / 2.0f * num_cameras;   // Y轴尺度因子
    
    // 对梯度进行尺度调整：
    // grads的形状为[N, 2]，其中第0维是x坐标梯度，第1维是y坐标梯度
    grads.select(-1, 0).mul_(scale_x);  // 调整x坐标梯度的尺度
    grads.select(-1, 1).mul_(scale_y);  // 调整y坐标梯度的尺度

    // =============================================================================
    // 步骤3：初始化状态张量（首次运行时）
    // =============================================================================
    
    const size_t num_gaussians = _splat_data.size();  // 获取当前高斯数量
    const c10::Device device = grads.device();        // 获取设备类型（CPU或GPU）
    
    // 初始化累积梯度张量
    if (!_grad2d.defined()) {
        _grad2d = torch::zeros(num_gaussians, torch::kFloat32).to(device);
    }
    
    // 初始化累积计数张量（记录每个高斯被渲染的次数）
    if (!_count.defined()) {
        _count = torch::zeros(num_gaussians, torch::kFloat32).to(device);
    }
    
    // 初始化最大半径张量（如果启用了基于尺度的细化停止条件）
    if (_params->stop_refine_scale2d > 0 && !_radii.defined()) {
        _radii = torch::zeros(num_gaussians, torch::kFloat32).to(device);
    }

    // =============================================================================
    // 步骤4：更新累积统计信息
    // =============================================================================
    
    torch::Tensor gaussian_ids;  // 有效高斯的索引
    torch::Tensor radii;         // 对应的半径值

    // grads的形状为[C, N, 2]，其中C是相机数量，N是高斯数量
    // 当前假设C = 1，render_output.radii的形状为[..., N]
    
    // 创建有效高斯的掩码：半径大于0表示该高斯在当前视图中可见
    const torch::Tensor valid_mask = render_output.radii > 0;  // [N] 布尔掩码
    
    // 获取所有可见高斯的索引
    gaussian_ids = valid_mask.nonzero().squeeze(-1);           // [nnz] 非零索引
    
    // 提取可见高斯对应的梯度和半径
    grads = grads.squeeze(0).index_select(0, gaussian_ids);    // [nnz, 2] 可见高斯的梯度
    radii = render_output.radii.index_select(0, gaussian_ids); // [nnz] 可见高斯的半径

    // =============================================================================
    // 步骤5：累积更新状态统计
    // =============================================================================
    
    // 累积每个高斯的梯度L2范数
    // norm(2, -1)计算每个梯度向量的L2范数，得到标量梯度强度
    _grad2d.index_add_(0, gaussian_ids, grads.norm(2, -1));
    
    // 累积每个高斯的渲染计数
    // 使用ones_like创建与gaussian_ids同形状的1张量进行累加
    _count.index_add_(0, gaussian_ids, torch::ones_like(gaussian_ids, torch::kFloat32));
    
    // 更新最大半径统计（如果启用）
    if (_params->stop_refine_scale2d > 0) {
        // 计算图像的最大尺寸，用于半径归一化
        const double max_wh = static_cast<double>(std::max(render_output.width, render_output.height));
        
        // 更新每个高斯的最大归一化半径
        // 使用torch::max确保半径只增不减，记录历史最大值
        _radii.index_put_({gaussian_ids},
                            torch::max(_radii.index_select(0, gaussian_ids), radii / max_wh));
    }
}

/**
 * [功能描述]：判断当前训练迭代是否应该执行高斯密集化操作
 * @param iter：当前训练迭代次数
 * @return 布尔值，true表示应该进行密集化，false表示跳过
 * 
 * 密集化操作包括：
 * - 高斯分裂（在高梯度区域增加更多细节）
 * - 高斯剪枝（移除不重要或过小的高斯）
 * - 透明度重置（定期清理低透明度高斯）
 * 
 * 该函数通过多重条件控制密集化的时机，确保：
 * 1. 训练初期有足够的稳定性
 * 2. 定期进行密集化以适应场景复杂度
 * 3. 在参数重置后给予缓冲时间
 */
bool DefaultStrategy::is_refining(int iter) const {
    return (iter > _params->start_refine &&                                        // 条件1：迭代超过开始阈值
            iter % _params->refine_every == 0 &&                                   // 条件2：达到密集化间隔
            iter % _params->reset_every >= _params->pause_refine_after_reset);     // 条件3：重置后缓冲期已过
    
    // =============================================================================
    // 条件解析：
    // =============================================================================
    
    // 条件1：iter > _params->start_refine
    // 目的：确保训练初期的稳定性
    // 原理：在训练开始阶段，高斯参数还不稳定，过早的密集化可能导致
    //       训练不收敛或产生不良的几何结构
    // 示例：如果start_refine = 500，则前500次迭代不进行密集化
    
    // 条件2：iter % _params->refine_every == 0
    // 目的：控制密集化的频率
    // 原理：密集化是计算密集型操作，不应该每次迭代都执行
    //       定期执行可以在性能和适应性之间取得平衡
    // 示例：如果refine_every = 100，则每100次迭代考虑一次密集化
    
    // 条件3：iter % _params->reset_every >= _params->pause_refine_after_reset
    // 目的：在参数重置后提供缓冲时间
    // 原理：训练过程中会定期重置某些参数（如透明度），重置后需要
    //       给高斯参数时间重新稳定，避免在不稳定状态下进行密集化
    // 示例：如果reset_every = 3000，pause_refine_after_reset = 200
    //       则在第3000、6000、9000...次迭代重置后的200次迭代内
    //       暂停密集化操作
    
    // =============================================================================
    // 综合效果：
    // =============================================================================
    // 这个函数实现了一个智能的密集化调度策略：
    // - 避免训练初期的不稳定性
    // - 定期适应场景复杂度变化
    // - 在参数重置后保持训练稳定性
    // - 平衡计算效率与渲染质量
}

/**
 * [功能描述]：复制指定的高斯散点以增加场景细节
 * @param is_duplicated：布尔掩码张量 [N]，标记需要复制的高斯（true表示复制）
 * 
 * 复制操作用于：
 * - 在高梯度区域增加更多细节表示
 * - 提高复杂场景的渲染质量
 * - 作为自适应密集化的重要组成部分
 * 
 * 该函数执行以下步骤：
 * 1. 提取需要复制的高斯索引
 * 2. 复制所有高斯参数（位置、颜色、几何属性等）
 * 3. 处理优化器状态以保持训练连续性
 * 4. 更新训练过程中的累积统计信息
 */
void DefaultStrategy::duplicate(const torch::Tensor is_duplicated) {
    // =============================================================================
    // 禁用梯度计算，因为这是几何操作而非训练步骤
    // =============================================================================
    torch::NoGradGuard no_grad;

    // =============================================================================
    // 步骤1：提取需要复制的高斯索引
    // =============================================================================
    const c10::Device device = is_duplicated.device();              // 获取设备类型（CPU或GPU）
    const torch::Tensor sampled_idxs = is_duplicated.nonzero().squeeze(-1);  // 获取所有为true的索引 [M]

    // =============================================================================
    // 步骤2：定义参数复制函数
    // =============================================================================
    // 该lambda函数定义如何复制每种类型的参数
    const auto param_fn = [&sampled_idxs](const int i, const torch::Tensor param) {
        // 从原始参数中选择需要复制的高斯
        const torch::Tensor new_param = param.index_select(0, sampled_idxs);
        
        // 将新复制的参数与原始参数连接，形状从[N, ...]变为[N+M, ...]
        // 其中M是复制的高斯数量
        return torch::cat({param, new_param}).set_requires_grad(param.requires_grad());
    };

    // =============================================================================
    // 步骤3：定义优化器状态更新函数
    // =============================================================================
    // 该lambda函数处理Adam优化器的动量状态，确保新复制的高斯有正确的优化状态
    const auto optimizer_fn = [&sampled_idxs](torch::optim::OptimizerParamState& state,
                                                const torch::Tensor full_param)
        -> std::unique_ptr<torch::optim::OptimizerParamState> {
        
        // 计算新参数的形状（增加复制的高斯数量）
        auto new_shape = full_param.sizes().vec();
        new_shape[0] = sampled_idxs.size(0);  // 设置为复制的高斯数量
        
        // 处理Adam优化器状态
        if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&state)) {
            // =============================================================================
            // Adam优化器状态包含：
            // - exp_avg: 梯度的指数移动平均（一阶动量）
            // - exp_avg_sq: 梯度平方的指数移动平均（二阶动量）
            // - max_exp_avg_sq: 最大二阶动量（可选，用于AMSGrad）
            // =============================================================================
            
            // 为新复制的高斯创建零初始化的状态
            // 这确保新高斯从干净的状态开始优化
            auto zeros_to_add = torch::zeros(new_shape, adam_state->exp_avg().options());
            
            // 扩展一阶动量：原始状态 + 零初始化的新状态
            auto new_exp_avg = torch::cat({adam_state->exp_avg(), zeros_to_add}, 0);
            
            // 扩展二阶动量：原始状态 + 零初始化的新状态
            auto new_exp_avg_sq = torch::cat({adam_state->exp_avg_sq(), zeros_to_add}, 0);

            // 创建新的Adam状态对象
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(adam_state->step());        // 保持相同的步数
            new_state->exp_avg(new_exp_avg);            // 设置扩展后的一阶动量
            new_state->exp_avg_sq(new_exp_avg_sq);      // 设置扩展后的二阶动量
            
            // 处理可选的最大二阶动量（AMSGrad变体）
            if (adam_state->max_exp_avg_sq().defined()) {
                auto new_max_exp_avg_sq = torch::cat({adam_state->max_exp_avg_sq(), zeros_to_add}, 0);
                new_state->max_exp_avg_sq(new_max_exp_avg_sq);
            }
            return new_state;
        }
        return nullptr;  // 不支持的优化器类型
    };

    // =============================================================================
    // 步骤4：执行参数和优化器状态的更新
    // =============================================================================
    // 调用策略工具函数，统一处理所有参数的复制和优化器状态更新
    // 这包括：位置、球谐系数、缩放、旋转、不透明度等所有高斯参数
    strategy::update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);

    // =============================================================================
    // 步骤5：更新训练过程中的累积统计信息
    // =============================================================================
    const int num_new_gaussians = sampled_idxs.size(0);  // 新增的高斯数量
    
    // 更新累积梯度统计
    if (_grad2d.defined()) {
        // 复制选定高斯的梯度统计，新高斯继承被复制高斯的梯度历史
        _grad2d = torch::cat({_grad2d, _grad2d.index_select(0, sampled_idxs)});
    }
    
    // 更新最大半径统计
    if (_radii.defined()) {
        // 复制选定高斯的半径统计，新高斯继承被复制高斯的半径历史
        _radii = torch::cat({_radii, _radii.index_select(0, sampled_idxs)});
    }
    
    // 更新渲染计数统计
    if (_count.defined()) {
        // 复制选定高斯的计数统计，新高斯继承被复制高斯的渲染历史
        _count = torch::cat({_count, _count.index_select(0, sampled_idxs)});
    }
    
    // =============================================================================
    // 复制操作完成后的状态：
    // - 高斯数量从N增加到N+M（M为复制的数量）
    // - 所有参数张量的第0维都相应扩展
    // - 优化器状态正确初始化，确保训练连续性
    // - 累积统计信息保持一致，支持后续的密集化决策
    // =============================================================================
}

/**
 * [功能描述]：分裂指定的高斯散点以增加局部细节表示
 * @param is_split：布尔掩码张量 [N]，标记需要分裂的高斯（true表示分裂）
 * 
 * 分裂操作用于：
 * - 在高梯度区域将大高斯分解为多个小高斯
 * - 提高复杂场景区域的表示精度
 * - 自适应地增加模型在重要区域的容量
 * 
 * 分裂策略：
 * - 每个被分裂的高斯产生2个新的子高斯
 * - 子高斯围绕原高斯的主轴方向进行位置偏移
 * - 子高斯的尺寸被适当缩小以避免过度覆盖
 */
void DefaultStrategy::split(const torch::Tensor is_split) {
    // =============================================================================
    // 禁用梯度计算，因为这是几何操作而非训练步骤
    // =============================================================================
    torch::NoGradGuard no_grad;

    // =============================================================================
    // 步骤1：提取分裂和保留的高斯索引
    // =============================================================================
    const c10::Device device = is_split.device();                                  // 获取设备类型
    const torch::Tensor sampled_idxs = is_split.nonzero().squeeze(-1);            // 需要分裂的高斯索引 [M]
    const torch::Tensor rest_idxs = is_split.logical_not().nonzero().squeeze(-1); // 保留不变的高斯索引 [N-M]

    // =============================================================================
    // 步骤2：获取分裂高斯的几何参数
    // =============================================================================
    // 获取需要分裂的高斯的缩放参数（已经过指数激活）
    const torch::Tensor sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs); // [M, 3]
    
    // 获取需要分裂的高斯的旋转四元数（已标准化）
    const torch::Tensor sampled_quats = _splat_data.get_rotation().index_select(0, sampled_idxs);  // [M, 4]
    
    // 将四元数转换为旋转矩阵，用于后续的空间变换
    const torch::Tensor rotmats = gsplat::quats_to_rotmats(sampled_quats); // [M, 3, 3]

    // =============================================================================
    // 步骤3：生成子高斯的空间分布
    // =============================================================================
    const auto num_split_gaussians = sampled_idxs.size(0);  // 需要分裂的高斯数量
    const auto split_size = 2;                              // 每个高斯分裂成2个子高斯
    
    // 使用Einstein求和约定生成子高斯的位置偏移
    // 公式：samples = rotmats @ (sampled_scales * random_vectors)
    // 这确保了子高斯沿着原高斯的主轴方向分布
    const torch::Tensor samples = torch::einsum( // [split_size, M, 3]
        "nij,nj,bnj->bni",
        {rotmats,                                                                   // 旋转矩阵 [M, 3, 3]
            sampled_scales,                                                            // 缩放参数 [M, 3]
            torch::randn({split_size, num_split_gaussians, 3}, sampled_quats.options().device(device))}); // 随机向量 [2, M, 3]

    // =============================================================================
    // 步骤4：定义参数更新函数
    // =============================================================================
    // 该lambda函数定义如何处理每种类型的参数在分裂操作中的变换
    const auto param_fn = [this, &sampled_idxs, &rest_idxs, &samples, &split_size, &sampled_scales](const int i, const torch::Tensor param) {
        // 为重复操作创建维度向量
        std::vector<int64_t> repeats(param.dim(), 1);
        repeats[0] = split_size;  // 在第0维重复split_size次

        // 获取被分裂高斯的原始参数
        const torch::Tensor sampled_param = param.index_select(0, sampled_idxs);
        torch::Tensor split_param;
        
        if (i == 0) {                                                               // 处理位置参数 (means)
            // 将原位置与生成的偏移相加，得到子高斯的新位置
            // unsqueeze(0)将形状从[M, 3]变为[1, M, 3]，便于与samples [2, M, 3]相加
            split_param = (sampled_param.unsqueeze(0) + samples).reshape({-1, 3});  // [split_size * M, 3]
            
        } else if (i == 3) {                                                        // 处理缩放参数 (scaling)
            // 缩小子高斯的尺寸，除以1.6是经验值，防止子高斯重叠过多
            // 使用log是因为缩放参数在对数空间中存储
            split_param = torch::log(sampled_scales / 1.6).repeat({split_size, 1}); // [split_size * M, 3]
            
        } else if (i == 5 && _params->revised_opacity) {                            // 处理不透明度参数 (opacity)
            // 使用修正的不透明度分裂策略
            // 公式：new_opacity = 1 - sqrt(1 - old_opacity)
            // 这确保了子高斯的总体不透明度贡献接近原高斯
            const torch::Tensor new_opacities = 1.0 - torch::sqrt(1.0 - torch::sigmoid(sampled_param));
            split_param = torch::logit(new_opacities).repeat(repeats); // [split_size * M, 1]
            
        } else {
            // 其他参数（旋转、球谐系数等）直接复制
            split_param = sampled_param.repeat(repeats);
        }

        // 组合保留的高斯参数和新分裂的参数
        const torch::Tensor rest_param = param.index_select(0, rest_idxs);
        return torch::cat({rest_param, split_param}, 0).set_requires_grad(param.requires_grad());
    };

    // =============================================================================
    // 步骤5：定义优化器状态更新函数
    // =============================================================================
    // 处理Adam优化器的动量状态，确保新分裂的子高斯有正确的优化状态
    const auto optimizer_fn = [&sampled_idxs, &rest_idxs, &split_size](
                                    torch::optim::OptimizerParamState& state,
                                    const torch::Tensor full_param)
        -> std::unique_ptr<torch::optim::OptimizerParamState> {
        
        // 计算新分裂子高斯的参数形状
        auto zero_shape = full_param.sizes().vec();
        zero_shape[0] = sampled_idxs.size(0) * split_size;  // 分裂产生的新参数数量
        
        if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&state)) {
            // =============================================================================
            // 处理Adam优化器状态的分裂
            // 策略：保留未分裂高斯的状态，为新分裂的子高斯创建零初始化状态
            // =============================================================================
            
            // 提取保留高斯的一阶动量
            auto rest_exp_avg = adam_state->exp_avg().index_select(0, rest_idxs);
            // 提取保留高斯的二阶动量
            auto rest_exp_avg_sq = adam_state->exp_avg_sq().index_select(0, rest_idxs);

            // 为新分裂的子高斯创建零初始化状态
            auto zeros_to_add = torch::zeros(zero_shape, adam_state->exp_avg().options());
            
            // 组合保留状态和新初始化状态
            auto new_exp_avg = torch::cat({rest_exp_avg, zeros_to_add}, 0);
            auto new_exp_avg_sq = torch::cat({rest_exp_avg_sq, zeros_to_add}, 0);

            // 创建新的Adam状态对象
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(adam_state->step());        // 保持相同的优化步数
            new_state->exp_avg(new_exp_avg);            // 设置新的一阶动量
            new_state->exp_avg_sq(new_exp_avg_sq);      // 设置新的二阶动量
            
            // 处理AMSGrad的最大二阶动量（如果存在）
            if (adam_state->max_exp_avg_sq().defined()) {
                auto rest_max_exp_avg_sq = adam_state->max_exp_avg_sq().index_select(0, rest_idxs);
                auto new_max_exp_avg_sq = torch::cat({rest_max_exp_avg_sq, zeros_to_add}, 0);
                new_state->max_exp_avg_sq(new_max_exp_avg_sq);
            }
            return new_state;
        }
        return nullptr;  // 不支持的优化器类型
    };

    // =============================================================================
    // 步骤6：执行参数和优化器状态的更新
    // =============================================================================
    // 调用策略工具函数，统一处理所有参数的分裂和优化器状态更新
    strategy::update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);

    // =============================================================================
    // 步骤7：更新训练过程中的累积统计信息
    // =============================================================================
    
    // 辅助函数：创建重复维度向量
    const auto make_repeats = [&split_size](const at::Tensor& t) {
        std::vector<int64_t> v(t.dim(), 1);
        v[0] = split_size;  // 在第0维重复split_size次
        return v;
    };
    
    // 更新累积梯度统计：保留未分裂高斯的统计，复制被分裂高斯的统计给子高斯
    if (_grad2d.defined()) {
        _grad2d = torch::cat({_grad2d.index_select(0, rest_idxs),                          // 保留的统计
                                _grad2d.index_select(0, sampled_idxs).repeat(make_repeats(_grad2d))}); // 复制的统计
    }
    
    // 更新最大半径统计：子高斯继承父高斯的半径历史
    if (_radii.defined()) {
        _radii = torch::cat({_radii.index_select(0, rest_idxs),                            // 保留的统计
                                _radii.index_select(0, sampled_idxs).repeat(make_repeats(_radii))}); // 复制的统计
    }
    
    // 更新渲染计数统计：子高斯继承父高斯的渲染历史
    if (_count.defined()) {
        _count = torch::cat({_count.index_select(0, rest_idxs),                            // 保留的统计
                                _count.index_select(0, sampled_idxs).repeat(make_repeats(_count))}); // 复制的统计
    }
    
    // =============================================================================
    // 分裂操作完成后的状态：
    // - 高斯数量从N变为(N-M)+2*M = N+M（M为分裂的高斯数量）
    // - 每个被分裂的高斯被2个子高斯替代
    // - 子高斯位置围绕原高斯的主轴分布
    // - 子高斯尺寸适当缩小，不透明度经过修正
    // - 优化器状态正确更新，确保训练连续性
    // - 累积统计信息保持一致性
    // =============================================================================
}

/**
 * [功能描述]：执行高斯密集化操作，通过复制和分裂增加高斯数量
 * @param iter：当前训练迭代次数
 * @return 元组 (复制的高斯数量, 分裂的高斯数量)
 * 
 * 密集化策略：
 * - 复制：对于小尺寸但高梯度的高斯，通过复制增加密度
 * - 分裂：对于大尺寸且高梯度的高斯，通过分裂增加细节
 * - 基于2D半径：过大的2D投影半径也会触发分裂
 * 
 * 该函数实现了自适应场景重建的核心算法，根据训练统计
 * 智能地决定在哪里增加表示容量
 */
std::tuple<int64_t, int64_t> DefaultStrategy::grow_gs(int iter) {
    // =============================================================================
    // 禁用梯度计算，因为这是几何操作而非训练步骤
    // =============================================================================
    torch::NoGradGuard no_grad;

    // =============================================================================
    // 步骤1：计算平均梯度强度
    // =============================================================================
    // 计算每个高斯的平均梯度强度
    // _grad2d是累积的梯度L2范数，_count是渲染次数
    // clamp_min(1)避免除零错误，确保至少除以1
    const torch::Tensor grads = _grad2d / _count.clamp_min(1);  // [N] 平均梯度强度
    const c10::Device device = grads.device();

    // =============================================================================
    // 步骤2：识别需要复制的高斯（小且高梯度）
    // =============================================================================
    // 识别梯度超过阈值的高斯
    const torch::Tensor is_grad_high = grads > _params->grad_threshold;  // [N] 布尔掩码
    
    // 获取每个高斯在三个轴向上的最大缩放值
    // get_scaling()返回激活后的缩放参数，max(-1)在最后一维上取最大值
    const auto max_values = std::get<0>(torch::max(_splat_data.get_scaling(), -1));  // [N]
    
    // 判断高斯是否足够小（需要通过复制而非分裂来增加密度）
    // grow_scale3d是3D尺寸阈值，get_scene_scale()提供场景相关的尺度归一化
    const torch::Tensor is_small = max_values <= _params->grow_scale3d * _splat_data.get_scene_scale();  // [N]
    
    // 复制条件：同时满足高梯度和小尺寸
    // 逻辑：小高斯无法通过分裂有效增加细节，需要通过复制增加密度
    const torch::Tensor is_duplicated = is_grad_high & is_small;  // [N] 布尔掩码
    const int64_t num_duplicates = is_duplicated.sum().item<int64_t>();  // 需要复制的高斯数量

    // =============================================================================
    // 步骤3：识别需要分裂的高斯（大且高梯度，或2D半径过大）
    // =============================================================================
    // 识别尺寸较大的高斯（与is_small相反）
    const torch::Tensor is_large = ~is_small;  // [N] 布尔掩码
    
    // 基础分裂条件：高梯度且大尺寸
    // 逻辑：大高斯可以通过分裂成多个小高斯来增加局部细节
    torch::Tensor is_split = is_grad_high & is_large;  // [N] 布尔掩码
    
    // 附加分裂条件：2D投影半径过大（仅在特定迭代范围内）
    if (iter < _params->stop_refine_scale2d) {
        // _radii记录了每个高斯的最大2D投影半径
        // grow_scale2d是2D半径阈值，超过此值需要分裂以避免过度覆盖
        is_split |= _radii > _params->grow_scale2d;
    }
    const int64_t num_split = is_split.sum().item<int64_t>();  // 需要分裂的高斯数量

    // =============================================================================
    // 步骤4：执行复制操作
    // =============================================================================
    // 首先执行复制操作，为小高斯增加密度
    if (num_duplicates > 0) {
        duplicate(is_duplicated);
        // 复制后，高斯总数从N增加到N + num_duplicates
    }

    // =============================================================================
    // 步骤5：更新分裂掩码并执行分裂操作
    // =============================================================================
    // 重要：新复制的高斯不应该被立即分裂
    // 因为它们刚刚被创建，还没有足够的训练统计来支持分裂决策
    is_split = torch::cat({is_split,  // 原有高斯的分裂掩码 [N]
                            torch::zeros(num_duplicates, c10::TensorOptions().dtype(torch::kBool).device(device))}); // 新复制高斯的掩码（全为false）[num_duplicates]
    // 现在is_split的形状为[N + num_duplicates]，对应复制后的高斯总数
    
    if (num_split > 0) {
        split(is_split);
        // 分裂后，高斯总数进一步增加
        // 每个被分裂的高斯被替换为2个子高斯
        // 所以总数从(N + num_duplicates)变为(N + num_duplicates - num_split + 2*num_split)
        // = N + num_duplicates + num_split
    }

    // =============================================================================
    // 返回操作统计信息
    // =============================================================================
    return {num_duplicates, num_split};
    
    // =============================================================================
    // 密集化操作完成后的效果：
    // - 在高梯度区域增加了表示容量
    // - 小高斯通过复制增加密度
    // - 大高斯通过分裂增加细节
    // - 过大的2D投影被分解为更合适的尺寸
    // - 总高斯数量增加了(num_duplicates + num_split)个
    // - 场景的表示能力在重要区域得到提升
    // =============================================================================
}

/**
 * [功能描述]：移除指定的高斯散点以减少模型复杂度
 * @param is_prune：布尔掩码张量 [N]，标记需要移除的高斯（true表示移除）
 * 
 * 剪枝操作用于：
 * - 移除不重要或冗余的高斯散点
 * - 减少模型复杂度和计算开销
 * - 防止过度拟合和提高泛化能力
 * - 清理训练过程中产生的低质量高斯
 * 
 * 剪枝策略通常基于：
 * - 低不透明度（对最终渲染贡献很小）
 * - 小尺寸（在屏幕上几乎不可见）
 * - 低梯度（训练过程中变化很少）
 * - 位置偏离（远离主要场景区域）
 */
void DefaultStrategy::remove(const torch::Tensor is_prune) {
    // =============================================================================
    // 禁用梯度计算，因为这是几何操作而非训练步骤
    // =============================================================================
    torch::NoGradGuard no_grad;

    // =============================================================================
    // 步骤1：获取需要保留的高斯索引
    // =============================================================================
    // 取is_prune的逻辑非，得到需要保留的高斯掩码，然后获取其索引
    // 逻辑：is_prune中为true的位置表示要移除，为false的位置表示要保留
    const torch::Tensor sampled_idxs = is_prune.logical_not().nonzero().squeeze(-1);  // [M] 保留的高斯索引

    // =============================================================================
    // 步骤2：定义参数裁剪函数
    // =============================================================================
    // 该lambda函数定义如何从每种类型的参数中移除不需要的高斯
    const auto param_fn = [&sampled_idxs](const int i, const torch::Tensor param) {
        // 仅选择需要保留的高斯对应的参数
        // index_select(0, sampled_idxs)在第0维（高斯维度）上选择指定索引的元素
        // 结果张量形状从[N, ...]缩减为[M, ...]，其中M < N
        return param.index_select(0, sampled_idxs).set_requires_grad(param.requires_grad());
    };

    // =============================================================================
    // 步骤3：定义优化器状态裁剪函数
    // =============================================================================
    // 该lambda函数处理Adam优化器状态的裁剪，确保优化器状态与参数保持一致
    const auto optimizer_fn = [&sampled_idxs](
                                    torch::optim::OptimizerParamState& state,
                                    const torch::Tensor new_param)
        -> std::unique_ptr<torch::optim::OptimizerParamState> {
        
        if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&state)) {
            // =============================================================================
            // 处理Adam优化器状态的裁剪
            // 策略：仅保留对应于保留高斯的优化器状态
            // =============================================================================
            
            // 从原始Adam状态中选择保留高斯对应的一阶动量
            auto new_exp_avg = adam_state->exp_avg().index_select(0, sampled_idxs);
            
            // 从原始Adam状态中选择保留高斯对应的二阶动量
            auto new_exp_avg_sq = adam_state->exp_avg_sq().index_select(0, sampled_idxs);

            // 创建新的Adam状态对象
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(adam_state->step());        // 保持相同的优化步数
            new_state->exp_avg(new_exp_avg);            // 设置裁剪后的一阶动量
            new_state->exp_avg_sq(new_exp_avg_sq);      // 设置裁剪后的二阶动量
            
            // 处理AMSGrad的最大二阶动量（如果存在）
            if (adam_state->max_exp_avg_sq().defined()) {
                auto new_max_exp_avg_sq = adam_state->max_exp_avg_sq().index_select(0, sampled_idxs);
                new_state->max_exp_avg_sq(new_max_exp_avg_sq);
            }
            return new_state;
        }
        return nullptr;  // 不支持的优化器类型
    };

    // =============================================================================
    // 步骤4：执行参数和优化器状态的更新
    // =============================================================================
    // 调用策略工具函数，统一处理所有参数的裁剪和优化器状态更新
    // 这包括：位置、球谐系数、缩放、旋转、不透明度等所有高斯参数
    strategy::update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);

    // =============================================================================
    // 步骤5：更新训练过程中的累积统计信息
    // =============================================================================
    
    // 裁剪累积梯度统计：仅保留未被移除高斯的梯度历史
    if (_grad2d.defined()) {
        _grad2d = _grad2d.index_select(0, sampled_idxs);
    }
    
    // 裁剪最大半径统计：仅保留未被移除高斯的半径历史
    if (_radii.defined()) {
        _radii = _radii.index_select(0, sampled_idxs);
    }
    
    // 裁剪渲染计数统计：仅保留未被移除高斯的计数历史
    if (_count.defined()) {
        _count = _count.index_select(0, sampled_idxs);
    }
    
    // =============================================================================
    // 剪枝操作完成后的状态：
    // - 高斯数量从N减少到M（M为保留的高斯数量）
    // - 所有参数张量的第0维都相应缩减
    // - 优化器状态正确裁剪，保持与新参数的对应关系
    // - 累积统计信息保持一致性，移除了被剪枝高斯的历史
    // - 模型复杂度降低，计算效率提升
    // - 内存使用量减少
    // =============================================================================
}

/**
 * [功能描述]：执行高斯散点剪枝操作，移除不重要或有害的高斯
 * @param iter：当前训练迭代次数
 * @return 被剪枝的高斯数量
 * 
 * 剪枝策略：
 * - 基础剪枝：移除不透明度过低的高斯（对渲染贡献很小）
 * - 尺寸剪枝：移除过大的高斯（可能产生伪影或过度模糊）
 * - 2D半径剪枝：移除屏幕投影过大的高斯（影响渲染效率）
 * 
 * 剪枝是自适应密集化的重要组成部分，与grow_gs形成平衡：
 * - grow_gs在重要区域增加高斯
 * - prune_gs移除不重要或有害的高斯
 */
int64_t DefaultStrategy::prune_gs(int iter) {
    // =============================================================================
    // 禁用梯度计算，因为这是几何操作而非训练步骤
    // =============================================================================
    torch::NoGradGuard no_grad;

    // =============================================================================
    // 步骤1：基础剪枝条件 - 低不透明度剪枝
    // =============================================================================
    // 识别不透明度过低的高斯
    // get_opacity()返回经过sigmoid激活的不透明度值，范围[0, 1]
    // prune_opacity是不透明度阈值，通常设置为较小值（如0.005）
    // 逻辑：不透明度太低的高斯对最终渲染几乎没有贡献，可以安全移除
    torch::Tensor is_prune = _splat_data.get_opacity() < _params->prune_opacity;  // [N] 布尔掩码

    // =============================================================================
    // 步骤2：尺寸相关剪枝条件（仅在训练进行一段时间后启用）
    // =============================================================================
    if (iter > _params->reset_every) {
        // 等待一定训练时间后再启用尺寸剪枝
        // 原因：训练初期高斯尺寸可能不稳定，过早剪枝可能移除重要高斯
        
        // =========================================================================
        // 3D尺寸剪枝：移除空间尺寸过大的高斯
        // =========================================================================
        // 获取每个高斯在三个轴向上的最大缩放值
        const auto max_values = std::get<0>(torch::max(_splat_data.get_scaling(), -1));  // [N]
        
        // 判断高斯是否在3D空间中过大
        // prune_scale3d是3D尺寸上限阈值
        // get_scene_scale()提供场景相关的尺度归一化
        // 逻辑：过大的3D高斯可能产生过度模糊或覆盖过多细节
        torch::Tensor is_too_big = max_values > _params->prune_scale3d * _splat_data.get_scene_scale();  // [N]

        // =========================================================================
        // 2D投影半径剪枝：移除屏幕投影过大的高斯（有时间限制）
        // =========================================================================
        if (iter < _params->stop_refine_scale2d) {
            // 仅在特定训练阶段检查2D投影半径
            // _radii记录了每个高斯的最大2D投影半径（归一化到屏幕尺寸）
            // prune_scale2d是2D投影半径上限阈值
            // 逻辑：屏幕投影过大的高斯影响渲染效率，且可能产生不自然的模糊
            is_too_big |= _radii > _params->prune_scale2d;
        }

        // 将尺寸剪枝条件加入总的剪枝掩码
        is_prune |= is_too_big;
    }

    // =============================================================================
    // 步骤3：执行剪枝操作
    // =============================================================================
    // 统计需要剪枝的高斯数量
    const int64_t num_prunes = is_prune.sum().item<int64_t>();
    
    if (num_prunes > 0) {
        // 调用remove函数执行实际的剪枝操作
        // remove函数会：
        // 1. 移除标记的高斯参数
        // 2. 更新优化器状态
        // 3. 清理相关统计信息
        remove(is_prune);
    }
    
    // 返回实际剪枝的高斯数量，用于训练日志和统计
    return num_prunes;
    
    // =============================================================================
    // 剪枝策略的设计理念：
    // =============================================================================
    // 1. 渐进式剪枝：从简单的不透明度剪枝开始，逐步增加复杂条件
    // 2. 时机控制：避免在训练初期过度剪枝，给高斯参数足够的稳定时间
    // 3. 多维度考量：结合不透明度、3D尺寸和2D投影等多个指标
    // 4. 场景自适应：使用场景尺度进行相对判断，适应不同场景
    // 5. 效率优化：移除对渲染效率有害的高斯（如过大投影）
    // 
    // =============================================================================
    // 与密集化的协同作用：
    // =============================================================================
    // - grow_gs：基于梯度在重要区域增加高斯
    // - prune_gs：基于质量和效率移除不良高斯
    // - 两者结合实现动态的模型复杂度管理
    // - 确保高斯数量在合理范围内，避免无限增长
    // - 提高整体渲染质量和训练效率
}

/**
 * [功能描述]：重置高斯散点的不透明度参数
 * 
 * 不透明度重置的作用：
 * - 防止高斯不透明度无限增长导致过度饱和
 * - 清理训练过程中累积的优化偏差
 * - 为后续的剪枝操作提供更清晰的判断基础
 * - 重新激活可能被"冻结"的高斯参数
 * 
 * 重置策略：
 * - 将过高的不透明度限制到合理范围
 * - 重置对应的Adam优化器动量状态
 * - 给高斯参数一个"重新开始"的机会
 */
void DefaultStrategy::reset_opacity() {
    // =============================================================================
    // 禁用梯度计算，因为这是参数重置操作而非训练步骤
    // =============================================================================
    torch::NoGradGuard no_grad;

    // =============================================================================
    // 设置不透明度重置阈值
    // =============================================================================
    // 使用剪枝阈值的2倍作为重置阈值
    // 逻辑：给高斯一些"生存空间"，避免重置后立即被剪枝
    const auto threshold = 2.0f * _params->prune_opacity;

    // =============================================================================
    // 定义参数重置函数
    // =============================================================================
    // 该lambda函数专门处理不透明度参数的重置
    const auto param_fn = [&threshold](const int i, const torch::Tensor param) {
        if (i == 5) {  // 参数索引5对应不透明度参数
            // 将不透明度参数限制在阈值以下
            // param是logit空间的原始不透明度值
            // torch::logit(threshold)将阈值转换到logit空间
            // clamp_max确保所有值不超过logit(threshold)
            const torch::Tensor new_opacities = torch::clamp_max(
                param,                                           // 原始不透明度参数（logit空间）
                torch::logit(torch::tensor(threshold)).item()); // 阈值的logit值
            
            return new_opacities.set_requires_grad(param.requires_grad());
        } else {
            // 这个函数只应该被调用来重置不透明度参数
            throw std::runtime_error("Invalid parameter index for reset_opacity: " + std::to_string(i));
        }
    };

    // =============================================================================
    // 定义优化器状态重置函数
    // =============================================================================
    // 该lambda函数重置Adam优化器的动量状态
    const auto optimizer_fn = [](torch::optim::OptimizerParamState& state,
                                    const torch::Tensor new_param)
        -> std::unique_ptr<torch::optim::OptimizerParamState> {
        
        if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&state)) {
            // =============================================================================
            // 重置Adam优化器状态
            // 策略：将所有动量项重置为零，给参数优化一个全新的开始
            // =============================================================================
            
            // 将一阶动量（梯度的指数移动平均）重置为零
            auto new_exp_avg = torch::zeros_like(adam_state->exp_avg());
            
            // 将二阶动量（梯度平方的指数移动平均）重置为零
            auto new_exp_avg_sq = torch::zeros_like(adam_state->exp_avg_sq());

            // 创建新的Adam状态对象
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(adam_state->step());        // 保持相同的优化步数
            new_state->exp_avg(new_exp_avg);            // 重置一阶动量
            new_state->exp_avg_sq(new_exp_avg_sq);      // 重置二阶动量
            
            // 处理AMSGrad的最大二阶动量（如果存在）
            if (adam_state->max_exp_avg_sq().defined()) {
                auto new_max_exp_avg_sq = torch::zeros_like(adam_state->max_exp_avg_sq());
                new_state->max_exp_avg_sq(new_max_exp_avg_sq);
            }
            return new_state;
        }

        return nullptr;  // 不支持的优化器类型
    };

    // =============================================================================
    // 执行不透明度参数和优化器状态的重置
    // =============================================================================
    // 调用策略工具函数，仅更新参数索引5（不透明度）
    strategy::update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data, {5});
}

/**
 * [功能描述]：反向传播后的处理函数，执行各种训练后操作
 * @param iter：当前训练迭代次数
 * @param render_output：渲染输出对象，包含梯度和统计信息
 * 
 * 该函数协调整个自适应训练过程的各个组件：
 * - 球谐函数阶数的渐进增加
 * - 训练状态的更新和统计
 * - 自适应密集化操作（增长和剪枝）
 * - 周期性的不透明度重置
 * - GPU内存管理和清理
 */
void DefaultStrategy::post_backward(int iter, gs::RenderOutput& render_output) {
    // =============================================================================
    // 球谐函数阶数的渐进增加
    // =============================================================================
    torch::NoGradGuard no_grad;  // 禁用梯度计算
    
    // 每隔指定间隔增加球谐函数的阶数
    // 策略：从简单的常数颜色逐步过渡到复杂的方向性颜色表示
    // 这样可以稳定训练过程，避免一开始就使用复杂的颜色模型
    if (iter % _params->sh_degree_interval == 0) {
        _splat_data.increment_sh_degree();
    }

    // =============================================================================
    // 检查是否停止细化操作
    // =============================================================================
    // 如果超过了停止细化的迭代次数，则不再执行后续的自适应操作
    if (iter >= _params->stop_refine) {
        return;  // 提前退出，仅保留球谐函数阶数的更新
    }

    // =============================================================================
    // 更新训练状态统计
    // =============================================================================
    // 收集当前训练步骤的梯度和渲染统计信息
    // 这些信息用于后续的密集化决策
    update_state(render_output);

    // =============================================================================
    // 自适应密集化操作
    // =============================================================================
    // 检查是否应该执行密集化（基于迭代次数和重置状态）
    if (is_refining(iter)) {
        // 执行高斯增长操作（复制和分裂）
        const auto [num_duplicates, num_splits] = grow_gs(iter);
        
        // 执行高斯剪枝操作
        const auto num_prunes = prune_gs(iter);

        // =============================================================================
        // 重置统计信息，为下一个密集化周期做准备
        // =============================================================================
        // 清零累积梯度统计，重新开始收集
        _grad2d.zero_();
        
        // 清零渲染计数统计
        _count.zero_();
        
        // 清零最大半径统计（如果启用）
        if (_params->stop_refine_scale2d > 0) {
            _radii.zero_();
        }

        // =============================================================================
        // GPU内存清理
        // =============================================================================
        // 清空CUDA缓存分配器，释放不再使用的GPU内存
        // 这对于密集化后的内存管理很重要，因为高斯数量可能发生显著变化
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // =============================================================================
    // 周期性不透明度重置
    // =============================================================================
    // 每隔一定迭代次数重置不透明度参数
    // 这有助于防止不透明度饱和并为剪枝提供更好的基础
    if (iter % _params->reset_every == 0 && iter > 0) {
        reset_opacity();
    }
}

/**
 * [功能描述]：执行单步优化操作
 * @param iter：当前训练迭代次数
 * 
 * 该函数执行标准的梯度下降优化步骤：
 * - 更新模型参数
 * - 清零梯度缓存
 * - 更新学习率
 * 
 * 这是训练循环中的核心优化步骤，与PyTorch的标准训练流程一致
 */
void DefaultStrategy::step(int iter) {
    // =============================================================================
    // 检查是否还在训练范围内
    // =============================================================================
    // 仅在指定的迭代次数内执行优化步骤
    if (iter < _params->iterations) {
        // =============================================================================
        // 执行优化器步骤
        // =============================================================================
        // 使用计算得到的梯度更新所有高斯参数
        // Adam优化器会根据一阶和二阶动量自动调整更新步长
        _optimizer->step();
        
        // =============================================================================
        // 清零梯度
        // =============================================================================
        // 清除所有参数的梯度缓存，为下一次迭代做准备
        // set_to_none=true 可以提高内存效率
        _optimizer->zero_grad(true);
        
        // =============================================================================
        // 更新学习率
        // =============================================================================
        // 根据学习率调度器更新当前的学习率
        // 通常使用指数衰减来逐步降低学习率
        _scheduler->step();
    }
    // 如果超过了指定的训练迭代次数，则不执行任何优化操作
}
