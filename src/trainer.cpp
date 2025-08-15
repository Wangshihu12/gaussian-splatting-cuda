#include "core/trainer.hpp"
#include "core/fast_rasterizer.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <numeric>
#include <print>

namespace gs {

    /**
     * [功能描述]：初始化双边网格滤波器及其优化器
     * @return 期望值类型，成功时返回void，失败时返回错误信息字符串
     * 
     * 双边网格功能：
     * - 用于图像后处理和色彩校正
     * - 可以在训练过程中自适应调整渲染图像的外观
     * - 通过学习仿射变换参数来优化图像质量
     */
    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        // 检查是否在训练参数中启用了双边网格功能
        if (!params_.optimization.use_bilateral_grid) {
            // 如果未启用双边网格，直接返回成功（空操作）
            return {};
        }

        try {
            // 创建双边网格实例
            bilateral_grid_ = std::make_unique<gs::BilateralGrid>(
                train_dataset_size_,                        // 训练数据集大小（图像数量）
                params_.optimization.bilateral_grid_X,      // 网格X维度分辨率（空间宽度）
                params_.optimization.bilateral_grid_Y,      // 网格Y维度分辨率（空间高度）
                params_.optimization.bilateral_grid_W       // 网格W维度分辨率（颜色/亮度引导）
            );

            // 为双边网格创建Adam优化器
            bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{bilateral_grid_->parameters()},  // 双边网格的可学习参数
                torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)  // Adam优化器选项
                    .eps(1e-15)                             // 数值稳定性参数，防止除零错误
            );

            return {};  // 初始化成功，返回空值
        } catch (const std::exception& e) {
            // 捕获初始化过程中的任何异常并返回错误信息
            return std::unexpected(std::format("Failed to initialize bilateral grid: {}", e.what()));
        }
    }

    /**
     * [功能描述]：计算光度测量损失（Photometric Loss）
     * @param render_output：渲染输出结果，包含渲染的图像和其他辅助信息
     * @param gt_image：真实参考图像（Ground Truth），用作训练目标
     * @param splatData：高斯散点数据，包含所有3D高斯的参数
     * @param opt_params：优化参数配置，包含损失函数的权重设置
     * @return 期望值类型，成功时返回计算得到的损失张量，失败时返回错误信息
     * 
     * 光度损失组成：
     * - L1损失：测量像素级别的绝对差异，对异常值敏感
     * - SSIM损失：结构相似性损失，关注图像的结构和纹理信息
     * - 最终损失 = (1-λ)×L1 + λ×SSIM，其中λ控制两者的权重平衡
     */
    std::expected<torch::Tensor, std::string> Trainer::compute_photometric_loss(
        const RenderOutput& render_output,
        const torch::Tensor& gt_image,
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        try {
            // =============================================================================
            // 图像预处理：确保维度一致性
            // =============================================================================
            
            // 提取渲染图像和真实图像
            torch::Tensor rendered = render_output.image;  // 神经渲染输出的图像
            torch::Tensor gt = gt_image;                   // 真实参考图像

            // 确保两个张量都是4D格式：[batch, height, width, channels]
            // 如果输入是3D张量 [height, width, channels]，则添加批次维度
            rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

            // 验证渲染图像和真实图像的尺寸是否匹配
            TORCH_CHECK(rendered.sizes() == gt.sizes(),
                        "ERROR: size mismatch – rendered ", rendered.sizes(),
                        " vs. ground truth ", gt.sizes());

            // =============================================================================
            // 损失函数计算：L1 + SSIM 组合损失
            // =============================================================================
            
            // 计算L1损失（平均绝对误差）
            // L1损失对像素级别的差异敏感，能够产生清晰的边缘但可能过于锐化
            auto l1_loss = torch::l1_loss(rendered, gt);
            
            // 计算SSIM损失（结构相似性损失）
            // SSIM关注图像的结构信息（亮度、对比度、结构），更符合人眼感知
            // fused_ssim返回相似性分数[0,1]，1-ssim得到损失值，越小表示越相似
            auto ssim_loss = 1.f - fused_ssim(rendered, gt, "valid", /*train=*/true);
            
            // 组合损失：加权平均L1和SSIM损失
            // lambda_dssim参数控制SSIM损失的权重：
            // - lambda_dssim=0：纯L1损失，关注像素精确性
            // - lambda_dssim=1：纯SSIM损失，关注结构相似性
            // - 0<lambda_dssim<1：平衡像素精度和结构相似性
            torch::Tensor loss = (1.f - opt_params.lambda_dssim) * l1_loss +
                                 opt_params.lambda_dssim * ssim_loss;
            
            return loss;  // 返回计算得到的总损失
            
        } catch (const std::exception& e) {
            // 捕获计算过程中的任何异常并返回详细错误信息
            return std::unexpected(std::format("Error computing photometric loss: {}", e.what()));
        }
    }

    /**
     * [功能描述]：计算缩放正则化损失（Scale Regularization Loss）
     * @param splatData：高斯散点数据，包含所有3D高斯的缩放参数
     * @param opt_params：优化参数配置，包含缩放正则化的权重设置
     * @return 期望值类型，成功时返回缩放正则化损失张量，失败时返回错误信息
     * 
     * 缩放正则化的作用：
     * - 防止高斯散点的尺寸变得过大，避免渲染结果过于模糊
     * - 鼓励模型使用较小的高斯来表示场景细节
     * - 提高渲染质量和几何表示的紧致性
     * - 防止训练过程中的数值不稳定问题
     */
    std::expected<torch::Tensor, std::string> Trainer::compute_scale_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        try {
            // 检查是否启用缩放正则化
            if (opt_params.scale_reg > 0.0f) {
                // =============================================================================
                // 计算缩放正则化损失
                // =============================================================================
                
                // 获取所有高斯散点的缩放参数并计算平均值
                // get_scaling()返回形状为[N, 3]的张量，表示每个高斯在x,y,z三个轴上的缩放
                // mean()计算所有缩放参数的平均值，得到一个标量
                auto scale_l1 = splatData.get_scaling().mean();
                
                // 应用正则化权重，返回加权的缩放损失
                // scale_reg是正则化强度参数：
                // - 值越大，对大尺寸高斯的惩罚越严厉
                // - 值为0时禁用缩放正则化
                return opt_params.scale_reg * scale_l1;
            }
            
            // =============================================================================
            // 缩放正则化未启用时的处理
            // =============================================================================
            
            // 返回零损失张量，但仍然需要梯度计算能力
            // requires_grad_()确保这个零张量参与反向传播，保持计算图的完整性
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
            
        } catch (const std::exception& e) {
            // 捕获计算过程中的任何异常并返回详细错误信息
            return std::unexpected(std::format("Error computing scale regularization loss: {}", e.what()));
        }
    }

    /**
     * [功能描述]：计算不透明度正则化损失（Opacity Regularization Loss）
     * @param splatData：高斯散点数据，包含所有3D高斯的不透明度参数
     * @param opt_params：优化参数配置，包含不透明度正则化的权重设置
     * @return 期望值类型，成功时返回不透明度正则化损失张量，失败时返回错误信息
     * 
     * 不透明度正则化的作用：
     * - 防止高斯散点变得过于不透明，鼓励使用半透明效果
     * - 提高渲染的真实感和视觉质量，避免"硬边缘"效果
     * - 促进模型学习更平滑的几何表示
     * - 减少过度拟合，提高泛化能力
     * - 保持训练过程的数值稳定性
     */
    std::expected<torch::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        try {
            // 检查是否启用不透明度正则化
            if (opt_params.opacity_reg > 0.0f) {
                // =============================================================================
                // 计算不透明度正则化损失
                // =============================================================================
                
                // 获取所有高斯散点的不透明度参数并计算平均值
                // get_opacity()返回形状为[N, 1]的张量，表示每个高斯的不透明度值[0,1]
                // mean()计算所有不透明度的平均值，得到一个标量
                // 较高的平均不透明度意味着高斯更"实体化"，较低则更"透明"
                auto opacity_l1 = splatData.get_opacity().mean();
                
                // 应用正则化权重，返回加权的不透明度损失
                // opacity_reg是正则化强度参数：
                // - 值越大，对高不透明度的惩罚越严厉，鼓励更透明的高斯
                // - 值为0时禁用不透明度正则化
                // - 适中的值有助于平衡渲染质量和几何表示
                return opt_params.opacity_reg * opacity_l1;
            }
            
            // =============================================================================
            // 不透明度正则化未启用时的处理
            // =============================================================================
            
            // 返回零损失张量，但保持梯度计算能力
            // requires_grad_()确保这个零张量参与反向传播，维持计算图的连续性
            // 这样即使正则化被禁用，总损失函数的梯度计算仍然正确
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
            
        } catch (const std::exception& e) {
            // 捕获计算过程中的任何异常并返回详细错误信息
            return std::unexpected(std::format("Error computing opacity regularization loss: {}", e.what()));
        }
    }

    /**
     * [功能描述]：计算双边网格全变分损失（Bilateral Grid Total Variation Loss）
     * @param bilateral_grid：双边网格智能指针，包含可学习的滤波参数
     * @param opt_params：优化参数配置，包含TV损失的权重设置
     * @return 期望值类型，成功时返回全变分损失张量，失败时返回错误信息
     * 
     * 全变分(TV)损失的作用：
     * - 促进双边网格参数的空间平滑性，避免突变和不连续
     * - 防止网格过拟合到训练数据，提高泛化能力
     * - 保持滤波效果的空间连续性，产生更自然的图像处理结果
     * - 减少噪声和伪影，提高渲染图像的视觉质量
     * - 正则化网格参数，防止训练过程中的数值不稳定
     */
    std::expected<torch::Tensor, std::string> Trainer::compute_bilateral_grid_tv_loss(
        const std::unique_ptr<gs::BilateralGrid>& bilateral_grid,
        const param::OptimizationParameters& opt_params) {

        try {
            // 检查是否启用双边网格功能
            if (opt_params.use_bilateral_grid) {
                // =============================================================================
                // 计算双边网格全变分损失
                // =============================================================================
                
                // 调用双边网格的tv_loss()方法计算全变分损失
                // TV损失测量相邻网格单元之间参数的变化程度：
                // - 在空间维度(X,Y)上：鼓励相邻位置的滤波参数相似
                // - 在引导维度(W)上：鼓励相邻亮度/颜色级别的参数平滑变化
                // - 数学表达：TV = Σ|∇x grid| + Σ|∇y grid| + Σ|∇w grid|
                // - 其中∇表示梯度算子，测量参数在各个维度上的变化率
                
                // 应用TV损失权重
                // tv_loss_weight控制空间平滑性的强度：
                // - 值越大，网格参数越平滑，但可能失去细节
                // - 值越小，保留更多细节，但可能产生伪影
                // - 需要在平滑性和细节保留之间找到平衡点
                return opt_params.tv_loss_weight * bilateral_grid->tv_loss();
            }
            
            // =============================================================================
            // 双边网格未启用时的处理
            // =============================================================================
            
            // 返回零损失张量，但保持梯度计算能力
            // 当双边网格功能被禁用时，TV损失不参与总损失计算
            // requires_grad_()确保计算图完整性，避免梯度传播中断
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
            
        } catch (const std::exception& e) {
            // 捕获计算过程中的任何异常并返回详细错误信息
            return std::unexpected(std::format("Error computing bilateral grid TV loss: {}", e.what()));
        }
    }

    /**
     * [功能描述]：Trainer类构造函数，初始化训练器的所有组件和配置
     * @param dataset：相机数据集智能指针，包含所有相机和对应图像
     * @param strategy：训练策略智能指针（MCMC或DefaultStrategy）
     * @param params：完整的训练参数配置
     * 
     * 构造函数完成以下初始化任务：
     * 1. CUDA环境检查
     * 2. 数据集分割（训练集/验证集）
     * 3. 训练策略初始化
     * 4. 双边网格设置
     * 5. 渲染环境配置
     * 6. 进度监控和评估器设置
     * 7. 相机缓存建立
     */
    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     const param::TrainingParameters& params)
        : strategy_(std::move(strategy)),        // 移动语义转移训练策略所有权
          params_(params) {                      // 保存训练参数配置

        // =============================================================================
        // CUDA环境验证
        // =============================================================================
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA不可用 – 中止。");
        }

        // =============================================================================
        // 数据集分割处理
        // =============================================================================
        // 根据评估标志决定是否进行训练/验证数据集分割
        if (params.optimization.enable_eval) {
            // 创建训练/验证数据集分割
            // 这样可以在训练过程中定期评估模型性能
            train_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(),          // 原始相机数据
                params.dataset,                  // 数据集参数
                CameraDataset::Split::TRAIN      // 训练集标记
            );
            val_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(),          // 原始相机数据
                params.dataset,                  // 数据集参数  
                CameraDataset::Split::VAL        // 验证集标记
            );

            std::println("创建了训练/验证集: {} 训练, {} 验证图像",
                         train_dataset_->size().value(),
                         val_dataset_->size().value());
        } else {
            // 使用所有图像进行训练（无验证）
            // 适用于数据量较小或不需要验证的场景
            train_dataset_ = dataset;
            val_dataset_ = nullptr;

            std::println("使用所有{}图像进行训练（无评估）",
                         train_dataset_->size().value());
        }

        // 缓存训练数据集大小，用于后续的批处理和进度计算
        train_dataset_size_ = train_dataset_->size().value();

        // =============================================================================
        // 训练策略初始化
        // =============================================================================
        // 使用优化参数初始化具体的训练策略（MCMC或其他）
        strategy_->initialize(params.optimization);

        // =============================================================================
        // 双边网格滤波器初始化
        // =============================================================================
        // 如果启用了双边网格，则初始化网格和对应的优化器
        if (auto result = initialize_bilateral_grid(); !result) {
            throw std::runtime_error(result.error());
        }

        // =============================================================================
        // 渲染环境配置
        // =============================================================================
        // 设置默认背景颜色为黑色[0, 0, 0]
        // 在CUDA设备上创建，用于渲染时的背景填充
        background_ = torch::tensor({0.f, 0.f, 0.f}, 
                                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // =============================================================================
        // 进度监控设置
        // =============================================================================
        // 根据是否为无头模式创建进度条
        if (params.optimization.headless) {
            // 无头模式：创建基于文本的进度监控
            progress_ = std::make_unique<TrainingProgress>(
                params.optimization.iterations,     // 总迭代次数
                /*update_frequency=*/100            // 每100次迭代更新一次进度
            );
        }
        // GUI模式不需要文本进度条，由可视化界面提供进度显示

        // =============================================================================
        // 评估器初始化
        // =============================================================================
        // 创建指标评估器，负责计算PSNR、SSIM、LPIPS等图像质量指标
        // 评估器内部处理所有指标的计算和统计
        evaluator_ = std::make_unique<metrics::MetricsEvaluator>(params);

        // =============================================================================
        // 相机缓存建立
        // =============================================================================
        // 建立相机ID到相机对象的映射，用于快速查找
        // 这样在训练过程中可以通过ID快速访问特定相机的参数
        for (const auto& cam : dataset->get_cameras()) {
            m_cam_id_to_cam[cam->uid()] = cam;
        }

        // =============================================================================
        // 配置信息输出
        // =============================================================================
        // 打印关键的训练配置信息，便于调试和日志记录
        std::println("渲染模式: {}", params.optimization.render_mode);
        std::println("可视化: {}", params.optimization.headless ? "禁用" : "启用");
        std::println("策略: {}", params.optimization.strategy);
    }

    /**
     * [功能描述]：Trainer类析构函数，负责安全地清理所有资源和停止训练过程
     * 
     * 析构函数执行以下清理任务：
     * 1. 发送训练停止信号
     * 2. 等待异步操作完成
     * 3. 同步CUDA流操作
     * 4. 取消事件系统订阅
     * 
     * 这些步骤确保对象销毁时不会产生悬挂指针、内存泄漏或崩溃
     */
    Trainer::~Trainer() {
        // =============================================================================
        // 步骤1：发送训练停止请求
        // =============================================================================
        // 设置原子标志，通知所有正在运行的训练线程停止执行
        // 这是一个线程安全的操作，确保训练循环能够优雅地退出
        stop_requested_ = true;

        // =============================================================================
        // 步骤2：等待异步回调操作完成
        // =============================================================================
        // 检查是否有回调函数正在执行（例如渲染回调、评估回调等）
        if (callback_busy_.load()) {
            // 如果回调正在执行，等待CUDA流中的所有操作完成
            // 这确保了所有GPU操作都已完成，避免在对象销毁后还有GPU任务运行
            callback_stream_.synchronize();
        }
        
        // =============================================================================
        // 步骤3：取消事件系统订阅
        // =============================================================================
        // 从事件总线中移除训练开始事件的处理器
        // 这非常重要：如果不取消订阅，当事件在对象销毁后触发时会导致崩溃
        // train_started_handle_是在构造时注册的事件处理句柄
        gs::event::bus().remove<gs::events::internal::TrainingReadyToStart>(train_started_handle_);
        
        // 注释解释了为什么需要取消订阅：
        // "当类被销毁时，如果事件仍然发射，我们会得到崩溃"
        // 这是因为事件处理器可能会尝试访问已销毁对象的成员函数或数据
    }

    /**
     * [功能描述]：处理训练过程中的各种控制请求
     * @param iter：当前训练迭代次数
     * @param stop_token：C++20停止令牌，用于协作式线程取消
     * 
     * 该函数处理以下控制操作：
     * 1. 外部停止信号检查
     * 2. 训练暂停和恢复
     * 3. 检查点保存请求
     * 4. 永久停止训练请求
     * 
     * 这些控制操作通常来自GUI界面或外部信号，确保训练过程的交互性和可控性
     */
    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // =============================================================================
        // 步骤1：检查外部停止令牌
        // =============================================================================
        // 首先检查C++20停止令牌，这是最高优先级的停止信号
        // 通常来自线程池或协作式多线程环境的取消请求
        if (stop_token.stop_requested()) {
            stop_requested_ = true;     // 设置内部停止标志
            return;                     // 立即退出，不处理其他请求
        }

        // =============================================================================
        // 步骤2：处理训练暂停和恢复逻辑
        // =============================================================================
        
        // 检查暂停请求：用户请求暂停但训练尚未暂停
        if (pause_requested_.load() && !is_paused_.load()) {
            // 设置暂停状态
            is_paused_ = true;
            
            // 如果有进度条，暂停进度显示
            if (progress_) {
                progress_->pause();
            }
            
            // 输出暂停信息，提供用户反馈
            std::println("\n训练暂停在迭代 {}", iter);
            std::println("点击'继续训练'以继续。");
            
        // 检查恢复请求：用户取消暂停请求且训练当前已暂停
        } else if (!pause_requested_.load() && is_paused_.load()) {
            // 取消暂停状态
            is_paused_ = false;
            
            // 如果有进度条，恢复进度显示并更新当前状态
            if (progress_) {
                progress_->resume(
                    iter,                                           // 当前迭代次数
                    current_loss_.load(),                          // 当前损失值
                    static_cast<int>(strategy_->get_model().size()) // 当前模型大小（高斯点数量）
                );
            }
            
            // 输出恢复信息
            std::println("\n训练恢复在迭代 {}", iter);
        }

        // =============================================================================
        // 步骤3：处理检查点保存请求
        // =============================================================================
        // 使用exchange操作原子性地检查并重置保存标志
        // exchange(false)返回原值并设置为false，避免重复保存
        if (save_requested_.exchange(false)) {
            std::println("\n保存检查点在迭代 {}...", iter);
            
            // 构建检查点保存路径
            auto checkpoint_path = params_.dataset.output_path / "checkpoints";
            
            // 保存当前模型状态为PLY文件
            // join=true表示等待保存操作完成后再继续
            strategy_->get_model().save_ply(checkpoint_path, iter, /*join=*/true);
            
            std::println("检查点保存到 {}", checkpoint_path.string());

            // 发射检查点保存完成事件
            // 这允许其他系统组件（如GUI）响应保存完成事件
            events::state::CheckpointSaved{
                .iteration = iter,                  // 保存时的迭代次数
                .path = checkpoint_path            // 保存路径
            }.emit();
        }

        // =============================================================================
        // 步骤4：处理永久停止训练请求
        // =============================================================================
        // 这是最终的停止操作，会保存模型并完全终止训练
        if (stop_requested_.load()) {
            std::println("\n永久停止训练在迭代 {}...", iter);
            std::println("保存最终模型...");
            
            // 保存最终模型到输出目录
            // 这确保了即使训练被中断，也能保留当前的训练成果
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
            
            // 设置运行状态为false，这将导致训练循环退出
            is_running_ = false;
        }
    }

    /**
     * [功能描述]：执行单个训练步骤的核心函数
     * @param iter：当前训练迭代次数
     * @param cam：当前用于训练的相机对象指针
     * @param gt_image：真实参考图像（Ground Truth）
     * @param render_mode：渲染模式（RGB、深度等）
     * @param stop_token：C++20停止令牌，用于协作式取消
     * @return 期望值类型，成功时返回训练步骤结果（Continue/Stop），失败时返回错误信息
     * 
     * 训练步骤流程：
     * 1. 相机兼容性验证
     * 2. 控制请求处理（暂停/停止/保存）
     * 3. 场景渲染
     * 4. 多种损失计算和反向传播
     * 5. 参数更新和模型优化
     * 6. 评估和检查点保存
     * 7. 返回继续/停止状态
     */
    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        Camera* cam,
        torch::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {

        try {
            // =============================================================================
            // 步骤1：相机兼容性验证
            // =============================================================================
            // 检查相机是否有径向畸变参数
            if (cam->radial_distortion().numel() != 0 ||
                cam->tangential_distortion().numel() != 0) {
                return std::unexpected("训练在有畸变的相机上不支持。");
            }
            // 检查相机模型类型，目前只支持针孔相机模型
            if (cam->camera_model_type() != gsplat::CameraModelType::PINHOLE) {
                return std::unexpected("训练在非针孔相机模型上不支持。");
            }

            // 更新当前迭代次数
            current_iteration_ = iter;

            // =============================================================================
            // 步骤2：控制请求处理和状态检查
            // =============================================================================
            
            // 在步骤开始时处理控制请求（暂停、停止、保存等）
            handle_control_requests(iter, stop_token);

            // 如果收到停止请求，立即返回
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // 暂停处理：在暂停状态下等待，但定期检查控制请求
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 休眠100ms避免忙等待
                handle_control_requests(iter, stop_token);
            }

            // 暂停结束后再次检查停止请求
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // =============================================================================
            // 步骤3：场景渲染
            // =============================================================================
            
            // 创建渲染函数lambda，封装渲染参数
            auto render_fn = [this, &cam, render_mode]() {
                return fast_rasterize(
                    *cam,                       // 当前相机
                    strategy_->get_model(),     // 高斯散点模型
                    background_                 // 背景颜色
                );
            };

            // 执行渲染，获得渲染输出
            RenderOutput r_output = render_fn();

            // 应用双边网格滤波（如果启用）
            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                // 使用相机ID作为图像索引，对渲染结果进行后处理
                r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
            }

            // 策略预反向传播处理（特定策略可能需要的预处理）
            strategy_->pre_backward(r_output);

            // =============================================================================
            // 步骤4：损失计算和反向传播
            // =============================================================================
            
            // 4.1 计算光度测量损失（主要损失）
            auto loss_result = compute_photometric_loss(r_output,
                                                        gt_image,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            if (!loss_result) {
                return std::unexpected(loss_result.error());
            }

            torch::Tensor loss = *loss_result;
            loss.backward();  // 反向传播计算梯度
            float loss_value = loss.item<float>();  // 提取损失数值

            // 4.2 缩放正则化损失
            auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), params_.optimization);
            if (!scale_loss_result) {
                return std::unexpected(scale_loss_result.error());
            }
            loss = *scale_loss_result;
            loss.backward();
            loss_value += loss.item<float>();  // 累加到总损失

            // 4.3 不透明度正则化损失
            auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), params_.optimization);
            if (!opacity_loss_result) {
                return std::unexpected(opacity_loss_result.error());
            }
            loss = *opacity_loss_result;
            loss.backward();
            loss_value += loss.item<float>();  // 累加到总损失

            // 4.4 双边网格全变分损失
            auto tv_loss_result = compute_bilateral_grid_tv_loss(bilateral_grid_, params_.optimization);
            if (!tv_loss_result) {
                return std::unexpected(tv_loss_result.error());
            }
            loss = *tv_loss_result;
            loss.backward();
            loss_value += loss.item<float>();  // 累加到总损失

            // =============================================================================
            // 步骤5：进度更新和事件发射
            // =============================================================================
            
            // 立即存储损失值，供其他组件使用
            current_loss_ = loss_value;

            // 同步更新进度条（如果存在）
            if (progress_) {
                progress_->update(iter, loss_value,
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // 发射训练进度事件（限制频率减少GUI更新负担）
            if (iter % 10 == 0 || iter == 1) { // 仅每10次迭代或第1次迭代更新
                events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)
                }.emit();
            }

            // =============================================================================
            // 步骤6：参数更新（无梯度上下文）
            // =============================================================================
            {
                torch::NoGradGuard no_grad;  // 禁用梯度计算，提高性能

                // 使用写锁保护参数更新过程，确保线程安全
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // 执行策略的后反向传播处理
                    strategy_->post_backward(iter, r_output);
                    
                    // 执行优化步骤（更新高斯参数）
                    strategy_->step(iter);

                    // 更新双边网格参数（如果启用）
                    if (params_.optimization.use_bilateral_grid) {
                        bilateral_grid_optimizer_->step();           // 执行优化步骤
                        bilateral_grid_optimizer_->zero_grad(true);  // 清零梯度
                    }

                    // 发射模型更新事件
                    events::state::ModelUpdated{
                        .iteration = iter,
                        .num_gaussians = static_cast<size_t>(strategy_->get_model().size())
                    }.emit();
                }

                // =============================================================================
                // 步骤7：评估和模型保存
                // =============================================================================
                
                // 模型评估（如果启用且到达评估时机）
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    
                    // 在验证集上评估模型性能
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    std::println("{}", metrics.to_string());
                }

                // 中间检查点保存（如果未禁用）
                if (!params_.optimization.skip_intermediate_saving) {
                    for (size_t save_step : params_.optimization.save_steps) {
                        // 检查是否到达保存步骤且不是最后一次迭代
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            // 如果是最后一个保存步骤，则等待保存完成
                            const bool join_threads = (iter == params_.optimization.save_steps.back());
                            auto save_path = params_.dataset.output_path;
                            
                            // 保存PLY格式的模型文件
                            strategy_->get_model().save_ply(save_path, iter, /*join=*/join_threads);

                            // 发射检查点保存事件
                            events::state::CheckpointSaved{
                                .iteration = iter,
                                .path = save_path
                            }.emit();
                        }
                    }
                }
            }

            // =============================================================================
            // 步骤8：返回训练状态
            // =============================================================================
            
            // 判断是否应该继续训练
            if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
                return StepResult::Continue;  // 继续下一次迭代
            } else {
                return StepResult::Stop;      // 训练完成或被停止
            }

        } catch (const std::exception& e) {
            // 捕获训练步骤中的任何异常
            return std::unexpected(std::format("Training step failed: {}", e.what()));
        }
    }

    /**
     * [功能描述]：训练主函数，控制整个训练过程的执行流程
     * @param stop_token：C++20停止令牌，用于协作式线程取消
     * @return 期望值类型，成功时返回void，失败时返回错误信息
     * 
     * 训练流程概述：
     * 1. 初始化训练状态和事件系统
     * 2. 等待GUI准备信号（非无头模式）
     * 3. 创建无限数据加载器
     * 4. 执行主要训练循环
     * 5. 处理异步回调和进度更新
     * 6. 完成训练并保存最终模型
     * 7. 生成训练报告和总结
     */
    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
        // =============================================================================
        // 步骤1：初始化训练状态
        // =============================================================================
        is_running_ = false;        // 训练尚未开始
        training_complete_ = false; // 训练尚未完成

        // =============================================================================
        // 步骤2：事件系统准备（非无头模式需要等待GUI信号）
        // =============================================================================
        if (!params_.optimization.headless) {
            std::atomic<bool> ready{false}; // 原子布尔值，线程安全的准备标志

            // 临时订阅训练开始信号事件
            // 当GUI发出开始信号时，将ready设置为true
            train_started_handle_ = events::internal::TrainingReadyToStart::when([&ready](const auto&) {
                ready = true;
            });

            // 向事件系统发出训练器准备完毕信号
            // 通知GUI等组件训练器已就绪，等待开始指令
            events::internal::TrainerReady{}.emit();

            // 等待开始信号，定期检查停止令牌
            while (!ready.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 休眠50ms避免忙等待
            }
        }

        // 设置运行状态为true，正式开始训练
        is_running_ = true;

        try {
            // =============================================================================
            // 步骤3：训练参数设置和初始化
            // =============================================================================
            int iter = 1;                    // 起始迭代次数
            const int num_workers = 16;     // 数据加载的工作线程数
            
            // 将字符串渲染模式转换为枚举类型
            const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

            // 更新初始进度显示（如果有进度条）
            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // =============================================================================
            // 步骤4：创建无限数据加载器
            // =============================================================================
            // 使用无限数据加载器避免epoch重启，确保连续的训练流程
            // 这样可以避免在epoch边界处的数据加载中断
            auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, num_workers);
            auto loader = train_dataloader->begin(); // 获取数据迭代器

            // =============================================================================
            // 步骤5：主要训练循环（单循环，无epoch概念）
            // =============================================================================
            while (iter <= params_.optimization.iterations) {
                // 检查停止请求，如果收到则立即退出循环
                if (stop_token.stop_requested() || stop_requested_.load()) {
                    break;
                }

                // 等待之前的异步回调完成（如果正在运行）
                // 这确保了回调不会与当前训练步骤冲突
                if (callback_busy_.load()) {
                    callback_stream_.synchronize(); // 同步CUDA流，等待回调完成
                }

                // =============================================================================
                // 步骤5.1：获取训练数据批次
                // =============================================================================
                auto& batch = *loader;                          // 解引用获取当前批次
                auto camera_with_image = batch[0].data;         // 获取第一个样本（相机+图像）
                Camera* cam = camera_with_image.camera;         // 提取相机对象
                
                // 将图像异步传输到CUDA设备，non_blocking=true提高性能
                torch::Tensor gt_image = std::move(camera_with_image.image).to(torch::kCUDA, /*non_blocking=*/true);

                // =============================================================================
                // 步骤5.2：执行单个训练步骤
                // =============================================================================
                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                if (!step_result) {
                    // 训练步骤失败，返回错误
                    return std::unexpected(step_result.error());
                }

                // 检查训练步骤返回的状态
                if (*step_result == StepResult::Stop) {
                    break; // 收到停止信号，退出训练循环
                }

                // =============================================================================
                // 步骤5.3：异步回调处理（用于进度更新等）
                // =============================================================================
                // 跳过第一次迭代，避免在初始化阶段启动回调
                if (iter > 1 && callback_) {
                    callback_busy_ = true; // 标记回调正在执行
                    
                    // 使用CUDA主机函数异步启动回调
                    // 这允许回调在独立的CUDA流中执行，不阻塞主训练循环
                    auto err = cudaLaunchHostFunc(
                        callback_stream_.stream(),
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_(); // 执行实际回调函数
                            }
                            trainer->callback_busy_ = false; // 标记回调完成
                        },
                        this // 传递trainer对象指针
                    );
                    
                    // 检查CUDA函数调用是否成功
                    if (err != cudaSuccess) {
                        std::cerr << "警告: 启动回调失败: " << cudaGetErrorString(err) << std::endl;
                        callback_busy_ = false; // 重置状态
                    }
                }

                // =============================================================================
                // 步骤5.4：迭代更新
                // =============================================================================
                ++iter;   // 增加迭代计数
                ++loader; // 移动到下一个数据批次
            }

            // =============================================================================
            // 步骤6：训练完成后的清理工作
            // =============================================================================
            
            // 确保最后的回调完成后再进行最终保存
            if (callback_busy_.load()) {
                callback_stream_.synchronize();
            }

            // 最终模型保存（如果不是由停止请求触发的）
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                
                // 保存最终训练的模型，join=true确保保存完成后再继续
                strategy_->get_model().save_ply(final_path, iter - 1, /*join=*/true);

                // 发射最终检查点保存事件
                events::state::CheckpointSaved{
                    .iteration = iter - 1,
                    .path = final_path
                }.emit();

                // 发射训练完成日志事件
                events::notify::Log{
                    .level = events::notify::Log::Level::Info,
                    .message = std::format("训练完成。最终模型保存在迭代 {}", iter - 1),
                    .source = "Trainer"
                }.emit();
            }

            // =============================================================================
            // 步骤7：生成训练报告和总结
            // =============================================================================
            
            // 完成进度条显示
            if (progress_) {
                progress_->complete();
            }
            
            // 保存评估报告（包含所有训练过程中的指标）
            evaluator_->save_report();
            
            // 打印最终训练总结
            if (progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }

            // =============================================================================
            // 步骤8：设置最终状态
            // =============================================================================
            is_running_ = false;        // 训练已停止
            training_complete_ = true;  // 训练已完成

            return {}; // 成功完成训练

        } catch (const std::exception& e) {
            // =============================================================================
            // 异常处理：确保状态正确设置
            // =============================================================================
            is_running_ = false; // 即使出现异常也要正确设置状态
            return std::unexpected(std::format("训练失败: {}", e.what()));
        }
    }

    /**
     * [功能描述]：根据相机ID获取对应的相机对象
     * @param camId：要查找的相机唯一标识符
     * @return 相机对象的常量共享指针，如果未找到则返回nullptr
     * 
     * 该函数用于：
     * - 在训练过程中根据ID快速查找特定相机
     * - 为渲染和评估提供相机参数访问
     * - 支持多线程安全的相机对象访问
     */
    std::shared_ptr<const Camera> Trainer::getCamById(int camId) const {
        // =============================================================================
        // 在相机缓存映射中查找指定ID的相机
        // =============================================================================
        // 使用std::map::find进行O(log n)复杂度的查找
        // m_cam_id_to_cam是在构造函数中建立的相机ID到相机对象的映射表
        const auto it = m_cam_id_to_cam.find(camId);
        
        // 检查是否找到了对应的相机
        if (it == m_cam_id_to_cam.end()) {
            // 未找到指定ID的相机，输出错误信息
            // 这通常表示传入了无效的相机ID，可能是程序逻辑错误
            std::cerr << "错误: getCamById - 找不到具有cam id的相机 " << camId << std::endl;
            return nullptr;  // 返回空指针表示查找失败
        }
        
        // 找到相机，返回对应的共享指针
        // it->second访问map中的值部分（相机对象的共享指针）
        // 返回const共享指针确保调用者不能修改相机对象
        return it->second;
    }

    /**
     * [功能描述]：获取训练器中所有相机对象的列表
     * @return 包含所有相机对象的常量共享指针向量
     * 
     * 该函数用于：
     * - 为可视化界面提供完整的相机列表
     * - 支持批量操作或统计分析
     * - 为外部组件提供相机数据的只读访问
     * - 用于调试和日志记录目的
     */
    std::vector<std::shared_ptr<const Camera>> Trainer::getCamList() const {

        // =============================================================================
        // 创建相机列表容器并预分配内存
        // =============================================================================
        std::vector<std::shared_ptr<const Camera>> cams;
        
        // 预分配内存空间，避免在插入过程中的多次内存重新分配
        // 这是一个重要的性能优化，因为我们已知最终容器的确切大小
        cams.reserve(m_cam_id_to_cam.size());
        
        // =============================================================================
        // 遍历相机缓存映射并提取所有相机对象
        // =============================================================================
        // 使用结构化绑定（C++17特性）遍历映射表
        // key是相机ID，value是相机对象的共享指针
        for (auto& [key, value] : m_cam_id_to_cam) {
            // 将相机对象的共享指针添加到结果向量中
            // 这里不需要key（相机ID），只需要相机对象本身
            cams.push_back(value);
        }

        // 返回包含所有相机对象的向量
        // 调用者获得所有相机的只读访问权限
        return cams;
    }

} // namespace gs