#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "components/poseopt.hpp"
#include "core/image_io.hpp"
#include "kernels/fused_ssim.cuh"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/rasterizer.hpp"
#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <print>

namespace gs::training {
    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            bilateral_grid_ = std::make_unique<BilateralGrid>(
                train_dataset_size_,
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W);

            bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{bilateral_grid_->parameters()},
                torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                    .eps(1e-15));

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize bilateral grid: {}", e.what()));
        }
    }

    /**
     * [功能描述]：计算光度损失函数，结合L1损失和SSIM损失来评估渲染图像与真实图像之间的差异
     * @param render_output 渲染输出结果，包含生成的图像
     * @param gt_image 真实图像（ground truth），用于比较的参考图像
     * @param splatData 样条数据，包含高斯样条的参数信息
     * @param opt_params 优化参数，包含损失函数的权重配置
     * @return 返回损失值张量，如果计算失败则返回错误信息
     */
    std::expected<torch::Tensor, std::string> Trainer::compute_photometric_loss(
        const RenderOutput& render_output,
        const torch::Tensor& gt_image,
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            // 确保图像具有相同的尺寸
            torch::Tensor rendered = render_output.image;  // 获取渲染输出的图像
            torch::Tensor gt = gt_image;                   // 获取真实图像

            // 确保两个张量都是4维的（批次、高度、宽度、通道）
            // 如果输入是3维（高度、宽度、通道），则在前面添加批次维度
            rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

            // 检查渲染图像和真实图像的尺寸是否匹配
            // 如果不匹配，抛出错误并显示详细的尺寸信息
            TORCH_CHECK(rendered.sizes() == gt.sizes(),
                        "ERROR: size mismatch – rendered ", rendered.sizes(),
                        " vs. ground truth ", gt.sizes());

            // 基础损失计算：L1损失 + SSIM损失
            auto l1_loss = torch::l1_loss(rendered, gt);  // 计算L1损失（平均绝对误差）
            auto ssim_loss = 1.f - fused_ssim(rendered, gt, "valid", /*train=*/true);  // 计算SSIM损失（结构相似性）
            
            // 组合损失：使用lambda_dssim参数来平衡L1损失和SSIM损失
            // lambda_dssim控制SSIM损失的权重，1-lambda_dssim控制L1损失的权重
            torch::Tensor loss = (1.f - opt_params.lambda_dssim) * l1_loss +
                                 opt_params.lambda_dssim * ssim_loss;
            
            return loss;  // 返回计算得到的损失值
        } catch (const std::exception& e) {
            // 异常处理：捕获任何计算过程中的异常
            // 返回包含错误信息的unexpected对象，使用std::format格式化错误消息
            return std::unexpected(std::format("Error computing photometric loss: {}", e.what()));
        }
    }

    /**
     * @brief 计算缩放参数正则化损失
     * @param splatData 高斯散射体数据，包含所有高斯体的参数
     * @param opt_params 优化参数，包含正则化权重等配置
     * @return 成功时返回正则化损失张量，失败时返回错误信息
     * @details 该函数计算缩放参数的正则化损失，用于防止缩放参数过大，提高模型稳定性。
     *          当scale_reg > 0时，计算缩放参数的L1正则化损失；否则返回零张量。
     */
    std::expected<torch::Tensor, std::string> Trainer::compute_scale_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            // 检查是否启用缩放正则化
            if (opt_params.scale_reg > 0.0f) {
                // 获取所有高斯体的缩放参数并计算平均值（L1正则化）
                auto scale_l1 = splatData.get_scaling().mean();
                // 返回正则化损失：权重系数 × L1损失
                return opt_params.scale_reg * scale_l1;
            }
            // 如果未启用正则化，返回零张量（需要梯度以保持计算图）
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            // 异常处理：捕获计算过程中的任何异常
            // 返回包含错误信息的unexpected对象，使用std::format格式化错误消息
            return std::unexpected(std::format("Error computing scale regularization loss: {}", e.what()));
        }
    }

    /**
     * @brief 计算不透明度参数正则化损失
     * @param splatData 高斯散射体数据，包含所有高斯体的参数
     * @param opt_params 优化参数，包含正则化权重等配置
     * @return 成功时返回正则化损失张量，失败时返回错误信息
     * @details 该函数计算不透明度参数的正则化损失，用于防止不透明度参数过大，提高模型稳定性。
     *          当opacity_reg > 0时，计算不透明度参数的L1正则化损失；否则返回零张量。
     */
    std::expected<torch::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            // 检查是否启用不透明度正则化
            if (opt_params.opacity_reg > 0.0f) {
                // 获取所有高斯体的不透明度参数并计算平均值（L1正则化）
                auto opacity_l1 = splatData.get_opacity().mean();
                // 返回正则化损失：权重系数 × L1损失
                return opt_params.opacity_reg * opacity_l1;
            }
            // 如果未启用正则化，返回零张量（需要梯度以保持计算图）
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            // 异常处理：捕获计算过程中的任何异常
            // 返回包含错误信息的unexpected对象，使用std::format格式化错误消息
            return std::unexpected(std::format("Error computing opacity regularization loss: {}", e.what()));
        }
    }

    /**
     * @brief 计算双边网格总变差损失
     * @param bilateral_grid 双边网格对象，用于计算TV损失
     * @param opt_params 优化参数，包含TV损失权重等配置
     * @return 成功时返回TV损失张量，失败时返回错误信息
     * @details 该函数计算双边网格的总变差(TV)损失，用于保持网格的平滑性。
     *          当use_bilateral_grid为true时，计算TV损失；否则返回零张量。
     */
    std::expected<torch::Tensor, std::string> Trainer::compute_bilateral_grid_tv_loss(
        const std::unique_ptr<BilateralGrid>& bilateral_grid,
        const param::OptimizationParameters& opt_params) {
        try {
            // 检查是否启用双边网格
            if (opt_params.use_bilateral_grid) {
                // 计算双边网格的TV损失并乘以权重系数
                return opt_params.tv_loss_weight * bilateral_grid->tv_loss();
            }
            // 如果未启用双边网格，返回零张量（需要梯度以保持计算图）
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            // 异常处理：捕获计算过程中的任何异常
            // 返回包含错误信息的unexpected对象，使用std::format格式化错误消息
            return std::unexpected(std::format("Error computing bilateral grid TV loss: {}", e.what()));
        }
    }

    /**
     * [功能描述]：训练器构造函数，负责初始化训练环境，包括数据集分割、策略初始化、双边网格设置、姿态优化模块等。
     * @param dataset [参数说明]：相机数据集，包含所有训练图像和相机信息。
     * @param strategy [参数说明]：训练策略，决定如何更新模型参数。
     * @param params [参数说明]：训练参数，包含各种配置选项和超参数。
     */
    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     const param::TrainingParameters& params)
        : strategy_(std::move(strategy)),  // 移动策略对象
          params_(params) {                // 复制训练参数
        
        // 检查CUDA可用性
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        // 根据评估标志处理数据集分割
        if (params.optimization.enable_eval) {
            // 创建训练/验证数据集分割
            train_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params.dataset, CameraDataset::Split::TRAIN);  // 训练集
            val_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params.dataset, CameraDataset::Split::VAL);    // 验证集

            std::println("Created train/val split: {} train, {} val images",
                         train_dataset_->size().value(),
                         val_dataset_->size().value());
        } else {
            // 使用所有图像进行训练
            train_dataset_ = dataset;
            val_dataset_ = nullptr;  // 无验证集

            std::println("Using all {} images for training (no evaluation)",
                         train_dataset_->size().value());
        }

        // 记录训练数据集大小
        train_dataset_size_ = train_dataset_->size().value();

        // 初始化训练策略
        strategy_->initialize(params.optimization);

        // 如果启用，初始化双边网格
        if (auto result = initialize_bilateral_grid(); !result) {
            throw std::runtime_error(result.error());
        }

        // 初始化背景颜色张量（黑色背景）
        background_ = torch::tensor({0.f, 0.f, 0.f},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        
        // 姿态优化模块初始化
        if (params.optimization.pose_optimization != "none") {
            // 检查评估与姿态优化的兼容性
            if (params.optimization.enable_eval) {
                throw std::runtime_error("Evaluating with pose optimization is not supported yet. "
                                         "Please disable pose optimization or evaluation.");
            }
            // 检查GUT与姿态优化的兼容性
            if (params.optimization.gut) {
                throw std::runtime_error("The 3DGUT rasterizer doesn't have camera gradients yet. "
                                         "Please disable pose optimization or disable gut.");
            }
            
            // 根据配置创建相应的姿态优化模块
            if (params.optimization.pose_optimization == "direct") {
                // 直接姿态优化模块
                poseopt_module_ = std::make_unique<DirectPoseOptimizationModule>(train_dataset_->get_cameras().size());
            } else if (params.optimization.pose_optimization == "mlp") {
                // MLP姿态优化模块
                poseopt_module_ = std::make_unique<MLPPoseOptimizationModule>(train_dataset_->get_cameras().size());
            } else {
                throw std::runtime_error("Invalid pose optimization type: " + params.optimization.pose_optimization);
            }
            
            // 创建姿态优化器的Adam优化器，学习率为1e-5
            poseopt_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{poseopt_module_->parameters()},
                torch::optim::AdamOptions(1e-5));
        } else {
            // 如果不使用姿态优化，创建默认模块
            poseopt_module_ = std::make_unique<PoseOptimizationModule>();
        }

        // 重新设置背景颜色张量并移动到CUDA设备
        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32));
        background_ = background_.to(torch::kCUDA);

        // 根据无头标志创建进度条
        if (params_.optimization.headless) {
            progress_ = std::make_unique<TrainingProgress>(
                params.optimization.iterations,  // 总迭代次数
                /*update_frequency=*/100);      // 更新频率：每100次迭代更新一次
        }

        // 初始化评估器 - 它内部处理所有指标
        evaluator_ = std::make_unique<MetricsEvaluator>(params);

        // 设置相机缓存：建立相机ID到相机对象的映射
        for (const auto& cam : dataset->get_cameras()) {
            m_cam_id_to_cam[cam->uid()] = cam;
        }

        // 打印渲染模式配置信息
        std::println("Render mode: {}", params.optimization.render_mode);
        std::println("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
        std::println("Strategy: {}", params.optimization.strategy);
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
        }
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (progress_) {
                progress_->pause();
            }
            std::println("\nTraining paused at iteration {}", iter);
            std::println("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            std::println("\nTraining resumed at iteration {}", iter);
        }

        // Handle save request
        if (save_requested_.exchange(false)) {
            std::println("\nSaving checkpoint at iteration {}...", iter);
            auto checkpoint_path = params_.dataset.output_path / "checkpoints";
            save_ply(checkpoint_path, iter, /*join=*/true);

            std::println("Checkpoint saved to {}", checkpoint_path.string());

            // Emit checkpoint saved event
            events::state::CheckpointSaved{
                .iteration = iter,
                .path = checkpoint_path}
                .emit();
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            std::println("\nStopping training permanently at iteration {}...", iter);
            std::println("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    /**
     * [功能描述]：执行单个训练步骤，包括相机验证、渲染、损失计算、反向传播、优化器更新和模型保存等。
     * @param iter [参数说明]：当前迭代次数。
     * @param cam [参数说明]：当前训练使用的相机对象。
     * @param gt_image [参数说明]：真实图像（ground truth），用于计算损失。
     * @param render_mode [参数说明]：渲染模式，决定使用哪种渲染方法。
     * @param stop_token [参数说明]：停止令牌，用于支持训练的中断和停止请求。
     * @return [返回值说明]：返回StepResult枚举值（Continue或Stop），失败时返回错误字符串。
     */
    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        Camera* cam,
        torch::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {
        try {
            // 相机模型验证：根据GUT选项检查相机类型和畸变
            if (params_.optimization.gut) {
                // GUT模式下不支持正交相机模型
                if (cam->camera_model_type() == gsplat::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else {
                // 非GUT模式下检查相机畸变和模型类型
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("You must use --gut option to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != gsplat::CameraModelType::PINHOLE) {
                    return std::unexpected("You must use --gut option to train on cameras with non-pinhole model.");
                }
            }

            // 更新当前迭代次数
            current_iteration_ = iter;

            // 在开始时检查控制请求
            handle_control_requests(iter, stop_token);

            // 如果请求停止，返回Stop
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // 如果暂停，等待恢复
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 100毫秒轮询间隔
                handle_control_requests(iter, stop_token);
            }

            // 暂停后再次检查停止请求
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // 应用姿态优化：调整相机位置
            auto adjusted_cam_pos = poseopt_module_->forward(cam->world_view_transform(), torch::tensor({cam->uid()}));
            auto adjusted_cam = Camera(*cam, adjusted_cam_pos);

            // 渲染输出
            RenderOutput r_output;
            // 根据参数选择渲染模式
            if (!params_.optimization.gut) {
                // 非GUT模式：使用快速光栅化
                r_output = fast_rasterize(adjusted_cam, strategy_->get_model(), background_);
            } else {
                // GUT模式：使用标准光栅化
                r_output = rasterize(adjusted_cam, strategy_->get_model(), background_, 1.0f, false, false, render_mode,
                                     nullptr, true);
            }

            // 如果启用双边网格，应用其效果
            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
            }

            // 计算损失：光度损失（主要损失）
            auto loss_result = compute_photometric_loss(r_output,
                                                        gt_image,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            if (!loss_result) {
                return std::unexpected(loss_result.error());
            }

            torch::Tensor loss = *loss_result;
            loss.backward();  // 反向传播
            float loss_value = loss.item<float>();

            // 缩放正则化损失
            auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), params_.optimization);
            if (!scale_loss_result) {
                return std::unexpected(scale_loss_result.error());
            }
            loss = *scale_loss_result;
            loss.backward();  // 反向传播
            loss_value += loss.item<float>();

            // 不透明度正则化损失
            auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), params_.optimization);
            if (!opacity_loss_result) {
                return std::unexpected(opacity_loss_result.error());
            }
            loss = *opacity_loss_result;
            loss.backward();  // 反向传播
            loss_value += loss.item<float>();

            // 双边网格总变分损失
            auto tv_loss_result = compute_bilateral_grid_tv_loss(bilateral_grid_, params_.optimization);
            if (!tv_loss_result) {
                return std::unexpected(tv_loss_result.error());
            }
            loss = *tv_loss_result;
            loss.backward();  // 反向传播
            loss_value += loss.item<float>();

            // 立即存储损失值
            current_loss_ = loss_value;

            // 如果需要，同步更新进度
            if (progress_) {
                progress_->update(iter, loss_value,
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // 发出训练进度事件（节流以减少GUI更新）
            if (iter % 10 == 0 || iter == 1) {
                // 每10次迭代更新一次
                events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }
            
            // 使用NoGradGuard确保不计算梯度
            {
                torch::NoGradGuard no_grad;

                DeferredEvents deferred;
                {
                    // 获取渲染互斥锁
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // 执行策略的后向传播后处理和步骤更新
                    strategy_->post_backward(iter, r_output);
                    strategy_->step(iter);

                    // 如果使用双边网格，更新其优化器
                    if (params_.optimization.use_bilateral_grid) {
                        bilateral_grid_optimizer_->step();
                        bilateral_grid_optimizer_->zero_grad(true);
                    }
                    
                    // 如果启用姿态优化，更新姿态优化器
                    if (params_.optimization.pose_optimization != "none") {
                        poseopt_optimizer_->step();
                        poseopt_optimizer_->zero_grad(true);
                    }

                    // 在锁释放后将事件加入队列
                    deferred.add(events::state::ModelUpdated{
                        .iteration = iter,
                        .num_gaussians = static_cast<size_t>(strategy_->get_model().size())});
                } // 锁在这里释放

                // 当deferred析构时，事件会自动发出

                // 清理评估：让评估器处理所有事情
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    std::println("{}", metrics.to_string());
                }

                // 在指定步骤保存模型
                if (!params_.optimization.skip_intermediate_saving) {
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            const bool join_threads = (iter == params_.optimization.save_steps.back());
                            auto save_path = params_.dataset.output_path;
                            save_ply(save_path, iter, /*join=*/join_threads);  // 保存PLY文件
                            // 发出检查点保存事件
                            events::state::CheckpointSaved{
                                .iteration = iter,
                                .path = save_path}
                                .emit();
                        }
                    }
                }

                // 时间轴图像生成：定期渲染指定图像
                if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                    for (const auto& img_name : params_.dataset.timelapse_images) {
                        // 获取训练和验证数据集中的相机
                        auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                        auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                        if (train_cam.has_value() || val_cam.has_value()) {
                            Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                            // 图像尺寸在图像加载一次之前不正确
                            // 如果我们在图像加载之前使用相机，它将以非缩放尺寸渲染图像
                            if (cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) {
                                cam_to_use->load_image_size(params_.dataset.resize_factor);
                            }

                            // 渲染时间轴图像
                            RenderOutput rendered_timelapse_output = fast_rasterize(
                                *cam_to_use, strategy_->get_model(), background_);

                            // 通过去除文件扩展名获取保存文件夹名称
                            std::string folder_name = img_name;
                            auto last_dot = folder_name.find_last_of('.');
                            if (last_dot != std::string::npos) {
                                folder_name = folder_name.substr(0, last_dot);
                            }

                            // 创建输出路径并保存图像
                            auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                            std::filesystem::create_directories(output_path);

                            image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
                                                       rendered_timelapse_output.image);
                        } else {
                            std::println("Warning: Timelapse image '{}' not found in dataset.", img_name);
                        }
                    }
                }
            }

            // 如果应该继续训练，返回Continue
            if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
                return StepResult::Continue;
            } else {
                return StepResult::Stop;
            }
        } catch (const std::exception& e) {
            // 异常处理：返回错误信息
            return std::unexpected(std::format("Training step failed: {}", e.what()));
        }
    }

    /**
     * [功能描述]：执行训练循环，包括数据加载、训练步骤、进度更新和模型保存等核心训练流程。
     * @param stop_token [参数说明]：停止令牌，用于支持训练的中断和停止请求。
     * @return [返回值说明]：成功时返回void，失败时返回错误字符串。
     */
    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
        // 重置训练状态标志
        is_running_ = false;           // 训练运行状态
        training_complete_ = false;     // 训练完成状态
        ready_to_start_ = false;       // 重置准备开始标志

        // 基于事件的准备信号处理（仅在非无头模式下）
        if (!params_.optimization.headless) {
            // 订阅开始信号（无需存储句柄）
            events::internal::TrainingReadyToStart::when([this](const auto&) {
                ready_to_start_ = true;
            });

            // 发出我们已准备好的信号
            events::internal::TrainerReady{}.emit();

            // 等待开始信号，同时检查停止请求
            while (!ready_to_start_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 50毫秒轮询间隔
            }
        }

        is_running_ = true; // 现在可以开始训练

        try {
            // 初始化训练变量
            int iter = 1;                           // 当前迭代次数
            const int num_workers = 16;             // 数据加载器工作线程数
            const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);  // 渲染模式

            // 更新进度信息（如果存在进度回调）
            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // 使用无限数据加载器避免epoch重启
            auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, num_workers);
            auto loader = train_dataloader->begin(); // 返回开始迭代器

            // 单循环训练，不使用epochs
            while (iter <= params_.optimization.iterations) {
                // 检查停止请求
                if (stop_token.stop_requested() || stop_requested_.load()) {
                    break;
                }

                // 等待前一个回调完成（如果仍在运行）
                if (callback_busy_.load()) {
                    callback_stream_.synchronize();
                }

                // 获取当前批次数据
                auto& batch = *loader;
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                // 将图像数据移动到CUDA设备，使用非阻塞传输
                torch::Tensor gt_image = std::move(camera_with_image.image).to(torch::kCUDA, /*non_blocking=*/true);

                // 执行训练步骤
                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                if (!step_result) {
                    return std::unexpected(step_result.error());
                }

                // 检查训练步骤结果，如果是停止信号则退出
                if (*step_result == StepResult::Stop) {
                    break;
                }

                // 启动异步进度更新回调（除第一次迭代外）
                if (iter > 1 && callback_) {
                    callback_busy_ = true;
                    // 使用CUDA主机函数启动异步回调
                    auto err = cudaLaunchHostFunc(
                        callback_stream_.stream(),
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_();
                            }
                            trainer->callback_busy_ = false;
                        },
                        this);
                    if (err != cudaSuccess) {
                        std::cerr << "Warning: Failed to launch callback: " << cudaGetErrorString(err) << std::endl;
                        callback_busy_ = false;
                    }
                }

                // 递增迭代次数和数据加载器
                ++iter;
                ++loader;
            }

            // 确保回调在最终保存前完成
            if (callback_busy_.load()) {
                callback_stream_.synchronize();
            }

            // 如果不是由停止请求触发的保存，则进行最终保存
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, iter - 1, /*join=*/true);  // 保存最终PLY文件
                // 发出最终检查点保存事件
                events::state::CheckpointSaved{
                    .iteration = iter - 1,
                    .path = final_path}
                    .emit();
            }

            // 完成进度更新
            if (progress_) {
                progress_->complete();
            }
            
            // 保存评估报告
            evaluator_->save_report();
            
            // 打印最终摘要
            if (progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }

            // 更新训练状态
            is_running_ = false;
            training_complete_ = true;

            return {};  // 训练成功完成
        } catch (const std::exception& e) {
            // 异常处理：更新状态并返回错误信息
            is_running_ = false;
            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    std::shared_ptr<const Camera> Trainer::getCamById(int camId) const {
        const auto it = m_cam_id_to_cam.find(camId);
        if (it == m_cam_id_to_cam.end()) {
            std::cerr << "error: getCamById - could not find cam with cam id " << camId << std::endl;
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<const Camera>> Trainer::getCamList() const {
        std::vector<std::shared_ptr<const Camera>> cams;
        cams.reserve(m_cam_id_to_cam.size());
        for (auto& [key, value] : m_cam_id_to_cam) {
            cams.push_back(value);
        }

        return cams;
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads) {
        strategy_->get_model().save_ply(save_path, iter_num + 1, /*join=*/join_threads);
        if (lf_project_) {
            const std::string ply_name = "splat_" + std::to_string(iter_num + 1);
            const std::filesystem::path ply_path = save_path / (ply_name + ".ply");
            lf_project_->addPly(gs::management::PlyData(false, ply_path, iter_num, ply_name));
        }
    }
} // namespace gs::training
