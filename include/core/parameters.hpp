// Copyright (c) 2023 Janusch Patas.

/**
 * @file parameters.hpp
 * @brief 高斯散射体训练参数定义文件
 * @details 定义了高斯散射体训练过程中所需的各种参数配置，包括优化参数、数据集配置和训练参数等
 */

#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

/**
 * @namespace gs
 * @brief 高斯散射体(Gaussian Splatting)相关功能的命名空间
 */
namespace gs {
    /**
     * @namespace param
     * @brief 参数配置相关的命名空间
     */
    namespace param {
        /**
         * @struct OptimizationParameters
         * @brief 优化参数结构体
         * @details 包含高斯散射体训练过程中的所有优化相关参数，如学习率、迭代次数、正则化参数等
         */
        struct OptimizationParameters {
            // 基础训练参数
            size_t iterations = 30'000;                    ///< 总训练迭代次数
            size_t sh_degree_interval = 1'000;             ///< 球谐函数度数增加的间隔步数
            float means_lr = 0.00016f;                     ///< 高斯体中心位置的学习率
            float shs_lr = 0.0025f;                        ///< 球谐函数系数的学习率
            float opacity_lr = 0.05f;                      ///< 不透明度参数的学习率
            float scaling_lr = 0.005f;                     ///< 缩放参数的学习率
            float rotation_lr = 0.001f;                    ///< 旋转参数的学习率
            float lambda_dssim = 0.2f;                     ///< DSSIM损失函数的权重系数
            float min_opacity = 0.005f;                    ///< 最小不透明度阈值
            size_t refine_every = 100;                     ///< 每隔多少步进行一次细化操作
            size_t start_refine = 500;                     ///< 开始细化的步数
            size_t stop_refine = 25'000;                   ///< 停止细化的步数
            float grad_threshold = 0.0002f;                ///< 梯度阈值，用于判断是否需要细化
            int sh_degree = 3;                             ///< 球谐函数的度数
            float opacity_reg = 0.01f;                     ///< 不透明度正则化权重
            float scale_reg = 0.01f;                       ///< 缩放参数正则化权重
            float init_opacity = 0.5f;                     ///< 初始不透明度值
            float init_scaling = 0.1f;                     ///< 初始缩放值
            int max_cap = 1000000;                         ///< 最大高斯体数量上限
            
            // 评估和保存相关参数
            std::vector<size_t> eval_steps = {7'000, 30'000}; ///< 模型评估的步数点
            std::vector<size_t> save_steps = {7'000, 30'000}; ///< 模型保存的步数点
            bool skip_intermediate_saving = false;            ///< 是否跳过中间结果保存，只保存最终输出
            bool enable_eval = false;                         ///< 是否启用评估功能
            bool enable_save_eval_images = true;              ///< 是否保存评估过程中的图像
            bool headless = false;                            ///< 是否禁用训练过程中的可视化
            
            // 渲染和策略相关参数
            std::string render_mode = "RGB";                  ///< 渲染模式：RGB, D, ED, RGB_D, RGB_ED
            std::string strategy = "mcmc";                    ///< 优化策略：mcmc, default
            bool preload_to_ram = false;                      ///< 是否在启动时将整个数据集加载到内存中
            std::string pose_optimization = "none";           ///< 位姿优化类型：none, direct, mlp

            // 双边网格参数
            bool use_bilateral_grid = false;                  ///< 是否使用双边网格
            int bilateral_grid_X = 16;                        ///< 双边网格X方向分辨率
            int bilateral_grid_Y = 16;                        ///< 双边网格Y方向分辨率
            int bilateral_grid_W = 8;                         ///< 双边网格W方向分辨率
            float bilateral_grid_lr = 2e-3f;                 ///< 双边网格的学习率
            float tv_loss_weight = 10.f;                      ///< 总变差损失权重

            // 默认策略特定参数
            float prune_opacity = 0.005f;                     ///< 不透明度修剪阈值
            float grow_scale3d = 0.01f;                      ///< 3D缩放增长参数
            float grow_scale2d = 0.05f;                      ///< 2D缩放增长参数
            float prune_scale3d = 0.1f;                      ///< 3D缩放修剪参数
            float prune_scale2d = 0.15f;                     ///< 2D缩放修剪参数
            size_t reset_every = 3'000;                      ///< 每隔多少步重置一次
            size_t pause_refine_after_reset = 0;             ///< 重置后暂停细化的步数
            bool revised_opacity = false;                    ///< 是否使用修正的不透明度计算
            bool gut = false;                                ///< 是否启用GUT（Gaussian Update Trick）
            float steps_scaler = 0.f;                        ///< 步长缩放因子，小于0时禁用步长缩放
            bool antialiasing = false;                       ///< 是否在渲染中启用抗锯齿

            // 随机初始化参数
            bool random = false;                              ///< 是否使用随机初始化而不是SfM
            int init_num_pts = 100'000;                       ///< 随机初始化点的数量
            float init_extent = 3.0f;                        ///< 随机点云的范围

            /**
             * @brief 将参数转换为JSON格式
             * @return 包含所有参数的JSON对象
             */
            nlohmann::json to_json() const;
            
            /**
             * @brief 从JSON格式解析参数
             * @param j JSON对象
             * @return 解析后的OptimizationParameters对象
             */
            static OptimizationParameters from_json(const nlohmann::json& j);
        };

        /**
         * @struct DatasetConfig
         * @brief 数据集配置结构体
         * @details 定义训练数据集的路径、输出路径、图像设置等配置信息
         */
        struct DatasetConfig {
            std::filesystem::path data_path = "";             ///< 数据集路径
            std::filesystem::path output_path = "";           ///< 输出路径
            std::filesystem::path project_path = "";          ///< 项目路径，如果为相对路径则保存到output_path/project_name.ls
            std::string images = "images";                    ///< 图像文件夹名称
            int resize_factor = -1;                          ///< 图像缩放因子，-1表示不缩放
            int test_every = 8;                               ///< 每隔多少张图像取一张作为测试集
            std::vector<std::string> timelapse_images = {};   ///< 时间流逝图像的路径列表
            int timelapse_every = 50;                         ///< 时间流逝图像生成的间隔
        };

        /**
         * @struct TrainingParameters
         * @brief 训练参数结构体
         * @details 组合数据集配置和优化参数，构成完整的训练参数配置
         */
        struct TrainingParameters {
            DatasetConfig dataset;                            ///< 数据集配置
            OptimizationParameters optimization;              ///< 优化参数

            // 查看器模式特定参数
            std::filesystem::path ply_path = "";              ///< PLY文件路径，用于查看器模式
        };

        /**
         * @brief 从JSON文件读取优化参数
         * @param strategy 优化策略名称
         * @return 成功时返回OptimizationParameters对象，失败时返回错误信息
         * @details 使用现代C++23的expected类型处理可能的错误情况
         */
        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::string strategy);

        /**
         * @brief 将训练参数保存到JSON文件
         * @param params 要保存的训练参数
         * @param output_path 输出文件路径
         * @return 成功时返回void，失败时返回错误信息
         * @details 使用现代C++23的expected类型处理可能的错误情况
         */
        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path);
    } // namespace param
} // namespace gs