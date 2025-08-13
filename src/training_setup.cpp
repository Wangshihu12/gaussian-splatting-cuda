/**
 * [文件描述]：训练环境设置实现文件
 * 功能：负责高斯散点训练的完整环境配置，包括数据加载、模型初始化、策略选择等
 * 用途：为高斯散点训练提供统一的设置接口，支持多种数据格式和训练策略
 */

#include "core/training_setup.hpp"      // 训练设置核心接口
#include "core/default_strategy.hpp"    // 默认训练策略（当前已禁用）
#include "core/mcmc.hpp"               // MCMC训练策略
#include "core/point_cloud.hpp"        // 点云数据处理
#include "loader/loader.hpp"           // 数据加载器接口
#include <format>                      // C++20字符串格式化
#include <print>                       // C++23标准打印功能

namespace gs {

    /**
     * [功能描述]：设置完整的高斯散点训练环境
     * @param params：训练参数配置，包含数据路径、优化设置、策略选择等所有训练相关参数
     * @return 期望值类型，成功时返回完整的训练设置对象，失败时返回错误信息
     * 
     * 训练设置流程：
     * 1. 创建数据加载器
     * 2. 配置加载选项
     * 3. 加载数据集
     * 4. 处理加载的数据（根据类型）
     * 5. 初始化点云和模型
     * 6. 选择训练策略
     * 7. 创建训练器
     */
    std::expected<TrainingSetup, std::string> setupTraining(const param::TrainingParameters& params) {
        
        // =============================================================================
        // 步骤1：创建数据加载器
        // =============================================================================
        // 使用工厂模式创建通用数据加载器，支持多种格式（COLMAP、Blender、PLY等）
        auto loader = loader::Loader::create();

        // =============================================================================
        // 步骤2：配置数据加载选项
        // =============================================================================
        loader::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,     // 图像缩放因子（1, 2, 4, 8）
            .images_folder = params.dataset.images,            // 图像文件夹名称
            .validate_only = false,                            // 不仅验证，执行实际加载
            .progress = [](float percentage, const std::string& message) {
                // 进度回调函数，实时显示加载进度
                std::println("[{:5.1f}%] {}", percentage, message);
            }
        };

        // =============================================================================
        // 步骤3：执行数据集加载
        // =============================================================================
        auto load_result = loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error()));
        }

        std::println("Dataset loaded successfully using {} loader", load_result->loader_used);

        // =============================================================================
        // 步骤4：根据加载的数据类型进行处理
        // =============================================================================
        // 使用std::visit进行类型安全的变体访问
        return std::visit([&params, &load_result](auto&& data) -> std::expected<TrainingSetup, std::string> {
            using T = std::decay_t<decltype(data)>;  // 获取数据的实际类型

            if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                // ---------------------------------------------------------------------
                // 处理直接PLY文件加载（不支持训练）
                // ---------------------------------------------------------------------
                return std::unexpected("Direct PLY loading is not supported for training. Please use a dataset format (COLMAP or Blender).");

            } else if constexpr (std::is_same_v<T, loader::LoadedScene>) {
                // ---------------------------------------------------------------------
                // 处理完整场景数据（支持训练）
                // ---------------------------------------------------------------------

                // 获取或生成初始点云
                PointCloud point_cloud_to_use;
                if (data.point_cloud && data.point_cloud->size() > 0) {
                    // 使用加载的点云数据（来自SfM重建）
                    point_cloud_to_use = *data.point_cloud;
                    std::println("Using point cloud with {} points", point_cloud_to_use.size());
                } else {
                    // 生成随机初始点云（当没有SfM点云时）
                    std::println("No point cloud provided, using random initialization");
                    
                    // 随机点云生成参数
                    int numInitGaussian = 10000;       // 初始高斯点数量
                    uint64_t seed = 8128;              // 随机种子，确保可重现性
                    torch::manual_seed(seed);

                    // 生成随机3D位置：[-1, 1]范围内的均匀分布
                    torch::Tensor positions = torch::rand({numInitGaussian, 3}); // 在[0, 1]范围内
                    positions = positions * 2.0 - 1.0;                           // 缩放到[-1, 1]范围

                    // 生成随机RGB颜色：[0, 255]范围内的整数
                    torch::Tensor colors = torch::randint(0, 256, {numInitGaussian, 3}, torch::kUInt8);

                    // 创建点云对象
                    point_cloud_to_use = PointCloud(positions, colors);
                }

                // ---------------------------------------------------------------------
                // 从点云初始化高斯散点模型
                // ---------------------------------------------------------------------
                auto splat_result = SplatData::init_model_from_pointcloud(
                    params,                         // 训练参数
                    load_result->scene_center,      // 场景中心点
                    point_cloud_to_use              // 初始点云数据
                );

                if (!splat_result) {
                    return std::unexpected(std::format("Failed to initialize model: {}", splat_result.error()));
                }

                // ---------------------------------------------------------------------
                // 步骤5：创建训练策略
                // ---------------------------------------------------------------------
                std::unique_ptr<IStrategy> strategy;
                if (params.optimization.strategy == "mcmc") {
                    // 使用MCMC（马尔可夫链蒙特卡洛）策略
                    strategy = std::make_unique<MCMC>(std::move(*splat_result));
                } else {
                    // 默认策略当前被禁用，等待新光栅化器集成
                    throw std::runtime_error("ADC (default strategy) is currently disabled until the new rasterizer is integrated. Please use MCMC strategy instead.");
                    // strategy = std::make_unique<DefaultStrategy>(std::move(*splat_result));
                }

                // ---------------------------------------------------------------------
                // 步骤6：创建训练器
                // ---------------------------------------------------------------------
                auto trainer = std::make_unique<Trainer>(
                    data.cameras,           // 相机数据集（用于多视角训练）
                    std::move(strategy),    // 训练策略（MCMC或其他）
                    params                  // 训练参数配置
                );

                // ---------------------------------------------------------------------
                // 返回完整的训练设置对象
                // ---------------------------------------------------------------------
                return TrainingSetup{
                    .trainer = std::move(trainer),              // 训练器实例
                    .dataset = data.cameras,                   // 相机数据集
                    .scene_center = load_result->scene_center  // 场景中心点坐标
                };
                
            } else {
                // 未知的数据类型
                return std::unexpected("Unknown data type returned from loader");
            }
        },
                        load_result->data);  // 传入加载结果中的数据变体
    }

} // namespace gs - 高斯散点项目命名空间结束
