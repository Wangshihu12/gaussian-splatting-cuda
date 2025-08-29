#include "training_setup.hpp"
#include "core/point_cloud.hpp"
#include "loader/loader.hpp"
#include "strategies/default_strategy.hpp"
#include "strategies/mcmc.hpp"
#include <format>
#include <print>

namespace gs::training {
    /**
     * [功能描述]：设置训练环境，包括数据加载、点云处理、模型初始化和训练器创建。
     * @param params [参数说明]：训练参数，包含数据集配置、优化策略等设置。
     * @return [返回值说明]：返回TrainingSetup结构，包含训练器、数据集和场景中心信息，失败时返回错误字符串。
     */
    std::expected<TrainingSetup, std::string> setupTraining(const param::TrainingParameters& params) {
        // 步骤1：创建数据加载器
        auto loader = loader::Loader::create();

        // 步骤2：设置加载选项，包括缩放因子、图像文件夹、验证模式和进度回调
        loader::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,  // 图像缩放因子
            .images_folder = params.dataset.images,         // 图像文件夹路径
            .validate_only = false,                         // 不进行验证，直接加载
            .progress = [](float percentage, const std::string& message) {
                // 进度回调函数，显示加载进度百分比和消息
                std::println("[{:5.1f}%] {}", percentage, message);
            }};

        // 步骤3：加载数据集
        auto load_result = loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            // 如果加载失败，返回错误信息
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error()));
        }

        std::println("Dataset loaded successfully using {} loader", load_result->loader_used);

        // 步骤4：根据加载的数据类型进行处理
        return std::visit([&params, &load_result](auto&& data) -> std::expected<TrainingSetup, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                // 直接PLY加载 - 不支持训练，需要数据集格式（COLMAP或Blender）
                return std::unexpected(
                    "Direct PLY loading is not supported for training. Please use a dataset format (COLMAP or Blender).");
            } else if constexpr (std::is_same_v<T, loader::LoadedScene>) {
                // 完整场景数据 - 设置训练环境

                // 获取点云或生成随机点云
                PointCloud point_cloud_to_use;
                if (data.point_cloud && data.point_cloud->size() > 0) {
                    // 如果存在点云数据，直接使用
                    point_cloud_to_use = *data.point_cloud;
                    std::println("Using point cloud with {} points", point_cloud_to_use.size());
                } else {
                    // 如果没有提供点云，生成随机点云作为初始化
                    std::println("No point cloud provided, using random initialization");
                    
                    // 设置随机点云生成参数
                    int numInitGaussian = 10000;  // 初始高斯数量
                    uint64_t seed = 8128;         // 随机种子
                    torch::manual_seed(seed);     // 设置PyTorch随机种子

                    // 生成随机位置：先在[0,1]范围内，然后转换到[-1,1]范围
                    torch::Tensor positions = torch::rand({numInitGaussian, 3}); // 在[0,1]范围内
                    positions = positions * 2.0 - 1.0;                           // 现在在[-1,1]范围内
                    
                    // 生成随机颜色：0-255范围内的整数
                    torch::Tensor colors =
                        torch::randint(0, 256, {numInitGaussian, 3}, torch::kUInt8);

                    // 创建点云对象
                    point_cloud_to_use = PointCloud(positions, colors);
                }

                // 使用点云直接初始化模型
                auto splat_result = SplatData::init_model_from_pointcloud(
                    params,
                    load_result->scene_center,  // 场景中心点
                    point_cloud_to_use);        // 点云数据

                if (!splat_result) {
                    // 如果模型初始化失败，返回错误信息
                    return std::unexpected(
                        std::format("Failed to initialize model: {}", splat_result.error()));
                }

                // 步骤5：创建训练策略
                std::unique_ptr<IStrategy> strategy;
                if (params.optimization.strategy == "mcmc") {
                    // 如果策略是MCMC，创建MCMC策略实例
                    strategy = std::make_unique<MCMC>(std::move(*splat_result));
                } else {
                    // 否则使用默认策略
                    strategy = std::make_unique<DefaultStrategy>(std::move(*splat_result));
                }

                // 创建训练器，传入相机数据、策略和参数
                auto trainer = std::make_unique<Trainer>(
                    data.cameras,      // 相机数据
                    std::move(strategy), // 训练策略
                    params);           // 训练参数

                // 返回训练设置，包含训练器、数据集和场景中心
                return TrainingSetup{
                    .trainer = std::move(trainer),
                    .dataset = data.cameras,
                    .scene_center = load_result->scene_center};
            } else {
                // 未知的数据类型，返回错误
                return std::unexpected("Unknown data type returned from loader");
            }
        },
                          load_result->data);  // 访问加载的数据
    }
} // namespace gs::training
