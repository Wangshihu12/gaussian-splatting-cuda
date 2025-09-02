#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "loader/loader.hpp"
#include <expected>
#include <format>
#include <memory>
#include <torch/torch.h>
#include <vector>

// Camera with loaded image
namespace gs::training {
    struct CameraWithImage {
        Camera* camera;
        torch::Tensor image;
    };

    using CameraExample = torch::data::Example<CameraWithImage, torch::Tensor>;

    class CameraDataset : public torch::data::Dataset<CameraDataset, CameraExample> {
    public:
        enum class Split {
            TRAIN,
            VAL,
            ALL
        };

        CameraDataset(std::vector<std::shared_ptr<Camera>> cameras,
                      const gs::param::DatasetConfig& params,
                      Split split = Split::ALL)
            : _cameras(std::move(cameras)),
              _datasetConfig(params),
              _split(split) {
            // Create indices based on split
            _indices.clear();
            for (size_t i = 0; i < _cameras.size(); ++i) {
                const bool is_test = (i % params.test_every) == 0;

                if (_split == Split::ALL ||
                    (_split == Split::TRAIN && !is_test) ||
                    (_split == Split::VAL && is_test)) {
                    _indices.push_back(i);
                }
            }

            std::cout << "Dataset created with " << _indices.size()
                      << " images (split: " << static_cast<int>(_split) << ")" << std::endl;
        }

        // Default copy constructor works with shared_ptr
        CameraDataset(const CameraDataset&) = default;

        CameraDataset(CameraDataset&&) noexcept = default;

        CameraDataset& operator=(CameraDataset&&) noexcept = default;

        CameraDataset& operator=(const CameraDataset&) = default;

        CameraExample get(size_t index) override {
            if (index >= _indices.size()) {
                throw std::out_of_range("Dataset index out of range");
            }

            size_t camera_idx = _indices[index];
            auto& cam = _cameras[camera_idx];

            torch::Tensor image = cam->load_and_get_image(_datasetConfig.resize_factor);
            return {{cam.get(), std::move(image)}, torch::empty({})};
        }

        torch::optional<size_t> size() const override {
            return _indices.size();
        }

        const std::vector<std::shared_ptr<Camera>>& get_cameras() const {
            return _cameras;
        }

        Split get_split() const { return _split; }

        size_t get_num_bytes() const {
            if (_cameras.empty()) {
                return 0;
            }
            size_t total_bytes = 0;
            for (const auto& cam : _cameras) {
                total_bytes += cam->get_num_bytes_from_file();
            }
            // Adjust for resolution factor if specified
            if (_datasetConfig.resize_factor > 0) {
                total_bytes /= _datasetConfig.resize_factor * _datasetConfig.resize_factor;
            }
            return total_bytes;
        }

        [[nodiscard]] std::optional<Camera*> get_camera_by_filename(const std::string& filename) const {
            for (const auto& cam : _cameras) {
                if (cam->image_name() == filename) {
                    return cam.get();
                }
            }
            return std::nullopt;
        }

    private:
        std::vector<std::shared_ptr<Camera>> _cameras;
        const gs::param::DatasetConfig _datasetConfig;
        Split _split;
        std::vector<size_t> _indices;
    };

    /**
     * @class InfiniteRandomSampler
     * @brief 无限随机采样器，用于实现连续的数据流
     * @details 继承自PyTorch的RandomSampler，当数据集遍历完毕后会自动重置并重新开始，
     *          确保训练过程中数据永不耗尽。这对于长时间训练特别有用。
     */
    class InfiniteRandomSampler : public torch::data::samplers::RandomSampler {
    public:
        using super = torch::data::samplers::RandomSampler;  ///< 父类类型别名

        /**
         * @brief 构造函数
         * @param dataset_size 数据集的大小
         * @details 初始化采样器，设置数据集大小
         */
        explicit InfiniteRandomSampler(size_t dataset_size)
            : super(dataset_size) {
        }

        /**
         * @brief 获取下一批数据的索引
         * @param batch_size 批次大小
         * @return 返回下一批数据的索引向量，如果数据用完则返回空
         * @details 重写父类的next方法，当数据用完时自动重置采样器并重新开始
         */
        std::optional<std::vector<size_t>> next(size_t batch_size) override {
            // 尝试从父类获取下一批索引
            auto indices = super::next(batch_size);
            // 如果没有更多数据（indices为空），则重置采样器并重新获取
            if (!indices) {
                super::reset();                    // 重置采样器状态
                indices = super::next(batch_size); // 重新获取索引
            }
            return indices;
        }

    private:
        size_t dataset_size_;  ///< 数据集大小，用于内部状态管理
    };

    inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
    create_dataset_from_colmap(const gs::param::DatasetConfig& datasetConfig) {
        try {
            if (!std::filesystem::exists(datasetConfig.data_path)) {
                return std::unexpected(std::format("Data path does not exist: {}",
                                                   datasetConfig.data_path.string()));
            }

            // Create loader
            auto loader = gs::loader::Loader::create();

            // Set up load options
            gs::loader::LoadOptions options{
                .resize_factor = datasetConfig.resize_factor,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load COLMAP dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit(
                [&result](
                    auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string> {
                    using T = std::decay_t<decltype(data)>;

                    if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                        return std::unexpected("Expected COLMAP dataset but got PLY file");
                    } else if constexpr (std::is_same_v<T, gs::loader::LoadedScene>) {
                        if (!data.cameras) {
                            return std::unexpected("Loaded scene has no cameras");
                        }
                        // Return the cameras that were already loaded
                        return std::make_tuple(data.cameras, result->scene_center);
                    } else {
                        return std::unexpected("Unknown data type returned from loader");
                    }
                },
                result->data);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to create dataset from COLMAP: {}", e.what()));
        }
    }

    inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
    create_dataset_from_transforms(const gs::param::DatasetConfig& datasetConfig) {
        try {
            if (!std::filesystem::exists(datasetConfig.data_path)) {
                return std::unexpected(std::format("Data path does not exist: {}",
                                                   datasetConfig.data_path.string()));
            }

            // Create loader
            auto loader = gs::loader::Loader::create();

            // Set up load options
            gs::loader::LoadOptions options{
                .resize_factor = datasetConfig.resize_factor,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load transforms dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit(
                [&datasetConfig, &result](
                    auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string> {
                    using T = std::decay_t<decltype(data)>;

                    if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                        return std::unexpected("Expected transforms.json dataset but got PLY file");
                    } else if constexpr (std::is_same_v<T, gs::loader::LoadedScene>) {
                        if (!data.cameras) {
                            return std::unexpected("Loaded scene has no cameras");
                        }
                        // Return the cameras that were already loaded
                        return std::make_tuple(data.cameras, result->scene_center);
                    } else {
                        return std::unexpected("Unknown data type returned from loader");
                    }
                },
                result->data);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to create dataset from transforms: {}", e.what()));
        }
    }

    inline auto create_dataloader_from_dataset(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers = 4) {
        const size_t dataset_size = dataset->size().value();

        return torch::data::make_data_loader(
            *dataset,
            torch::data::samplers::RandomSampler(dataset_size),
            torch::data::DataLoaderOptions()
                .batch_size(1)
                .workers(num_workers)
                .enforce_ordering(false));
    }

    /**
     * @brief 从数据集创建无限循环的数据加载器
     * @param dataset 相机数据集，包含所有训练用的相机和图像
     * @param num_workers 数据加载的工作线程数量，默认为4
     * @return 返回一个无限循环的PyTorch数据加载器
     * @details 该函数创建一个无限循环的数据加载器，当数据集遍历完毕后会自动重新开始，
     *          确保训练过程中数据不会用完。使用InfiniteRandomSampler实现无限循环功能。
     */
    inline auto create_infinite_dataloader_from_dataset(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers = 4) {
        // 获取数据集的大小，用于初始化采样器
        const size_t dataset_size = dataset->size().value();

        // 创建PyTorch数据加载器，配置无限随机采样
        return torch::data::make_data_loader(
            *dataset,                                    // 传入数据集
            InfiniteRandomSampler(dataset_size),        // 使用无限随机采样器，确保数据永不耗尽
            torch::data::DataLoaderOptions()
                .batch_size(1)                          // 批次大小为1，每次返回一个样本
                .workers(num_workers)                    // 设置工作线程数量，用于并行数据加载
                .enforce_ordering(false));              // 不强制保持数据顺序，提高性能
    }
} // namespace gs::training
