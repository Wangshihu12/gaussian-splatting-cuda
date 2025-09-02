#include "core/splat_data.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"

#include "external/nanoflann.hpp"
#include "external/tinyply.hpp"
#include <algorithm>
#include <cmath>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <print>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

namespace {
    /**
     * @brief 将张量尺寸转换为字符串表示
     * @param sizes 张量的尺寸数组引用
     * @return 返回格式化的尺寸字符串，如"[batch_size, height, width]"
     * @details 该函数将PyTorch张量的尺寸信息转换为可读的字符串格式，
     *          用于调试和日志输出。输出格式为方括号包围的逗号分隔列表。
     */
    std::string tensor_sizes_to_string(const c10::ArrayRef<int64_t>& sizes) {
        std::ostringstream oss;  // 创建字符串流用于构建输出字符串
        oss << "[";              // 开始方括号
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (i > 0)
                oss << ", ";     // 在第一个元素后添加逗号分隔符
            oss << sizes[i];     // 输出当前尺寸值
        }
        oss << "]";             // 结束方括号
        return oss.str();       // 返回构建的字符串
    }

    /**
     * @struct PointCloudAdaptor
     * @brief 点云适配器，用于nanoflann库的KD树索引
     * @details 该结构体实现了nanoflann库所需的接口，将原始点云数据适配为KD树可处理的格式。
     *          支持3D点云数据的快速最近邻搜索。
     */
    struct PointCloudAdaptor {
        const float* points;     ///< 指向点云数据的指针，每个点包含3个float值(x,y,z)
        size_t num_points;      ///< 点云中点的总数

        /**
         * @brief 构造函数
         * @param pts 指向点云数据的指针
         * @param n 点的数量
         * @details 初始化点云适配器，设置数据指针和点数量
         */
        PointCloudAdaptor(const float* pts, size_t n) : points(pts),
                                                        num_points(n) {}

        /**
         * @brief 获取点云中点的数量
         * @return 返回点的总数
         * @details nanoflann库要求的接口函数，用于KD树构建
         */
        inline size_t kdtree_get_point_count() const { return num_points; }

        /**
         * @brief 获取指定点的指定维度坐标
         * @param idx 点的索引
         * @param dim 维度索引(0=x, 1=y, 2=z)
         * @return 返回指定点的指定维度坐标值
         * @details nanoflann库要求的接口函数，用于访问点云数据。
         *          假设数据按行优先存储，每个点占用3个连续的float值。
         */
        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            return points[idx * 3 + dim];  // 计算内存偏移：点索引*3 + 维度
        }

        /**
         * @brief 获取点云的边界框（未实现）
         * @param bb 边界框参数（未使用）
         * @return 始终返回false，表示不提供边界框信息
         * @details nanoflann库的可选接口函数，此处未实现边界框功能
         */
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    };

    /**
     * @typedef KDTree
     * @brief KD树类型定义，用于3D点云的快速最近邻搜索
     * @details 使用nanoflann库实现，支持L2距离度量的3D点云KD树索引。
     *          模板参数说明：
     *          - nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>: L2距离适配器
     *          - PointCloudAdaptor: 点云数据适配器
     *          - 3: 3D空间维度
     */
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor, 3>;

    /**
     * @brief 计算每个点到其3个最近邻的平均距离
     * @param points 输入点云张量，形状为[N, 3]，包含N个3D点的坐标
     * @return 返回每个点的平均邻居距离张量，形状为[N]
     * @details 该函数使用KD树快速计算每个点到其最近3个邻居的平均距离。
     *          这个距离信息常用于高斯散射体的初始化，帮助确定高斯体的初始缩放参数。
     *          函数会自动处理边界情况，如点数量过少或距离过近的情况。
     */
    torch::Tensor compute_mean_neighbor_distances(const torch::Tensor& points) {
        // 将点云数据转移到CPU并确保内存连续，以便nanoflann库处理
        auto cpu_points = points.to(torch::kCPU).contiguous();
        const int num_points = cpu_points.size(0);

        // 输入验证：确保点云形状正确[N, 3]且数据类型为float32
        TORCH_CHECK(cpu_points.dim() == 2 && cpu_points.size(1) == 3,
                    "Input points must have shape [N, 3]");
        TORCH_CHECK(cpu_points.dtype() == torch::kFloat32,
                    "Input points must be float32");

        // 边界情况处理：如果点数量太少，返回默认距离值
        if (num_points <= 1) {
            return torch::full({num_points}, 0.01f, points.options());
        }

        // 获取点云数据的原始指针，用于nanoflann库
        const float* data = cpu_points.data_ptr<float>();

        // 创建点云适配器和KD树索引
        PointCloudAdaptor cloud(data, num_points);
        KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));  // 3D空间，10个叶子节点
        index.buildIndex();  // 构建KD树索引

        // 创建结果张量，用于存储每个点的平均邻居距离
        auto result = torch::zeros({num_points}, torch::kFloat32);
        float* result_data = result.data_ptr<float>();

        // 并行处理每个点（当点数量大于1000时启用OpenMP并行）
#pragma omp parallel for if (num_points > 1000)
        for (int i = 0; i < num_points; i++) {
            // 提取当前查询点的3D坐标
            const float query_pt[3] = {data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]};

            // 设置搜索结果数量（最多4个，但只使用前3个有效邻居）
            const size_t num_results = std::min(4, num_points);
            std::vector<size_t> ret_indices(num_results);      // 邻居点的索引
            std::vector<float> out_dists_sqr(num_results);    // 距离的平方值

            // 创建KNN结果集并初始化
            nanoflann::KNNResultSet<float> resultSet(num_results);
            resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
            // 执行最近邻搜索
            index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));

            // 计算有效邻居的平均距离
            float sum_dist = 0.0f;
            int valid_neighbors = 0;

            // 遍历搜索结果，计算前3个有效邻居的平均距离
            for (size_t j = 0; j < num_results && valid_neighbors < 3; j++) {
                // 过滤掉距离过近的点（避免数值精度问题）
                if (out_dists_sqr[j] > 1e-8f) {
                    sum_dist += std::sqrt(out_dists_sqr[j]);  // 将平方距离转换为欧几里得距离
                    valid_neighbors++;
                }
            }

            // 计算平均距离，如果没有有效邻居则使用默认值
            result_data[i] = (valid_neighbors > 0) ? (sum_dist / valid_neighbors) : 0.01f;
        }

        // 将结果张量移回原始设备（GPU/CPU）
        return result.to(points.device());
    }

    void write_ply_impl(const gs::PointCloud& pc,
                        const std::filesystem::path& root,
                        int iteration) {
        namespace fs = std::filesystem;
        fs::create_directories(root);

        std::vector<torch::Tensor> tensors;
        tensors.push_back(pc.means);

        if (pc.normals.defined())
            tensors.push_back(pc.normals);
        if (pc.sh0.defined())
            tensors.push_back(pc.sh0);
        if (pc.shN.defined())
            tensors.push_back(pc.shN);
        if (pc.opacity.defined())
            tensors.push_back(pc.opacity);
        if (pc.scaling.defined())
            tensors.push_back(pc.scaling);
        if (pc.rotation.defined())
            tensors.push_back(pc.rotation);

        auto write_output_ply =
            [](const fs::path& file_path,
               const std::vector<torch::Tensor>& data,
               const std::vector<std::string>& attr_names) {
                tinyply::PlyFile ply;
                size_t attr_off = 0;

                for (const auto& tensor : data) {
                    const size_t cols = tensor.size(1);
                    std::vector<std::string> attrs(attr_names.begin() + attr_off,
                                                   attr_names.begin() + attr_off + cols);

                    ply.add_properties_to_element(
                        "vertex",
                        attrs,
                        tinyply::Type::FLOAT32,
                        tensor.size(0),
                        reinterpret_cast<uint8_t*>(tensor.data_ptr<float>()),
                        tinyply::Type::INVALID, 0);

                    attr_off += cols;
                }

                std::filebuf fb;
                fb.open(file_path, std::ios::out | std::ios::binary);
                std::ostream out_stream(&fb);
                ply.write(out_stream, /*binary=*/true);
            };

        write_output_ply(root / ("splat_" + std::to_string(iteration) + ".ply"), tensors, pc.attribute_names);
    }
} // namespace

namespace gs {
    SplatData::~SplatData() {
        // Wait for all save threads to complete
        std::lock_guard<std::mutex> lock(_threads_mutex);
        for (auto& t : _save_threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    // Move constructor
    SplatData::SplatData(SplatData&& other) noexcept
        : _active_sh_degree(other._active_sh_degree),
          _max_sh_degree(other._max_sh_degree),
          _scene_scale(other._scene_scale),
          _means(std::move(other._means)),
          _sh0(std::move(other._sh0)),
          _shN(std::move(other._shN)),
          _scaling(std::move(other._scaling)),
          _rotation(std::move(other._rotation)),
          _opacity(std::move(other._opacity)),
          _densification_info(std::move(other._densification_info)) {
        // Move threads under lock
        std::lock_guard<std::mutex> lock(other._threads_mutex);
        _save_threads = std::move(other._save_threads);
    }

    // Move assignment operator
    SplatData& SplatData::operator=(SplatData&& other) noexcept {
        if (this != &other) {
            // First, wait for our own threads to complete
            {
                std::lock_guard<std::mutex> lock(_threads_mutex);
                for (auto& t : _save_threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
            }

            // Move scalar members
            _active_sh_degree = other._active_sh_degree;
            _max_sh_degree = other._max_sh_degree;
            _scene_scale = other._scene_scale;

            // Move tensors
            _means = std::move(other._means);
            _sh0 = std::move(other._sh0);
            _shN = std::move(other._shN);
            _scaling = std::move(other._scaling);
            _rotation = std::move(other._rotation);
            _opacity = std::move(other._opacity);
            _densification_info = other._densification_info;

            // Move threads under lock
            std::lock_guard<std::mutex> lock(other._threads_mutex);
            _save_threads = std::move(other._save_threads);
        }
        return *this;
    }

    // Constructor from tensors
    SplatData::SplatData(int sh_degree,
                         torch::Tensor means,
                         torch::Tensor sh0,
                         torch::Tensor shN,
                         torch::Tensor scaling,
                         torch::Tensor rotation,
                         torch::Tensor opacity,
                         float scene_scale)
        : _max_sh_degree{sh_degree},
          _active_sh_degree{0},
          _scene_scale{scene_scale},
          _means{std::move(means)},
          _sh0{std::move(sh0)},
          _shN{std::move(shN)},
          _scaling{std::move(scaling)},
          _rotation{std::move(rotation)},
          _opacity{std::move(opacity)} {}

    // Computed getters
    torch::Tensor SplatData::get_means() const {
        return _means;
    }

    torch::Tensor SplatData::get_opacity() const {
        return torch::sigmoid(_opacity).squeeze(-1);
    }

    torch::Tensor SplatData::get_rotation() const {
        return torch::nn::functional::normalize(_rotation,
                                                torch::nn::functional::NormalizeFuncOptions().dim(-1));
    }

    torch::Tensor SplatData::get_scaling() const {
        return torch::exp(_scaling);
    }

    torch::Tensor SplatData::get_shs() const {
        return torch::cat({_sh0, _shN}, 1);
    }

    SplatData& SplatData::transform(const glm::mat4& transform_matrix) {
        LOG_TIMER("SplatData::transform");

        if (_means.size(0) == 0) {
            return *this; // Nothing to transform
        }

        const int num_points = _means.size(0);

        // Keep everything on GPU for efficiency
        auto device = _means.device();

        // 1. Transform positions (means)
        // Convert transform matrix to tensor
        auto transform_tensor = torch::tensor({transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2], transform_matrix[0][3],
                                               transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2], transform_matrix[1][3],
                                               transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2], transform_matrix[2][3],
                                               transform_matrix[3][0], transform_matrix[3][1], transform_matrix[3][2], transform_matrix[3][3]},
                                              torch::TensorOptions().dtype(torch::kFloat32).device(device))
                                    .reshape({4, 4});

        // Add homogeneous coordinate
        auto means_homo = torch::cat({_means, torch::ones({num_points, 1}, _means.options())}, 1);

        // Apply transform: (4x4) @ (Nx4)^T = (4xN), then transpose back
        auto transformed_means = torch::matmul(transform_tensor, means_homo.t()).t();

        // Extract xyz and update in-place
        _means.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                          transformed_means.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));

        // 2. Extract rotation from transform matrix (simple method without decompose)
        glm::mat3 rot_mat(transform_matrix);

        // Normalize columns to remove scale
        glm::vec3 scale;
        for (int i = 0; i < 3; ++i) {
            scale[i] = glm::length(rot_mat[i]);
            if (scale[i] > 0.0f) {
                rot_mat[i] /= scale[i];
            }
        }

        // Convert rotation matrix to quaternion
        glm::quat rotation = glm::quat_cast(rot_mat);

        // 3. Transform rotations (quaternions) if there's rotation
        if (std::abs(rotation.w - 1.0f) > 1e-6f) {
            auto rot_tensor = torch::tensor({rotation.w, rotation.x, rotation.y, rotation.z},
                                            torch::TensorOptions().dtype(torch::kFloat32).device(device));

            // Quaternion multiplication: q_new = q_transform * q_original
            auto q = _rotation; // Shape: [N, 4] in [w, x, y, z] format

            // Expand rotation quaternion to match batch size
            auto q_rot = rot_tensor.unsqueeze(0).expand({num_points, 4});

            // Quaternion multiplication formula
            auto w1 = q_rot.index({torch::indexing::Slice(), 0});
            auto x1 = q_rot.index({torch::indexing::Slice(), 1});
            auto y1 = q_rot.index({torch::indexing::Slice(), 2});
            auto z1 = q_rot.index({torch::indexing::Slice(), 3});

            auto w2 = q.index({torch::indexing::Slice(), 0});
            auto x2 = q.index({torch::indexing::Slice(), 1});
            auto y2 = q.index({torch::indexing::Slice(), 2});
            auto z2 = q.index({torch::indexing::Slice(), 3});

            _rotation.index_put_({torch::indexing::Slice(), 0}, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2);
            _rotation.index_put_({torch::indexing::Slice(), 1}, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2);
            _rotation.index_put_({torch::indexing::Slice(), 2}, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2);
            _rotation.index_put_({torch::indexing::Slice(), 3}, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2);
        }

        // 4. Transform scaling (if non-uniform scale is present)
        if (std::abs(scale.x - 1.0f) > 1e-6f ||
            std::abs(scale.y - 1.0f) > 1e-6f ||
            std::abs(scale.z - 1.0f) > 1e-6f) {

            // Average scale factor (for isotropic gaussian scaling)
            float avg_scale = (scale.x + scale.y + scale.z) / 3.0f;

            // Since _scaling is log(scale), we add log of the scale factor
            _scaling = _scaling + std::log(avg_scale);
        }

        // 5. Update scene scale if significant change
        torch::Tensor scene_center = _means.mean(0);
        torch::Tensor dists = torch::norm(_means - scene_center, 2, 1);
        float new_scene_scale = dists.median().item<float>();
        if (std::abs(new_scene_scale - _scene_scale) > _scene_scale * 0.1f) {
            _scene_scale = new_scene_scale;
        }

        LOG_DEBUG("Transformed {} gaussians", num_points);
        return *this;
    }

    // Utility method
    void SplatData::increment_sh_degree() {
        if (_active_sh_degree < _max_sh_degree) {
            _active_sh_degree++;
        }
    }

    // Get attribute names for PLY format
    std::vector<std::string> SplatData::get_attribute_names() const {
        std::vector<std::string> a{"x", "y", "z", "nx", "ny", "nz"};

        for (int i = 0; i < _sh0.size(1) * _sh0.size(2); ++i)
            a.emplace_back("f_dc_" + std::to_string(i));
        for (int i = 0; i < _shN.size(1) * _shN.size(2); ++i)
            a.emplace_back("f_rest_" + std::to_string(i));

        a.emplace_back("opacity");

        for (int i = 0; i < _scaling.size(1); ++i)
            a.emplace_back("scale_" + std::to_string(i));
        for (int i = 0; i < _rotation.size(1); ++i)
            a.emplace_back("rot_" + std::to_string(i));

        return a;
    }

    void SplatData::cleanup_finished_threads() const {
        std::lock_guard<std::mutex> lock(_threads_mutex);

        // Remove threads that have finished
        _save_threads.erase(
            std::remove_if(_save_threads.begin(), _save_threads.end(),
                           [](std::thread& t) {
                               if (t.joinable()) {
                                   // Try to join with zero timeout to check if finished
                                   // Since C++11 doesn't have try_join, we'll keep all threads
                                   return false;
                               }
                               return true;
                           }),
            _save_threads.end());
    }

    // Export to PLY
    void SplatData::save_ply(const std::filesystem::path& root, int iteration, bool join_thread) const {
        auto pc = to_point_cloud();

        if (join_thread) {
            // Synchronous save
            write_ply_impl(pc, root, iteration);
        } else {
            // Clean up any finished threads first
            cleanup_finished_threads();

            // Asynchronous save with thread tracking
            std::lock_guard<std::mutex> lock(_threads_mutex);
            _save_threads.emplace_back([pc = std::move(pc), root, iteration]() {
                write_ply_impl(pc, root, iteration);
            });
        }
    }

    PointCloud SplatData::to_point_cloud() const {
        PointCloud pc;

        // Basic attributes
        pc.means = _means.cpu().contiguous();
        pc.normals = torch::zeros_like(pc.means);

        // Gaussian attributes
        pc.sh0 = _sh0.transpose(1, 2).flatten(1).cpu();
        pc.shN = _shN.transpose(1, 2).flatten(1).cpu();
        pc.opacity = _opacity.cpu();
        pc.scaling = _scaling.cpu();
        pc.rotation = _rotation.cpu();

        // Set attribute names for PLY export
        pc.attribute_names = get_attribute_names();

        return pc;
    }

    std::expected<SplatData, std::string> SplatData::init_model_from_pointcloud(
        const param::TrainingParameters& params,
        torch::Tensor scene_center,
        const PointCloud& pcd) {

        try {
            // Generate positions and colors based on init type
            torch::Tensor positions, colors;
            if (params.optimization.random) {
                const int num_points = params.optimization.init_num_pts;
                const float extent = params.optimization.init_extent;
                const auto f32_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

                positions = (torch::rand({num_points, 3}, f32_cuda) * 2.0f - 1.0f) * extent;
                colors = torch::rand({num_points, 3}, f32_cuda);
            } else {
                positions = pcd.means;
                colors = pcd.colors / 255.0f; // Normalize directly
            }

            scene_center = scene_center.to(positions.device());
            const torch::Tensor dists = torch::norm(positions - scene_center, 2, 1);
            const auto scene_scale = dists.median().item<float>();

            auto rgb_to_sh = [](const torch::Tensor& rgb) {
                constexpr float kInvSH = 0.28209479177387814f;
                return (rgb - 0.5f) / kInvSH;
            };

            const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
            const auto f32_cuda = f32.device(torch::kCUDA);

            // 1. means
            torch::Tensor means;
            if (params.optimization.random) {
                // Scale positions before setting requires_grad
                means = (positions * scene_scale).to(torch::kCUDA).set_requires_grad(true);
            } else {
                means = positions.to(torch::kCUDA).set_requires_grad(true);
            }

            // 2. scaling (log(σ))
            auto nn_dist = torch::clamp_min(compute_mean_neighbor_distances(means), 1e-7);
            auto scaling = torch::log(torch::sqrt(nn_dist) * params.optimization.init_scaling)
                               .unsqueeze(-1)
                               .repeat({1, 3})
                               .to(f32_cuda)
                               .set_requires_grad(true);

            // 3. rotation (quaternion, identity) - split into multiple lines to avoid compilation error
            auto rotation = torch::zeros({means.size(0), 4}, f32_cuda);
            rotation.index_put_({torch::indexing::Slice(), 0}, 1);
            rotation = rotation.set_requires_grad(true);

            // 4. opacity (inverse sigmoid of 0.5)
            auto opacity = torch::logit(params.optimization.init_opacity * torch::ones({means.size(0), 1}, f32_cuda))
                               .set_requires_grad(true);

            // 5. shs (SH coefficients)
            auto colors_float = colors.to(torch::kCUDA);
            auto fused_color = rgb_to_sh(colors_float);

            const int64_t feature_shape = static_cast<int64_t>(std::pow(params.optimization.sh_degree + 1, 2));
            auto shs = torch::zeros({fused_color.size(0), 3, feature_shape}, f32_cuda);

            // Set DC coefficients
            shs.index_put_({torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            0},
                           fused_color);

            auto sh0 = shs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(0, 1)})
                           .transpose(1, 2)
                           .contiguous()
                           .set_requires_grad(true);

            auto shN = shs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(1, torch::indexing::None)})
                           .transpose(1, 2)
                           .contiguous()
                           .set_requires_grad(true);

            std::println("Scene scale: {}", scene_scale);
            std::println("Initialized SplatData with:");
            std::println("  - {} points", means.size(0));
            std::println("  - Max SH degree: {}", params.optimization.sh_degree);
            std::println("  - Total SH coefficients: {}", feature_shape);
            std::cout << std::format("  - sh0 shape: {}\n", tensor_sizes_to_string(sh0.sizes()));
            std::cout << std::format("  - shN shape: {}\n", tensor_sizes_to_string(shN.sizes()));

            return SplatData(
                params.optimization.sh_degree,
                means.contiguous(),
                sh0.contiguous(),
                shN.contiguous(),
                scaling.contiguous(),
                rotation.contiguous(),
                opacity.contiguous(),
                scene_scale);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize SplatData: {}", e.what()));
        }
    }
} // namespace gs