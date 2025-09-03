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

    /**
     * [功能描述]：将高斯点云数据写入PLY格式文件的实现函数
     * @param pc：高斯点云数据结构，包含位置、法线、球谐系数、不透明度、缩放和旋转等属性
     * @param root：输出目录路径，PLY文件将保存在此目录下
     * @param iteration：当前训练迭代次数，用于生成唯一的文件名
     * @details 该函数负责将高斯散射体的所有参数数据序列化保存为PLY格式文件，
     *          便于后续可视化或重新加载。PLY格式是一种常用的3D点云存储格式。
     */
    void write_ply_impl(const gs::PointCloud& pc,
                        const std::filesystem::path& root,
                        int iteration) {
        namespace fs = std::filesystem;
        // 确保输出目录存在，如果不存在则创建
        fs::create_directories(root);

        // 收集所有需要保存的张量数据，按照PLY格式的属性顺序排列
        std::vector<torch::Tensor> tensors;
        
        // 位置信息（x, y, z坐标）是必需的，始终添加
        tensors.push_back(pc.means);

        // 条件性添加可选属性：只有当张量已定义时才添加到输出列表中
        if (pc.normals.defined())        // 法线向量（nx, ny, nz）
            tensors.push_back(pc.normals);
        if (pc.sh0.defined())           // 0阶球谐系数（直流分量，用于基础颜色）
            tensors.push_back(pc.sh0);
        if (pc.shN.defined())           // 高阶球谐系数（用于复杂光照效果）
            tensors.push_back(pc.shN);
        if (pc.opacity.defined())       // 不透明度值
            tensors.push_back(pc.opacity);
        if (pc.scaling.defined())       // 高斯体的缩放参数
            tensors.push_back(pc.scaling);
        if (pc.rotation.defined())      // 高斯体的旋转四元数
            tensors.push_back(pc.rotation);

        // 定义Lambda函数用于实际的PLY文件写入操作
        auto write_output_ply =
            [](const fs::path& file_path,                           // PLY文件的完整路径
               const std::vector<torch::Tensor>& data,             // 要写入的张量数据数组
               const std::vector<std::string>& attr_names) {       // 对应的属性名称数组
                
                // 创建tinyply库的PLY文件对象
                tinyply::PlyFile ply;
                size_t attr_off = 0;    // 属性名称偏移量，用于跟踪当前处理的属性位置

                // 遍历每个张量，将其添加为PLY文件的属性
                for (const auto& tensor : data) {
                    const size_t cols = tensor.size(1);    // 获取张量的列数（属性维度）
                    
                    // 从属性名称数组中提取当前张量对应的属性名称
                    std::vector<std::string> attrs(attr_names.begin() + attr_off,
                                                   attr_names.begin() + attr_off + cols);

                    // 将张量数据添加到PLY文件的"vertex"元素中
                    ply.add_properties_to_element(
                        "vertex",                                           // 元素类型为顶点
                        attrs,                                             // 属性名称列表
                        tinyply::Type::FLOAT32,                           // 数据类型为32位浮点数
                        tensor.size(0),                                   // 顶点数量（张量的行数）
                        reinterpret_cast<uint8_t*>(tensor.data_ptr<float>()), // 原始数据指针
                        tinyply::Type::INVALID, 0);                       // 列表类型和列表计数（未使用）

                    // 更新属性偏移量，指向下一个张量的属性名称起始位置
                    attr_off += cols;
                }

                // 创建文件缓冲区并以二进制模式打开文件
                std::filebuf fb;
                fb.open(file_path, std::ios::out | std::ios::binary);
                std::ostream out_stream(&fb);
                
                // 将PLY数据以二进制格式写入文件（二进制格式更紧凑高效）
                ply.write(out_stream, /*binary=*/true);
            };

        // 调用Lambda函数执行实际的文件写入操作
        // 文件名格式：splat_[迭代次数].ply，例如：splat_1000.ply
        write_output_ply(root / ("splat_" + std::to_string(iteration) + ".ply"), tensors, pc.attribute_names);
    }
} // namespace

namespace gs {
    /**
     * [功能描述]：SplatData类的析构函数，负责清理资源并等待所有保存线程完成
     * @details 析构函数确保在对象销毁前，所有异步保存PLY文件的线程都已完成执行，
     *          避免在对象销毁后仍有线程访问已释放的内存，防止数据竞争和内存泄漏。
     */
    SplatData::~SplatData() {
        // 等待所有保存线程完成执行
        std::lock_guard<std::mutex> lock(_threads_mutex);  // 获取线程互斥锁，确保线程安全
        for (auto& t : _save_threads) {                     // 遍历所有保存线程
            if (t.joinable()) {                             // 检查线程是否可以join（尚未完成）
                t.join();                                   // 等待线程完成执行
            }
        }
    }

    /**
     * [功能描述]：SplatData类的移动构造函数，实现高效的资源转移
     * @param other：要移动的源对象（右值引用）
     * @details 移动构造函数通过转移所有权而不是复制来高效地创建新对象。
     *          所有张量数据通过std::move转移，避免不必要的内存拷贝。
     *          线程列表在锁保护下进行转移，确保线程安全。
     */
    SplatData::SplatData(SplatData&& other) noexcept
        : _active_sh_degree(other._active_sh_degree),      // 复制标量成员：当前激活的球谐度数
          _max_sh_degree(other._max_sh_degree),            // 复制标量成员：最大球谐度数
          _scene_scale(other._scene_scale),                // 复制标量成员：场景缩放因子
          _means(std::move(other._means)),                 // 移动张量：高斯体的位置坐标
          _sh0(std::move(other._sh0)),                     // 移动张量：0阶球谐系数（直流分量）
          _shN(std::move(other._shN)),                     // 移动张量：高阶球谐系数
          _scaling(std::move(other._scaling)),             // 移动张量：高斯体的缩放参数（对数形式）
          _rotation(std::move(other._rotation)),           // 移动张量：高斯体的旋转四元数
          _opacity(std::move(other._opacity)),             // 移动张量：高斯体的不透明度（对数形式）
          _densification_info(std::move(other._densification_info)) { // 移动：密度化信息
        // 在锁保护下移动线程列表，确保线程安全
        std::lock_guard<std::mutex> lock(other._threads_mutex);
        _save_threads = std::move(other._save_threads);    // 移动保存线程列表
    }

    /**
     * [功能描述]：SplatData类的移动赋值运算符，实现高效的资源转移
     * @param other：要移动的源对象（右值引用）
     * @return 返回当前对象的引用
     * @details 移动赋值运算符首先等待当前对象的保存线程完成，然后转移源对象的所有资源。
     *          通过移动语义避免深拷贝，提高性能。包含自赋值检查以防止资源丢失。
     */
    SplatData& SplatData::operator=(SplatData&& other) noexcept {
        if (this != &other) {  // 自赋值检查，防止资源丢失
            // 首先等待当前对象的所有保存线程完成
            {
                std::lock_guard<std::mutex> lock(_threads_mutex);  // 获取当前对象的线程锁
                for (auto& t : _save_threads) {                   // 遍历当前对象的保存线程
                    if (t.joinable()) {                           // 检查线程是否可以join
                        t.join();                                 // 等待线程完成
                    }
                }
            }

            // 移动标量成员变量
            _active_sh_degree = other._active_sh_degree;           // 当前激活的球谐度数
            _max_sh_degree = other._max_sh_degree;                 // 最大球谐度数
            _scene_scale = other._scene_scale;                     // 场景缩放因子

            // 移动所有张量数据，避免深拷贝
            _means = std::move(other._means);                      // 高斯体位置坐标
            _sh0 = std::move(other._sh0);                         // 0阶球谐系数
            _shN = std::move(other._shN);                         // 高阶球谐系数
            _scaling = std::move(other._scaling);                 // 缩放参数（对数形式）
            _rotation = std::move(other._rotation);               // 旋转四元数
            _opacity = std::move(other._opacity);                 // 不透明度（对数形式）
            _densification_info = other._densification_info;      // 密度化信息（非移动，因为可能是简单类型）

            // 在锁保护下移动线程列表
            std::lock_guard<std::mutex> lock(other._threads_mutex);
            _save_threads = std::move(other._save_threads);        // 移动保存线程列表
        }
        return *this;  // 返回当前对象的引用，支持链式赋值
    }

    /**
     * [功能描述]：SplatData类的构造函数，从张量数据初始化高斯散射体
     * @param sh_degree：最大球谐度数，控制光照表示的复杂度
     * @param means：高斯体的位置坐标张量，形状为[N, 3]
     * @param sh0：0阶球谐系数张量，表示基础颜色信息
     * @param shN：高阶球谐系数张量，表示复杂光照效果
     * @param scaling：高斯体的缩放参数张量（对数形式），形状为[N, 3]
     * @param rotation：高斯体的旋转四元数张量，形状为[N, 4]
     * @param opacity：高斯体的不透明度张量（对数形式），形状为[N, 1]
     * @param scene_scale：场景的缩放因子，用于坐标系统标准化
     * @details 构造函数初始化高斯散射体的所有核心参数。所有张量通过std::move转移，
     *          避免不必要的拷贝。初始时激活的球谐度数设为0，表示从最简单的光照模型开始。
     */
    SplatData::SplatData(int sh_degree,
                         torch::Tensor means,
                         torch::Tensor sh0,
                         torch::Tensor shN,
                         torch::Tensor scaling,
                         torch::Tensor rotation,
                         torch::Tensor opacity,
                         float scene_scale)
        : _max_sh_degree{sh_degree},                              // 设置最大球谐度数
          _active_sh_degree{0},                                  // 初始激活球谐度数为0
          _scene_scale{scene_scale},                             // 设置场景缩放因子
          _means{std::move(means)},                              // 移动位置坐标张量
          _sh0{std::move(sh0)},                                  // 移动0阶球谐系数张量
          _shN{std::move(shN)},                                  // 移动高阶球谐系数张量
          _scaling{std::move(scaling)},                          // 移动缩放参数张量
          _rotation{std::move(rotation)},                        // 移动旋转四元数张量
          _opacity{std::move(opacity)} {}                        // 移动不透明度张量

    /**
     * [功能描述]：获取高斯体的位置坐标
     * @return 返回位置坐标张量，形状为[N, 3]
     * @details 直接返回内部存储的位置坐标，无需额外计算
     */
    torch::Tensor SplatData::get_means() const {
        return _means;
    }

    /**
     * [功能描述]：获取高斯体的不透明度值（经过sigmoid激活函数处理）
     * @return 返回不透明度张量，形状为[N]，值域为[0,1]
     * @details 内部存储的不透明度是对数形式，通过sigmoid函数转换为[0,1]范围。
     *          squeeze(-1)移除最后一个维度，将[N,1]转换为[N]。
     */
    torch::Tensor SplatData::get_opacity() const {
        return torch::sigmoid(_opacity).squeeze(-1);  // sigmoid激活 + 维度压缩
    }

    /**
     * [功能描述]：获取高斯体的旋转四元数（经过归一化处理）
     * @return 返回归一化的旋转四元数张量，形状为[N, 4]
     * @details 对四元数进行L2归一化，确保其为单位四元数，保证旋转矩阵的正交性。
     *          归一化在最后一个维度（dim=-1）上进行，即对每个四元数分别归一化。
     */
    torch::Tensor SplatData::get_rotation() const {
        return torch::nn::functional::normalize(_rotation,
                                                torch::nn::functional::NormalizeFuncOptions().dim(-1));
    }

    /**
     * [功能描述]：获取高斯体的缩放参数（经过指数函数处理）
     * @return 返回缩放参数张量，形状为[N, 3]，值为正数
     * @details 内部存储的缩放参数是对数形式，通过exp函数转换为正数。
     *          这确保了高斯体的缩放参数始终为正值，符合物理意义。
     */
    torch::Tensor SplatData::get_scaling() const {
        return torch::exp(_scaling);  // 将对数缩放转换为实际缩放值
    }

    /**
     * [功能描述]：获取完整的球谐系数张量（0阶和高阶系数合并）
     * @return 返回合并后的球谐系数张量，形状为[N, 3, (sh_degree+1)²]
     * @details 将0阶球谐系数（直流分量）和高阶球谐系数在维度1上拼接，
     *          形成完整的球谐系数表示，用于复杂光照计算。
     */
    torch::Tensor SplatData::get_shs() const {
        return torch::cat({_sh0, _shN}, 1);  // 在维度1上拼接0阶和高阶系数
    }

    /**
     * [功能描述]：对高斯散射体数据进行3D变换，包括位置、旋转和缩放的变换
     * @param transform_matrix：4x4变换矩阵，包含旋转、平移、缩放等变换信息
     * @return 返回当前对象的引用，支持链式调用
     * @details 该函数对高斯散射体的所有属性进行完整的3D变换：
     *          1. 变换位置坐标（使用齐次坐标）
     *          2. 提取并应用旋转变换（四元数乘法）
     *          3. 处理缩放变换（对数缩放）
     *          4. 更新场景缩放因子
     *          所有计算都在GPU上进行以提高效率。
     */
    SplatData& SplatData::transform(const glm::mat4& transform_matrix) {
        LOG_TIMER("SplatData::transform");  // 记录函数执行时间

        // 边界检查：如果没有高斯体数据，直接返回
        if (_means.size(0) == 0) {
            return *this; // 没有数据需要变换
        }

        const int num_points = _means.size(0);  // 获取高斯体的数量

        // 保持所有计算在GPU上进行以提高效率
        auto device = _means.device();

        // 步骤1：变换位置坐标（means）
        // 将GLM变换矩阵转换为PyTorch张量
        auto transform_tensor = torch::tensor({transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2], transform_matrix[0][3],
                                               transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2], transform_matrix[1][3],
                                               transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2], transform_matrix[2][3],
                                               transform_matrix[3][0], transform_matrix[3][1], transform_matrix[3][2], transform_matrix[3][3]},
                                              torch::TensorOptions().dtype(torch::kFloat32).device(device))
                                    .reshape({4, 4});  // 重塑为4x4矩阵

        // 添加齐次坐标（w=1），将3D坐标转换为4D齐次坐标
        auto means_homo = torch::cat({_means, torch::ones({num_points, 1}, _means.options())}, 1);

        // 应用变换矩阵：使用矩阵乘法 (4x4) @ (Nx4)^T = (4xN)，然后转置回 (Nx4)
        auto transformed_means = torch::matmul(transform_tensor, means_homo.t()).t();

        // 提取xyz坐标并原地更新位置数据（忽略齐次坐标的w分量）
        _means.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                          transformed_means.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));

        // 步骤2：从变换矩阵中提取旋转信息（简化方法，不进行完整的矩阵分解）
        glm::mat3 rot_mat(transform_matrix);  // 提取3x3旋转部分

        // 归一化列向量以移除缩放影响，提取纯旋转矩阵
        glm::vec3 scale;  // 存储各轴的缩放因子
        for (int i = 0; i < 3; ++i) {
            scale[i] = glm::length(rot_mat[i]);  // 计算列向量的长度（缩放因子）
            if (scale[i] > 0.0f) {
                rot_mat[i] /= scale[i];  // 归一化列向量，移除缩放
            }
        }

        // 将旋转矩阵转换为四元数表示
        glm::quat rotation = glm::quat_cast(rot_mat);

        // 步骤3：变换旋转四元数（如果存在旋转）
        if (std::abs(rotation.w - 1.0f) > 1e-6f) {  // 检查是否有实际旋转（w接近1表示无旋转）
            // 将GLM四元数转换为PyTorch张量
            auto rot_tensor = torch::tensor({rotation.w, rotation.x, rotation.y, rotation.z},
                                            torch::TensorOptions().dtype(torch::kFloat32).device(device));

            // 四元数乘法：q_new = q_transform * q_original
            auto q = _rotation; // 原始四元数，形状：[N, 4]，格式为[w, x, y, z]

            // 将变换四元数扩展到批次大小以匹配所有高斯体
            auto q_rot = rot_tensor.unsqueeze(0).expand({num_points, 4});

            // 提取四元数的各个分量，用于四元数乘法计算
            auto w1 = q_rot.index({torch::indexing::Slice(), 0});  // 变换四元数的w分量
            auto x1 = q_rot.index({torch::indexing::Slice(), 1});  // 变换四元数的x分量
            auto y1 = q_rot.index({torch::indexing::Slice(), 2});  // 变换四元数的y分量
            auto z1 = q_rot.index({torch::indexing::Slice(), 3});  // 变换四元数的z分量

            auto w2 = q.index({torch::indexing::Slice(), 0});      // 原始四元数的w分量
            auto x2 = q.index({torch::indexing::Slice(), 1});      // 原始四元数的x分量
            auto y2 = q.index({torch::indexing::Slice(), 2});      // 原始四元数的y分量
            auto z2 = q.index({torch::indexing::Slice(), 3});      // 原始四元数的z分量

            // 执行四元数乘法：q_result = q1 * q2
            // 四元数乘法公式：
            // w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            // x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            // y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            // z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            _rotation.index_put_({torch::indexing::Slice(), 0}, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2);  // w分量
            _rotation.index_put_({torch::indexing::Slice(), 1}, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2);  // x分量
            _rotation.index_put_({torch::indexing::Slice(), 2}, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2);  // y分量
            _rotation.index_put_({torch::indexing::Slice(), 3}, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2);  // z分量
        }

        // 步骤4：变换缩放参数（如果存在非均匀缩放）
        if (std::abs(scale.x - 1.0f) > 1e-6f ||
            std::abs(scale.y - 1.0f) > 1e-6f ||
            std::abs(scale.z - 1.0f) > 1e-6f) {  // 检查是否有缩放变换

            // 计算平均缩放因子（用于各向同性高斯体缩放）
            float avg_scale = (scale.x + scale.y + scale.z) / 3.0f;

            // 由于_scaling存储的是log(scale)，我们添加缩放因子的对数
            _scaling = _scaling + std::log(avg_scale);
        }

        // 步骤5：更新场景缩放因子（如果变化显著）
        torch::Tensor scene_center = _means.mean(0);  // 计算场景中心点
        torch::Tensor dists = torch::norm(_means - scene_center, 2, 1);  // 计算所有点到中心的距离
        float new_scene_scale = dists.median().item<float>();  // 使用中位数作为新的场景缩放因子
        if (std::abs(new_scene_scale - _scene_scale) > _scene_scale * 0.1f) {  // 如果变化超过10%
            _scene_scale = new_scene_scale;  // 更新场景缩放因子
        }

        LOG_DEBUG("Transformed {} gaussians", num_points);  // 记录变换的高斯体数量
        return *this;  // 返回当前对象引用，支持链式调用
    }

    /**
     * [功能描述]：增加当前激活的球谐度数
     * @details 该函数用于渐进式训练，逐步增加球谐系数的复杂度。
     *          从0阶开始，逐步增加到最大球谐度数，这样可以：
     *          1. 加快训练收敛速度
     *          2. 避免训练初期过度拟合
     *          3. 逐步学习复杂的光照效果
     */
    void SplatData::increment_sh_degree() {
        if (_active_sh_degree < _max_sh_degree) {  // 检查是否已达到最大球谐度数
            _active_sh_degree++;                   // 增加当前激活的球谐度数
        }
    }

    /**
     * [功能描述]：获取PLY格式文件的属性名称列表
     * @return 返回属性名称的字符串向量，用于PLY文件导出
     * @details 该函数生成PLY文件中每个顶点属性的名称，包括：
     *          1. 位置坐标：x, y, z
     *          2. 法线向量：nx, ny, nz（通常为零）
     *          3. 球谐系数：f_dc_0, f_dc_1, ..., f_rest_0, f_rest_1, ...
     *          4. 不透明度：opacity
     *          5. 缩放参数：scale_0, scale_1, scale_2
     *          6. 旋转四元数：rot_0, rot_1, rot_2, rot_3
     */
    std::vector<std::string> SplatData::get_attribute_names() const {
        std::vector<std::string> a{"x", "y", "z", "nx", "ny", "nz"};  // 基础几何属性

        // 添加0阶球谐系数属性名称（直流分量，用于基础颜色）
        for (int i = 0; i < _sh0.size(1) * _sh0.size(2); ++i)
            a.emplace_back("f_dc_" + std::to_string(i));  // f_dc_0, f_dc_1, f_dc_2, ...
        
        // 添加高阶球谐系数属性名称（用于复杂光照效果）
        for (int i = 0; i < _shN.size(1) * _shN.size(2); ++i)
            a.emplace_back("f_rest_" + std::to_string(i));  // f_rest_0, f_rest_1, f_rest_2, ...

        a.emplace_back("opacity");  // 不透明度属性

        // 添加缩放参数属性名称（3个分量）
        for (int i = 0; i < _scaling.size(1); ++i)
            a.emplace_back("scale_" + std::to_string(i));  // scale_0, scale_1, scale_2
        
        // 添加旋转四元数属性名称（4个分量）
        for (int i = 0; i < _rotation.size(1); ++i)
            a.emplace_back("rot_" + std::to_string(i));  // rot_0, rot_1, rot_2, rot_3

        return a;  // 返回完整的属性名称列表
    }

    /**
     * [功能描述]：清理已完成的保存线程
     * @details 该函数用于管理异步保存PLY文件的线程池。
     *          由于C++11没有try_join功能，当前实现保留所有线程。
     *          这是一个占位实现，实际清理逻辑需要更复杂的线程管理。
     */
    void SplatData::cleanup_finished_threads() const {
        std::lock_guard<std::mutex> lock(_threads_mutex);  // 获取线程互斥锁

        // 移除已完成的线程（当前实现保留所有线程）
        _save_threads.erase(
            std::remove_if(_save_threads.begin(), _save_threads.end(),
                           [](std::thread& t) {
                               if (t.joinable()) {
                                   // 尝试以零超时时间join来检查是否完成
                                   // 由于C++11没有try_join，我们保留所有线程
                                   return false;  // 当前实现不删除任何线程
                               }
                               return true;  // 不可join的线程会被删除
                           }),
            _save_threads.end());
    }

    /**
     * [功能描述]：将高斯散射体数据导出为PLY格式文件
     * @param root：输出目录路径
     * @param iteration：当前训练迭代次数，用于生成文件名
     * @param join_thread：是否同步保存（true=同步，false=异步）
     * @details 该函数支持两种保存模式：
     *          1. 同步保存：立即执行，阻塞当前线程
     *          2. 异步保存：在后台线程执行，不阻塞主线程
     *          异步模式用于训练过程中不中断训练流程。
     */
    void SplatData::save_ply(const std::filesystem::path& root, int iteration, bool join_thread) const {
        auto pc = to_point_cloud();  // 将SplatData转换为PointCloud格式

        if (join_thread) {
            // 同步保存：直接在当前线程执行
            write_ply_impl(pc, root, iteration);
        } else {
            // 异步保存：在后台线程执行
            // 首先清理已完成的线程
            cleanup_finished_threads();

            // 创建新的保存线程并添加到线程池
            std::lock_guard<std::mutex> lock(_threads_mutex);
            _save_threads.emplace_back([pc = std::move(pc), root, iteration]() {
                write_ply_impl(pc, root, iteration);  // 在新线程中执行保存操作
            });
        }
    }

    /**
     * [功能描述]：将SplatData转换为PointCloud格式
     * @return 返回转换后的PointCloud对象
     * @details 该函数将内部张量数据转换为适合PLY导出的PointCloud格式。
     *          所有数据都会转移到CPU并确保内存连续，以便PLY库处理。
     *          球谐系数会进行维度重排以匹配PLY格式要求。
     */
    PointCloud SplatData::to_point_cloud() const {
        PointCloud pc;  // 创建PointCloud对象

        // 基础几何属性
        pc.means = _means.cpu().contiguous();                    // 位置坐标：转移到CPU并确保连续
        pc.normals = torch::zeros_like(pc.means);                // 法线向量：初始化为零（高斯散射体通常不需要法线）

        // 高斯散射体属性
        pc.sh0 = _sh0.transpose(1, 2).flatten(1).cpu();         // 0阶球谐系数：转置+展平+转移到CPU
        pc.shN = _shN.transpose(1, 2).flatten(1).cpu();         // 高阶球谐系数：转置+展平+转移到CPU
        pc.opacity = _opacity.cpu();                             // 不透明度：转移到CPU
        pc.scaling = _scaling.cpu();                            // 缩放参数：转移到CPU
        pc.rotation = _rotation.cpu();                           // 旋转四元数：转移到CPU

        // 设置PLY导出的属性名称
        pc.attribute_names = get_attribute_names();              // 获取属性名称列表

        return pc;  // 返回转换后的PointCloud对象
    }

    /**
     * [功能描述]：从点云数据初始化高斯散射体模型
     * @param params：训练参数，包含初始化配置
     * @param scene_center：场景中心点，用于坐标系统标准化
     * @param pcd：输入点云数据，包含位置和颜色信息
     * @return 返回SplatData对象或错误信息
     * @details 该函数是高斯散射体系统的核心初始化函数，支持两种初始化模式：
     *          1. 随机初始化：在指定范围内随机生成高斯体
     *          2. 点云初始化：基于输入点云数据创建高斯体
     *          所有参数都会设置为可训练状态，为后续优化做准备。
     */
    std::expected<SplatData, std::string> SplatData::init_model_from_pointcloud(
        const param::TrainingParameters& params,
        torch::Tensor scene_center,
        const PointCloud& pcd) {

        try {
            // 根据初始化类型生成位置和颜色数据
            torch::Tensor positions, colors;
            if (params.optimization.random) {
                // 随机初始化模式：在指定范围内随机生成高斯体
                const int num_points = params.optimization.init_num_pts;      // 高斯体数量
                const float extent = params.optimization.init_extent;         // 初始化范围
                const auto f32_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

                // 生成随机位置：在[-extent, extent]范围内均匀分布
                positions = (torch::rand({num_points, 3}, f32_cuda) * 2.0f - 1.0f) * extent;
                // 生成随机颜色：在[0, 1]范围内均匀分布
                colors = torch::rand({num_points, 3}, f32_cuda);
            } else {
                // 点云初始化模式：使用输入点云的位置和颜色
                positions = pcd.means;                    // 使用点云的位置坐标
                colors = pcd.colors / 255.0f;            // 将颜色从[0,255]归一化到[0,1]
            }

            // 计算场景缩放因子：使用到场景中心距离的中位数
            scene_center = scene_center.to(positions.device());              // 确保设备一致
            const torch::Tensor dists = torch::norm(positions - scene_center, 2, 1);  // 计算欧几里得距离
            const auto scene_scale = dists.median().item<float>();           // 使用中位数作为场景缩放

            // RGB到球谐系数的转换函数
            auto rgb_to_sh = [](const torch::Tensor& rgb) {
                constexpr float kInvSH = 0.28209479177387814f;  // 球谐系数的归一化常数
                return (rgb - 0.5f) / kInvSH;                   // 将RGB转换为球谐系数
            };

            // 设置张量选项
            const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
            const auto f32_cuda = f32.device(torch::kCUDA);

            // 步骤1：初始化位置坐标（means）
            torch::Tensor means;
            if (params.optimization.random) {
                // 随机模式：将位置缩放到场景尺度
                means = (positions * scene_scale).to(torch::kCUDA).set_requires_grad(true);
            } else {
                // 点云模式：直接使用点云位置
                means = positions.to(torch::kCUDA).set_requires_grad(true);
            }

            // 步骤2：初始化缩放参数（对数形式）
            // 计算每个点到最近邻居的平均距离，用于确定高斯体的初始大小
            auto nn_dist = torch::clamp_min(compute_mean_neighbor_distances(means), 1e-7);  // 避免零距离
            auto scaling = torch::log(torch::sqrt(nn_dist) * params.optimization.init_scaling)  // 对数缩放
                               .unsqueeze(-1)                                                    // 添加维度：[N] -> [N, 1]
                               .repeat({1, 3})                                                   // 扩展到3D：[N, 1] -> [N, 3]
                               .to(f32_cuda)
                               .set_requires_grad(true);

            // 步骤3：初始化旋转四元数（单位四元数，表示无旋转）
            auto rotation = torch::zeros({means.size(0), 4}, f32_cuda);  // 创建零张量
            rotation.index_put_({torch::indexing::Slice(), 0}, 1);       // 设置w分量为1（单位四元数）
            rotation = rotation.set_requires_grad(true);

            // 步骤4：初始化不透明度（sigmoid的逆函数）
            auto opacity = torch::logit(params.optimization.init_opacity * torch::ones({means.size(0), 1}, f32_cuda))
                               .set_requires_grad(true);
            // logit函数是sigmoid的逆函数：logit(x) = log(x/(1-x))

            // 步骤5：初始化球谐系数（SH coefficients）
            auto colors_float = colors.to(torch::kCUDA);                    // 确保在GPU上
            auto fused_color = rgb_to_sh(colors_float);                    // 转换为球谐系数

            // 计算球谐系数的总数量：(sh_degree + 1)²
            const int64_t feature_shape = static_cast<int64_t>(std::pow(params.optimization.sh_degree + 1, 2));
            auto shs = torch::zeros({fused_color.size(0), 3, feature_shape}, f32_cuda);  // 创建球谐系数张量

            // 设置0阶球谐系数（直流分量，用于基础颜色）
            shs.index_put_({torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            0},
                           fused_color);

            // 提取0阶球谐系数（第0个系数）
            auto sh0 = shs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(0, 1)})  // 形状：[N, 3, 1]
                           .transpose(1, 2)                       // 转置：形状：[N, 1, 3]
                           .contiguous()
                           .set_requires_grad(true);

            // 提取高阶球谐系数（第1个及以后的系数）
            auto shN = shs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(1, torch::indexing::None)})  // 形状：[N, 3, feature_shape-1]
                           .transpose(1, 2)                                           // 转置：形状：[N, feature_shape-1, 3]
                           .contiguous()
                           .set_requires_grad(true);

            // 输出初始化信息
            std::println("Scene scale: {}", scene_scale);
            std::println("Initialized SplatData with:");
            std::println("  - {} points", means.size(0));
            std::println("  - Max SH degree: {}", params.optimization.sh_degree);
            std::println("  - Total SH coefficients: {}", feature_shape);
            std::cout << std::format("  - sh0 shape: {}\n", tensor_sizes_to_string(sh0.sizes()));
            std::cout << std::format("  - shN shape: {}\n", tensor_sizes_to_string(shN.sizes()));

            // 创建并返回SplatData对象
            return SplatData(
                params.optimization.sh_degree,    // 最大球谐度数
                means.contiguous(),              // 位置坐标
                sh0.contiguous(),                // 0阶球谐系数
                shN.contiguous(),                // 高阶球谐系数
                scaling.contiguous(),            // 缩放参数
                rotation.contiguous(),           // 旋转四元数
                opacity.contiguous(),            // 不透明度
                scene_scale);                   // 场景缩放因子

        } catch (const std::exception& e) {
            // 异常处理：返回错误信息
            return std::unexpected(std::format("Failed to initialize SplatData: {}", e.what()));
        }
    }
} // namespace gs