#pragma once

#include "core/point_cloud.hpp"
#include <expected>
#include <filesystem>
#include <glm/glm.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

namespace gs {
    /**
     * @brief 训练参数命名空间，包含高斯散射体训练的相关配置
     * @details 该命名空间定义了训练过程中需要的各种参数，
     *          如球谐度数、初始化参数、优化设置等。
     */
    namespace param {
        struct TrainingParameters;
    }

    /**
     * @class SplatData
     * @brief 高斯散射体数据管理类，存储和管理3D高斯散射体的所有属性
     * @details 该类是高斯散射体系统的核心数据结构，负责管理：
     *          1. 位置坐标：每个高斯体的3D位置
     *          2. 球谐系数：用于表示光照和颜色信息
     *          3. 缩放参数：控制高斯体的大小
     *          4. 旋转四元数：控制高斯体的方向
     *          5. 不透明度：控制高斯体的可见性
     *          6. 场景缩放：用于坐标系统标准化
     *          支持移动语义、异步保存、3D变换等高级功能。
     */
    class SplatData {
    public:
        /**
         * @brief 默认构造函数
         * @details 创建一个空的SplatData对象，所有成员变量使用默认值
         */
        SplatData() = default;

        /**
         * @brief 析构函数
         * @details 清理资源，等待所有异步保存线程完成
         */
        ~SplatData();

        /**
         * @brief 禁用拷贝构造函数
         * @details 由于包含大量张量数据，拷贝成本很高，因此禁用拷贝构造
         */
        SplatData(const SplatData&) = delete;

        /**
         * @brief 禁用拷贝赋值运算符
         * @details 禁用拷贝赋值，强制使用移动语义
         */
        SplatData& operator=(const SplatData&) = delete;

        /**
         * @brief 移动构造函数
         * @param other 要移动的源对象
         * @details 高效地转移所有权，避免深拷贝
         */
        SplatData(SplatData&& other) noexcept;

        /**
         * @brief 移动赋值运算符
         * @param other 要移动的源对象
         * @return 返回当前对象的引用
         * @details 高效地转移所有权，支持链式赋值
         */
        SplatData& operator=(SplatData&& other) noexcept;

        /**
         * @brief 构造函数，从张量数据初始化
         * @param sh_degree 最大球谐度数
         * @param means 位置坐标张量 [N, 3]
         * @param sh0 0阶球谐系数张量 [N, 1, 3]
         * @param shN 高阶球谐系数张量 [N, K, 3]
         * @param scaling 缩放参数张量 [N, 3]（对数形式）
         * @param rotation 旋转四元数张量 [N, 4]
         * @param opacity 不透明度张量 [N, 1]（对数形式）
         * @param scene_scale 场景缩放因子
         * @details 使用移动语义初始化所有成员变量
         */
        SplatData(int sh_degree,
                  torch::Tensor means,
                  torch::Tensor sh0,
                  torch::Tensor shN,
                  torch::Tensor scaling,
                  torch::Tensor rotation,
                  torch::Tensor opacity,
                  float scene_scale);

        /**
         * @brief 静态工厂方法，从点云数据创建SplatData对象
         * @param params 训练参数配置
         * @param scene_center 场景中心点
         * @param point_cloud 输入点云数据
         * @return 返回SplatData对象或错误信息
         * @details 支持随机初始化和点云初始化两种模式
         */
        static std::expected<SplatData, std::string> init_model_from_pointcloud(
            const gs::param::TrainingParameters& params,
            torch::Tensor scene_center,
            const PointCloud& point_cloud);

        // ========== 计算属性getter方法（在cpp文件中实现）==========

        /**
         * @brief 获取位置坐标
         * @return 返回位置坐标张量 [N, 3]
         * @details 直接返回内部存储的位置数据
         */
        torch::Tensor get_means() const;

        /**
         * @brief 获取不透明度值（经过sigmoid处理）
         * @return 返回不透明度张量 [N]，值域[0,1]
         * @details 将对数不透明度转换为[0,1]范围
         */
        torch::Tensor get_opacity() const;

        /**
         * @brief 获取旋转四元数（经过归一化处理）
         * @return 返回归一化四元数张量 [N, 4]
         * @details 确保四元数为单位四元数
         */
        torch::Tensor get_rotation() const;

        /**
         * @brief 获取缩放参数（经过指数处理）
         * @return 返回缩放张量 [N, 3]，值为正数
         * @details 将对数缩放转换为实际缩放值
         */
        torch::Tensor get_scaling() const;

        /**
         * @brief 获取完整球谐系数（0阶和高阶合并）
         * @return 返回合并后的球谐系数张量 [N, 3, (sh_degree+1)²]
         * @details 将0阶和高阶系数在维度1上拼接
         */
        torch::Tensor get_shs() const;

        /**
         * @brief 3D变换函数（临时实现，未来将移至CUDA内核）
         * @param transform_matrix 4x4变换矩阵
         * @return 返回当前对象的引用，支持链式调用
         * @details 对高斯体进行位置、旋转、缩放的完整3D变换
         */
        SplatData& transform(const glm::mat4& transform_matrix);

        // ========== 简单内联getter方法 ==========

        /**
         * @brief 获取当前激活的球谐度数
         * @return 返回当前激活的球谐度数
         * @details 用于渐进式训练，控制当前使用的球谐系数复杂度
         */
        int get_active_sh_degree() const { return _active_sh_degree; }

        /**
         * @brief 获取场景缩放因子
         * @return 返回场景缩放因子
         * @details 用于坐标系统标准化
         */
        float get_scene_scale() const { return _scene_scale; }

        /**
         * @brief 获取高斯体的数量
         * @return 返回高斯体的总数
         * @details 基于位置张量的第一个维度
         */
        int64_t size() const { return _means.size(0); }

        // ========== 原始张量访问方法（内联，用于性能优化）==========

        /**
         * @brief 获取位置坐标张量的引用（可修改）
         * @return 返回位置张量的引用
         * @details 用于直接修改位置数据，提高性能
         */
        inline torch::Tensor& means() { return _means; }

        /**
         * @brief 获取位置坐标张量的常量引用
         * @return 返回位置张量的常量引用
         * @details 用于只读访问位置数据
         */
        inline const torch::Tensor& means() const { return _means; }

        /**
         * @brief 获取不透明度张量的引用（可修改）
         * @return 返回不透明度张量的引用
         * @details 用于直接修改不透明度数据
         */
        inline torch::Tensor& opacity_raw() { return _opacity; }

        /**
         * @brief 获取不透明度张量的常量引用
         * @return 返回不透明度张量的常量引用
         * @details 用于只读访问不透明度数据
         */
        inline const torch::Tensor& opacity_raw() const { return _opacity; }

        /**
         * @brief 获取旋转四元数张量的引用（可修改）
         * @return 返回旋转张量的引用
         * @details 用于直接修改旋转数据
         */
        inline torch::Tensor& rotation_raw() { return _rotation; }

        /**
         * @brief 获取旋转四元数张量的常量引用
         * @return 返回旋转张量的常量引用
         * @details 用于只读访问旋转数据
         */
        inline const torch::Tensor& rotation_raw() const { return _rotation; }

        /**
         * @brief 获取缩放参数张量的引用（可修改）
         * @return 返回缩放张量的引用
         * @details 用于直接修改缩放数据
         */
        inline torch::Tensor& scaling_raw() { return _scaling; }

        /**
         * @brief 获取缩放参数张量的常量引用
         * @return 返回缩放张量的常量引用
         * @details 用于只读访问缩放数据
         */
        inline const torch::Tensor& scaling_raw() const { return _scaling; }

        /**
         * @brief 获取0阶球谐系数张量的引用（可修改）
         * @return 返回0阶球谐系数张量的引用
         * @details 用于直接修改0阶球谐系数数据
         */
        inline torch::Tensor& sh0() { return _sh0; }

        /**
         * @brief 获取0阶球谐系数张量的常量引用
         * @return 返回0阶球谐系数张量的常量引用
         * @details 用于只读访问0阶球谐系数数据
         */
        inline const torch::Tensor& sh0() const { return _sh0; }

        /**
         * @brief 获取高阶球谐系数张量的引用（可修改）
         * @return 返回高阶球谐系数张量的引用
         * @details 用于直接修改高阶球谐系数数据
         */
        inline torch::Tensor& shN() { return _shN; }

        /**
         * @brief 获取高阶球谐系数张量的常量引用
         * @return 返回高阶球谐系数张量的常量引用
         * @details 用于只读访问高阶球谐系数数据
         */
        inline const torch::Tensor& shN() const { return _shN; }

        // ========== 实用方法 ==========

        /**
         * @brief 增加当前激活的球谐度数
         * @details 用于渐进式训练，逐步增加球谐系数的复杂度
         */
        void increment_sh_degree();

        // ========== 导出方法 - 干净的公共接口 ==========

        /**
         * @brief 将高斯散射体数据导出为PLY格式文件
         * @param root 输出目录路径
         * @param iteration 当前训练迭代次数
         * @param join_thread 是否同步保存（默认false，异步保存）
         * @details 支持同步和异步两种保存模式
         */
        void save_ply(const std::filesystem::path& root, int iteration, bool join_thread = false) const;

        /**
         * @brief 获取PLY格式文件的属性名称列表
         * @return 返回属性名称的字符串向量
         * @details 生成PLY文件中每个顶点属性的名称
         */
        std::vector<std::string> get_attribute_names() const;

    public:
        /**
         * @brief 密度化信息张量
         * @details 存储屏幕空间梯度幅值，用于密度化策略
         *          初始化为空张量，在训练过程中动态填充
         */
        torch::Tensor _densification_info = torch::empty({0});

    private:
        // ========== 私有成员变量 ==========

        int _active_sh_degree = 0;    ///< 当前激活的球谐度数，用于渐进式训练
        int _max_sh_degree = 0;       ///< 最大球谐度数，限制球谐系数的复杂度
        float _scene_scale = 0.f;     ///< 场景缩放因子，用于坐标系统标准化

        // ========== 核心数据张量 ==========

        torch::Tensor _means;         ///< 位置坐标张量 [N, 3]，存储每个高斯体的3D位置
        torch::Tensor _sh0;           ///< 0阶球谐系数张量 [N, 1, 3]，表示基础颜色
        torch::Tensor _shN;           ///< 高阶球谐系数张量 [N, K, 3]，表示复杂光照
        torch::Tensor _scaling;       ///< 缩放参数张量 [N, 3]（对数形式），控制高斯体大小
        torch::Tensor _rotation;      ///< 旋转四元数张量 [N, 4]，控制高斯体方向
        torch::Tensor _opacity;       ///< 不透明度张量 [N, 1]（对数形式），控制可见性

        // ========== 线程管理 ==========

        mutable std::vector<std::thread> _save_threads;  ///< 异步保存线程池
        mutable std::mutex _threads_mutex;             ///< 线程互斥锁，保护线程池

        // ========== 私有方法 ==========

        /**
         * @brief 转换为点云格式（用于导出）
         * @return 返回PointCloud对象
         * @details 将内部张量数据转换为适合PLY导出的格式
         */
        PointCloud to_point_cloud() const;

        /**
         * @brief 清理已完成的保存线程
         * @details 管理异步保存线程池，避免内存泄漏
         */
        void cleanup_finished_threads() const;
    };
} // namespace gs
