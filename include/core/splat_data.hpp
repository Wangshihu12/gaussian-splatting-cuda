#pragma once

#include "core/point_cloud.hpp"
#include <expected>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

/**
 * [文件描述]：高斯散点数据类头文件
 * 功能：定义SplatData类，管理3D高斯散点的所有参数和操作
 * 用途：为高斯散点渲染和训练提供核心数据结构
 */

namespace gs {
    namespace param {
        struct TrainingParameters;  // 前向声明训练参数结构体
    }

    /**
     * [类描述]：高斯散点数据管理类
     * 
     * 该类是整个高斯散点系统的核心数据结构，负责：
     * - 存储和管理所有3D高斯的参数（位置、颜色、几何属性等）
     * - 提供参数的访问和修改接口
     * - 支持从点云数据初始化高斯散点
     * - 提供模型的导入导出功能
     * - 管理球谐函数的阶数和激活状态
     * - 支持异步文件保存操作
     */
    class SplatData {
    public:
        // =============================================================================
        // 构造函数和析构函数
        // =============================================================================
        
        SplatData() = default;  // 默认构造函数
        ~SplatData();          // 析构函数，清理资源和线程

        // =============================================================================
        // 拷贝控制：禁用拷贝，启用移动
        // =============================================================================
        
        SplatData(const SplatData&) = delete;               // 禁用拷贝构造（避免昂贵的张量拷贝）
        SplatData& operator=(const SplatData&) = delete;    // 禁用拷贝赋值（避免昂贵的张量拷贝）
        SplatData(SplatData&& other) noexcept;              // 移动构造函数（高效的资源转移）
        SplatData& operator=(SplatData&& other) noexcept;   // 移动赋值运算符（高效的资源转移）

        // =============================================================================
        // 参数化构造函数
        // =============================================================================
        
        /**
         * [功能描述]：使用指定参数构造SplatData对象
         * @param sh_degree：球谐函数的最大阶数，控制颜色表示的复杂度
         * @param means：高斯中心位置张量 [N, 3]
         * @param sh0：球谐0阶系数张量 [N, 3]，直流颜色分量
         * @param shN：球谐高阶系数张量 [N, (degree+1)²-1, 3]，方向性颜色
         * @param scaling：缩放参数张量 [N, 3]，控制高斯在各轴的大小
         * @param rotation：旋转参数张量 [N, 4]，四元数表示的朝向
         * @param opacity：不透明度参数张量 [N, 1]，控制透明度
         * @param scene_scale：场景缩放因子，用于归一化坐标
         */
        SplatData(int sh_degree,
                  torch::Tensor means,
                  torch::Tensor sh0,
                  torch::Tensor shN,
                  torch::Tensor scaling,
                  torch::Tensor rotation,
                  torch::Tensor opacity,
                  float scene_scale);

        // =============================================================================
        // 静态工厂方法
        // =============================================================================
        
        /**
         * [功能描述]：从点云数据创建SplatData对象的静态工厂方法
         * @param params：训练参数配置，包含初始化设置
         * @param scene_center：场景中心点坐标，用于坐标归一化
         * @param point_cloud：输入点云数据，包含3D点和颜色信息
         * @return 期望值类型，成功时返回SplatData对象，失败时返回错误信息
         * 
         * 该方法执行以下初始化过程：
         * - 将点云位置转换为高斯中心位置
         * - 基于点云颜色初始化球谐系数
         * - 设置默认的几何参数（缩放、旋转、不透明度）
         * - 计算场景边界和归一化参数
         */
        static std::expected<SplatData, std::string> init_model_from_pointcloud(
            const gs::param::TrainingParameters& params,
            torch::Tensor scene_center,
            const PointCloud& point_cloud);

        // =============================================================================
        // 计算型获取器方法（经过处理的参数）
        // =============================================================================
        
        /**
         * [功能描述]：获取高斯中心位置（已处理）
         * @return 处理后的位置张量，可能包含场景归一化等变换
         */
        torch::Tensor get_means() const;
        
        /**
         * [功能描述]：获取不透明度值（已激活）
         * @return 经过sigmoid激活后的不透明度张量 [N, 1]，值域[0, 1]
         */
        torch::Tensor get_opacity() const;
        
        /**
         * [功能描述]：获取旋转矩阵（已标准化）
         * @return 从四元数转换并标准化后的旋转参数
         */
        torch::Tensor get_rotation() const;
        
        /**
         * [功能描述]：获取缩放参数（已激活）
         * @return 经过指数激活后的缩放张量，确保正值
         */
        torch::Tensor get_scaling() const;
        
        /**
         * [功能描述]：获取球谐系数（已整理）
         * @return 完整的球谐系数张量，包含0阶和高阶系数
         */
        torch::Tensor get_shs() const;

        // =============================================================================
        // 简单内联获取器方法
        // =============================================================================
        
        /**
         * [功能描述]：获取当前激活的球谐函数阶数
         * @return 当前使用的球谐阶数，影响颜色表示的复杂度
         */
        int get_active_sh_degree() const { return _active_sh_degree; }
        
        /**
         * [功能描述]：获取场景缩放因子
         * @return 场景归一化使用的缩放参数
         */
        float get_scene_scale() const { return _scene_scale; }
        
        /**
         * [功能描述]：获取高斯数量
         * @return 当前模型中高斯散点的总数量
         */
        int64_t size() const { return _means.size(0); }

        // =============================================================================
        // 原始张量访问方法（用于优化）
        // =============================================================================
        
        // 位置参数的直接访问（内联以提高性能）
        inline torch::Tensor& means() { return _means; }                           // 可修改的位置张量
        inline const torch::Tensor& means() const { return _means; }               // 只读的位置张量
        
        // 不透明度原始参数（训练时使用，未经激活函数处理）
        inline torch::Tensor& opacity_raw() { return _opacity; }                   // 可修改的原始不透明度
        inline const torch::Tensor& opacity_raw() const { return _opacity; }       // 只读的原始不透明度
        
        // 旋转原始参数（四元数形式，未标准化）
        inline torch::Tensor& rotation_raw() { return _rotation; }                 // 可修改的原始旋转参数
        inline const torch::Tensor& rotation_raw() const { return _rotation; }     // 只读的原始旋转参数
        
        // 缩放原始参数（对数空间，未经指数激活）
        inline torch::Tensor& scaling_raw() { return _scaling; }                   // 可修改的原始缩放参数
        inline const torch::Tensor& scaling_raw() const { return _scaling; }       // 只读的原始缩放参数
        
        // 球谐系数直接访问
        inline torch::Tensor& sh0() { return _sh0; }                              // 可修改的0阶球谐系数
        inline const torch::Tensor& sh0() const { return _sh0; }                  // 只读的0阶球谐系数
        inline torch::Tensor& shN() { return _shN; }                              // 可修改的高阶球谐系数
        inline const torch::Tensor& shN() const { return _shN; }                  // 只读的高阶球谐系数
        
        // 2D投影最大半径（用于光栅化优化）
        inline torch::Tensor& max_radii2D() { return _max_radii2D; }              // 可修改的2D最大半径

        // =============================================================================
        // 工具方法
        // =============================================================================
        
        /**
         * [功能描述]：增加激活的球谐函数阶数
         * 用途：在训练过程中逐步增加颜色表示的复杂度
         * 效果：从简单的常数颜色过渡到复杂的方向性颜色变化
         */
        void increment_sh_degree();

        // =============================================================================
        // 导出方法 - 清洁的公共接口
        // =============================================================================
        
        /**
         * [功能描述]：保存模型为PLY格式文件
         * @param root：保存的根目录路径
         * @param iteration：当前训练迭代次数，用于文件命名
         * @param join_thread：是否等待保存线程完成，默认false（异步保存）
         * 
         * 该方法支持：
         * - 异步保存，不阻塞训练过程
         * - 自动文件命名（包含迭代次数）
         * - 完整的高斯参数导出
         * - 线程安全的并发保存
         */
        void save_ply(const std::filesystem::path& root, int iteration, bool join_thread = false) const;

        /**
         * [功能描述]：获取PLY格式的属性名称列表
         * @return 包含所有导出属性名称的字符串向量
         * 用途：为PLY文件头部提供属性定义信息
         */
        std::vector<std::string> get_attribute_names() const;

    private:
        // =============================================================================
        // 私有成员变量 - 核心数据存储
        // =============================================================================
        
        // 球谐函数相关参数
        int _active_sh_degree = 0;     // 当前激活的球谐阶数
        int _max_sh_degree = 0;        // 最大支持的球谐阶数
        float _scene_scale = 0.f;      // 场景归一化缩放因子

        // 高斯参数张量（所有张量都是[N, ...]格式，N为高斯数量）
        torch::Tensor _means;          // 3D位置 [N, 3]
        torch::Tensor _sh0;            // 球谐0阶系数 [N, 3]，直流颜色
        torch::Tensor _shN;            // 球谐高阶系数 [N, (degree+1)²-1, 3]
        torch::Tensor _scaling;        // 缩放参数 [N, 3]，对数空间
        torch::Tensor _rotation;       // 旋转四元数 [N, 4]
        torch::Tensor _opacity;        // 不透明度 [N, 1]，logit空间
        torch::Tensor _max_radii2D;    // 2D最大投影半径 [N, 1]，光栅化优化用

        // =============================================================================
        // 线程管理 - 异步文件保存
        // =============================================================================
        
        mutable std::vector<std::thread> _save_threads;  // 异步保存线程容器
        mutable std::mutex _threads_mutex;               // 线程容器访问互斥锁

        // =============================================================================
        // 私有辅助方法
        // =============================================================================
        
        /**
         * [功能描述]：将高斯散点转换为点云格式
         * @return 转换后的PointCloud对象
         * 用途：为导出和可视化提供点云表示
         */
        PointCloud to_point_cloud() const;

        /**
         * [功能描述]：清理已完成的保存线程
         * 用途：释放已完成线程的资源，防止内存泄漏
         * 调用：在新的异步保存操作前自动调用
         */
        void cleanup_finished_threads() const;
    };
} // namespace gs - 高斯散点项目命名空间结束
