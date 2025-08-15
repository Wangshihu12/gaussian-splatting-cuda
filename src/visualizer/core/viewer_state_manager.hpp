#pragma once

#include "core/events.hpp"
#include <atomic>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>

/**
 * [文件描述]：查看器状态管理器头文件
 * 功能：定义可视化器的全局状态管理系统，包括训练信息、渲染配置、查看器模式等
 * 用途：为整个可视化系统提供统一的状态管理和数据共享机制
 */

 namespace gs::visualizer {

    /**
     * [结构体描述]：训练信息数据结构
     * 
     * 存储和管理训练过程中的实时信息，包括：
     * - 迭代进度跟踪
     * - 高斯散点数量统计
     * - 损失值历史记录
     * 
     * 该结构体从VisualizerImpl中移出，作为独立的数据容器
     * 支持多线程安全访问，使用原子变量和互斥锁保护数据
     */
    // 从VisualizerImpl中移出TrainingInfo
    struct TrainingInfo {
        // =============================================================================
        // 训练进度相关（原子变量，支持多线程安全访问）
        // =============================================================================
        
        std::atomic<int> curr_iterations_{0};    // 当前训练迭代次数，原子操作确保线程安全
        std::atomic<int> total_iterations_{0};   // 总预计迭代次数，用于计算训练进度百分比
        std::atomic<int> num_splats_{0};         // 当前高斯散点总数，实时跟踪模型复杂度

        // =============================================================================
        // 损失值历史记录（受互斥锁保护）
        // =============================================================================
        
        int max_loss_points_ = 200;              // 损失值缓冲区最大容量，控制内存使用
        std::deque<float> loss_buffer_;          // 损失值历史记录，双端队列便于添加和删除
        std::mutex loss_buffer_mutex_;           // 损失缓冲区互斥锁，保护多线程访问

        // =============================================================================
        // 公共接口方法
        // =============================================================================
        
        /**
         * [功能描述]：更新训练进度信息
         * @param iter：当前迭代次数
         * @param total_iterations：总迭代次数
         */
        void updateProgress(int iter, int total_iterations);
        
        /**
         * [功能描述]：更新高斯散点数量
         * @param num_splats：当前高斯散点总数
         */
        void updateNumSplats(int num_splats);
        
        /**
         * [功能描述]：添加新的损失值到历史记录
         * @param loss：当前迭代的损失值
         */
        void updateLoss(float loss);
        
        /**
         * [功能描述]：获取损失值历史记录的副本
         * @return 损失值队列的副本，线程安全
         */
        std::deque<float> getLossBuffer() const;
    };

    /**
     * [结构体描述]：渲染配置数据结构
     * 
     * 存储和管理渲染相关的配置参数：
     * - 视野角度（FOV）控制
     * - 缩放修正因子
     * - 线程安全的参数访问
     * 
     * 该结构体从VisualizerImpl中移出，作为独立的配置容器
     * 使用互斥锁保护配置参数的并发访问
     */
    // 从VisualizerImpl中移出RenderingConfig
    struct RenderingConfig {
        mutable std::mutex mtx;                  // 可变互斥锁，保护配置参数的线程安全访问
        float fov = 60.0f;                      // 视野角度（度），默认60度，符合大多数3D应用习惯
        float scaling_modifier = 1.0f;          // 缩放修正因子，默认1.0无修正，用于调整高斯散点尺寸

        // =============================================================================
        // 配置访问方法（线程安全）
        // =============================================================================
        
        /**
         * [功能描述]：根据分辨率计算实际FOV值
         * @param reso_x：屏幕水平分辨率
         * @param reso_y：屏幕垂直分辨率
         * @return glm::vec2：水平和垂直方向的FOV值
         */
        glm::vec2 getFov(size_t reso_x, size_t reso_y) const;
        
        /**
         * [功能描述]：获取FOV角度值
         * @return float：当前设置的视野角度（度）
         */
        float getFovDegrees() const;
        
        /**
         * [功能描述]：获取缩放修正因子
         * @return float：当前设置的缩放修正因子
         */
        float getScalingModifier() const;
        
        /**
         * [功能描述]：设置新的FOV角度
         * @param f：新的视野角度值（度）
         */
        void setFov(float f);
        
        /**
         * [功能描述]：设置新的缩放修正因子
         * @param s：新的缩放修正因子值
         */
        void setScalingModifier(float s);
    };

    /**
     * [枚举描述]：查看器工作模式
     * 
     * 定义可视化器的不同工作状态：
     * - Empty：空状态，未加载任何数据
     * - PLYViewer：PLY文件查看模式，显示静态点云
     * - Training：训练模式，显示动态训练过程
     */
    enum class ViewerMode {
        Empty,        // 空模式：没有加载任何数据，显示空场景
        PLYViewer,    // PLY查看模式：加载并显示PLY点云文件，只读模式
        Training      // 训练模式：执行高斯散点训练，支持实时参数调整和监控
    };

    /**
     * [类描述]：查看器状态管理器
     * 
     * ViewerStateManager是整个可视化系统的状态中枢，负责：
     * - 管理查看器的工作模式切换
     * - 维护当前加载的文件路径信息
     * - 提供训练信息和渲染配置的统一访问接口
     * - 管理渲染相关的全局状态（如抗锯齿）
     * - 处理状态变更事件和通知
     * 
     * 该类实现了单例模式的变种，通过共享指针提供全局状态访问
     * 所有状态变更都通过事件系统通知其他组件，保持组件间的松耦合
     */
    class ViewerStateManager {
    public:
        // =============================================================================
        // 构造和析构函数
        // =============================================================================
        
        /**
         * [功能描述]：构造函数，初始化状态管理器和相关组件
         */
        ViewerStateManager();
        
        /**
         * [功能描述]：析构函数，清理资源并取消事件订阅
         */
        ~ViewerStateManager();

        // =============================================================================
        // 模式管理方法
        // =============================================================================
        
        /**
         * [功能描述]：获取当前查看器模式
         * @return ViewerMode：当前的工作模式
         */
        ViewerMode getCurrentMode() const { return current_mode_; }
        
        /**
         * [功能描述]：设置查看器模式
         * @param mode：要切换到的新模式
         */
        void setMode(ViewerMode mode);

        // =============================================================================
        // 路径管理方法
        // =============================================================================
        
        /**
         * [功能描述]：获取当前PLY文件路径
         * @return const std::filesystem::path&：当前加载的PLY文件路径引用
         */
        const std::filesystem::path& getCurrentPLYPath() const { return current_ply_path_; }
        
        /**
         * [功能描述]：获取当前数据集路径
         * @return const std::filesystem::path&：当前加载的数据集根目录路径引用
         */
        const std::filesystem::path& getCurrentDatasetPath() const { return current_dataset_path_; }
        
        /**
         * [功能描述]：设置PLY文件路径
         * @param path：新的PLY文件路径
         */
        void setPLYPath(const std::filesystem::path& path);
        
        /**
         * [功能描述]：设置数据集路径
         * @param path：新的数据集根目录路径
         */
        void setDatasetPath(const std::filesystem::path& path);
        
        /**
         * [功能描述]：清空所有路径信息
         */
        void clearPaths();

        // =============================================================================
        // 状态组件访问方法
        // =============================================================================
        
        /**
         * [功能描述]：获取训练信息共享指针
         * @return std::shared_ptr<TrainingInfo>：训练信息对象的共享指针
         */
        std::shared_ptr<TrainingInfo> getTrainingInfo() { return training_info_; }
        
        /**
         * [功能描述]：获取渲染配置共享指针
         * @return std::shared_ptr<RenderingConfig>：渲染配置对象的共享指针
         */
        std::shared_ptr<RenderingConfig> getRenderingConfig() { return rendering_config_; }

        // =============================================================================
        // 渲染状态管理方法
        // =============================================================================
        
        /**
         * [功能描述]：检查抗锯齿是否启用
         * @return bool：true表示抗锯齿已启用，false表示已禁用
         */
        bool isAntiAliasingEnabled() const { return anti_aliasing_; }
        
        /**
         * [功能描述]：设置抗锯齿开关状态
         * @param enabled：true启用抗锯齿，false禁用抗锯齿
         */
        void setAntiAliasing(bool enabled);

        // =============================================================================
        // 状态重置方法
        // =============================================================================
        
        /**
         * [功能描述]：重置所有状态到初始值
         * 清空路径信息、重置模式、清理训练数据等
         */
        void reset();

    private:
        // =============================================================================
        // 内部方法
        // =============================================================================
        
        /**
         * [功能描述]：设置事件处理器，监听相关事件
         */
        void setupEventHandlers();
        
        /**
         * [功能描述]：发布状态变更事件，通知其他组件
         */
        void publishStateChange();

        // =============================================================================
        // 当前状态数据
        // =============================================================================
        
        std::atomic<ViewerMode> current_mode_{ViewerMode::Empty};    // 当前查看器模式，原子变量保证线程安全
        std::filesystem::path current_ply_path_;                    // 当前PLY文件路径
        std::filesystem::path current_dataset_path_;                // 当前数据集路径

        // =============================================================================
        // 状态组件（共享指针，支持多处访问）
        // =============================================================================
        
        std::shared_ptr<TrainingInfo> training_info_;        // 训练信息共享对象
        std::shared_ptr<RenderingConfig> rendering_config_;  // 渲染配置共享对象

        // =============================================================================
        // 渲染状态
        // =============================================================================
        
        std::atomic<bool> anti_aliasing_{false};             // 抗锯齿开关，原子变量保证线程安全

        // =============================================================================
        // 同步原语
        // =============================================================================
        
        mutable std::mutex paths_mutex_;                     // 路径信息互斥锁，保护路径操作的线程安全
    };

} // namespace gs::visualizer
