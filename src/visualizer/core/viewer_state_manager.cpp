#include "core/viewer_state_manager.hpp"
#include <algorithm>

/**
 * [文件描述]：查看器状态管理器实现文件
 * 功能：实现训练信息、渲染配置和状态管理器的核心逻辑
 * 用途：为可视化系统提供统一的状态管理、事件处理和数据同步功能
 */

 namespace gs::visualizer {

    // =============================================================================
    // TrainingInfo 实现部分 - 训练信息管理
    // =============================================================================
    
    /**
     * [功能描述]：更新训练进度信息
     * @param iter：当前训练迭代次数
     * @param total_iterations：总预计迭代次数
     * 
     * 该方法线程安全地更新训练进度，用于计算训练完成百分比
     * 使用原子变量确保多线程环境下的数据一致性
     */
    void TrainingInfo::updateProgress(int iter, int total_iterations) {
        curr_iterations_ = iter;           // 原子操作，更新当前迭代数
        total_iterations_ = total_iterations;  // 原子操作，更新总迭代数
    }

    /**
     * [功能描述]：更新当前高斯散点数量
     * @param num_splats：当前场景中的高斯散点总数
     * 
     * 高斯散点数量在训练过程中会动态变化（通过分裂和剪枝操作）
     * 该信息用于监控模型复杂度和内存使用情况
     */
    void TrainingInfo::updateNumSplats(int num_splats) {
        num_splats_ = num_splats;  // 原子操作，更新高斯散点数量
    }

    /**
     * [功能描述]：添加新的损失值到历史记录缓冲区
     * @param loss：当前迭代的损失值
     * 
     * 维护一个固定长度的损失值历史队列，用于：
     * - GUI中绘制损失曲线图
     * - 监控训练收敛情况
     * - 检测训练异常（如损失爆炸）
     * 
     * 使用互斥锁确保多线程安全访问
     */
    void TrainingInfo::updateLoss(float loss) {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);  // 获取互斥锁
        
        loss_buffer_.push_back(loss);  // 在队列末尾添加新的损失值
        
        // 维护缓冲区大小限制，移除最老的数据点
        while (loss_buffer_.size() > static_cast<size_t>(max_loss_points_)) {
            loss_buffer_.pop_front();  // 从队列前端移除最老的损失值
        }
    }

    /**
     * [功能描述]：获取损失值历史记录的副本
     * @return std::deque<float>：损失值历史队列的完整副本
     * 
     * 返回损失缓冲区的副本而非引用，确保调用方可以安全地
     * 访问数据而不会与后续的更新操作产生竞争条件
     */
    std::deque<float> TrainingInfo::getLossBuffer() const {
        // 注意：这里需要移除const以获取互斥锁，这是const成员函数的常见模式
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(loss_buffer_mutex_));
        return loss_buffer_;  // 返回队列的副本（拷贝构造）
    }

    // =============================================================================
    // RenderingConfig 实现部分 - 渲染配置管理
    // =============================================================================

    /**
     * [功能描述]：根据屏幕分辨率计算实际的视野角度
     * @param reso_x：屏幕水平分辨率（像素）
     * @param reso_y：屏幕垂直分辨率（像素）
     * @return glm::vec2：(水平FOV弧度, 垂直FOV弧度)
     * 
     * 将配置的垂直FOV转换为水平和垂直方向的实际FOV值
     * 考虑屏幕宽高比，确保图像不会被拉伸变形
     */
    glm::vec2 RenderingConfig::getFov(size_t reso_x, size_t reso_y) const {
        std::lock_guard<std::mutex> lock(mtx);  // 线程安全访问配置
        
        // 计算水平FOV：根据宽高比调整垂直FOV得到水平FOV
        float horizontal_fov = atan(tan(glm::radians(fov) / 2.0f) * reso_x / reso_y) * 2.0f;
        float vertical_fov = glm::radians(fov);  // 垂直FOV直接转换为弧度
        
        return glm::vec2(horizontal_fov, vertical_fov);
    }

    /**
     * [功能描述]：获取当前设置的视野角度（度）
     * @return float：视野角度值，单位为度
     */
    float RenderingConfig::getFovDegrees() const {
        std::lock_guard<std::mutex> lock(mtx);  // 线程安全访问
        return fov;  // 返回以度为单位的FOV值
    }

    /**
     * [功能描述]：获取当前的缩放修正因子
     * @return float：缩放修正因子，1.0表示无修正
     * 
     * 缩放修正因子用于微调高斯散点的显示尺寸，
     * 可以在不重新训练的情况下调整视觉效果
     */
    float RenderingConfig::getScalingModifier() const {
        std::lock_guard<std::mutex> lock(mtx);  // 线程安全访问
        return scaling_modifier;
    }

    /**
     * [功能描述]：设置新的视野角度
     * @param f：新的视野角度值（度），通常在10-120度范围内
     */
    void RenderingConfig::setFov(float f) {
        std::lock_guard<std::mutex> lock(mtx);  // 线程安全修改
        fov = f;
    }

    /**
     * [功能描述]：设置新的缩放修正因子
     * @param s：新的缩放修正因子，建议在0.1-10.0范围内
     */
    void RenderingConfig::setScalingModifier(float s) {
        std::lock_guard<std::mutex> lock(mtx);  // 线程安全修改
        scaling_modifier = s;
    }

    // =============================================================================
    // ViewerStateManager 实现部分 - 状态管理器核心
    // =============================================================================

    /**
     * [功能描述]：状态管理器构造函数
     * 
     * 初始化状态管理器的所有组件：
     * - 创建训练信息和渲染配置的共享对象
     * - 设置事件处理器，建立与其他组件的通信
     */
    ViewerStateManager::ViewerStateManager() {
        // 创建训练信息共享对象，多个组件可以同时访问
        training_info_ = std::make_shared<TrainingInfo>();
        
        // 创建渲染配置共享对象，支持实时配置更新
        rendering_config_ = std::make_shared<RenderingConfig>();
        
        // 设置事件处理器，建立与事件系统的连接
        setupEventHandlers();
    }

    /**
     * [功能描述]：状态管理器析构函数
     * 使用默认析构函数，共享指针会自动清理资源
     */
    ViewerStateManager::~ViewerStateManager() = default;

    /**
     * [功能描述]：设置事件处理器，建立状态管理器与事件系统的连接
     * 
     * 监听以下类型的事件：
     * - 渲染设置变更事件
     * - 训练进度更新事件
     * - 训练开始事件
     * 
     * 通过事件驱动的方式实现组件间的松耦合通信
     */
    void ViewerStateManager::setupEventHandlers() {
        using namespace events;  // 简化事件命名空间访问

        // ==========================================================================
        // 渲染设置变更事件处理器
        // ==========================================================================
        // 当GUI中的渲染设置发生变化时，同步更新内部配置
        ui::RenderSettingsChanged::when([this](const auto& event) {
            // 如果事件包含FOV变更，更新渲染配置
            if (event.fov) {
                rendering_config_->setFov(*event.fov);
            }
            
            // 如果事件包含缩放修正因子变更，更新配置
            if (event.scaling_modifier) {
                rendering_config_->setScalingModifier(*event.scaling_modifier);
            }
            
            // 如果事件包含抗锯齿设置变更，更新状态
            if (event.antialiasing) {
                setAntiAliasing(*event.antialiasing);
            }
        });

        // ==========================================================================
        // 训练进度事件处理器
        // ==========================================================================
        // 当训练器报告进度更新时，同步更新训练信息
        state::TrainingProgress::when([this](const auto& event) {
            // 更新迭代进度（使用当前总迭代数）
            training_info_->updateProgress(event.iteration, training_info_->total_iterations_);
            
            // 更新高斯散点数量（用于监控模型复杂度）
            training_info_->updateNumSplats(event.num_gaussians);
            
            // 添加新的损失值到历史记录
            training_info_->updateLoss(event.loss);
        });

        // ==========================================================================
        // 训练开始事件处理器
        // ==========================================================================
        // 当训练开始时，设置总迭代数等初始信息
        state::TrainingStarted::when([this](const auto& event) {
            training_info_->total_iterations_ = event.total_iterations;
        });
    }

    /**
     * [功能描述]：设置查看器工作模式
     * @param mode：新的查看器模式（Empty、PLYViewer、Training等）
     * 
     * 模式切换会触发状态变更通知，其他组件可以据此调整行为
     * 使用原子操作确保模式切换的线程安全性
     */
    void ViewerStateManager::setMode(ViewerMode mode) {
        // 原子交换操作，获取旧模式并设置新模式
        ViewerMode old_mode = current_mode_.exchange(mode);
        
        // 只有当模式确实发生变化时才发布状态变更事件
        if (old_mode != mode) {
            publishStateChange();  // 通知其他组件模式已变更
        }
    }

    /**
     * [功能描述]：设置PLY文件路径并切换到PLY查看模式
     * @param path：PLY文件的完整路径
     * 
     * 设置路径的同时自动切换到PLYViewer模式，
     * 表示当前正在查看静态点云数据
     */
    void ViewerStateManager::setPLYPath(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(paths_mutex_);  // 线程安全路径操作
        current_ply_path_ = path;                        // 保存PLY文件路径
        setMode(ViewerMode::PLYViewer);                  // 自动切换模式
    }

    /**
     * [功能描述]：设置数据集路径并切换到训练模式
     * @param path：数据集根目录的完整路径
     * 
     * 设置路径的同时自动切换到Training模式，
     * 表示当前正在进行或准备进行高斯散点训练
     */
    void ViewerStateManager::setDatasetPath(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(paths_mutex_);  // 线程安全路径操作
        current_dataset_path_ = path;                    // 保存数据集路径
        setMode(ViewerMode::Training);                   // 自动切换模式
    }

    /**
     * [功能描述]：清空所有路径信息并回到空状态
     * 
     * 清理当前加载的所有文件路径，并将查看器
     * 重置为Empty模式，表示没有加载任何数据
     */
    void ViewerStateManager::clearPaths() {
        std::lock_guard<std::mutex> lock(paths_mutex_);  // 线程安全路径操作
        current_ply_path_.clear();                       // 清空PLY路径
        current_dataset_path_.clear();                   // 清空数据集路径
        setMode(ViewerMode::Empty);                      // 切换到空模式
    }

    /**
     * [功能描述]：设置抗锯齿开关状态
     * @param enabled：true启用抗锯齿，false禁用
     * 
     * 抗锯齿设置会影响渲染质量和性能，
     * 使用原子变量确保设置的线程安全性
     */
    void ViewerStateManager::setAntiAliasing(bool enabled) {
        anti_aliasing_ = enabled;  // 原子操作，设置抗锯齿状态
    }

    /**
     * [功能描述]：重置所有状态到初始值
     * 
     * 清理操作包括：
     * - 清空所有文件路径
     * - 重置训练进度信息
     * - 清空损失值历史记录
     * 
     * 通常在开始新的训练会话或切换项目时调用
     */
    void ViewerStateManager::reset() {
        clearPaths();  // 清空路径并重置模式
        
        // 重置训练信息到初始状态
        training_info_->curr_iterations_ = 0;      // 清零当前迭代数
        training_info_->total_iterations_ = 0;     // 清零总迭代数
        training_info_->num_splats_ = 0;           // 清零高斯散点数
        training_info_->loss_buffer_.clear();     // 清空损失历史记录
    }

    /**
     * [功能描述]：发布状态变更通知事件
     * 
     * 当重要状态（如查看器模式）发生变化时，
     * 通过事件系统通知其他组件进行相应调整
     * 
     * 目前使用日志事件作为通知机制，未来可能
     * 会添加专门的状态变更事件类型
     */
    void ViewerStateManager::publishStateChange() {
        // 由于ViewerModeChanged事件尚未定义，暂时使用日志事件
        // 未来可以创建专门的状态变更事件以提供更详细的信息
        events::notify::Log{
            .level = events::notify::Log::Level::Debug,     // 调试级别日志
            .message = "Viewer mode changed",               // 状态变更消息
            .source = "ViewerStateManager"                  // 事件来源标识
        }.emit();  // 发布事件到事件系统
    }

} // namespace gs::visualizer