#pragma once

#include "core/error_handler.hpp"
#include "core/main_loop.hpp"
#include "core/memory_monitor.hpp"
#include "core/parameters.hpp"
#include "core/viewer_state_manager.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_manager.hpp"
#include "internal/viewport.hpp"
#include "rendering/render_bounding_box.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene.hpp"
#include "scene/scene_manager.hpp"
#include "tools/tool_manager.hpp"
#include "training/training_manager.hpp"
#include "visualizer/visualizer.hpp"
#include "window/window_manager.hpp"
#include <memory>
#include <string>

/**
 * [文件描述]：可视化器实现类头文件
 * 功能：定义VisualizerImpl类，实现高斯散点3D可视化系统的核心功能
 * 用途：为高斯散点训练和推理提供实时3D查看器界面
 */

// GLFW窗口系统的前向声明
struct GLFWwindow;

namespace gs {
    class CommandProcessor;  // 命令处理器前向声明
}

namespace gs::visualizer {
    class DataLoadingService;  // 数据加载服务前向声明

    /**
     * [类描述]：可视化器实现类
     * 
     * VisualizerImpl是整个3D可视化系统的核心实现类，负责：
     * - 窗口管理和OpenGL上下文初始化
     * - GUI界面的创建和管理（ImGui集成）
     * - 场景渲染和相机控制
     * - 训练过程的实时监控和可视化
     * - 数据加载（PLY文件、数据集）
     * - 用户交互处理（鼠标、键盘输入）
     * - 工具系统（裁剪框、背景设置等）
     * 
     * 该类继承自Visualizer基类，实现了所有抽象接口
     */
    class VisualizerImpl : public Visualizer {
    public:
        // =============================================================================
        // 构造和析构函数
        // =============================================================================
        
        /**
         * [功能描述]：构造函数，初始化可视化器实例
         * @param options：查看器配置选项，包含窗口大小、标题、渲染设置等
         */
        explicit VisualizerImpl(const ViewerOptions& options);
        
        /**
         * [功能描述]：析构函数，清理所有资源和组件
         */
        ~VisualizerImpl() override;

        // =============================================================================
        // 核心接口实现（继承自Visualizer基类）
        // =============================================================================
        
        /**
         * [功能描述]：启动可视化器主循环
         * 该函数会阻塞执行，直到用户关闭窗口
         */
        void run() override;
        
        /**
         * [功能描述]：设置训练参数配置
         * @param params：训练参数对象，包含所有训练相关设置
         */
        void setParameters(const param::TrainingParameters& params) override;
        
        /**
         * [功能描述]：加载PLY格式的点云文件
         * @param path：PLY文件的文件系统路径
         * @return 期望值类型，成功时返回void，失败时返回错误信息
         */
        std::expected<void, std::string> loadPLY(const std::filesystem::path& path) override;
        
        /**
         * [功能描述]：加载训练数据集
         * @param path：数据集根目录路径（支持COLMAP、Blender等格式）
         * @return 期望值类型，成功时返回void，失败时返回错误信息
         */
        std::expected<void, std::string> loadDataset(const std::filesystem::path& path) override;
        
        /**
         * [功能描述]：清空当前场景，移除所有加载的数据
         */
        void clearScene() override;

        // =============================================================================
        // GUI界面访问器方法（委托给状态管理器）
        // =============================================================================
        
        /**
         * [功能描述]：获取当前查看器模式
         * @return ViewerMode：当前模式（查看、训练等）
         */
        ViewerMode getCurrentMode() const { return state_manager_->getCurrentMode(); }
        
        /**
         * [功能描述]：获取训练器对象指针
         * @return Trainer指针：当前活跃的训练器实例，可能为nullptr
         */
        Trainer* getTrainer() const { return trainer_manager_->getTrainer(); }
        
        /**
         * [功能描述]：获取训练信息共享指针
         * @return 训练信息对象，包含训练进度、损失值、性能指标等
         */
        std::shared_ptr<TrainingInfo> getTrainingInfo() const { return state_manager_->getTrainingInfo(); }
        
        /**
         * [功能描述]：获取渲染配置共享指针
         * @return 渲染配置对象，包含渲染模式、质量设置等
         */
        std::shared_ptr<RenderingConfig> getRenderingConfig() const { return state_manager_->getRenderingConfig(); }
        
        /**
         * [功能描述]：获取当前加载的PLY文件路径
         * @return 文件系统路径对象
         */
        const std::filesystem::path& getCurrentPLYPath() const { return state_manager_->getCurrentPLYPath(); }
        
        /**
         * [功能描述]：获取当前加载的数据集路径
         * @return 文件系统路径对象
         */
        const std::filesystem::path& getCurrentDatasetPath() const { return state_manager_->getCurrentDatasetPath(); }
        
        /**
         * [功能描述]：获取训练器管理器指针
         * @return TrainerManager指针：负责训练器生命周期管理
         */
        TrainerManager* getTrainerManager() { return trainer_manager_.get(); }
        
        /**
         * [功能描述]：获取场景管理器指针
         * @return SceneManager指针：负责3D场景管理
         */
        SceneManager* getSceneManager() { return scene_manager_.get(); }
        
        /**
         * [功能描述]：获取GLFW窗口句柄
         * @return GLFWwindow指针：底层窗口句柄，用于低级操作
         */
        ::GLFWwindow* getWindow() const { return window_manager_->getWindow(); }
        
        /**
         * [功能描述]：获取工具管理器指针
         * @return ToolManager指针：负责各种交互工具的管理
         */
        ToolManager* getToolManager() { return tool_manager_.get(); }
        
        /**
         * [功能描述]：获取渲染管理器指针
         * @return RenderingManager指针：负责OpenGL渲染管理
         */
        RenderingManager* getRenderingManager() { return rendering_manager_.get(); }
        
        /**
         * [功能描述]：获取视口配置引用
         * @return Viewport引用：包含视口尺寸、投影矩阵等信息
         */
        const Viewport& getViewport() const { return viewport_; }

        // =============================================================================
        // 性能监控方法
        // =============================================================================
        
        /**
         * [功能描述]：获取当前帧率
         * @return float：当前瞬时FPS值
         */
        [[nodiscard]] float getCurrentFPS() const {
            return rendering_manager_ ? rendering_manager_->getCurrentFPS() : 0.0f;
        }

        /**
         * [功能描述]：获取平均帧率
         * @return float：一段时间内的平均FPS值
         */
        [[nodiscard]] float getAverageFPS() const {
            return rendering_manager_ ? rendering_manager_->getAverageFPS() : 0.0f;
        }

        // =============================================================================
        // 垂直同步控制方法
        // =============================================================================
        
        /**
         * [功能描述]：设置垂直同步状态
         * @param enabled：true启用垂直同步，false禁用
         */
        void setVSync(bool enabled) {
            if (window_manager_) {
                window_manager_->setVSync(enabled);
            }
        }

        /**
         * [功能描述]：获取当前垂直同步状态
         * @return bool：true表示已启用，false表示已禁用
         */
        [[nodiscard]] bool getVSyncEnabled() const {
            return window_manager_ ? window_manager_->getVSync() : true;
        }

        // =============================================================================
        // 场景组件访问方法
        // =============================================================================
        
        /**
         * [功能描述]：获取裁剪框渲染对象
         * @return RenderBoundingBox共享指针：用于场景裁剪的边界框
         */
        std::shared_ptr<RenderBoundingBox> getCropBox() const;
        
        /**
         * [功能描述]：获取坐标轴渲染对象
         * @return RenderCoordinateAxes共享指针：用于显示坐标系参考
         */
        std::shared_ptr<const RenderCoordinateAxes> getAxes() const;
        
        /**
         * [功能描述]：获取世界到用户坐标变换
         * @return EuclideanTransform共享指针：坐标系变换矩阵
         */
        std::shared_ptr<const geometry::EuclideanTransform> getWorldToUser() const;

        // =============================================================================
        // 兼容性公共成员（为GUI提供临时访问）
        // =============================================================================
        
        std::shared_ptr<TrainingInfo> info_;        // 训练信息对象（兼容性）
        std::shared_ptr<RenderingConfig> config_;   // 渲染配置对象（兼容性）
        bool anti_aliasing_ = false;                // 抗锯齿设置（临时兼容性）

        // 场景管理（临时公开以保持兼容性）
        std::unique_ptr<Scene> scene_;              // 场景对象
        std::shared_ptr<TrainerManager> trainer_manager_;  // 训练器管理器

        // GUI管理器
        std::unique_ptr<gui::GuiManager> gui_manager_;     // GUI管理器

        // 友元类声明，允许这些类访问私有成员
        friend class gui::GuiManager;   // GUI管理器友元
        friend class ToolManager;       // 工具管理器友元

    private:
        // =============================================================================
        // 主循环回调函数
        // =============================================================================
        
        /**
         * [功能描述]：初始化所有组件和OpenGL上下文
         * @return bool：true表示初始化成功，false表示失败
         */
        bool initialize();
        
        /**
         * [功能描述]：更新应用程序状态和逻辑
         */
        void update();
        
        /**
         * [功能描述]：执行渲染操作
         */
        void render();
        
        /**
         * [功能描述]：清理资源和关闭组件
         */
        void shutdown();

        // =============================================================================
        // 事件系统设置
        // =============================================================================
        
        /**
         * [功能描述]：设置事件处理器
         */
        void setupEventHandlers();
        
        /**
         * [功能描述]：设置组件间连接
         */
        void setupComponentConnections();

        // =============================================================================
        // 配置选项
        // =============================================================================
        
        ViewerOptions options_;  // 查看器配置选项

        // =============================================================================
        // 核心组件（使用unique_ptr管理生命周期）
        // =============================================================================
        
        Viewport viewport_;                                      // 视口配置
        std::unique_ptr<WindowManager> window_manager_;          // 窗口管理器
        std::unique_ptr<InputManager> input_manager_;            // 输入管理器
        std::unique_ptr<RenderingManager> rendering_manager_;    // 渲染管理器
        std::unique_ptr<SceneManager> scene_manager_;            // 场景管理器
        std::unique_ptr<ViewerStateManager> state_manager_;      // 状态管理器
        std::unique_ptr<CommandProcessor> command_processor_;    // 命令处理器
        std::unique_ptr<DataLoadingService> data_loader_;        // 数据加载服务
        std::unique_ptr<MainLoop> main_loop_;                    // 主循环管理器
        std::unique_ptr<ToolManager> tool_manager_;              // 工具管理器

        // =============================================================================
        // 支持组件
        // =============================================================================
        
        std::unique_ptr<ErrorHandler> error_handler_;    // 错误处理器
        std::unique_ptr<MemoryMonitor> memory_monitor_;  // 内存监控器
        
        // =============================================================================
        // 状态变量
        // =============================================================================
        
        bool gui_initialized_ = false;  // GUI初始化状态标志
    };

} // namespace gs::visualizer