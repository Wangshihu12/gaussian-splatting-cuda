#include "visualizer_impl.hpp"
#include "core/command_processor.hpp"
#include "core/data_loading_service.hpp"
#include "core/model_providers.hpp"
#include "tools/background_tool.hpp"
#include "tools/crop_box_tool.hpp"

#include <tools/world_transform_tool.hpp>

namespace gs::visualizer {

    /**
     * [功能描述]：VisualizerImpl类构造函数，初始化整个3D可视化系统
     * @param options：查看器配置选项，包含窗口尺寸、标题、抗锯齿等设置
     * 
     * 构造函数按照依赖关系顺序初始化各个组件：
     * 1. 基础组件：视口、窗口管理器
     * 2. 状态管理：状态管理器、兼容性指针
     * 3. 核心管理器：场景、训练器、工具管理器
     * 4. 支持组件：GUI、错误处理、内存监控
     * 5. 渲染系统：渲染管理器
     * 6. 服务组件：命令处理器、数据加载器
     * 7. 主循环和事件系统设置
     */
    VisualizerImpl::VisualizerImpl(const ViewerOptions& options)
        : options_(options),                                                    // 保存配置选项
          viewport_(options.width, options.height),                           // 初始化视口尺寸
          window_manager_(std::make_unique<WindowManager>(options.title, options.width, options.height)) {  // 创建窗口管理器

        // =============================================================================
        // 步骤1：创建状态管理器（最高优先级，其他组件依赖它）
        // =============================================================================
        // 状态管理器是核心组件，负责管理应用程序的全局状态
        // 包括训练信息、渲染配置、查看器模式等
        state_manager_ = std::make_unique<ViewerStateManager>();

        // =============================================================================
        // 步骤2：设置兼容性指针，为GUI提供直接访问
        // =============================================================================
        // 这些指针提供对状态管理器内部对象的直接访问，用于GUI组件
        info_ = state_manager_->getTrainingInfo();      // 训练信息共享指针
        config_ = state_manager_->getRenderingConfig(); // 渲染配置共享指针
        anti_aliasing_ = options.antialiasing;          // 保存抗锯齿设置到成员变量
        state_manager_->setAntiAliasing(options.antialiasing);  // 同步设置到状态管理器

        // =============================================================================
        // 步骤3：创建场景管理器和默认场景
        // =============================================================================
        // 场景管理器负责3D场景的生命周期管理，包括高斯散点、相机等
        scene_manager_ = std::make_unique<SceneManager>();
        
        // 创建默认的空场景对象
        auto scene = std::make_unique<Scene>();
        scene_manager_->setScene(std::move(scene));     // 转移场景所有权给管理器

        // =============================================================================
        // 步骤4：创建训练器管理器并建立双向关联
        // =============================================================================
        // 训练器管理器负责高斯散点训练过程的管理和监控
        trainer_manager_ = std::make_shared<TrainerManager>();
        trainer_manager_->setViewer(this);              // 设置可视化器反向引用
        scene_manager_->setTrainerManager(trainer_manager_.get());  // 建立场景与训练器的连接

        // =============================================================================
        // 步骤5：创建工具管理器并注册内置工具
        // =============================================================================
        // 工具管理器负责各种交互工具的管理（裁剪框、变换工具等）
        tool_manager_ = std::make_unique<ToolManager>(this);
        tool_manager_->registerBuiltinTools();          // 注册所有内置工具类型
        
        // 添加默认启用的工具
        tool_manager_->addTool("Crop Box");             // 裁剪框工具，用于场景裁剪
        tool_manager_->addTool("World Transform");      // 世界变换工具，用于坐标系调整
        tool_manager_->addTool("Background");           // 背景设置工具

        // =============================================================================
        // 步骤6：创建支持组件
        // =============================================================================
        // GUI管理器：负责ImGui界面的创建和管理
        gui_manager_ = std::make_unique<gui::GuiManager>(this);
        
        // 错误处理器：统一的错误处理和日志记录
        error_handler_ = std::make_unique<ErrorHandler>();
        
        // 内存监控器：实时监控GPU和CPU内存使用情况
        memory_monitor_ = std::make_unique<MemoryMonitor>();
        memory_monitor_->start();                       // 启动内存监控线程

        // =============================================================================
        // 步骤7：创建渲染管理器
        // =============================================================================
        // 渲染管理器负责OpenGL渲染管线、着色器管理、帧率控制等
        rendering_manager_ = std::make_unique<RenderingManager>();

        // =============================================================================
        // 步骤8：创建命令处理器
        // =============================================================================
        // 命令处理器实现命令模式，处理用户操作的撤销/重做功能
        command_processor_ = std::make_unique<CommandProcessor>(scene_manager_.get());

        // =============================================================================
        // 步骤9：创建数据加载服务
        // =============================================================================
        // 数据加载服务负责异步加载PLY文件、数据集等，避免阻塞UI
        data_loader_ = std::make_unique<DataLoadingService>(scene_manager_.get(), state_manager_.get());

        // =============================================================================
        // 步骤10：创建主循环管理器
        // =============================================================================
        // 主循环管理器负责应用程序的渲染循环控制
        main_loop_ = std::make_unique<MainLoop>();

        // =============================================================================
        // 步骤11：设置组件间连接和事件处理
        // =============================================================================
        // 建立各组件间的事件连接和回调关系
        setupEventHandlers();       // 设置事件处理器（鼠标、键盘、窗口事件等）
        setupComponentConnections(); // 设置组件间的连接和通信机制
    }

    VisualizerImpl::~VisualizerImpl() {
        trainer_manager_.reset();
        if (gui_manager_) {
            gui_manager_->shutdown();
        }
        std::cout << "Visualizer destroyed." << std::endl;
    }

    /**
     * [功能描述]：设置各组件间的连接和回调关系
     * 
     * 该函数建立整个可视化系统的通信机制：
     * 1. 配置主循环的生命周期回调函数
     * 2. 设置GUI管理器的命令执行和文件操作回调
     * 3. 建立事件驱动的组件间通信
     * 
     * 这些连接使得各个独立的组件能够协同工作，形成一个完整的可视化系统
     */
    void VisualizerImpl::setupComponentConnections() {
        // =============================================================================
        // 步骤1：设置主循环回调函数
        // =============================================================================
        // 这些回调函数定义了应用程序的完整生命周期，使用lambda表达式绑定到当前对象的方法
        
        /**
         * 初始化回调：在主循环开始前执行一次
         * 负责OpenGL上下文初始化、资源加载、组件启动等
         */
        main_loop_->setInitCallback([this]() { return initialize(); });
        
        /**
         * 更新回调：每帧执行一次，在渲染前调用
         * 负责处理用户输入、更新动画、处理事件、更新GUI状态等
         */
        main_loop_->setUpdateCallback([this]() { update(); });
        
        /**
         * 渲染回调：每帧执行一次，在更新后调用
         * 负责执行所有OpenGL渲染操作，包括3D场景、GUI界面等
         */
        main_loop_->setRenderCallback([this]() { render(); });
        
        /**
         * 关闭回调：在主循环结束后执行一次
         * 负责资源清理、状态保存、线程终止等善后工作
         */
        main_loop_->setShutdownCallback([this]() { shutdown(); });
        
        /**
         * 窗口关闭检查回调：每帧检查窗口是否应该关闭
         * 通过窗口管理器检查用户是否点击了关闭按钮或按下了退出快捷键
         */
        main_loop_->setShouldCloseCallback([this]() { return window_manager_->shouldClose(); });

        // =============================================================================
        // 步骤2：设置GUI管理器连接
        // =============================================================================
        
        /**
         * 脚本执行器回调：为GUI提供命令行接口
         * 当用户在GUI控制台输入命令时，将命令传递给命令处理器执行
         * @param command：用户输入的命令字符串
         * @return 命令执行结果的字符串表示
         */
        gui_manager_->setScriptExecutor([this](const std::string& command) -> std::string {
            return command_processor_->processCommand(command);
        });

        /**
         * 文件选择回调：处理GUI文件浏览器的文件选择事件
         * 当用户通过GUI选择文件时，发送加载文件的事件
         * @param path：用户选择的文件或目录路径
         * @param is_dataset：true表示选择的是数据集目录，false表示选择的是PLY文件
         */
        gui_manager_->setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            // 使用事件系统异步处理文件加载，避免阻塞GUI线程
            events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();
        });

        // =============================================================================
        // 注意事项：输入管理器连接
        // =============================================================================
        // 移除输入管理器的设置 - 因为它还没有被创建！
        // 输入管理器的连接将在initialize()函数中设置，
        // 那时OpenGL上下文已经创建，可以安全地初始化输入系统
    }

    /**
     * [功能描述]：设置事件处理器，建立事件驱动的系统通信机制
     * 
     * 该函数注册各种事件的处理回调，实现组件间的松耦合通信：
     * 1. 训练控制命令事件（开始、暂停、恢复、停止、保存）
     * 2. 渲染设置变更事件
     * 3. UI交互事件
     * 4. 状态变更通知事件
     * 5. 系统通知事件（警告、错误）
     * 6. 内部协调事件
     */
    void VisualizerImpl::setupEventHandlers() {
        using namespace events;  // 简化事件命名空间访问

        // =============================================================================
        // 训练控制命令事件处理器
        // =============================================================================
        
        /**
         * 开始训练命令事件处理器
         * 当用户点击GUI中的"开始训练"按钮或发送开始训练命令时触发
         */
        cmd::StartTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->startTraining();  // 委托给训练器管理器执行
            }
            // 添加计时开始
            if (auto training_info = state_manager_->getTrainingInfo()) {
                training_info->startTiming();
            }
        });

        /**
         * 暂停训练命令事件处理器
         * 允许用户临时暂停训练过程，保持当前状态，可以稍后恢复
         */
        cmd::PauseTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->pauseTraining();  // 暂停训练循环，保留训练状态
            }
            // 添加计时暂停
            if (auto training_info = state_manager_->getTrainingInfo()) {
                training_info->pauseTiming();
            }
        });

        /**
         * 恢复训练命令事件处理器
         * 从暂停状态恢复训练，继续之前的训练进度
         */
        cmd::ResumeTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->resumeTraining();  // 恢复训练循环执行
            }
            // 添加计时恢复
            if (auto training_info = state_manager_->getTrainingInfo()) {
                training_info->resumeTiming();
            }
        });

        /**
         * 停止训练命令事件处理器
         * 完全停止训练过程，清理训练资源，回到查看模式
         */
        cmd::StopTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->stopTraining();   // 停止训练并清理资源
            }
            // 添加计时停止
            if (auto training_info = state_manager_->getTrainingInfo()) {
                training_info->stopTiming();
            }
        });

        /**
         * 保存检查点命令事件处理器
         * 保存当前训练状态和模型参数，用于断点恢复或结果导出
         */
        cmd::SaveCheckpoint::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->requestSaveCheckpoint();  // 异步保存检查点
            }
        });

        // =============================================================================
        // 渲染设置变更事件处理器
        // =============================================================================
        
        /**
         * 渲染设置变更事件处理器
         * 当用户在GUI中修改渲染参数时，同步更新渲染管理器的设置
         * 
         * 注意：渲染设置的变更由ViewerStateManager处理状态更新，
         * 但我们需要将这些变更同步到渲染管理器以生效
         */
        ui::RenderSettingsChanged::when([this](const auto&) {
            // 获取渲染管理器当前设置的副本
            auto settings = rendering_manager_->getSettings();

            // 从状态管理器获取最新的渲染配置值
            settings.fov = state_manager_->getRenderingConfig()->getFovDegrees();              // 视野角度
            settings.scaling_modifier = state_manager_->getRenderingConfig()->getScalingModifier();  // 缩放修正因子
            settings.antialiasing = state_manager_->isAntiAliasingEnabled();                   // 抗锯齿开关

            // 更新兼容性标志，确保与旧代码的兼容性
            anti_aliasing_ = settings.antialiasing;

            // 将更新后的设置应用到渲染管理器
            rendering_manager_->updateSettings(settings);
        });

        // =============================================================================
        // UI交互事件处理器
        // =============================================================================
        
        /**
         * 相机移动事件处理器
         * 当用户通过鼠标或键盘移动相机时触发
         * 目前预留用于自动保存相机位置等功能
         */
        ui::CameraMove::when([this](const auto&) {
            // 可以用于自动保存相机位置或触发重新渲染
            // 当前为空实现，预留接口
        });

        // =============================================================================
        // 状态变更通知事件处理器
        // =============================================================================
        
        /**
         * 评估完成事件处理器
         * 当训练过程中的模型评估完成时，将结果显示在GUI控制台中
         */
        state::EvaluationCompleted::when([this](const auto& event) {
            if (gui_manager_) {
                // 在控制台显示评估指标：PSNR（峰值信噪比）、SSIM（结构相似性）、LPIPS（感知损失）
                gui_manager_->addConsoleLog(
                    "Evaluation completed - PSNR: %.2f, SSIM: %.3f, LPIPS: %.3f",
                    event.psnr, event.ssim, event.lpips);
            }
        });

        // =============================================================================
        // 系统通知事件处理器
        // =============================================================================
        
        /**
         * 内存警告事件处理器
         * 当系统检测到内存使用过高时，在GUI控制台显示警告信息
         */
        notify::MemoryWarning::when([this](const auto& event) {
            if (gui_manager_) {
                gui_manager_->addConsoleLog("WARNING: %s", event.message.c_str());
            }
        });

        /**
         * 错误通知事件处理器
         * 当系统发生错误时，在GUI控制台显示错误信息和详细描述
         */
        notify::Error::when([this](const auto& event) {
            if (gui_manager_) {
                // 显示错误主要信息
                gui_manager_->addConsoleLog("ERROR: %s", event.message.c_str());
                
                // 如果有详细信息，也一并显示
                if (!event.details.empty()) {
                    gui_manager_->addConsoleLog("Details: %s", event.details.c_str());
                }
            }
        });

        // =============================================================================
        // 内部协调事件处理器
        // =============================================================================
        
        /**
         * 训练器就绪事件处理器
         * 当训练器初始化完成并准备开始训练时，发送训练准备开始的信号
         * 这是一个内部协调事件，用于确保组件间的正确同步
         */
        internal::TrainerReady::when([this](const auto&) {
            // 发送训练准备开始的内部事件，通知其他组件可以开始训练流程
            internal::TrainingReadyToStart{}.emit();
        });
    }

    /**
     * [功能描述]：初始化可视化器的所有组件和OpenGL环境
     * @return bool：true表示初始化成功，false表示初始化失败
     * 
     * 初始化流程按照依赖关系顺序执行：
     * 1. 窗口管理器和OpenGL上下文
     * 2. 输入管理器和事件回调
     * 3. 渲染管理器和OpenGL资源
     * 4. 工具管理器
     * 5. GUI管理器和ImGui初始化
     * 
     * 任何步骤失败都会导致整个初始化过程失败
     */
    bool VisualizerImpl::initialize() {
        // =============================================================================
        // 步骤1：初始化窗口管理器和OpenGL上下文
        // =============================================================================
        // 窗口管理器负责创建GLFW窗口和OpenGL上下文
        // 这是所有其他组件的前提，必须首先成功初始化
        if (!window_manager_->init()) {
            return false;  // 窗口创建失败，直接返回失败
        }

        // =============================================================================
        // 步骤2：创建和初始化输入管理器
        // =============================================================================
        // 输入管理器需要有效的GLFW窗口句柄，所以在窗口管理器初始化后创建
        // 它负责处理键盘、鼠标输入，以及相机控制
        input_manager_ = std::make_unique<InputManager>(window_manager_->getWindow(), viewport_);
        input_manager_->initialize();  // 设置GLFW输入回调和初始状态
        
        // 建立输入管理器与训练器管理器的连接，用于训练过程中的键盘快捷键
        input_manager_->setTrainingManager(trainer_manager_);

        // =============================================================================
        // 步骤3：设置输入管理器的回调函数
        // =============================================================================
        
        /**
         * 设置视口焦点检查回调
         * 用于判断当前视口是否获得焦点，影响相机控制的激活状态
         * 当GUI窗口处于活跃状态时，相机控制可能需要被禁用
         */
        input_manager_->setViewportFocusCheck([this]() {
            return gui_manager_ && gui_manager_->isViewportFocused();
        });

        /**
         * 设置鼠标位置检查回调
         * 用于判断鼠标事件是否发生在有效的视口区域内
         * 防止GUI区域的鼠标操作影响3D场景交互
         * @param x, y：鼠标在窗口中的像素坐标
         * @return bool：true表示鼠标在视口内，false表示在GUI区域
         */
        input_manager_->setPositionCheck([this](double x, double y) {
            return gui_manager_ && gui_manager_->isPositionInViewport(x, y);
        });

        // =============================================================================
        // 步骤4：设置输入管理器的高级回调
        // =============================================================================
        // 这些回调必须在input_manager_创建后设置，处理复杂的用户交互
        input_manager_->setupCallbacks(
            /**
             * GUI活跃状态检查回调
             * 当任何GUI窗口处于活跃状态时，返回true
             * 用于决定是否应该处理某些输入事件（如文件拖拽）
             */
            [this]() { return gui_manager_ && gui_manager_->isAnyWindowActive(); },
            
            /**
             * 文件拖拽处理回调
             * 当用户将文件拖拽到窗口时触发
             * @param path：被拖拽文件的路径
             * @param is_dataset：true表示是数据集目录，false表示是PLY文件
             * @return bool：true表示处理成功
             */
            [this](const std::filesystem::path& path, bool is_dataset) {
                // 文件加载现在通过DataLoadingService异步处理，使用事件系统
                // 避免在输入线程中执行耗时的文件加载操作
                events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();
                return true;  // 始终返回true，因为加载是异步的
            });

        // =============================================================================
        // 步骤5：初始化渲染管理器
        // =============================================================================
        // 渲染管理器负责OpenGL资源管理、着色器编译、渲染管线设置等
        // 需要有效的OpenGL上下文，所以在窗口管理器初始化后执行
        rendering_manager_->initialize();

        // =============================================================================
        // 步骤6：初始化工具管理器
        // =============================================================================
        // 工具管理器负责各种交互工具的初始化
        // 如裁剪框工具、坐标变换工具、背景设置工具等
        tool_manager_->initialize();

        // =============================================================================
        // 步骤7：初始化GUI管理器
        // =============================================================================
        // GUI管理器负责ImGui的初始化和所有GUI界面的创建
        // 必须在OpenGL上下文创建后进行，因为ImGui需要OpenGL资源
        gui_manager_->init();
        gui_initialized_ = true;  // 设置GUI初始化完成标志

        return true;  // 所有组件初始化成功
    }

    void VisualizerImpl::update() {
        window_manager_->updateWindowSize();

        // Update the main viewport with window size
        viewport_.windowSize = window_manager_->getWindowSize();
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();

        // 如果正在训练，定期更新GUI以刷新时间显示
        if (auto training_info = state_manager_->getTrainingInfo()) {
            if (training_info->getCurrentTrainingTimeSeconds() > 0) {
                // GUI会在下一帧自动刷新，无需特殊处理
            }
        }

        // Update tools
        tool_manager_->update();
    }

    /**
     * [功能描述]：执行单帧渲染操作，绘制整个3D场景和GUI界面
     * 
     * 渲染流程：
     * 1. 更新输入路由状态
     * 2. 收集和更新渲染设置
     * 3. 从各个工具获取状态信息
     * 4. 构建渲染上下文
     * 5. 执行主场景渲染
     * 6. 渲染工具覆盖层
     * 7. 渲染GUI界面
     * 8. 交换帧缓冲区并处理事件
     */
    void VisualizerImpl::render() {
        // =============================================================================
        // 步骤1：更新输入路由状态
        // =============================================================================
        // 根据当前的焦点状态更新输入事件的路由
        // 确保输入事件被正确地分发到相应的组件（3D视口 vs GUI）
        if (input_manager_) {
            input_manager_->updateInputRouting();
        }

        // =============================================================================
        // 步骤2：收集和更新渲染设置
        // =============================================================================
        // 从渲染管理器获取当前设置的副本
        RenderSettings settings = rendering_manager_->getSettings();

        // =============================================================================
        // 步骤3：从裁剪框工具获取状态
        // =============================================================================
        // 检查裁剪框工具是否存在，并获取其显示和使用状态
        if (auto* crop_tool = dynamic_cast<CropBoxTool*>(tool_manager_->getTool("Crop Box"))) {
            settings.show_crop_box = crop_tool->shouldShowBox();    // 是否显示裁剪框线框
            settings.use_crop_box = crop_tool->shouldUseBox();      // 是否应用裁剪效果
        } else {
            // 如果裁剪框工具不存在，禁用相关功能
            settings.show_crop_box = false;
            settings.use_crop_box = false;
        }

        // =============================================================================
        // 步骤4：从世界变换工具获取坐标轴显示状态
        // =============================================================================
        // 检查世界变换工具是否存在，并获取坐标轴的显示状态
        if (auto* world_trans = dynamic_cast<WorldTransformTool*>(tool_manager_->getTool("World Transform"))) {
            settings.show_coord_axes = world_trans->ShouldShowAxes();  // 是否显示坐标轴
        } else {
            // 如果世界变换工具不存在，不显示坐标轴
            settings.show_coord_axes = false;
        }

        // =============================================================================
        // 步骤5：从状态管理器同步渲染参数
        // =============================================================================
        // 将状态管理器中的最新设置同步到渲染设置中
        settings.antialiasing = state_manager_->isAntiAliasingEnabled();                      // 抗锯齿开关
        settings.fov = state_manager_->getRenderingConfig()->getFovDegrees();                 // 视野角度
        settings.scaling_modifier = state_manager_->getRenderingConfig()->getScalingModifier(); // 缩放修正因子
        
        // 将更新后的设置应用到渲染管理器
        rendering_manager_->updateSettings(settings);

        // =============================================================================
        // 步骤6：获取裁剪框渲染对象
        // =============================================================================
        // 获取裁剪框的渲染对象指针，用于在渲染时绘制裁剪框
        RenderBoundingBox* crop_box_ptr = nullptr;
        if (auto crop_box = getCropBox()) {
            crop_box_ptr = crop_box.get();  // 获取原始指针
        }

        // =============================================================================
        // 步骤7：从GUI管理器获取视口区域信息
        // =============================================================================
        // 获取GUI中3D视口的实际位置和尺寸，用于正确的视口渲染
        ViewportRegion viewport_region;
        bool has_viewport_region = false;
        if (gui_manager_) {
            ImVec2 pos = gui_manager_->getViewportPos();   // 视口在窗口中的位置
            ImVec2 size = gui_manager_->getViewportSize(); // 视口的实际尺寸

            // 注意：pos和size已经是相对于窗口的坐标！
            // 直接传递给渲染系统即可，无需额外转换
            viewport_region.x = pos.x;          // 视口左上角X坐标
            viewport_region.y = pos.y;          // 视口左上角Y坐标
            viewport_region.width = size.x;     // 视口宽度
            viewport_region.height = size.y;    // 视口高度

            has_viewport_region = true;         // 标记视口区域有效
        }

        // =============================================================================
        // 步骤8：获取坐标轴和世界变换信息
        // =============================================================================
        // 获取坐标轴渲染对象，用于在场景中显示坐标系参考
        const RenderCoordinateAxes* coord_axes_ptr = nullptr;
        if (auto coord_axes = getAxes()) {
            coord_axes_ptr = coord_axes.get();
        }

        // 获取世界到用户坐标的变换矩阵
        const geometry::EuclideanTransform* world_to_user = nullptr;
        if (auto coord_axes = getWorldToUser()) {
            world_to_user = coord_axes.get();
        }

        // =============================================================================
        // 步骤9：获取背景工具
        // =============================================================================
        // 获取背景工具的指针，用于应用自定义背景设置
        const BackgroundTool* background_tool = nullptr;
        if (auto* bg_tool = dynamic_cast<BackgroundTool*>(tool_manager_->getTool("Background"))) {
            background_tool = bg_tool;
        }

        // =============================================================================
        // 步骤10：构建渲染上下文
        // =============================================================================
        // 将所有收集到的信息组装成渲染上下文对象
        RenderingManager::RenderContext context{
            .viewport = viewport_,                                                          // 基础视口配置
            .settings = rendering_manager_->getSettings(),                                 // 渲染设置
            .crop_box = crop_box_ptr,                                                      // 裁剪框对象
            .coord_axes = coord_axes_ptr,                                                  // 坐标轴对象
            .world_to_user = world_to_user,                                               // 坐标变换
            .viewport_region = has_viewport_region ? &viewport_region : nullptr,          // 视口区域
            .has_focus = gui_manager_ && gui_manager_->isViewportFocused(),              // 焦点状态
            .background_tool = background_tool                                            // 背景工具
        };

        // =============================================================================
        // 步骤11：执行主场景渲染
        // =============================================================================
        // 使用构建好的上下文渲染整个3D场景
        // 包括高斯散点、背景、裁剪框、坐标轴等所有3D元素
        rendering_manager_->renderFrame(context, scene_manager_.get());

        // =============================================================================
        // 步骤12：渲染工具覆盖层
        // =============================================================================
        // 在3D场景之上渲染各种工具的可视化元素
        // 如工具手柄、辅助线、选择框等
        tool_manager_->render();

        // =============================================================================
        // 步骤13：渲染GUI界面
        // =============================================================================
        // 在所有3D内容之上渲染ImGui界面
        // 包括控制面板、工具窗口、调试信息等
        gui_manager_->render();

        // =============================================================================
        // 步骤14：交换缓冲区和处理事件
        // =============================================================================
        // 将后台缓冲区的内容显示到屏幕上（双缓冲技术）
        window_manager_->swapBuffers();
        
        // 处理GLFW事件队列中的所有待处理事件
        // 包括窗口事件、输入事件等
        window_manager_->pollEvents();
    }

    void VisualizerImpl::shutdown() {
        tool_manager_->shutdown();
    }

    void VisualizerImpl::run() {
        main_loop_->run();
    }

    void VisualizerImpl::setParameters(const param::TrainingParameters& params) {
        data_loader_->setParameters(params);
    }

    std::expected<void, std::string> VisualizerImpl::loadPLY(const std::filesystem::path& path) {
        return data_loader_->loadPLY(path);
    }

    std::expected<void, std::string> VisualizerImpl::loadDataset(const std::filesystem::path& path) {
        return data_loader_->loadDataset(path);
    }

    void VisualizerImpl::clearScene() {
        data_loader_->clearScene();
    }

    std::shared_ptr<RenderBoundingBox> VisualizerImpl::getCropBox() const {
        if (auto* crop_tool = dynamic_cast<CropBoxTool*>(tool_manager_->getTool("Crop Box"))) {
            return crop_tool->getBoundingBox();
        }
        return nullptr;
    }

    std::shared_ptr<const RenderCoordinateAxes> VisualizerImpl::getAxes() const {
        if (auto* world_transform = dynamic_cast<WorldTransformTool*>(tool_manager_->getTool("World Transform"))) {
            return world_transform->getAxes();
        }
        return nullptr;
    }

    std::shared_ptr<const geometry::EuclideanTransform> VisualizerImpl::getWorldToUser() const {
        if (auto* world_transform = dynamic_cast<WorldTransformTool*>(tool_manager_->getTool("World Transform"))) {
            if (world_transform->IsTrivialTrans()) {
                return nullptr;
            }
            return world_transform->GetTransform();
        }
        return nullptr;
    }

} // namespace gs::visualizer