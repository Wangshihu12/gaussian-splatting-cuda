#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "project/project.hpp"
#include "training/training_setup.hpp"
#include "visualizer/visualizer.hpp"
#include <print>

namespace gs {

    /**
     * [功能描述]：运行无头模式（无GUI）的训练应用程序，用于批量训练或服务器环境。
     * @param params [参数说明]：训练参数，包含数据集路径和优化配置。
     * @return [返回值说明]：程序退出码，0表示成功，-1表示失败。
     */
    int run_headless_app(std::unique_ptr<param::TrainingParameters> params) {
        // 检查数据集路径是否为空，无头模式必须指定数据路径
        if (params->dataset.data_path.empty()) {
            std::println(stderr, "Error: Headless mode requires --data-path");
            return -1;
        }

        std::println("Starting headless training...");

        // 创建新项目
        auto project = gs::management::CreateNewProject(params->dataset, params->optimization);
        if (!project) {
            std::println(stderr, "project creation failed");
            return -1;
        }

        // 设置训练环境
        auto setup_result = gs::training::setupTraining(*params);
        if (!setup_result) {
            std::println(stderr, "Error: {}", setup_result.error());
            return -1;
        }

        // 将项目设置到训练器中并开始训练
        setup_result->trainer->setProject(project);
        auto train_result = setup_result->trainer->train();
        if (!train_result) {
            std::println(stderr, "Training error: {}", train_result.error());
            return -1;
        }

        return 0;  // 训练成功完成
    }

    /**
     * [功能描述]：运行图形用户界面应用程序，提供交互式的3D场景查看和训练功能。
     * @param params [参数说明]：训练参数，包含数据集路径、项目路径和优化配置。
     * @return [返回值说明]：程序退出码，0表示成功，-1表示失败。
     */
    int run_gui_app(std::unique_ptr<param::TrainingParameters> params) {

        // 启动GUI应用程序
        std::println("Starting viewer mode...");

        // 创建可视化器，设置窗口标题、尺寸、抗锯齿和CUDA互操作
        auto viewer = visualizer::Visualizer::create({.title = "LichtFeld Studio",
                                                      .width = 1280,
                                                      .height = 720,
                                                      .antialiasing = params->optimization.antialiasing,
                                                      .enable_cuda_interop = true});

        // 检查项目文件路径是否存在
        if (!params->dataset.project_path.empty() &&
            !std::filesystem::exists(params->dataset.project_path)) {
            std::println(stderr, "project file does not exists {}", params->dataset.project_path.string());
            return -1;
        }

        // 如果项目文件存在，则打开现有项目
        if (std::filesystem::exists(params->dataset.project_path)) {
            bool success = viewer->openProject(params->dataset.project_path);
            if (!success) {
                std::println(stderr, "error opening existing project");
                return -1;
            }
            // 不能同时从命令行打开PLY文件和项目文件
            if (!params->ply_path.empty()) {
                std::println(stderr, "can not open ply and open project from commandline");
                return -1;
            }
            // 不能同时从命令行指定新的数据路径和项目文件
            if (!params->dataset.data_path.empty()) {
                std::println(stderr, "cannot open new data_path and project from commandline");
                return -1;
            }
        } else { 
            // 创建临时项目，直到用户在期望位置保存它
            std::shared_ptr<gs::management::Project> project = nullptr;
            if (params->dataset.output_path.empty()) {
                // 如果没有指定输出路径，创建临时项目
                project = gs::management::CreateTempNewProject(params->dataset, params->optimization);
                if (!project) {
                    LOG_ERROR("project creation failed");
                    return -1;
                }
                params->dataset.output_path = project->getProjectOutputFolder();
            } else {
                // 如果指定了输出路径，创建正式项目
                project = gs::management::CreateNewProject(params->dataset, params->optimization);
                if (!project) {
                    LOG_ERROR("project creation failed");
                    return -1;
                }
            }
            // 将项目附加到可视化器
            viewer->attachProject(project);
        }

        // 设置参数到可视化器
        viewer->setParameters(*params);

        // 如果指定了PLY文件路径，则加载PLY数据
        if (!params->ply_path.empty()) {
            auto result = viewer->loadPLY(params->ply_path);
            if (!result) {
                std::println(stderr, "Error: {}", result.error());
                return -1;
            }
        } else if (!params->dataset.data_path.empty()) {
            // 如果指定了数据集路径，则加载数据集
            auto result = viewer->loadDataset(params->dataset.data_path);
            if (!result) {
                std::println(stderr, "Error: {}", result.error());
                return -1;
            }
        }

        // 显示抗锯齿设置状态
        std::println("Anti-aliasing: {}", params->optimization.antialiasing ? "enabled" : "disabled");

        // 运行可视化器主循环
        viewer->run();

        std::println("Viewer closed.");
        return 0;  // 应用程序正常关闭
    }

    /**
     * [功能描述]：应用程序主运行函数，根据参数决定运行无头模式还是GUI模式。
     * @param params [参数说明]：训练参数，包含优化配置和数据集信息。
     * @return [返回值说明]：程序退出码，0表示成功，-1表示失败。
     */
    int Application::run(std::unique_ptr<param::TrainingParameters> params) {
        // 根据headless标志决定运行模式
        if (params->optimization.headless) {
            return run_headless_app(std::move(params));  // 无GUI模式
        }
        // GUI模式
        return run_gui_app(std::move(params));
    }
} // namespace gs
