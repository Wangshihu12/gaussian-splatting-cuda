/**
 * [文件描述]：应用程序主类实现文件
 * 功能：负责高斯散点可视化应用程序的启动和运行流程
 * 作者：高斯散点CUDA项目团队
 */

#include "core/application.hpp"         // 应用程序主类头文件
#include "core/argument_parser.hpp"     // 命令行参数解析器头文件
#include "visualizer/visualizer.hpp"    // 可视化器组件头文件
#include <print>                        // C++23标准打印功能

namespace gs {
    /**
     * [功能描述]：运行高斯散点可视化应用程序的主入口函数
     * @param params：训练参数的智能指针，包含所有配置信息如数据路径、优化设置等
     * @return 程序退出状态码：0表示成功，-1表示失败
     */
    int Application::run(std::unique_ptr<param::TrainingParameters> params) {
        // 创建可视化器实例，配置窗口属性和渲染选项
        auto viewer = visualizer::Visualizer::create({
            .title = "LichtFeld Studio",                                    // 窗口标题
            .width = 1280,                                                  // 窗口宽度（像素）
            .height = 720,                                                  // 窗口高度（像素）
            .antialiasing = params->optimization.antialiasing,             // 抗锯齿设置（从参数获取）
            .enable_cuda_interop = true                                     // 启用CUDA-OpenGL互操作加速
        });

        // 将训练参数传递给可视化器，用于配置渲染和优化设置
        viewer->setParameters(*params);

        // 数据加载逻辑：优先加载PLY文件，其次加载数据集
        if (!params->ply_path.empty()) {
            // 加载PLY格式的点云文件（预训练的高斯散点模型）
            auto result = viewer->loadPLY(params->ply_path);
            if (!result) {
                // 加载失败时输出错误信息并退出
                std::println(stderr, "Error: {}", result.error());
                return -1;
            }
        } else if (!params->dataset.data_path.empty()) {
            // 加载训练数据集（用于从头开始训练）
            auto result = viewer->loadDataset(params->dataset.data_path);
            if (!result) {
                // 加载失败时输出错误信息并退出
                std::println(stderr, "Error: {}", result.error());
                return -1;
            }
        }

        // 输出抗锯齿配置状态信息
        std::println("抗锯齿: {}", params->optimization.antialiasing ? "启用" : "禁用");

        // 启动可视化器主循环，开始渲染和交互
        viewer->run();

        // 可视化器关闭后的清理信息
        std::println("查看器已关闭.");
        return 0;                                                           // 成功退出
    }
} // namespace gs - 高斯散点项目命名空间结束
