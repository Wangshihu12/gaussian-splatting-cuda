/**
 * [文件描述]：高斯散点CUDA应用程序主入口文件
 * 功能：程序启动点，负责初始化CUDA环境、解析命令行参数、启动训练或查看器模式
 * 作者：高斯散点CUDA项目团队
 */

#include "core/application.hpp"         // 应用程序主类
#include "core/argument_parser.hpp"     // 命令行参数解析器
#include "core/training_setup.hpp"      // 训练环境设置
#include <c10/cuda/CUDAAllocatorConfig.h>  // CUDA内存分配器配置
#include <print>                        // C++23标准打印功能

/**
 * [功能描述]：程序主入口函数
 * @param argc：命令行参数个数
 * @param argv：命令行参数数组
 * @return 程序退出状态码：0表示成功，-1表示失败
 */
int main(int argc, char* argv[]) {
    
    // =============================================================================
    // CUDA 内存分配器优化配置
    // =============================================================================
    /*
    * 设置CUDA缓存分配器参数以避免内存碎片化问题
    * 这样可以避免在密集化步骤后重复调用emptyCache()函数
    * 我们在这里手动调用相应的函数，而不是设置环境变量，
    * 希望这样也能在Windows上正常工作
    * 
    * 通过环境变量设置的方式：
    * setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", 1);
    * 这种方式在Linux上可以工作，但在Windows上不行，所以我们使用C++ API
    * 
    * 如果将来这个方法失效，我们总是可以回退到旧方法：
    * 在每个密集化步骤后调用emptyCache()
    */
#ifndef _WIN32
    // Windows系统不支持CUDACachingAllocator的expandable_segments功能
    c10::cuda::CUDACachingAllocator::setAllocatorSettings("expandable_segments:True");
#endif

    // =============================================================================
    // 命令行参数解析
    // =============================================================================
    
    // 解析命令行参数并创建训练参数对象
    auto params_result = gs::args::parse_args_and_params(argc, argv);
    if (!params_result) {
        // 参数解析失败，输出错误信息并退出
        std::println(stderr, "错误: {}", params_result.error());
        return -1;
    }
    
    // 获取解析成功的参数对象（使用移动语义避免拷贝）
    auto params = std::move(*params_result);
    
    // =============================================================================
    // 运行模式判断和执行
    // =============================================================================
    
    // 检查是否为无头模式（无图形界面的训练模式）
    if (params->optimization.headless) {
        
        // 无头模式必须提供数据路径
        if (params->dataset.data_path.empty()) {
            std::println(stderr, "错误: Headless 模式需要 --data-path");
            return -1;
        }

        std::println("开始无图形界面训练...");

        // 保存训练配置到JSON文件，便于后续分析和复现
        auto save_result = gs::param::save_training_parameters_to_json(*params, params->dataset.output_path);
        if (!save_result) {
            std::println(stderr, "错误: 保存配置时出错: {}", save_result.error());
            return -1;
        }

        // 设置训练环境（初始化数据加载器、模型、优化器等）
        auto setup_result = gs::setupTraining(*params);
        if (!setup_result) {
            std::println(stderr, "错误: {}", setup_result.error());
            return -1;
        }

        // 执行训练过程
        auto train_result = setup_result->trainer->train();
        if (!train_result) {
            std::println(stderr, "训练错误: {}", train_result.error());
            return -1;
        }

        // 训练完成，成功退出
        return 0;
    }

    // =============================================================================
    // GUI 查看器模式
    // =============================================================================
    
    // 启动带图形界面的查看器应用程序
    std::println("开始查看器模式...");
    
    // 创建应用程序实例
    gs::Application app;
    
    // 运行应用程序并返回退出状态码
    return app.run(std::move(params));
}
