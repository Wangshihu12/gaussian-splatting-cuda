#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <print>

/**
 * [功能描述]：高斯溅射CUDA应用程序的主入口函数，负责初始化CUDA内存分配器、解析命令行参数、启动应用程序。
 * @param argc [参数说明]：命令行参数的数量。
 * @param argv [参数说明]：命令行参数字符串数组。
 * @return [返回值说明]：程序退出码，0表示成功，-1表示失败。
 */
int main(int argc, char* argv[]) {
//----------------------------------------------------------------------
// 0. 设置CUDA缓存分配器以避免内存碎片问题
// 这避免了在密集化步骤后重复调用emptyCache()的需要。
// 我们在这里手动调用适当的函数，而不是设置环境变量，
// 希望这也能在Windows上工作。使用setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", 1);
// 在Linux上可以工作，但在Windows上不行，所以我们使用C++ API。
// 如果将来出现问题，我们总是可以恢复到在每个密集化步骤后调用emptyCache()的旧方法。
//----------------------------------------------------------------------
#ifndef _WIN32
    // Windows不支持CUDACachingAllocator的expandable_segments功能
    c10::cuda::CUDACachingAllocator::setAllocatorSettings("expandable_segments:True");
#endif

    // 解析命令行参数（这会根据--log-level标志自动初始化日志记录器）
    auto params_result = gs::args::parse_args_and_params(argc, argv);
    if (!params_result) {
        // 日志记录器已经初始化，所以我们可以用它来记录错误
        LOG_ERROR("Failed to parse arguments: {}", params_result.error());
        std::println(stderr, "Error: {}", params_result.error());
        return -1;  // 返回错误退出码
    }

    // 日志记录器现在可以使用了
    LOG_INFO("========================================");
    LOG_INFO("LichtFeld Studio");  // 显示应用程序名称
    LOG_INFO("========================================");

    // 移动参数结果到params变量中
    auto params = std::move(*params_result);

    // 创建应用程序实例
    gs::Application app;
    // 运行应用程序并返回结果
    return app.run(std::move(params));
}
