/**
 * [文件描述]：命令行参数解析器实现文件
 * 功能：解析高斯散点CUDA应用程序的命令行参数，支持训练模式和查看模式
 * 版权：Copyright (c) 2025 Janusch Patas.
 */

#include "core/argument_parser.hpp"     // 参数解析器头文件
#include "core/parameters.hpp"          // 参数结构定义
#include <args.hxx>                     // 第三方命令行参数解析库
#include <expected>                     // C++23期望值类型
#include <filesystem>                   // 文件系统操作
#include <format>                       // C++20格式化库
#include <print>                        // C++23打印功能
#include <set>                          // 标准集合容器

namespace {  // 匿名命名空间 - 限制内部实现的作用域

    /**
     * [枚举描述]：参数解析结果状态
     */
    enum class ParseResult {
        Success,    // 解析成功
        Help        // 显示帮助信息
    };

    // 有效的渲染模式常量集合
    const std::set<std::string> VALID_RENDER_MODES = {
        "RGB",      // RGB颜色渲染
        "D",        // 深度渲染
        "ED",       // 期望深度渲染
        "RGB_D",    // RGB+深度渲染
        "RGB_ED"    // RGB+期望深度渲染
    };
    
    // 有效的优化策略常量集合
    const std::set<std::string> VALID_STRATEGIES = {
        "mcmc",     // 马尔可夫链蒙特卡洛策略
        "default"   // 默认优化策略
    };

    /**
     * [功能描述]：对训练步骤向量进行缩放处理
     * @param steps：要缩放的步骤向量（引用传递，会被修改）
     * @param scaler：缩放因子，将为每个原始步骤生成1到scaler倍的步骤
     */
    void scale_steps_vector(std::vector<size_t>& steps, size_t scaler) {
        std::set<size_t> unique_steps(steps.begin(), steps.end());  // 使用集合去重
        for (const auto& step : steps) {
            // 为每个步骤生成1倍到scaler倍的所有倍数
            for (size_t i = 1; i <= scaler; ++i) {
                unique_steps.insert(step * i);
            }
        }
        // 将去重后的结果重新赋值给原向量
        steps.assign(unique_steps.begin(), unique_steps.end());
    }

    /**
     * [功能描述]：解析命令行参数并验证其有效性
     * @param args：命令行参数字符串向量
     * @param params：训练参数对象引用，用于存储解析结果
     * @return 期望值类型，成功时返回解析结果和参数覆盖函数，失败时返回错误信息
     */
    std::expected<std::tuple<ParseResult, std::function<void()>>, std::string> parse_arguments(
        const std::vector<std::string>& args,
        gs::param::TrainingParameters& params) {

        try {
            // 创建参数解析器，设置程序描述和使用说明
            ::args::ArgumentParser parser(
                "3D Gaussian Splatting CUDA Implementation\n",
                "Lightning-fast CUDA implementation of 3D Gaussian Splatting algorithm.\n\n"
                "Usage:\n"
                "  Training:  gs_cuda --data-path <path> --output-path <path> [options]\n"
                "  Viewing:   gs_cuda --view <path_to_ply> [options]\n");

            // =============================================================================
            // 定义所有命令行参数
            // =============================================================================
            
            // 帮助和完成相关参数
            ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
            ::args::CompletionFlag completion(parser, {"complete"});

            // PLY文件查看模式参数
            ::args::ValueFlag<std::string> view_ply(parser, "ply_file", "View a PLY file", {'v', "view"});

            // 训练模式必需参数
            ::args::ValueFlag<std::string> data_path(parser, "data_path", "Path to training data", {'d', "data-path"});
            ::args::ValueFlag<std::string> output_path(parser, "output_path", "Path to output", {'o', "output-path"});

            // 可选数值参数
            ::args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations", {'i', "iter"});
            ::args::ValueFlag<int> max_cap(parser, "max_cap", "Max Gaussians for MCMC", {"max-cap"});
            ::args::ValueFlag<std::string> images_folder(parser, "images", "Images folder name", {"images"});
            ::args::ValueFlag<int> test_every(parser, "test_every", "Use every Nth image as test", {"test-every"});
            ::args::ValueFlag<float> steps_scaler(parser, "steps_scaler", "Scale training steps by factor", {"steps-scaler"});
            ::args::ValueFlag<int> sh_degree_interval(parser, "sh_degree_interval", "SH degree interval", {"sh-degree-interval"});
            ::args::ValueFlag<int> sh_degree(parser, "sh_degree", "Max SH degree [1-3]", {"sh-degree"});
            ::args::ValueFlag<float> min_opacity(parser, "min_opacity", "Minimum opacity threshold", {"min-opacity"});
            ::args::ValueFlag<std::string> render_mode(parser, "render_mode", "Render mode: RGB, D, ED, RGB_D, RGB_ED", {"render-mode"});
            ::args::ValueFlag<std::string> strategy(parser, "strategy", "Optimization strategy: mcmc, default", {"strategy"});
            ::args::ValueFlag<int> init_num_pts(parser, "init_num_pts", "Number of random initialization points", {"init-num-pts"});
            ::args::ValueFlag<float> init_extent(parser, "init_extent", "Extent of random initialization", {"init-extent"});

            // 可选布尔标志参数
            ::args::Flag use_bilateral_grid(parser, "bilateral_grid", "Enable bilateral grid filtering", {"bilateral-grid"});
            ::args::Flag enable_eval(parser, "eval", "Enable evaluation during training", {"eval"});
            ::args::Flag headless(parser, "headless", "Disable visualization during training", {"headless"});
            ::args::Flag antialiasing(parser, "antialiasing", "Enable antialiasing", {'a', "antialiasing"});
            ::args::Flag enable_save_eval_images(parser, "save_eval_images", "Save eval images and depth maps", {"save-eval-images"});
            ::args::Flag save_depth(parser, "save_depth", "Save depth maps during training", {"save-depth"});
            ::args::Flag skip_intermediate_saving(parser, "skip_intermediate", "Skip saving intermediate results and only save final output", {"skip-intermediate"});
            ::args::Flag random(parser, "random", "Use random initialization instead of SfM", {"random"});

            // 分辨率缩放参数（映射类型）
            ::args::MapFlag<std::string, int> resize_factor(parser, "resize_factor",
                                                            "resize resolution by this factor. Options: auto, 1, 2, 4, 8 (default: auto)",
                                                            {'r', "resize_factor"},
                                                            // load_image函数仅支持这些缩放比例
                                                            std::unordered_map<std::string, int>{
                                                                {"auto", 1},
                                                                {"1", 1},
                                                                {"2", 2},
                                                                {"4", 4},
                                                                {"8", 8}});

            // =============================================================================
            // 执行参数解析
            // =============================================================================
            try {
                parser.Prog(args.front());  // 设置程序名称
                parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));  // 解析参数
            } catch (const ::args::Help&) {
                // 用户请求帮助
                std::print("{}", parser.Help());
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::Completion& e) {
                // 命令行补全请求
                std::print("{}", e.what());
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::ParseError& e) {
                // 参数解析错误
                return std::unexpected(std::format("Parse error: {}\n{}", e.what(), parser.Help()));
            }

            // 检查是否显式请求帮助
            if (help) {
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            }

            // =============================================================================
            // 参数模式判断和验证
            // =============================================================================
            
            // 无参数 = 查看器模式（空参数）
            if (args.size() == 1) {
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // 检查PLY文件查看模式
            if (view_ply) {
                const auto ply_path = ::args::get(view_ply);
                if (!ply_path.empty()) {
                    params.ply_path = ply_path;

                    // 检查PLY文件是否存在
                    if (!std::filesystem::exists(params.ply_path)) {
                        return std::unexpected(std::format("PLY file does not exist: {}", params.ply_path.string()));
                    }
                }

                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // 训练模式路径检查
            bool has_data_path = data_path && !::args::get(data_path).empty();
            bool has_output_path = output_path && !::args::get(output_path).empty();

            // 无头模式必须提供数据路径
            if (headless && !has_data_path) {
                return std::unexpected(std::format(
                    "ERROR: Headless mode requires --data-path\n\n{}",
                    parser.Help()));
            }

            // 训练模式：两个路径都提供时为训练模式
            if (has_data_path && has_output_path) {
                params.dataset.data_path = ::args::get(data_path);
                params.dataset.output_path = ::args::get(output_path);

                // 创建输出目录
                std::error_code ec;
                std::filesystem::create_directories(params.dataset.output_path, ec);
                if (ec) {
                    return std::unexpected(std::format(
                        "Failed to create output directory '{}': {}",
                        params.dataset.output_path.string(), ec.message()));
                }
            } else if (has_data_path != has_output_path) {
                // 只提供一个路径是错误的
                return std::unexpected(std::format(
                    "ERROR: Training mode requires both --data-path and --output-path\n\n{}",
                    parser.Help()));
            }

            // =============================================================================
            // 参数值验证
            // =============================================================================
            
            // 验证渲染模式
            if (render_mode) {
                const auto mode = ::args::get(render_mode);
                if (VALID_RENDER_MODES.find(mode) == VALID_RENDER_MODES.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid render mode '{}'. Valid modes are: RGB, D, ED, RGB_D, RGB_ED",
                        mode));
                }
            }
            
            // 验证优化策略
            if (strategy) {
                const auto strat = ::args::get(strategy);
                if (VALID_STRATEGIES.find(strat) == VALID_STRATEGIES.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid optimization strategy '{}'. Valid strategies are: mcmc, default",
                        strat));
                }

                // 策略参数必须立即设置，以确保JSON加载时的正确性
                // 与其他参数不同，策略不能作为覆盖参数延迟设置
                params.optimization.strategy = strat;
            }

            // =============================================================================
            // 创建参数覆盖Lambda函数
            // =============================================================================
            
            // 创建lambda函数以在JSON加载后应用命令行覆盖
            auto apply_cmd_overrides = [&params,
                                        // 捕获参数值（而非引用）以避免悬挂引用
                                        iterations_val = iterations ? std::optional<uint32_t>(::args::get(iterations)) : std::optional<uint32_t>(),
                                        resize_factor_val = resize_factor ? std::optional<int>(::args::get(resize_factor)) : std::optional<int>(1), // 默认值1
                                        max_cap_val = max_cap ? std::optional<int>(::args::get(max_cap)) : std::optional<int>(),
                                        images_folder_val = images_folder ? std::optional<std::string>(::args::get(images_folder)) : std::optional<std::string>(),
                                        test_every_val = test_every ? std::optional<int>(::args::get(test_every)) : std::optional<int>(),
                                        steps_scaler_val = steps_scaler ? std::optional<float>(::args::get(steps_scaler)) : std::optional<float>(),
                                        sh_degree_interval_val = sh_degree_interval ? std::optional<int>(::args::get(sh_degree_interval)) : std::optional<int>(),
                                        sh_degree_val = sh_degree ? std::optional<int>(::args::get(sh_degree)) : std::optional<int>(),
                                        min_opacity_val = min_opacity ? std::optional<float>(::args::get(min_opacity)) : std::optional<float>(),
                                        render_mode_val = render_mode ? std::optional<std::string>(::args::get(render_mode)) : std::optional<std::string>(),
                                        init_num_pts_val = init_num_pts ? std::optional<int>(::args::get(init_num_pts)) : std::optional<int>(),
                                        init_extent_val = init_extent ? std::optional<float>(::args::get(init_extent)) : std::optional<float>(),
                                        // 捕获标志状态
                                        use_bilateral_grid_flag = bool(use_bilateral_grid),
                                        enable_eval_flag = bool(enable_eval),
                                        headless_flag = bool(headless),
                                        antialiasing_flag = bool(antialiasing),
                                        enable_save_eval_images_flag = bool(enable_save_eval_images),
                                        skip_intermediate_saving_flag = bool(skip_intermediate_saving),
                                        random_flag = bool(random)]() {
                // 获取参数结构的引用
                auto& opt = params.optimization;
                auto& ds = params.dataset;

                // 辅助lambda函数：设置数值类型参数
                auto setVal = [](const auto& flag, auto& target) {
                    if (flag)  // 如果命令行提供了该参数
                        target = *flag;  // 设置目标值
                };

                // 辅助lambda函数：设置布尔标志参数
                auto setFlag = [](bool flag, auto& target) {
                    if (flag)  // 如果命令行设置了该标志
                        target = true;  // 启用该功能
                };

                // 应用所有参数覆盖
                setVal(iterations_val, opt.iterations);                        // 迭代次数
                setVal(resize_factor_val, ds.resize_factor);                   // 分辨率缩放因子
                setVal(max_cap_val, opt.max_cap);                             // MCMC最大高斯数量
                setVal(images_folder_val, ds.images);                         // 图像文件夹名称
                setVal(test_every_val, ds.test_every);                        // 测试图像间隔
                setVal(steps_scaler_val, opt.steps_scaler);                   // 训练步骤缩放因子
                setVal(sh_degree_interval_val, opt.sh_degree_interval);       // 球谐函数度数间隔
                setVal(sh_degree_val, opt.sh_degree);                         // 最大球谐函数度数
                setVal(min_opacity_val, opt.min_opacity);                     // 最小不透明度阈值
                setVal(render_mode_val, opt.render_mode);                     // 渲染模式
                setVal(init_num_pts_val, opt.init_num_pts);                   // 随机初始化点数
                setVal(init_extent_val, opt.init_extent);                     // 随机初始化范围

                setFlag(use_bilateral_grid_flag, opt.use_bilateral_grid);     // 双边网格滤波
                setFlag(enable_eval_flag, opt.enable_eval);                   // 训练期间评估
                setFlag(headless_flag, opt.headless);                         // 无头模式
                setFlag(antialiasing_flag, opt.antialiasing);                 // 抗锯齿
                setFlag(enable_save_eval_images_flag, opt.enable_save_eval_images);  // 保存评估图像
                setFlag(skip_intermediate_saving_flag, opt.skip_intermediate_saving);  // 跳过中间保存
                setFlag(random_flag, opt.random);                             // 随机初始化
            };

            return std::make_tuple(ParseResult::Success, apply_cmd_overrides);

        } catch (const std::exception& e) {
            // 捕获其他未预期的异常
            return std::unexpected(std::format("Unexpected error during argument parsing: {}", e.what()));
        }
    }

    /**
     * [功能描述]：应用训练步骤缩放到各种训练相关的参数
     * @param params：训练参数对象引用，包含需要缩放的各种步骤参数
     */
    void apply_step_scaling(gs::param::TrainingParameters& params) {
        auto& opt = params.optimization;
        const float scaler = opt.steps_scaler;  // 获取缩放因子

        if (scaler > 0) {  // 仅在缩放因子有效时应用缩放
            std::println("Scaling training steps by factor: {}", scaler);

            // 缩放主要训练参数
            opt.iterations *= scaler;           // 总迭代次数
            opt.start_refine *= scaler;         // 开始细化的步骤
            opt.stop_refine *= scaler;          // 停止细化的步骤
            opt.refine_every *= scaler;         // 细化间隔
            opt.sh_degree_interval *= scaler;   // 球谐函数度数更新间隔

            // 缩放步骤向量
            scale_steps_vector(opt.eval_steps, scaler);  // 评估步骤
            scale_steps_vector(opt.save_steps, scaler);  // 保存步骤
        }
    }

    /**
     * [功能描述]：将C风格命令行参数转换为C++字符串向量
     * @param argc：参数个数
     * @param argv：参数数组指针
     * @return 包含所有参数的字符串向量
     */
    std::vector<std::string> convert_args(int argc, const char* const argv[]) {
        return std::vector<std::string>(argv, argv + argc);
    }

} // anonymous namespace - 匿名命名空间结束

// =============================================================================
// 公共接口实现
// =============================================================================

/**
 * [功能描述]：解析命令行参数并创建训练参数对象的公共接口函数
 * @param argc：命令行参数个数
 * @param argv：命令行参数数组
 * @return 期望值类型，成功时返回训练参数的智能指针，失败时返回错误信息
 */
std::expected<std::unique_ptr<gs::param::TrainingParameters>, std::string>
gs::args::parse_args_and_params(int argc, const char* const argv[]) {

    // 创建训练参数对象
    auto params = std::make_unique<gs::param::TrainingParameters>();

    // 解析命令行参数
    auto parse_result = parse_arguments(convert_args(argc, argv), *params);
    if (!parse_result) {
        return std::unexpected(parse_result.error());  // 返回解析错误
    }

    auto [result, apply_overrides] = *parse_result;  // 解构返回值

    // 处理帮助请求
    if (result == ParseResult::Help) {
        std::exit(0);  // 显示帮助后退出程序
    }

    // 训练模式：首先加载JSON配置
    if (!params->dataset.data_path.empty()) {
        auto opt_params_result = gs::param::read_optim_params_from_json(params->optimization.strategy);
        if (!opt_params_result) {
            return std::unexpected(std::format("Failed to load optimization parameters: {}",
                                            opt_params_result.error()));
        }
        params->optimization = *opt_params_result;  // 应用JSON配置
    }

    // 应用命令行参数覆盖（优先级高于JSON配置）
    if (apply_overrides) {
        apply_overrides();
    }

    // 应用训练步骤缩放
    apply_step_scaling(*params);

    return params;  // 返回配置完成的参数对象
}