// Copyright (c) 2025 Janusch Patas.

#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include <args.hxx>
#include <expected>
#include <filesystem>
#include <format>
#include <print>
#include <set>
#include <unordered_map>

namespace {

    enum class ParseResult {
        Success,
        Help
    };

    const std::set<std::string> VALID_RENDER_MODES = {"RGB", "D", "ED", "RGB_D", "RGB_ED"};
    const std::set<std::string> VALID_POSE_OPTS = {"none", "direct", "mlp"};
    const std::set<std::string> VALID_STRATEGIES = {"mcmc", "default"};

    void scale_steps_vector(std::vector<size_t>& steps, size_t scaler) {
        std::set<size_t> unique_steps(steps.begin(), steps.end());
        for (const auto& step : steps) {
            for (size_t i = 1; i <= scaler; ++i) {
                unique_steps.insert(step * i);
            }
        }
        steps.assign(unique_steps.begin(), unique_steps.end());
    }

    // Parse log level from string
    gs::core::LogLevel parse_log_level(const std::string& level_str) {
        if (level_str == "trace")
            return gs::core::LogLevel::Trace;
        if (level_str == "debug")
            return gs::core::LogLevel::Debug;
        if (level_str == "info")
            return gs::core::LogLevel::Info;
        if (level_str == "warn" || level_str == "warning")
            return gs::core::LogLevel::Warn;
        if (level_str == "error")
            return gs::core::LogLevel::Error;
        if (level_str == "critical")
            return gs::core::LogLevel::Critical;
        if (level_str == "off")
            return gs::core::LogLevel::Off;
        return gs::core::LogLevel::Info; // Default
    }

    /**
     * [功能描述]：解析命令行参数并配置训练参数对象
     * @param args 命令行参数字符串向量
     * @param params 训练参数对象的引用，将被填充
     * @return 返回解析结果和覆盖函数的元组，如果解析失败则返回错误信息
     */
    std::expected<std::tuple<ParseResult, std::function<void()>>, std::string> parse_arguments(
        const std::vector<std::string>& args,
        gs::param::TrainingParameters& params) {

        try {
            // 创建参数解析器，设置程序描述和用法说明
            ::args::ArgumentParser parser(
                "3D Gaussian Splatting CUDA Implementation\n",
                "Lightning-fast CUDA implementation of 3D Gaussian Splatting algorithm.\n\n"
                "Usage:\n"
                "  Training:  gs_cuda --data-path <path> --output-path <path> [options]\n"
                "  Viewing:   gs_cuda --view <path_to_ply> [options]\n");

            // 定义所有命令行参数
            ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});  // 帮助标志
            ::args::CompletionFlag completion(parser, {"complete"});  // 自动补全标志

            // PLY查看模式参数
            ::args::ValueFlag<std::string> view_ply(parser, "ply_file", "View a PLY file", {'v', "view"});

            // LichtFeldStudio项目参数
            ::args::ValueFlag<std::string> project_name(parser, "proj_path", "LichtFeldStudio project path. Path must end with .ls", {"proj_path"});

            // 训练模式参数
            ::args::ValueFlag<std::string> data_path(parser, "data_path", "Path to training data", {'d', "data-path"});  // 训练数据路径
            ::args::ValueFlag<std::string> output_path(parser, "output_path", "Path to output", {'o', "output-path"});  // 输出路径

            // 可选值参数
            ::args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations", {'i', "iter"});  // 迭代次数
            ::args::ValueFlag<int> max_cap(parser, "max_cap", "Max Gaussians for MCMC", {"max-cap"});  // MCMC最大高斯数
            ::args::ValueFlag<std::string> images_folder(parser, "images", "Images folder name", {"images"});  // 图像文件夹名
            ::args::ValueFlag<int> test_every(parser, "test_every", "Use every Nth image as test", {"test-every"});  // 测试图像间隔
            ::args::ValueFlag<float> steps_scaler(parser, "steps_scaler", "Scale training steps by factor", {"steps-scaler"});  // 步长缩放因子
            ::args::ValueFlag<int> sh_degree_interval(parser, "sh_degree_interval", "SH degree interval", {"sh-degree-interval"});  // 球谐函数度数间隔
            ::args::ValueFlag<int> sh_degree(parser, "sh_degree", "Max SH degree [1-3]", {"sh-degree"});  // 最大球谐函数度数
            ::args::ValueFlag<float> min_opacity(parser, "min_opacity", "Minimum opacity threshold", {"min-opacity"});  // 最小不透明度阈值
            ::args::ValueFlag<std::string> render_mode(parser, "render_mode", "Render mode: RGB, D, ED, RGB_D, RGB_ED", {"render-mode"});  // 渲染模式
            ::args::ValueFlag<std::string> pose_opt(parser, "pose_opt", "Enable pose optimization type: none, direct, mlp", {"pose-opt"});  // 位姿优化类型
            ::args::ValueFlag<std::string> strategy(parser, "strategy", "Optimization strategy: mcmc, default", {"strategy"});  // 优化策略
            ::args::ValueFlag<int> init_num_pts(parser, "init_num_pts", "Number of random initialization points", {"init-num-pts"});  // 随机初始化点数
            ::args::ValueFlag<float> init_extent(parser, "init_extent", "Extent of random initialization", {"init-extent"});  // 随机初始化范围
            ::args::ValueFlagList<std::string> timelapse_images(parser, "timelapse_images", "Image filenames to render timelapse images for", {"timelapse-images"});  // 延时摄影图像文件名列表
            ::args::ValueFlag<int> timelapse_every(parser, "timelapse_every", "Render timelapse image every N iterations (default: 50)", {"timelapse-every"});  // 延时摄影渲染间隔

            // 日志选项
            ::args::ValueFlag<std::string> log_level(parser, "level", "Log level: trace, debug, info, warn, error, critical, off (default: info)", {"log-level"});  // 日志级别
            ::args::ValueFlag<std::string> log_file(parser, "file", "Optional log file path", {"log-file"});  // 可选日志文件路径

            // 可选标志参数
            ::args::Flag use_bilateral_grid(parser, "bilateral_grid", "Enable bilateral grid filtering", {"bilateral-grid"});  // 启用双边网格滤波
            ::args::Flag enable_eval(parser, "eval", "Enable evaluation during training", {"eval"});  // 训练期间启用评估
            ::args::Flag headless(parser, "headless", "Disable visualization during training", {"headless"});  // 训练期间禁用可视化
            ::args::Flag antialiasing(parser, "antialiasing", "Enable antialiasing", {'a', "antialiasing"});  // 启用抗锯齿
            ::args::Flag enable_save_eval_images(parser, "save_eval_images", "Save eval images and depth maps", {"save-eval-images"});  // 保存评估图像和深度图
            ::args::Flag save_depth(parser, "save_depth", "Save depth maps during training", {"save-depth"});  // 训练期间保存深度图
            ::args::Flag skip_intermediate_saving(parser, "skip_intermediate", "Skip saving intermediate results and only save final output", {"skip-intermediate"});  // 跳过保存中间结果
            ::args::Flag random(parser, "random", "Use random initialization instead of SfM", {"random"});  // 使用随机初始化而非SfM
            ::args::Flag gut(parser, "gut", "Enable GUT mode", {"gut"});  // 启用GUT模式

            // 图像缩放因子参数，支持映射到预定义值
            ::args::MapFlag<std::string, int> resize_factor(parser, "resize_factor",
                                                            "resize resolution by this factor. Options: auto, 1, 2, 4, 8 (default: auto)",
                                                            {'r', "resize_factor"},
                                                            // load_image只支持这些缩放值
                                                            std::unordered_map<std::string, int>{
                                                                {"auto", 1},
                                                                {"1", 1},
                                                                {"2", 2},
                                                                {"4", 4},
                                                                {"8", 8}});

            // 解析命令行参数
            try {
                parser.Prog(args.front());  // 设置程序名
                parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));  // 解析除程序名外的所有参数
            } catch (const ::args::Help&) {
                // 用户请求帮助信息
                std::print("{}", parser.Help());
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::Completion& e) {
                // 自动补全请求
                std::print("{}", e.what());
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::ParseError& e) {
                // 参数解析错误
                return std::unexpected(std::format("Parse error: {}\n{}", e.what(), parser.Help()));
            }

            // 根据命令行参数初始化日志记录器
            {
                auto level = gs::core::LogLevel::Info; // 默认日志级别
                std::string log_file_path;

                if (log_level) {
                    level = parse_log_level(::args::get(log_level));  // 解析用户指定的日志级别
                }

                if (log_file) {
                    log_file_path = ::args::get(log_file);  // 获取日志文件路径
                }

                // 使用指定的级别和可选的日志文件初始化日志记录器
                gs::core::Logger::get().init(level, log_file_path);

                // 记录日志记录器初始化信息（不使用gs::前缀）
                LOG_DEBUG("Logger initialized with level: {}", static_cast<int>(level));
                if (!log_file_path.empty()) {
                    LOG_DEBUG("Logging to file: {}", log_file_path);
                }
            }

            // 检查是否明确显示帮助信息
            if (help) {
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            }

            // 无参数 = 查看器模式（空）
            if (args.size() == 1) {
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // 检查PLY查看模式
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

            // 训练模式
            bool has_data_path = data_path && !::args::get(data_path).empty();  // 检查是否有数据路径
            bool has_output_path = output_path && !::args::get(output_path).empty();  // 检查是否有输出路径

            // 如果是无头模式，必须提供数据路径
            if (headless && !has_data_path) {
                return std::unexpected(std::format(
                    "ERROR: Headless mode requires --data-path\n\n{}",
                    parser.Help()));
            }

            // 如果两个路径都提供了，则为训练模式
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
                // 如果只提供了一个路径，则报错
                return std::unexpected(std::format(
                    "ERROR: Training mode requires both --data-path and --output-path\n\n{}",
                    parser.Help()));
            }

            // 验证渲染模式（如果提供）
            if (render_mode) {
                const auto mode = ::args::get(render_mode);
                if (VALID_RENDER_MODES.find(mode) == VALID_RENDER_MODES.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid render mode '{}'. Valid modes are: RGB, D, ED, RGB_D, RGB_ED",
                        mode));
                }
            }
            
            // 验证优化策略（如果提供）
            if (strategy) {
                const auto strat = ::args::get(strategy);
                if (VALID_STRATEGIES.find(strat) == VALID_STRATEGIES.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid optimization strategy '{}'. Valid strategies are: mcmc, default",
                        strat));
                }

                // 与其他稍后作为覆盖设置的参数不同，
                // 策略必须立即设置，以确保在`read_optim_params_from_json()`中正确加载JSON
                params.optimization.strategy = strat;
            }

            // 验证位姿优化选项（如果提供）
            if (pose_opt) {
                const auto opt = ::args::get(pose_opt);
                if (VALID_POSE_OPTS.find(opt) == VALID_POSE_OPTS.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid pose optimization '{}'. Valid options are: none, direct, mlp",
                        opt));
                }
            }

            // 创建lambda函数，在JSON加载后应用命令行覆盖
            auto apply_cmd_overrides = [&params,
                                        // 捕获值，而不是引用
                                        iterations_val = iterations ? std::optional<uint32_t>(::args::get(iterations)) : std::optional<uint32_t>(),
                                        resize_factor_val = resize_factor ? std::optional<int>(::args::get(resize_factor)) : std::optional<int>(1), // 默认值1
                                        max_cap_val = max_cap ? std::optional<int>(::args::get(max_cap)) : std::optional<int>(),
                                        project_name_val = project_name ? std::optional<std::string>(::args::get(project_name)) : std::optional<std::string>(),
                                        images_folder_val = images_folder ? std::optional<std::string>(::args::get(images_folder)) : std::optional<std::string>(),
                                        test_every_val = test_every ? std::optional<int>(::args::get(test_every)) : std::optional<int>(),
                                        steps_scaler_val = steps_scaler ? std::optional<float>(::args::get(steps_scaler)) : std::optional<float>(),
                                        sh_degree_interval_val = sh_degree_interval ? std::optional<int>(::args::get(sh_degree_interval)) : std::optional<int>(),
                                        sh_degree_val = sh_degree ? std::optional<int>(::args::get(sh_degree)) : std::optional<int>(),
                                        min_opacity_val = min_opacity ? std::optional<float>(::args::get(min_opacity)) : std::optional<float>(),
                                        render_mode_val = render_mode ? std::optional<std::string>(::args::get(render_mode)) : std::optional<std::string>(),
                                        init_num_pts_val = init_num_pts ? std::optional<int>(::args::get(init_num_pts)) : std::optional<int>(),
                                        init_extent_val = init_extent ? std::optional<float>(::args::get(init_extent)) : std::optional<float>(),
                                        pose_opt_val = pose_opt ? std::optional<std::string>(::args::get(pose_opt)) : std::optional<std::string>(),
                                        strategy_val = strategy ? std::optional<std::string>(::args::get(strategy)) : std::optional<std::string>(),
                                        timelapse_images_val = timelapse_images ? std::optional<std::vector<std::string>>(::args::get(timelapse_images)) : std::optional<std::vector<std::string>>(),
                                        timelapse_every_val = timelapse_every ? std::optional<int>(::args::get(timelapse_every)) : std::optional<int>(),
                                        // 捕获标志状态
                                        use_bilateral_grid_flag = bool(use_bilateral_grid),
                                        enable_eval_flag = bool(enable_eval),
                                        headless_flag = bool(headless),
                                        antialiasing_flag = bool(antialiasing),
                                        enable_save_eval_images_flag = bool(enable_save_eval_images),
                                        skip_intermediate_saving_flag = bool(skip_intermediate_saving),
                                        random_flag = bool(random),
                                        gut_flag = bool(gut)]() {
                auto& opt = params.optimization;  // 优化参数引用
                auto& ds = params.dataset;        // 数据集参数引用

                // 简单的lambda函数，如果标志/值存在则应用
                auto setVal = [](const auto& flag, auto& target) {
                    if (flag)
                        target = *flag;
                };

                auto setFlag = [](bool flag, auto& target) {
                    if (flag)
                        target = true;
                };

                // 应用所有覆盖
                setVal(iterations_val, opt.iterations);                    // 迭代次数
                setVal(resize_factor_val, ds.resize_factor);               // 缩放因子
                setVal(max_cap_val, opt.max_cap);                         // 最大高斯数
                setVal(project_name_val, ds.project_path);                 // 项目路径
                setVal(images_folder_val, ds.images);                      // 图像文件夹
                setVal(test_every_val, ds.test_every);                     // 测试间隔
                setVal(steps_scaler_val, opt.steps_scaler);                // 步长缩放
                setVal(sh_degree_interval_val, opt.sh_degree_interval);   // 球谐函数度数间隔
                setVal(sh_degree_val, opt.sh_degree);                      // 球谐函数度数
                setVal(min_opacity_val, opt.min_opacity);                  // 最小不透明度
                setVal(render_mode_val, opt.render_mode);                  // 渲染模式
                setVal(init_num_pts_val, opt.init_num_pts);               // 初始化点数
                setVal(init_extent_val, opt.init_extent);                 // 初始化范围
                setVal(pose_opt_val, opt.pose_optimization);              // 位姿优化
                setVal(strategy_val, opt.strategy);                        // 优化策略
                setVal(timelapse_images_val, ds.timelapse_images);        // 延时摄影图像
                setVal(timelapse_every_val, ds.timelapse_every);          // 延时摄影间隔

                setFlag(use_bilateral_grid_flag, opt.use_bilateral_grid);           // 双边网格滤波
                setFlag(enable_eval_flag, opt.enable_eval);                         // 启用评估
                setFlag(headless_flag, opt.headless);                               // 无头模式
                setFlag(antialiasing_flag, opt.antialiasing);                       // 抗锯齿
                setFlag(enable_save_eval_images_flag, opt.enable_save_eval_images); // 保存评估图像
                setFlag(skip_intermediate_saving_flag, opt.skip_intermediate_saving); // 跳过中间保存
                setFlag(random_flag, opt.random);                                   // 随机初始化
                setFlag(gut_flag, opt.gut);                                        // GUT模式
            };

            // 返回成功结果和覆盖函数
            return std::make_tuple(ParseResult::Success, apply_cmd_overrides);

        } catch (const std::exception& e) {
            // 捕获参数解析过程中的意外错误
            return std::unexpected(std::format("Unexpected error during argument parsing: {}", e.what()));
        }
    }

    void apply_step_scaling(gs::param::TrainingParameters& params) {
        auto& opt = params.optimization;
        const float scaler = opt.steps_scaler;

        if (scaler > 0) {
            LOG_INFO("Scaling training steps by factor: {}", scaler);

            opt.iterations *= scaler;
            opt.start_refine *= scaler;
            opt.reset_every *= scaler;
            opt.stop_refine *= scaler;
            opt.refine_every *= scaler;
            opt.sh_degree_interval *= scaler;

            scale_steps_vector(opt.eval_steps, scaler);
            scale_steps_vector(opt.save_steps, scaler);
        }
    }

    std::vector<std::string> convert_args(int argc, const char* const argv[]) {
        return std::vector<std::string>(argv, argv + argc);
    }

} // anonymous namespace

// 公共接口
/**
 * [功能描述]：解析命令行参数和配置文件，创建训练参数对象
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return 返回训练参数对象的智能指针，如果解析失败则返回错误信息
 */
std::expected<std::unique_ptr<gs::param::TrainingParameters>, std::string>
gs::args::parse_args_and_params(int argc, const char* const argv[]) {

    // 创建训练参数对象的智能指针
    auto params = std::make_unique<gs::param::TrainingParameters>();

    // 解析命令行参数
    // 将argc和argv转换为字符串向量，然后解析到params对象中
    auto parse_result = parse_arguments(convert_args(argc, argv), *params);
    if (!parse_result) {
        // 如果解析失败，返回错误信息
        return std::unexpected(parse_result.error());
    }

    // 解构解析结果，获取解析状态和覆盖函数
    auto [result, apply_overrides] = *parse_result;

    // 处理帮助选项
    if (result == ParseResult::Help) {
        std::exit(0);  // 显示帮助信息后退出程序
    }

    // 训练模式 - 首先加载JSON配置文件
    if (!params->dataset.data_path.empty()) {
        // 从JSON文件读取优化参数，策略由params->optimization.strategy指定
        auto opt_params_result = gs::param::read_optim_params_from_json(params->optimization.strategy);
        if (!opt_params_result) {
            // 如果加载失败，返回格式化的错误信息
            return std::unexpected(std::format("Failed to load optimization parameters: {}",
                                               opt_params_result.error()));
        }
        // 将JSON中的优化参数覆盖到params对象中
        params->optimization = *opt_params_result;
    }

    // 应用命令行覆盖
    // 如果存在覆盖函数，则执行它来覆盖默认值
    // 首先加载 json 文件中的参数，然后用命令行的参数对其进行覆盖
    if (apply_overrides) {
        apply_overrides();
    }

    // 应用步长缩放
    // 根据配置调整各种步长参数
    apply_step_scaling(*params);

    // 返回配置完成的训练参数对象
    return params;
}