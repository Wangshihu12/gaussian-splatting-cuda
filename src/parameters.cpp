// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#include "core/parameters.hpp"
#include <chrono>
#include <ctime>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <print>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace gs {
    namespace param {
        namespace {
            /**
             * @brief Get the path to a configuration file
             * @param filename Name of the configuration file
             * @return std::filesystem::path Full path to the configuration file
             */
            std::filesystem::path get_config_path(const std::string& filename) {
#ifdef _WIN32
                char executablePathWindows[MAX_PATH];
                GetModuleFileNameA(nullptr, executablePathWindows, MAX_PATH);
                std::filesystem::path executablePath = std::filesystem::path(executablePathWindows);
                std::filesystem::path searchDir = executablePath.parent_path();
                while (!searchDir.empty() && !std::filesystem::exists(searchDir / "parameter" / filename)) {
                    searchDir = searchDir.parent_path();
                }

                if (searchDir.empty()) {
                    throw std::runtime_error("could not find " + (std::filesystem::path("parameter") / filename).string());
                }
#else
                std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
                std::filesystem::path searchDir = executablePath.parent_path().parent_path();
#endif
                return searchDir / "parameter" / filename;
            }

            /**
             * @brief Read and parse a JSON configuration file
             * @param path Path to the JSON file
             * @return Expected JSON object or error message
             */
            std::expected<nlohmann::json, std::string> read_json_file(const std::filesystem::path& path) {
                if (!std::filesystem::exists(path)) {
                    return std::unexpected(std::format("Configuration file does not exist: {}", path.string()));
                }

                std::ifstream file(path);
                if (!file.is_open()) {
                    return std::unexpected(std::format("Could not open configuration file: {}", path.string()));
                }

                try {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    return nlohmann::json::parse(buffer.str());
                } catch (const nlohmann::json::parse_error& e) {
                    return std::unexpected(std::format("JSON parsing error in {}: {}", path.string(), e.what()));
                }
            }

            /**
             * @brief Verify optimization parameters between JSON and struct defaults
             *
             * This function performs a comprehensive verification of optimization parameters:
             * 1. Checks if all struct parameters exist in JSON
             * 2. Verifies values match between JSON and struct defaults
             * 3. Identifies unknown parameters in JSON (with warnings)
             * 4. Provides detailed error reporting
             *
             * @param defaults The default parameters from the struct
             * @param json The JSON configuration to verify
             * @param strict If true, unknown parameters are treated as errors
             * @return bool True if verification passed, false otherwise
             */
            bool verify_optimization_parameters(const OptimizationParameters& defaults,
                                                const nlohmann::json& json,
                                                bool strict = false) {
                bool all_match = true;
                std::vector<std::string> missing_in_json;
                std::vector<std::string> unknown_params;
                std::vector<std::string> mismatched_values;

                // Define all expected parameters and their types
                struct ParamInfo {
                    std::string name;
                    std::variant<int, size_t, float, int64_t, std::string, bool> value;
                    std::string description; // Added for better documentation
                };

                const std::vector<ParamInfo> expected_params = {
                    {"iterations", defaults.iterations, "Total number of training iterations"},
                    {"means_lr", defaults.means_lr, "Initial learning rate for position updates"},
                    {"shs_lr", defaults.shs_lr, "Learning rate for spherical harmonics updates"},
                    {"opacity_lr", defaults.opacity_lr, "Learning rate for opacity updates"},
                    {"scaling_lr", defaults.scaling_lr, "Learning rate for scaling updates"},
                    {"rotation_lr", defaults.rotation_lr, "Learning rate for rotation updates"},
                    {"lambda_dssim", defaults.lambda_dssim, "DSSIM loss weight"},
                    {"min_opacity", defaults.min_opacity, "Minimum opacity threshold"},
                    {"refine_every", defaults.refine_every, "Interval between densification steps"},
                    {"start_refine", defaults.start_refine, "Starting iteration for densification"},
                    {"stop_refine", defaults.stop_refine, "Ending iteration for densification"},
                    {"grad_threshold", defaults.grad_threshold, "Gradient threshold for densification"},
                    {"opacity_reg", defaults.opacity_reg, "Opacity L1 regularization weight"},
                    {"scale_reg", defaults.scale_reg, "Scale L1 regularization weight"},
                    {"init_opacity", defaults.init_opacity, "Initial opacity value for new Gaussians"},
                    {"init_scaling", defaults.init_scaling, "Initial scaling value for new Gaussians"},
                    {"sh_degree", defaults.sh_degree, "Spherical harmonics degree"},
                    {"max_cap", defaults.max_cap, "Maximum number of Gaussians for MCMC strategy"},
                    {"render_mode", defaults.render_mode, "Render mode: RGB, D, ED, RGB_D, RGB_ED"},
                    {"strategy", defaults.strategy, "Optimization strategy: mcmc, default"},
                    {"enable_eval", defaults.enable_eval, "Enable evaluation during training"},
                    {"enable_save_eval_images", defaults.enable_save_eval_images, "Save images during evaluation"},
                    {"skip_intermediate", defaults.skip_intermediate_saving, "Skip saving intermediate results and only save final output"},
                    {"use_bilateral_grid", defaults.use_bilateral_grid, "Enable bilateral grid for appearance modeling"},
                    {"bilateral_grid_X", defaults.bilateral_grid_X, "Bilateral grid X dimension"},
                    {"bilateral_grid_Y", defaults.bilateral_grid_Y, "Bilateral grid Y dimension"},
                    {"bilateral_grid_W", defaults.bilateral_grid_W, "Bilateral grid W dimension"},
                    {"bilateral_grid_lr", defaults.bilateral_grid_lr, "Learning rate for bilateral grid"},
                    {"tv_loss_weight", defaults.tv_loss_weight, "Weight for total variation loss"},
                    {"prune_opacity", defaults.prune_opacity, "Opacity pruning threshold"},
                    {"grow_scale3d", defaults.grow_scale3d, "3D scale threshold for duplication"},
                    {"grow_scale2d", defaults.grow_scale2d, "2D scale threshold for splitting"},
                    {"prune_scale3d", defaults.prune_scale3d, "3D scale threshold for pruning"},
                    {"prune_scale2d", defaults.prune_scale2d, "2D scale threshold for pruning"},
                    {"stop_refine_scale2d", defaults.stop_refine_scale2d, "Stop refining Gaussians based on 2D scale at this iteration"},
                    {"reset_every", defaults.reset_every, "Reset opacity every this many iterations"},
                    {"pause_refine_after_reset", defaults.pause_refine_after_reset, "Pause refinement after reset for N iterations"},
                    {"revised_opacity", defaults.revised_opacity, "Use revised opacity heuristic"},
                    {"steps_scaler", defaults.steps_scaler, "Scales the training steps and values"},
                    {"antialiasing", defaults.antialiasing, "Enables antialiasing"},
                    {"sh_degree_interval", defaults.sh_degree_interval, "Interval for increasing SH degree"},
                    {"random", defaults.random, "Use random initialization instead of SfM"},
                    {"init_num_pts", defaults.init_num_pts, "Number of random initialization points"},
                    {"init_extent", defaults.init_extent, "Extent of random initialization"}};

                // Check all expected parameters
                for (const auto& param : expected_params) {
                    if (!json.contains(param.name)) {
                        // Skip eval_steps and save_steps as they are handled separately
                        if (param.name != "eval_steps" && param.name != "save_steps") {
                            missing_in_json.push_back(param.name);
                            all_match = false;
                        }
                        continue;
                    }

                    // Compare values based on type
                    std::visit([&](const auto& default_val) {
                        using T = std::decay_t<decltype(default_val)>;
                        try {
                            if (json[param.name].get<T>() != default_val) {
                                mismatched_values.push_back(param.name);
                                all_match = false;
                            }
                        } catch (...) {
                            // Type mismatch
                            mismatched_values.push_back(param.name + " (type mismatch)");
                            all_match = false;
                        }
                    },
                               param.value);
                }

                // Check for any unknown parameters in JSON
                for (const auto& [key, value] : json.items()) {
                    bool found = false;
                    for (const auto& param : expected_params) {
                        if (key == param.name) {
                            found = true;
                            break;
                        }
                    }
                    if (key == "eval_steps" || key == "save_steps") {
                        found = true;
                    }
                    if (!found) {
                        unknown_params.push_back(key);
                        if (strict) {
                            all_match = false;
                        }
                    }
                }

                // Print report
                if (!all_match || !unknown_params.empty()) {
                    std::println(stderr, "\n参数验证报告:");

                    if (!mismatched_values.empty()) {
                        std::println(stderr, "\n不匹配的值:");
                        for (const auto& param : mismatched_values) {
                            std::print(stderr, "  - {}: JSON={}", param, json[param].dump());
                            // Find and print the default value and description
                            for (const auto& p : expected_params) {
                                if (p.name == param) {
                                    std::print(stderr, ", Default=");
                                    std::visit([&](const auto& val) {
                                        std::print(stderr, "{}", val);
                                    },
                                               p.value);
                                    std::println(stderr, " ({})", p.description);
                                    break;
                                }
                            }
                        }
                    }

                    if (!missing_in_json.empty()) {
                        std::println(stderr, "\n结构体中存在但JSON中不存在的参数:");
                        for (const auto& param : missing_in_json) {
                            // Find and print the description
                            for (const auto& p : expected_params) {
                                if (p.name == param) {
                                    std::println(stderr, "  - {} ({})", param, p.description);
                                    break;
                                }
                            }
                        }
                    }

                    if (!unknown_params.empty()) {
                        std::println(stderr, "\nJSON中存在未知的参数 (将被忽略):");
                        for (const auto& param : unknown_params) {
                            std::println(stderr, "  - {}", param);
                        }
                    }
                } else {
                    std::println("参数验证成功!");
                }

                return all_match;
            }
        } // namespace

        /**
         * @brief Read optimization parameters from JSON file
         * @param[in] strategy Optimization strategy to load parameters for
         * @return Expected OptimizationParameters or error message
         */
        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::string strategy) {
            auto json_result = read_json_file(get_config_path(strategy + "_optimization_params.json"));
            if (!json_result) {
                return std::unexpected(json_result.error());
            }

            auto json = *json_result;

            // Create default parameters for verification
            OptimizationParameters defaults;

            // Verify parameters before reading
            verify_optimization_parameters(defaults, json);

            try {
                OptimizationParameters params;
                params.iterations = json["iterations"];
                params.means_lr = json["means_lr"];
                params.shs_lr = json["shs_lr"];
                params.opacity_lr = json["opacity_lr"];
                params.scaling_lr = json["scaling_lr"];
                params.rotation_lr = json["rotation_lr"];
                params.lambda_dssim = json["lambda_dssim"];
                params.min_opacity = json["min_opacity"];
                params.refine_every = json["refine_every"];
                params.start_refine = json["start_refine"];
                params.stop_refine = json["stop_refine"];
                params.grad_threshold = json["grad_threshold"];
                params.sh_degree = json["sh_degree"];

                if (json.contains("opacity_reg")) {
                    params.opacity_reg = json["opacity_reg"];
                }
                if (json.contains("scale_reg")) {
                    params.scale_reg = json["scale_reg"];
                }
                if (json.contains("init_opacity")) {
                    params.init_opacity = json["init_opacity"];
                }
                if (json.contains("init_scaling")) {
                    params.init_scaling = json["init_scaling"];
                }
                if (json.contains("max_cap")) {
                    params.max_cap = json["max_cap"];
                }

                // Handle render mode
                if (json.contains("render_mode")) {
                    std::string mode = json["render_mode"];
                    // Validate render mode
                    if (mode == "RGB" || mode == "D" || mode == "ED" ||
                        mode == "RGB_D" || mode == "RGB_ED") {
                        params.render_mode = mode;
                    } else {
                        std::println(stderr, "警告: 无效的渲染模式 '{}' in JSON. 使用默认值 'RGB'", mode);
                    }
                }

                if (json.contains("strategy")) {
                    std::string strategy = json["strategy"];
                    if (strategy == "mcmc" || strategy == "default") {
                        params.strategy = strategy;
                    } else {
                        std::println(stderr, "警告: 无效的优化策略 '{}' in JSON. 使用默认值 'default'", strategy);
                    }
                }

                if (json.contains("eval_steps")) {
                    params.eval_steps.clear();
                    for (const auto& step : json["eval_steps"]) {
                        params.eval_steps.push_back(step.get<size_t>());
                    }
                }

                if (json.contains("save_steps")) {
                    params.save_steps.clear();
                    for (const auto& step : json["save_steps"]) {
                        params.save_steps.push_back(step.get<size_t>());
                    }
                }

                if (json.contains("enable_eval")) {
                    params.enable_eval = json["enable_eval"];
                }
                if (json.contains("enable_save_eval_images")) {
                    params.enable_save_eval_images = json["enable_save_eval_images"];
                }
                if (json.contains("skip_intermediate")) {
                    params.skip_intermediate_saving = json["skip_intermediate"];
                }
                if (json.contains("use_bilateral_grid")) {
                    params.use_bilateral_grid = json["use_bilateral_grid"];
                }
                if (json.contains("bilateral_grid_X")) {
                    params.bilateral_grid_X = json["bilateral_grid_X"];
                }
                if (json.contains("bilateral_grid_Y")) {
                    params.bilateral_grid_Y = json["bilateral_grid_Y"];
                }
                if (json.contains("bilateral_grid_W")) {
                    params.bilateral_grid_W = json["bilateral_grid_W"];
                }
                if (json.contains("bilateral_grid_lr")) {
                    params.bilateral_grid_lr = json["bilateral_grid_lr"];
                }
                if (json.contains("tv_loss_weight")) {
                    params.tv_loss_weight = json["tv_loss_weight"];
                }
                if (json.contains("prune_opacity")) {
                    params.prune_opacity = json["prune_opacity"];
                }
                if (json.contains("grow_scale3d")) {
                    params.grow_scale3d = json["grow_scale3d"];
                }
                if (json.contains("grow_scale2d")) {
                    params.grow_scale2d = json["grow_scale2d"];
                }
                if (json.contains("prune_scale3d")) {
                    params.prune_scale3d = json["prune_scale3d"];
                }
                if (json.contains("prune_scale2d")) {
                    params.prune_scale2d = json["prune_scale2d"];
                }
                if (json.contains("stop_refine_scale2d")) {
                    params.stop_refine_scale2d = json["stop_refine_scale2d"];
                }
                if (json.contains("reset_every")) {
                    params.reset_every = json["reset_every"];
                }
                if (json.contains("pause_refine_after_reset")) {
                    params.pause_refine_after_reset = json["pause_refine_after_reset"];
                }
                if (json.contains("revised_opacity")) {
                    params.revised_opacity = json["revised_opacity"];
                }
                if (json.contains("steps_scaler")) {
                    params.steps_scaler = json["steps_scaler"];
                }
                if (json.contains("antialiasing")) {
                    params.antialiasing = json["antialiasing"];
                }
                if (json.contains("skip_intermediate")) {
                    params.antialiasing = json["skip_intermediate"];
                }
                if (json.contains("sh_degree_interval")) {
                    params.sh_degree_interval = json["sh_degree_interval"];
                }
                if (json.contains("random")) {
                    params.random = json["random"];
                }
                if (json.contains("init_num_pts")) {
                    params.init_num_pts = json["init_num_pts"];
                }
                if (json.contains("init_extent")) {
                    params.init_extent = json["init_extent"];
                }

                return params;

            } catch (const std::exception& e) {
                return std::unexpected(std::format("解析优化参数时出错: {}", e.what()));
            }
        }

        /**
         * @brief Save full training parameters (dataset + optimization) to JSON
         * @param params The full training parameters
         * @param output_path Path to the output directory
         * @return Expected void or error message
         */
        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path) {

            try {
                nlohmann::json json;

                // Dataset configuration
                json["dataset"]["data_path"] = params.dataset.data_path.string();
                json["dataset"]["output_path"] = params.dataset.output_path.string();
                json["dataset"]["images"] = params.dataset.images;
                json["dataset"]["resize_factor"] = params.dataset.resize_factor;
                json["dataset"]["test_every"] = params.dataset.test_every;

                // Optimization configuration
                nlohmann::json opt_json;
                opt_json["iterations"] = params.optimization.iterations;
                opt_json["means_lr"] = params.optimization.means_lr;
                opt_json["shs_lr"] = params.optimization.shs_lr;
                opt_json["opacity_lr"] = params.optimization.opacity_lr;
                opt_json["scaling_lr"] = params.optimization.scaling_lr;
                opt_json["rotation_lr"] = params.optimization.rotation_lr;
                opt_json["lambda_dssim"] = params.optimization.lambda_dssim;
                opt_json["min_opacity"] = params.optimization.min_opacity;
                opt_json["refine_every"] = params.optimization.refine_every;
                opt_json["start_refine"] = params.optimization.start_refine;
                opt_json["stop_refine"] = params.optimization.stop_refine;
                opt_json["grad_threshold"] = params.optimization.grad_threshold;
                opt_json["sh_degree"] = params.optimization.sh_degree;
                opt_json["opacity_reg"] = params.optimization.opacity_reg;
                opt_json["scale_reg"] = params.optimization.scale_reg;
                opt_json["init_opacity"] = params.optimization.init_opacity;
                opt_json["init_scaling"] = params.optimization.init_scaling;
                opt_json["max_cap"] = params.optimization.max_cap;
                opt_json["render_mode"] = params.optimization.render_mode;
                opt_json["eval_steps"] = params.optimization.eval_steps;
                opt_json["save_steps"] = params.optimization.save_steps;
                opt_json["enable_eval"] = params.optimization.enable_eval;
                opt_json["enable_save_eval_images"] = params.optimization.enable_save_eval_images;
                opt_json["strategy"] = params.optimization.strategy;
                opt_json["skip_intermediate"] = params.optimization.skip_intermediate_saving;
                opt_json["use_bilateral_grid"] = params.optimization.use_bilateral_grid;
                opt_json["bilateral_grid_X"] = params.optimization.bilateral_grid_X;
                opt_json["bilateral_grid_Y"] = params.optimization.bilateral_grid_Y;
                opt_json["bilateral_grid_W"] = params.optimization.bilateral_grid_W;
                opt_json["bilateral_grid_lr"] = params.optimization.bilateral_grid_lr;
                opt_json["tv_loss_weight"] = params.optimization.tv_loss_weight;
                opt_json["prune_opacity"] = params.optimization.prune_opacity;
                opt_json["grow_scale3d"] = params.optimization.grow_scale3d;
                opt_json["grow_scale2d"] = params.optimization.grow_scale2d;
                opt_json["prune_scale3d"] = params.optimization.prune_scale3d;
                opt_json["prune_scale2d"] = params.optimization.prune_scale2d;
                opt_json["stop_refine_scale2d"] = params.optimization.stop_refine_scale2d;
                opt_json["reset_every"] = params.optimization.reset_every;
                opt_json["pause_refine_after_reset"] = params.optimization.pause_refine_after_reset;
                opt_json["revised_opacity"] = params.optimization.revised_opacity;
                opt_json["steps_scaler"] = params.optimization.steps_scaler;
                opt_json["antialiasing"] = params.optimization.antialiasing;
                opt_json["sh_degree_interval"] = params.optimization.sh_degree_interval;
                opt_json["random"] = params.optimization.random;
                opt_json["init_num_pts"] = params.optimization.init_num_pts;
                opt_json["init_extent"] = params.optimization.init_extent;

                json["optimization"] = opt_json;

                // Add timestamp
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                json["timestamp"] = ss.str();

                // Save to file
                std::filesystem::path filepath = output_path / "training_config.json";
                std::ofstream file(filepath);
                if (!file.is_open()) {
                    return std::unexpected(std::format("无法打开文件进行写入: {}", filepath.string()));
                }

                file << json.dump(4); // Pretty print with 4 spaces
                file.close();

                std::println("训练配置已保存到: {}", filepath.string());
                return {};

            } catch (const std::exception& e) {
                return std::unexpected(std::format("保存训练参数时出错: {}", e.what()));
            }
        }

    } // namespace param
} // namespace gs