#pragma once

#include "../dataset.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <vector>

// 前向声明：高斯溅射数据类
class splatData;

namespace gs::training {
    // 工具函数：SSIM窗口创建
    /**
     * [功能描述]：创建高斯窗口函数，用于SSIM计算。
     * 生成指定大小和标准差的高斯核，用于结构相似性指数的计算。
     * 
     * @param window_size [参数说明]：窗口大小，必须是奇数。
     * @param sigma [参数说明]：高斯分布的标准差，控制窗口的平滑程度。
     * @return [返回值说明]：返回高斯窗口张量，形状为[window_size, window_size]。
     */
    torch::Tensor gaussian(const int window_size, const float sigma);

    /**
     * [功能描述]：创建SSIM计算窗口函数。
     * 根据窗口大小和通道数创建适合SSIM计算的窗口张量。
     * 
     * @param window_size [参数说明]：窗口大小，用于局部区域分析。
     * @param channel [参数说明]：图像通道数，通常为3（RGB）。
     * @return [返回值说明]：返回SSIM窗口张量，形状为[channel, window_size, window_size]。
     */
    torch::Tensor create_window(const int window_size, const int channel);

    /**
     * [功能描述]：峰值信噪比（PSNR）指标类。
     * PSNR是衡量图像质量的重要指标，值越高表示图像质量越好。
     * 常用于评估图像重建、压缩和去噪等任务的质量。
     */
    class PSNR {
    public:
        /**
         * [功能描述]：构造函数，初始化PSNR计算器。
         * @param data_range [参数说明]：数据范围，通常为1.0（归一化图像）或255（8位图像）。
         */
        explicit PSNR(const float data_range = 1.0f) : data_range_(data_range) {
        }

        /**
         * [功能描述]：计算PSNR值。
         * @param pred [参数说明]：预测图像张量。
         * @param target [参数说明]：目标图像张量。
         * @return [返回值说明]：返回PSNR值，单位为dB。
         */
        float compute(const torch::Tensor& pred, const torch::Tensor& target) const;

    private:
        const float data_range_;  // 数据范围，用于PSNR计算
    };

    /**
     * [功能描述]：结构相似性指数（SSIM）指标类。
     * SSIM衡量两个图像在结构、亮度和对比度方面的相似性，
     * 比PSNR更接近人眼视觉感知，值范围在[-1, 1]之间，1表示完全相同。
     */
    class SSIM {
    public:
        /**
         * [功能描述]：构造函数，初始化SSIM计算器。
         * @param window_size [参数说明]：局部分析窗口大小，默认为11。
         * @param channel [参数说明]：图像通道数，默认为3（RGB）。
         */
        SSIM(const int window_size = 11, const int channel = 3);

        /**
         * [功能描述]：计算SSIM值。
         * @param pred [参数说明]：预测图像张量。
         * @param target [参数说明]：目标图像张量。
         * @return [返回值说明]：返回SSIM值，范围[-1, 1]。
         */
        float compute(const torch::Tensor& pred, const torch::Tensor& target);

    private:
        const int window_size_;           // 局部分析窗口大小
        const int channel_;               // 图像通道数
        torch::Tensor window_;            // 预计算的SSIM窗口
        static constexpr float C1 = 0.01f * 0.01f;  // 数值稳定性常数C1
        static constexpr float C2 = 0.03f * 0.03f;  // 数值稳定性常数C2
    };

    /**
     * [功能描述]：学习感知图像块相似度（LPIPS）指标类。
     * LPIPS使用预训练的深度神经网络来评估图像相似性，
     * 更接近人类视觉感知，广泛用于图像质量评估。
     */
    class LPIPS {
    public:
        /**
         * [功能描述]：构造函数，初始化LPIPS计算器。
         * @param model_path [参数说明]：预训练LPIPS模型文件路径，空字符串使用默认路径。
         */
        explicit LPIPS(const std::string& model_path = "");

        /**
         * [功能描述]：计算LPIPS值。
         * @param pred [参数说明]：预测图像张量。
         * @param target [参数说明]：目标图像张量。
         * @return [返回值说明]：返回LPIPS值，值越小表示越相似。
         */
        float compute(const torch::Tensor& pred, const torch::Tensor& target);

        /**
         * [功能描述]：检查LPIPS模型是否已成功加载。
         * @return [返回值说明]：如果模型已加载返回true，否则返回false。
         */
        bool is_loaded() const { return model_loaded_; }

    private:
        torch::jit::script::Module model_;  // PyTorch JIT脚本模块，存储LPIPS模型
        bool model_loaded_ = false;          // 模型加载状态标志

        /**
         * [功能描述]：加载LPIPS预训练模型。
         * @param model_path [参数说明]：模型文件路径。
         */
        void load_model(const std::string& model_path);
    };

    /**
     * [功能描述]：评估指标结果结构体。
     * 存储单次评估的所有指标结果，包括图像质量指标、性能指标和训练状态信息。
     * 提供多种输出格式，便于结果分析和报告生成。
     */
    struct EvalMetrics {
        float psnr;           // 峰值信噪比，衡量图像质量
        float ssim;           // 结构相似性指数，衡量结构相似性
        float lpips;          // 学习感知图像块相似度，衡量感知相似性
        float elapsed_time;   // 单张图像处理时间（秒）
        int num_gaussians;    // 当前高斯点数量
        int iteration;        // 当前训练迭代次数

        /**
         * [功能描述]：将评估结果转换为可读的字符串格式。
         * @return [返回值说明]：返回格式化的结果字符串。
         */
        [[nodiscard]] std::string to_string() const {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4);
            ss << "PSNR: " << psnr
               << ", SSIM: " << ssim
               << ", LPIPS: " << lpips
               << ", Time: " << elapsed_time << "s/image"
               << ", #GS: " << num_gaussians;
            return ss.str();
        }

        /**
         * [功能描述]：生成CSV文件的表头。
         * @return [返回值说明]：返回CSV格式的表头字符串。
         */
        static std::string to_csv_header() {
            return "iteration,psnr,ssim,lpips,time_per_image,num_gaussians";
        }

        /**
         * [功能描述]：将评估结果转换为CSV行格式。
         * @return [返回值说明]：返回CSV格式的数据行字符串。
         */
        [[nodiscard]] std::string to_csv_row() const {
            std::stringstream ss;
            ss << iteration << ","
               << std::fixed << std::setprecision(6)
               << psnr << ","
               << ssim << ","
               << lpips << ","
               << elapsed_time << ","
               << num_gaussians;
            return ss.str();
        }
    };

    /**
     * [功能描述]：指标报告器类。
     * 负责收集、存储和保存训练过程中的所有评估指标，
     * 支持CSV和文本格式的输出，便于后续分析和可视化。
     */
    class MetricsReporter {
    public:
        /**
         * [功能描述]：构造函数，初始化指标报告器。
         * @param output_dir [参数说明]：输出目录路径，用于保存报告文件。
         */
        explicit MetricsReporter(const std::filesystem::path& output_dir);

        /**
         * [功能描述]：添加新的评估指标到报告器中。
         * @param metrics [参数说明]：要添加的评估指标结果。
         */
        void add_metrics(const EvalMetrics& metrics);

        /**
         * [功能描述]：保存完整的评估报告。
         * 将收集的所有指标保存为CSV和文本文件。
         */
        void save_report() const;

    private:
        const std::filesystem::path output_dir_;           // 输出目录路径
        std::vector<EvalMetrics> all_metrics_;             // 存储所有评估指标的向量
        const std::filesystem::path csv_path_;             // CSV文件路径
        const std::filesystem::path txt_path_;             // 文本文件路径
    };

    /**
     * [功能描述]：主评估器类，处理所有指标计算和可视化。
     * 这是整个评估系统的核心类，协调各种指标的计算、
     * 管理评估时机、生成报告，并提供完整的评估流程。
     */
    class MetricsEvaluator {
    public:
        /**
         * [功能描述]：构造函数，初始化指标评估器。
         * @param params [参数说明]：训练参数，包含评估相关的配置信息。
         */
        explicit MetricsEvaluator(const param::TrainingParameters& params);

        /**
         * [功能描述]：检查评估功能是否已启用。
         * @return [返回值说明]：如果评估已启用返回true，否则返回false。
         */
        bool is_enabled() const { return _params.optimization.enable_eval; }

        /**
         * [功能描述]：检查是否应该在当前迭代进行评估。
         * @param iteration [参数说明]：当前训练迭代次数。
         * @return [返回值说明]：如果应该评估返回true，否则返回false。
         */
        bool should_evaluate(const int iteration) const;

        /**
         * [功能描述]：主要的评估方法，计算所有指标。
         * @param iteration [参数说明]：当前训练迭代次数。
         * @param splatData [参数说明]：高斯溅射数据，包含当前模型状态。
         * @param val_dataset [参数说明]：验证数据集，用于评估。
         * @param background [参数说明]：背景颜色张量。
         * @return [返回值说明]：返回包含所有评估指标的结果结构体。
         */
        EvalMetrics evaluate(const int iteration,
                             const SplatData& splatData,
                             std::shared_ptr<CameraDataset> val_dataset,
                             torch::Tensor& background);

        /**
         * [功能描述]：保存最终评估报告。
         * 调用报告器保存所有收集的指标数据。
         */
        void save_report() const {
            if (_reporter)
                _reporter->save_report();
        }

        /**
         * [功能描述]：打印评估头部信息。
         * @param iteration [参数说明]：当前评估的迭代次数。
         */
        void print_evaluation_header(const int iteration) const {
            std::cout << std::endl;
            std::cout << "[Evaluation at step " << iteration << "]" << std::endl;
        }

    private:
        // 配置相关
        const param::TrainingParameters _params;  // 训练参数配置

        // 指标计算器
        std::unique_ptr<PSNR> _psnr_metric;      // PSNR指标计算器
        std::unique_ptr<SSIM> _ssim_metric;      // SSIM指标计算器
        std::unique_ptr<LPIPS> _lpips_metric;    // LPIPS指标计算器
        std::unique_ptr<MetricsReporter> _reporter;  // 指标报告器

        // 辅助函数
        /**
         * [功能描述]：将深度图转换为彩色映射。
         * @param depth_normalized [参数说明]：归一化的深度张量。
         * @return [返回值说明]：返回彩色映射后的深度图张量。
         */
        torch::Tensor apply_depth_colormap(const torch::Tensor& depth_normalized) const;

        /**
         * [功能描述]：检查是否有RGB数据。
         * @return [返回值说明]：如果有RGB数据返回true，否则返回false。
         */
        bool has_rgb() const;

        /**
         * [功能描述]：检查是否有深度数据。
         * @return [返回值说明]：如果有深度数据返回true，否则返回false。
         */
        bool has_depth() const;

        /**
         * [功能描述]：从数据集创建数据加载器。
         * @param dataset [参数说明]：相机数据集。
         * @param workers [参数说明]：工作线程数，默认为1。
         * @return [返回值说明]：返回数据加载器对象。
         */
        auto make_dataloader(std::shared_ptr<CameraDataset> dataset, const int workers = 1) const;
    };
} // namespace gs::training
