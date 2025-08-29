#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>

namespace gs::training {
    /**
     * [功能描述]：FusedAdam优化器类，专门为高斯溅射模型优化的融合Adam优化器。
     * 这个类继承自PyTorch的torch::optim::Optimizer基类，实现了Adam优化算法的融合版本。
     * 使用融合的CUDA内核来提高性能，特别适合处理大规模的高斯溅射参数优化。
     * 
     * 主要特点：
     * 1. 继承PyTorch优化器基类，保持API兼容性
     * 2. 使用融合CUDA内核提升性能
     * 3. 支持参数组级别的配置
     * 4. 完整的序列化支持
     * 5. 支持AMSGrad变体（当前未使用）
     */
    class FusedAdam : public torch::optim::Optimizer {
    public:
        /**
         * [功能描述]：FusedAdam优化器的配置选项结构体。
         * 继承自torch::optim::OptimizerCloneableOptions，提供可克隆的优化器选项。
         * 支持链式调用语法，方便配置各种优化参数。
         */
        struct Options : public torch::optim::OptimizerCloneableOptions<Options> {
            /**
             * [功能描述]：构造函数，初始化优化器选项。
             * @param lr [参数说明]：学习率，默认值为1e-3。
             */
            Options(double lr = 1e-3) : lr_(lr) {
            }

            /**
             * [功能描述]：设置学习率，支持链式调用。
             * @param lr [参数说明]：新的学习率值。
             * @return [返回值说明]：返回当前Options对象的引用，支持链式调用。
             */
            Options& lr(double lr) {
                lr_ = lr;
                return *this;
            }

            /**
             * [功能描述]：设置Adam算法的beta参数，控制动量估计的衰减率。
             * @param betas [参数说明]：包含两个beta值的元组，通常为(0.9, 0.999)。
             * @return [返回值说明]：返回当前Options对象的引用，支持链式调用。
             */
            Options& betas(const std::tuple<double, double>& betas) {
                betas_ = betas;
                return *this;
            }

            /**
             * [功能描述]：设置epsilon值，用于数值稳定性，防止除零错误。
             * @param eps [参数说明]：epsilon值，通常设置为很小的正数。
             * @return [返回值说明]：返回当前Options对象的引用，支持链式调用。
             */
            Options& eps(double eps) {
                eps_ = eps;
                return *this;
            }

            /**
             * [功能描述]：设置权重衰减参数，用于L2正则化。
             * @param weight_decay [参数说明]：权重衰减系数，控制正则化强度。
             * @return [返回值说明]：返回当前Options对象的引用，支持链式调用。
             */
            Options& weight_decay(double weight_decay) {
                weight_decay_ = weight_decay;
                return *this;
            }

            // 获取器函数
            double lr() const { return lr_; }                                    // 获取学习率
            const std::tuple<double, double>& betas() const { return betas_; }  // 获取beta参数
            double eps() const { return eps_; }                                 // 获取epsilon值
            double weight_decay() const { return weight_decay_; }                // 获取权重衰减

        private:
            double lr_ = 1e-3;                                    // 学习率，默认1e-3
            std::tuple<double, double> betas_ = std::make_tuple(0.9, 0.999);  // beta参数，默认(0.9, 0.999)
            double eps_ = 1e-8;                                   // epsilon值，默认1e-8
            double weight_decay_ = 0;                             // 权重衰减，默认0
        };

        /**
         * [功能描述]：Adam优化器的参数状态结构体。
         * 继承自torch::optim::OptimizerParamState，存储每个参数的优化状态。
         * 包含动量估计、二阶矩估计和步数计数等关键信息。
         */
        struct AdamParamState : public torch::optim::OptimizerParamState {
            torch::Tensor exp_avg;        // 一阶矩估计（动量），用于Adam算法的第一个动量项
            torch::Tensor exp_avg_sq;     // 二阶矩估计，用于Adam算法的第二个动量项
            torch::Tensor max_exp_avg_sq; // 最大二阶矩估计，用于AMSGrad变体（当前未使用）
            int64_t step_count = 0;       // 步数计数，记录该参数被更新的次数

            /**
             * [功能描述]：序列化函数，将优化器状态保存到归档中。
             * 支持模型的保存和加载，确保训练状态的持久化。
             * @param archive [参数说明]：输出归档对象，用于序列化数据。
             */
            void serialize(torch::serialize::OutputArchive& archive) const override {
                archive.write("exp_avg", exp_avg);           // 保存一阶矩估计
                archive.write("exp_avg_sq", exp_avg_sq);     // 保存二阶矩估计
                archive.write("step", step_count);           // 保存步数计数
                if (max_exp_avg_sq.defined()) {
                    archive.write("max_exp_avg_sq", max_exp_avg_sq);  // 如果定义了最大二阶矩估计，也保存它
                }
            }
        };

        /**
         * [功能描述]：构造函数，使用参数组和选项初始化FusedAdam优化器。
         * @param param_groups [参数说明]：参数组向量，每个参数组包含相关的参数张量。
         * @param options [参数说明]：优化器选项的智能指针，包含学习率、beta等配置。
         */
        explicit FusedAdam(std::vector<torch::optim::OptimizerParamGroup> param_groups,
                           std::unique_ptr<Options> options)
            : Optimizer(std::move(param_groups),
                        std::unique_ptr<torch::optim::OptimizerOptions>(std::move(options))) {
        }

        /**
         * [功能描述]：构造函数，使用参数张量向量和选项初始化FusedAdam优化器。
         * @param params [参数说明]：参数张量向量，所有参数将组成一个参数组。
         * @param options [参数说明]：优化器选项的智能指针。
         */
        explicit FusedAdam(std::vector<torch::Tensor> params, std::unique_ptr<Options> options)
            : Optimizer({torch::optim::OptimizerParamGroup(std::move(params))},
                        std::unique_ptr<torch::optim::OptimizerOptions>(std::move(options))) {
        }

        /**
         * [功能描述]：重写基类的step函数，执行优化步骤。
         * 这是PyTorch优化器接口的标准实现，接受损失闭包作为参数。
         * @param closure [参数说明]：损失计算闭包，用于计算当前损失值。
         * @return [返回值说明]：返回损失值张量。
         */
        torch::Tensor step(LossClosure closure) override;

        /**
         * [功能描述]：执行优化步骤，使用迭代次数参数。
         * 这是FusedAdam的专用step函数，提供更好的性能和功能。
         * @param iteration [参数说明]：当前迭代次数，用于优化器的内部状态管理。
         */
        void step(int iteration);

        /**
         * [功能描述]：清零所有参数的梯度。
         * @param set_to_none [参数说明]：是否将梯度设置为None（true）或零张量（false）。
         * @param iteration [参数说明]：当前迭代次数，用于优化器状态管理。
         */
        void zero_grad(bool set_to_none, int iteration);

    private:
        /**
         * [功能描述]：获取优化器选项的常量引用。
         * 将基类的默认选项转换为FusedAdam::Options类型。
         * @return [返回值说明]：返回Options对象的常量引用。
         */
        const Options& options() const {
            return static_cast<const Options&>(defaults());
        }
    };
} // namespace gs::training
