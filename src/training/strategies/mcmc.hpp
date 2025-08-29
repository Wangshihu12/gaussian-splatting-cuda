#pragma once

#include "istrategy.hpp"
#include <memory>
#include <torch/torch.h>

namespace gs::training {
    /**
     * [功能描述]：MCMC（马尔可夫链蒙特卡洛）训练策略类。
     * 这个类实现了基于MCMC的高斯溅射模型训练策略，继承自IStrategy接口。
     * MCMC策略通过随机采样和状态转移来优化高斯溅射模型的参数，
     * 特别适用于处理复杂的多模态优化问题和避免局部最优解。
     * 
     * 主要特点：
     * 1. 支持高斯点的动态添加和移除（密集化）
     * 2. 使用指数学习率调度器
     * 3. 集成噪声注入机制
     * 4. 支持SelectiveAdam优化器
     */
    class MCMC : public IStrategy {
    public:
        /**
         * [功能描述]：删除默认构造函数，MCMC类必须通过参数构造。
         */
        MCMC() = delete;

        /**
         * [功能描述]：构造函数，使用移动语义初始化高斯溅射数据。
         * @param splat_data [参数说明]：高斯溅射数据，使用移动语义转移所有权。
         */
        MCMC(gs::SplatData&& splat_data);

        /**
         * [功能描述]：删除拷贝构造函数，防止意外的对象拷贝。
         */
        MCMC(const MCMC&) = delete;

        /**
         * [功能描述]：删除拷贝赋值操作符，防止意外的对象拷贝。
         */
        MCMC& operator=(const MCMC&) = delete;

        /**
         * [功能描述]：移动构造函数，支持对象的移动语义。
         */
        MCMC(MCMC&&) = default;

        /**
         * [功能描述]：移动赋值操作符，支持对象的移动语义。
         */
        MCMC& operator=(MCMC&&) = default;

        // IStrategy接口实现
        /**
         * [功能描述]：初始化MCMC策略，设置优化参数和训练环境。
         * @param optimParams [参数说明]：优化参数配置，包含学习率、迭代次数等设置。
         */
        void initialize(const gs::param::OptimizationParameters& optimParams) override;

        /**
         * [功能描述]：后向传播后的处理函数，在每个训练步骤后执行。
         * 这个函数负责处理渲染输出，可能包括损失计算、梯度更新等操作。
         * @param iter [参数说明]：当前迭代次数。
         * @param render_output [参数说明]：渲染输出结果，包含图像和透明度等信息。
         */
        void post_backward(int iter, RenderOutput& render_output) override;

        /**
         * [功能描述]：判断当前迭代是否处于细化阶段。
         * 细化阶段通常指模型参数已经相对稳定，开始进行精细调整的阶段。
         * @param iter [参数说明]：当前迭代次数。
         * @return [返回值说明]：如果处于细化阶段返回true，否则返回false。
         */
        bool is_refining(int iter) const override;

        /**
         * [功能描述]：执行一个训练步骤，包括参数更新、密集化等操作。
         * 这是MCMC策略的核心函数，实现了一个完整的训练迭代。
         * @param iter [参数说明]：当前迭代次数。
         */
        void step(int iter) override;

        /**
         * [功能描述]：获取模型的高斯溅射数据（非常量版本）。
         * @return [返回值说明]：返回高斯溅射数据的引用，允许修改。
         */
        gs::SplatData& get_model() override { return _splat_data; }
        
        /**
         * [功能描述]：获取模型的高斯溅射数据（常量版本）。
         * @return [返回值说明]：返回高斯溅射数据的常量引用，不允许修改。
         */
        const gs::SplatData& get_model() const override { return _splat_data; }

    private:
        /**
         * [功能描述]：指数学习率调度器类。
         * 由于C++ API与Python API不同，这里实现了一个简单的指数学习率调度器。
         * 该调度器按照指数衰减的方式调整学习率，帮助模型收敛。
         */
        class ExponentialLR {
        public:
            /**
             * [功能描述]：构造函数，初始化指数学习率调度器。
             * @param optimizer [参数说明]：要调度的优化器引用。
             * @param gamma [参数说明]：学习率衰减因子，每次更新后学习率乘以这个值。
             * @param param_group_index [参数说明]：参数组索引，-1表示所有参数组。
             */
            ExponentialLR(torch::optim::Optimizer& optimizer, double gamma, int param_group_index = -1)
                : optimizer_(optimizer),
                  gamma_(gamma),
                  param_group_index_(param_group_index) {
            }

            /**
             * [功能描述]：执行学习率更新步骤。
             * 将当前学习率乘以衰减因子gamma，实现学习率的指数衰减。
             */
            void step();

        private:
            torch::optim::Optimizer& optimizer_;  // 要调度的优化器引用
            double gamma_;                        // 学习率衰减因子
            int param_group_index_;               // 参数组索引
        };

        // 辅助函数
        /**
         * [功能描述]：从权重分布中进行多项分布采样。
         * 用于MCMC策略中的随机采样，支持有放回和无放回采样。
         * @param weights [参数说明]：权重张量，表示每个元素的采样概率。
         * @param n [参数说明]：要采样的元素数量。
         * @param replacement [参数说明]：是否允许重复采样，true表示有放回。
         * @return [返回值说明]：采样结果的索引张量。
         */
        torch::Tensor multinomial_sample(const torch::Tensor& weights, int n, bool replacement = true);

        /**
         * [功能描述]：重新定位高斯点。
         * 在MCMC策略中，可能需要重新调整高斯点的位置以改善模型质量。
         * @return [返回值说明]：重新定位的高斯点数量。
         */
        int relocate_gs();

        /**
         * [功能描述]：添加新的高斯点。
         * 这是密集化过程的一部分，根据训练需要动态增加高斯点数量。
         * @return [返回值说明]：新添加的高斯点数量。
         */
        int add_new_gs();

        /**
         * [功能描述]：注入噪声到模型参数中。
         * 噪声注入是MCMC策略的重要组成部分，帮助模型跳出局部最优解。
         */
        void inject_noise();

        /**
         * [功能描述]：为重新定位操作更新优化器。
         * 当高斯点被重新定位时，需要相应地更新优化器的状态。
         * @param optimizer [参数说明]：要更新的优化器指针。
         * @param sampled_indices [参数说明]：采样的索引张量。
         * @param dead_indices [参数说明]：被移除的高斯点索引张量。
         * @param param_position [参数说明]：参数在优化器中的位置。
         */
        void update_optimizer_for_relocate(torch::optim::Optimizer* optimizer,
                                           const torch::Tensor& sampled_indices,
                                           const torch::Tensor& dead_indices,
                                           int param_position);

        // 成员变量
        std::unique_ptr<torch::optim::Optimizer> _optimizer;        // 优化器，负责参数更新
        std::unique_ptr<ExponentialLR> _scheduler;                  // 学习率调度器
        gs::SplatData _splat_data;                                  // 高斯溅射模型数据
        std::unique_ptr<const gs::param::OptimizationParameters> _params;  // 优化参数配置

        // MCMC特定参数
        const float _noise_lr = 5e5;                                // 噪声学习率，控制噪声注入的强度

        // 状态变量
        torch::Tensor _binoms;                                      // 二项分布相关张量，用于MCMC采样

        // SelectiveAdam支持
        torch::Tensor _last_visibility_mask;                        // 上一次的可见性掩码，用于SelectiveAdam优化器
    };
} // namespace gs::training
