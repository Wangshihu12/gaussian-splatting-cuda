#pragma once

#include <torch/torch.h>

namespace gs::training {
    /**
     * [功能描述]：指数学习率调度器类，实现学习率的指数衰减策略。
     * 由于PyTorch C++ API与Python API不同，这里实现了一个简单的指数学习率调度器。
     * 该调度器按照指数衰减的方式调整优化器的学习率，帮助模型在训练过程中更好地收敛。
     * 
     * 主要特点：
     * 1. 支持单个参数组或所有参数组的学习率调整
     * 2. 使用指数衰减公式：new_lr = current_lr * gamma
     * 3. 与PyTorch优化器无缝集成
     * 4. 轻量级实现，适合C++环境
     */
    class ExponentialLR {
    public:
        /**
         * [功能描述]：构造函数，初始化指数学习率调度器。
         * 设置优化器引用、衰减因子和参数组索引，为后续的学习率调整做准备。
         * 
         * @param optimizer [参数说明]：要调度的PyTorch优化器引用，调度器将直接修改其学习率。
         * @param gamma [参数说明]：学习率衰减因子，每次调用step()后学习率将乘以这个值。
         *                         通常gamma < 1，实现学习率的递减。
         * @param param_group_index [参数说明]：参数组索引，指定要调整哪个参数组的学习率。
         *                                   -1表示调整所有参数组，>=0表示调整特定参数组。
         */
        ExponentialLR(torch::optim::Optimizer& optimizer, double gamma, int param_group_index = -1)
            : optimizer_(optimizer),           // 初始化优化器引用
              gamma_(gamma),                   // 初始化衰减因子
              param_group_index_(param_group_index) {  // 初始化参数组索引
        }

        /**
         * [功能描述]：执行学习率更新步骤。
         * 根据设置的衰减因子gamma更新优化器的学习率。
         * 如果param_group_index_为-1，则更新所有参数组的学习率；
         * 否则只更新指定参数组的学习率。
         * 
         * 更新公式：new_lr = current_lr * gamma
         */
        void step();

    private:
        /**
         * [功能说明]：PyTorch优化器的引用，调度器直接操作这个优化器来调整学习率。
         * 使用引用类型避免不必要的拷贝，提高性能。
         */
        torch::optim::Optimizer& optimizer_;
        
        /**
         * [功能说明]：学习率衰减因子，控制学习率的衰减速度。
         * 典型值：
         * - gamma = 0.9：每次更新后学习率变为原来的90%
         * - gamma = 0.5：每次更新后学习率变为原来的50%
         * - gamma = 0.1：每次更新后学习率变为原来的10%
         * 通常gamma < 1，实现学习率的递减策略。
         */
        double gamma_;
        
        /**
         * [功能说明]：参数组索引，指定要调整学习率的参数组。
         * 取值范围：
         * - -1：调整所有参数组的学习率（默认行为）
         * - >=0：只调整指定索引的参数组学习率
         * 
         * 这种设计允许对不同参数组使用不同的学习率调整策略，
         * 例如：位置参数使用快速衰减，颜色参数使用慢速衰减。
         */
        int param_group_index_;
    };
} // namespace gs::training
