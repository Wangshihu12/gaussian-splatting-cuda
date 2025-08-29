#pragma once
#include <torch/nn/module.h>           // PyTorch神经网络模块基类
#include <torch/nn/modules/container/sequential.h>  // 顺序容器模块，用于构建MLP
#include <torch/nn/modules/embedding.h>             // 嵌入层模块，用于相机姿态编码

namespace gs::training {
    /**
     * [功能描述]：姿态优化模块的基类，继承自PyTorch的nn::Module，提供相机姿态优化的基础接口。
     * 这是一个虚函数基类，定义了姿态优化的通用接口，子类可以实现不同的优化策略。
     */
    struct PoseOptimizationModule : torch::nn::Module {
        /**
         * [功能描述]：默认构造函数，不执行任何初始化操作。
         */
        PoseOptimizationModule() {}
        
        /**
         * [功能描述]：前向传播函数，处理相机变换矩阵并返回结果。
         * 在基类中，这是一个虚函数，默认实现是直接返回输入的变换矩阵，不进行任何修改。
         * @param camera_transforms [参数说明]：输入的相机变换矩阵，包含位置和旋转信息。
         * @param embedding_ids [参数说明]：相机嵌入ID，用于标识不同的相机（可能未使用）。
         * @return [返回值说明]：返回处理后的相机变换矩阵，基类中直接返回输入。
         */
        virtual torch::Tensor forward(torch::Tensor camera_transforms, [[maybe_unused]] torch::Tensor embedding_ids) {
            // 无操作，直接返回输入的变换矩阵
            return camera_transforms;
        }
    };
    
    /**
     * [功能描述]：直接姿态优化模块，继承自PoseOptimizationModule。
     * 使用嵌入层直接学习每个相机的姿态参数，适用于相机数量较少且姿态变化相对简单的情况。
     * 每个相机都有独立的9维嵌入向量，包含3维平移和6维旋转表示。
     */
    struct DirectPoseOptimizationModule : PoseOptimizationModule {
        /**
         * [功能描述]：构造函数，初始化直接姿态优化模块。
         * @param number_of_cameras [参数说明]：相机数量，决定嵌入层的维度。
         */
        explicit DirectPoseOptimizationModule(int number_of_cameras);
        
        /**
         * [功能描述]：重写的前向传播函数，实现直接姿态优化逻辑。
         * @param camera_transforms [参数说明]：输入的相机变换矩阵。
         * @param embedding_ids [参数说明]：相机嵌入ID，用于查找对应的嵌入向量。
         * @return [返回值说明]：返回优化后的相机变换矩阵。
         */
        torch::Tensor forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids) override;
        
        /**
         * [功能说明]：相机嵌入层，存储每个相机的姿态参数。
         * 维度为[C, 9]，其中C是相机数量，9包含3维平移和6维旋转表示。
         * 每个相机都有独立的嵌入向量，可以直接学习其姿态参数。
         */
        torch::nn::Embedding camera_embeddings;
        
        /**
         * [功能说明]：旋转单位矩阵，用于6D旋转表示。
         * 维度为[6]，表示6D旋转表示中的单位旋转（无旋转状态）。
         * 作为旋转优化的参考基准。
         */
        torch::Tensor rot_identity;
    };
    
    /**
     * [功能描述]：MLP姿态优化模块，继承自PoseOptimizationModule。
     * 使用多层感知机（MLP）网络来学习相机姿态的复杂非线性变换。
     * 适用于相机数量较多或姿态变化复杂的情况，可以学习更复杂的姿态关系。
     */
    struct MLPPoseOptimizationModule : PoseOptimizationModule {
        /**
         * [功能描述]：构造函数，初始化MLP姿态优化模块。
         * @param number_of_cameras [参数说明]：相机数量，决定嵌入层的维度。
         * @param width [参数说明]：MLP隐藏层的宽度，默认为64个神经元。
         * @param depth [参数说明]：MLP的深度（层数），默认为2层。
         */
        explicit MLPPoseOptimizationModule(int number_of_cameras, int width = 64, int depth = 2);
        
        /**
         * [功能描述]：重写的前向传播函数，实现MLP姿态优化逻辑。
         * @param camera_transforms [参数说明]：输入的相机变换矩阵。
         * @param embedding_ids [参数说明]：相机嵌入ID，用于查找对应的嵌入向量。
         * @return [返回值说明]：返回通过MLP优化后的相机变换矩阵。
         */
        torch::Tensor forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids) override;
        
        /**
         * [功能说明]：相机嵌入层，存储每个相机的特征表示。
         * 维度为[C, F]，其中C是相机数量，F是特征维度。
         * 相比直接模块，这里的嵌入向量作为MLP的输入特征。
         */
        torch::nn::Embedding camera_embeddings;
        
        /**
         * [功能说明]：旋转单位矩阵，用于6D旋转表示。
         * 维度为[6]，表示6D旋转表示中的单位旋转（无旋转状态）。
         * 作为旋转优化的参考基准，与直接模块相同。
         */
        torch::Tensor rot_identity;
        
        /**
         * [功能说明]：多层感知机网络，用于学习相机姿态的非线性变换。
         * 网络结构由构造函数中的width和depth参数决定。
         * 输入是相机嵌入特征，输出是优化后的姿态参数。
         */
        torch::nn::Sequential mlp;
    };
} // namespace gs::training