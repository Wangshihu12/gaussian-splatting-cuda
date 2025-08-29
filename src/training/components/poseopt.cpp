#include "poseopt.hpp"
#include <torch/torch.h>

namespace gs::training {
    // 为PyTorch函数式接口创建别名，简化代码
    namespace F = torch::nn::functional;

    /**
     * [功能描述]：将6D旋转表示转换为3x3旋转矩阵。
     * 6D旋转表示是一种避免万向锁问题的旋转表示方法，通过两个3D向量来定义旋转。
     * @param rot_6d [参数说明]：6D旋转表示张量，形状为[..., 6]。
     * @return [返回值说明]：3x3旋转矩阵，形状为[..., 3, 3]。
     */
    torch::Tensor rotation_6d_to_matrix(torch::Tensor rot_6d) {
        // 提取前3个元素作为第一个向量a1
        auto a1 = rot_6d.index({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3)});
        // 提取后3个元素作为第二个向量a2
        auto a2 = rot_6d.index({at::indexing::Ellipsis, at::indexing::Slice(3, at::indexing::None)});
        
        // 将a1归一化为单位向量b1（第一个基向量）
        auto b1 = F::normalize(a1, F::NormalizeFuncOptions().dim(-1));
        
        // 计算b2：从a2中减去在b1方向上的投影，然后归一化
        // 这确保了b2与b1正交
        auto b2 = a2 - (b1 * a2).sum(-1, true) * b1;
        b2 = F::normalize(b2, F::NormalizeFuncOptions().dim(-1));
        
        // 计算b3：b1和b2的叉积，确保右手坐标系
        auto b3 = torch::cross(b1, b2, -1);
        
        // 将三个正交基向量堆叠成3x3旋转矩阵
        return torch::stack({b1, b2, b3}, -2);
    }

    /**
     * [功能描述]：直接姿态优化模块的构造函数，初始化相机嵌入层和旋转单位矩阵。
     * @param number_of_cameras [参数说明]：相机数量，决定嵌入层的输入维度。
     */
    DirectPoseOptimizationModule::DirectPoseOptimizationModule(int number_of_cameras)
        : camera_embeddings(register_module("camera_embeddings",
                                            torch::nn::Embedding(number_of_cameras, 9))),  // 每个相机9维嵌入（3维平移+6维旋转）
          rot_identity(register_buffer(
              "rot_identity",
              torch::tensor({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}))) {  // 6D旋转的单位矩阵
        // 将嵌入层权重初始化为零
        torch::nn::init::zeros_(camera_embeddings->weight);
    }
    
    /**
     * [功能描述]：直接姿态优化模块的前向传播函数，直接学习每个相机的姿态参数。
     * 注意：仅支持1维批次大小。
     * @param camera_transforms [参数说明]：输入的相机变换矩阵，形状为[bs, 4, 4]。
     * @param embedding_ids [参数说明]：相机嵌入ID，用于查找对应的嵌入向量。
     * @return [返回值说明]：优化后的相机变换矩阵，形状为[bs, 4, 4]。
     */
    torch::Tensor DirectPoseOptimizationModule::forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids) {
        auto bs = camera_transforms.size(0);  // 获取批次大小
        
        // 通过嵌入ID查找对应的姿态参数
        auto delta_transformation = camera_embeddings(embedding_ids);
        
        // 提取平移部分：前3个元素
        auto delta_translation = delta_transformation.index({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3)});
        // 提取旋转部分：后6个元素
        auto delta_rotation = delta_transformation.index({at::indexing::Ellipsis, at::indexing::Slice(3, at::indexing::None)});
        
        // 将旋转参数与单位旋转相加，然后转换为旋转矩阵
        auto delta_rotation_matrix = rotation_6d_to_matrix(delta_rotation + rot_identity.expand({bs, -1}));

        // 创建4x4变换矩阵：从单位矩阵开始
        auto transform = torch::eye(4, camera_transforms.options()).repeat({bs, 1, 1});
        
        // 设置旋转部分：左上角3x3子矩阵
        transform.index_put_({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3), at::indexing::Slice(at::indexing::None, 3)},
                             delta_rotation_matrix);
        // 设置平移部分：最后一列的前3个元素
        transform.index_put_({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3), 3},
                             delta_translation);
        
        // 将增量变换与原始变换相乘，得到最终的变换矩阵
        return torch::matmul(camera_transforms, transform);
    }
    
    /**
     * [功能描述]：MLP姿态优化模块的构造函数，初始化相机嵌入层、旋转单位矩阵和多层感知机网络。
     * @param number_of_cameras [参数说明]：相机数量，决定嵌入层的输入维度。
     * @param width [参数说明]：MLP隐藏层的宽度（神经元数量）。
     * @param depth [参数说明]：MLP的深度（隐藏层数量）。
     */
    MLPPoseOptimizationModule::MLPPoseOptimizationModule(int number_of_cameras, int width, int depth) 
        : camera_embeddings(register_module("camera_embeddings",
                                            torch::nn::Embedding(number_of_cameras, width))),  // 每个相机width维嵌入
          rot_identity(register_buffer(
              "rot_identity",
              torch::tensor({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}))),  // 6D旋转的单位矩阵
          mlp(register_module("mlp", torch::nn::Sequential())) {  // 多层感知机网络
        
        // 将嵌入层权重初始化为零
        torch::nn::init::zeros_(camera_embeddings->weight);
        
        // 构建MLP网络：添加隐藏层
        for (int i = 0; i < depth; ++i) {
            mlp->push_back(torch::nn::Linear(width, width));  // 线性层
            mlp->push_back(torch::nn::ReLU());                // ReLU激活函数
        }
        
        // 添加输出层：将特征映射到9维输出（3维平移+6维旋转）
        auto last_layer = torch::nn::Linear(width, 9);
        torch::nn::init::zeros_(last_layer->weight);  // 权重初始化为零
        torch::nn::init::zeros_(last_layer->bias);    // 偏置初始化为零
        mlp->push_back(last_layer);
    }

    /**
     * [功能描述]：MLP姿态优化模块的前向传播函数，通过MLP网络学习相机姿态的非线性变换。
     * @param camera_transforms [参数说明]：输入的相机变换矩阵，形状为[bs, 4, 4]。
     * @param embedding_ids [参数说明]：相机嵌入ID，用于查找对应的嵌入向量。
     * @return [返回值说明]：通过MLP优化后的相机变换矩阵，形状为[bs, 4, 4]。
     */
    torch::Tensor MLPPoseOptimizationModule::forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids) {
        auto bs = camera_transforms.size(0);  // 获取批次大小
        
        // 通过嵌入ID查找对应的相机特征
        auto camera_embedding = camera_embeddings(embedding_ids);
        
        // 通过MLP网络计算姿态增量变换
        auto delta_transformation = mlp->forward(camera_embedding);
        
        // 提取平移部分：前3个元素
        auto delta_translation = delta_transformation.index({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3)});
        // 提取旋转部分：后6个元素
        auto delta_rotation = delta_transformation.index({at::indexing::Ellipsis, at::indexing::Slice(3, at::indexing::None)});
        
        // 将旋转参数与单位旋转相加，然后转换为旋转矩阵
        auto delta_rotation_matrix = rotation_6d_to_matrix(delta_rotation + rot_identity.expand({bs, -1}));
        
        // 创建4x4变换矩阵：从单位矩阵开始
        auto transform = torch::eye(4, camera_transforms.options()).repeat({bs, 1, 1});
        
        // 设置旋转部分：左上角3x3子矩阵
        transform.index_put_({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3), at::indexing::Slice(at::indexing::None, 3)},
                             delta_rotation_matrix);
        // 设置平移部分：最后一列的前3个元素
        transform.index_put_({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3), 3},
                             delta_translation);
        
        // 将增量变换与原始变换相乘，得到最终的变换矩阵
        return torch::matmul(camera_transforms, transform);
    }

} // namespace gs::training
