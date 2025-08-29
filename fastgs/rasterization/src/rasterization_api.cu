#include "backward.h"
#include "forward.h"
#include "helper_math.h"
#include "rasterization_api.h"
#include "rasterization_config.h"
#include "torch_utils.h"
#include <functional>
#include <stdexcept>
#include <tuple>

/**
 * [功能描述]：快速高斯溅射光栅化的前向传播包装函数。
 * 这个函数是PyTorch张量和底层CUDA光栅化内核之间的桥梁，
 * 负责输入验证、输出张量创建、缓冲区管理和核心光栅化函数的调用。
 * 
 * 返回值是一个包含11个元素的元组，包含渲染结果和各种缓冲区信息。
 * 
 * @param means [参数说明]：高斯中心位置张量，形状为[N, 3]，N是高斯数量。
 * @param scales_raw [参数说明]：原始缩放参数张量，形状为[N, 3]。
 * @param rotations_raw [参数说明]：原始旋转参数张量，形状为[N, 4]，使用四元数表示。
 * @param opacities_raw [参数说明]：原始不透明度张量，形状为[N, 1]。
 * @param sh_coefficients_0 [参数说明]：球谐函数0阶系数，形状为[N, 1, 3]。
 * @param sh_coefficients_rest [参数说明]：球谐函数高阶系数，形状为[N, K, 3]，K是基函数数量。
 * @param w2c [参数说明]：世界坐标系到相机坐标系的变换矩阵，形状为[4, 4]。
 * @param cam_position [参数说明]：相机在世界坐标系中的位置，形状为[3]。
 * @param active_sh_bases [参数说明]：当前活跃的球谐函数基函数数量。
 * @param width [参数说明]：输出图像的宽度。
 * @param height [参数说明]：输出图像的高度。
 * @param focal_x [参数说明]：x方向焦距。
 * @param focal_y [参数说明]：y方向焦距。
 * @param center_x [参数说明]：x方向主点坐标。
 * @param center_y [参数说明]：y方向主点坐标。
 * @param near_plane [参数说明]：近裁剪平面距离。
 * @param far_plane [参数说明]：远裁剪平面距离。
 * @return [返回值说明]：返回包含11个元素的元组，详见函数末尾的返回语句。
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int>
fast_gs::rasterization::forward_wrapper(
    const torch::Tensor& means,                    // 高斯中心位置
    const torch::Tensor& scales_raw,               // 原始缩放参数
    const torch::Tensor& rotations_raw,            // 原始旋转参数
    const torch::Tensor& opacities_raw,            // 原始不透明度
    const torch::Tensor& sh_coefficients_0,        // 球谐函数0阶系数
    const torch::Tensor& sh_coefficients_rest,     // 球谐函数高阶系数
    const torch::Tensor& w2c,                      // 世界到相机的变换矩阵
    const torch::Tensor& cam_position,             // 相机位置
    const int active_sh_bases,                     // 活跃的球谐函数基函数数量
    const int width,                               // 图像宽度
    const int height,                              // 图像高度
    const float focal_x,                           // x方向焦距
    const float focal_y,                           // y方向焦距
    const float center_x,                          // x方向主点坐标
    const float center_y,                          // y方向主点坐标
    const float near_plane,                        // 近裁剪平面
    const float far_plane) {                       // 远裁剪平面
    
    // 步骤1：输入张量验证
    // 检查所有可优化张量必须是连续的CUDA浮点张量
    // 这些检查确保输入数据的格式和类型正确，避免运行时错误
    CHECK_INPUT(config::debug, means, "means");                    // 检查高斯中心位置
    CHECK_INPUT(config::debug, scales_raw, "scales_raw");          // 检查缩放参数
    CHECK_INPUT(config::debug, rotations_raw, "rotations_raw");    // 检查旋转参数
    CHECK_INPUT(config::debug, opacities_raw, "opacities_raw");    // 检查不透明度
    CHECK_INPUT(config::debug, sh_coefficients_0, "sh_coefficients_0");      // 检查0阶球谐系数
    CHECK_INPUT(config::debug, sh_coefficients_rest, "sh_coefficients_rest"); // 检查高阶球谐系数

    // 步骤2：提取关键参数和设置张量选项
    const int n_primitives = means.size(0);                        // 高斯图元数量
    const int total_bases_sh_rest = sh_coefficients_rest.size(1);  // 高阶球谐函数的基函数总数
    
    // 创建CUDA浮点张量选项，用于输出张量
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    // 创建CUDA字节张量选项，用于缓冲区张量
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    
    // 步骤3：创建输出张量
    // 图像张量：3通道RGB，高度×宽度
    torch::Tensor image = torch::empty({3, height, width}, float_options);
    // 透明度张量：1通道，高度×宽度
    torch::Tensor alpha = torch::empty({1, height, width}, float_options);
    
    // 步骤4：创建缓冲区张量（初始为空）
    // 每个图元的缓冲区：存储图元级别的信息
    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    // 每个瓦片的缓冲区：存储瓦片级别的信息
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    // 每个实例的缓冲区：存储实例级别的信息
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    // 每个桶的缓冲区：存储桶级别的信息
    torch::Tensor per_bucket_buffers = torch::empty({0}, byte_options);
    
    // 步骤5：创建缓冲区调整函数
    // 这些函数用于动态调整缓冲区大小，支持光栅化过程中的内存管理
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);
    const std::function<char*(size_t)> per_bucket_buffers_func = resize_function_wrapper(per_bucket_buffers);

    // 步骤6：调用核心光栅化函数
    // 使用结构化绑定获取返回的多个值
    auto [n_visible_primitives, n_instances, n_buckets, primitive_primitive_indices_selector, instance_primitive_indices_selector] = forward(
        // 缓冲区调整函数
        per_primitive_buffers_func,      // 图元缓冲区调整函数
        per_tile_buffers_func,           // 瓦片缓冲区调整函数
        per_instance_buffers_func,       // 实例缓冲区调整函数
        per_bucket_buffers_func,         // 桶缓冲区调整函数
        
        // 高斯参数数据指针（转换为CUDA类型）
        reinterpret_cast<float3*>(means.data_ptr<float>()),                    // 高斯中心位置，转换为float3指针
        reinterpret_cast<float3*>(scales_raw.data_ptr<float>()),               // 缩放参数，转换为float3指针
        reinterpret_cast<float4*>(rotations_raw.data_ptr<float>()),            // 旋转参数，转换为float4指针
        opacities_raw.data_ptr<float>(),                                       // 不透明度，保持float指针
        
        // 球谐函数系数数据指针
        reinterpret_cast<float3*>(sh_coefficients_0.data_ptr<float>()),        // 0阶系数，转换为float3指针
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),     // 高阶系数，转换为float3指针
        
        // 相机参数数据指针
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),         // 变换矩阵，确保连续并转换为float4指针
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()), // 相机位置，确保连续并转换为float3指针
        
        // 输出缓冲区指针
        image.data_ptr<float>(),                                               // 图像输出缓冲区
        alpha.data_ptr<float>(),                                               // 透明度输出缓冲区
        
        // 数量和配置参数
        n_primitives,                                                          // 图元总数
        active_sh_bases,                                                       // 活跃的球谐函数基函数数量
        total_bases_sh_rest,                                                   // 高阶球谐函数的基函数总数
        width,                                                                 // 图像宽度
        height,                                                                // 图像高度
        focal_x,                                                               // x方向焦距
        focal_y,                                                               // y方向焦距
        center_x,                                                              // x方向主点坐标
        center_y,                                                              // y方向主点坐标
        near_plane,                                                            // 近裁剪平面
        far_plane);                                                            // 远裁剪平面

    // 步骤7：返回结果元组
    // 包含所有输出张量、缓冲区和统计信息
    return {
        image, alpha,                                                          // 渲染结果：图像和透明度
        per_primitive_buffers, per_tile_buffers, per_instance_buffers, per_bucket_buffers,  // 各种缓冲区
        n_visible_primitives, n_instances, n_buckets,                          // 统计信息：可见图元数、实例数、桶数
        primitive_primitive_indices_selector, instance_primitive_indices_selector};  // 索引选择器
}

/**
 * [功能描述]：快速高斯溅射光栅化的反向传播包装函数。
 * 这个函数负责计算所有高斯参数的梯度，用于训练过程中的参数更新。
 * 它是前向传播的逆过程，通过链式法则计算损失函数对各个参数的偏导数。
 * 
 * 返回值是一个包含7个梯度张量的元组，对应所有可训练的高斯参数。
 * 
 * @param densification_info [参数说明]：密集化信息张量，用于动态调整高斯数量，可修改。
 * @param grad_image [参数说明]：图像梯度的输入张量，来自损失函数对渲染图像的偏导数。
 * @param grad_alpha [参数说明]：透明度梯度的输入张量，来自损失函数对透明度的偏导数。
 * @param image [参数说明]：前向传播生成的图像张量，用于梯度计算。
 * @param alpha [参数说明]：前向传播生成的透明度张量，用于梯度计算。
 * @param means [参数说明]：高斯中心位置张量，形状为[N, 3]，N是高斯数量。
 * @param scales_raw [参数说明]：原始缩放参数张量，形状为[N, 3]。
 * @param rotations_raw [参数说明]：原始旋转参数张量，形状为[N, 4]，使用四元数表示。
 * @param sh_coefficients_rest [参数说明]：球谐函数高阶系数，形状为[N, K, 3]，K是基函数数量。
 * @param per_primitive_buffers [参数说明]：图元级别的缓冲区张量，包含前向传播的中间结果。
 * @param per_tile_buffers [参数说明]：瓦片级别的缓冲区张量。
 * @param per_instance_buffers [参数说明]：实例级别的缓冲区张量。
 * @param per_bucket_buffers [参数说明]：桶级别的缓冲区张量。
 * @param w2c [参数说明]：世界坐标系到相机坐标系的变换矩阵，形状为[4, 4]。
 * @param cam_position [参数说明]：相机在世界坐标系中的位置，形状为[3]。
 * @param active_sh_bases [参数说明]：当前活跃的球谐函数基函数数量。
 * @param width [参数说明]：图像宽度。
 * @param height [参数说明]：图像高度。
 * @param focal_x [参数说明]：x方向焦距。
 * @param focal_y [参数说明]：y方向焦距。
 * @param center_x [参数说明]：x方向主点坐标。
 * @param center_y [参数说明]：y方向主点坐标。
 * @param near_plane [参数说明]：近裁剪平面距离。
 * @param far_plane [参数说明]：远裁剪平面距离。
 * @param n_visible_primitives [参数说明]：可见图元的数量。
 * @param n_instances [参数说明]：实例的数量。
 * @param n_buckets [参数说明]：桶的数量。
 * @param primitive_primitive_indices_selector [参数说明]：图元索引选择器。
 * @param instance_primitive_indices_selector [参数说明]：实例索引选择器。
 * @return [返回值说明]：返回包含7个梯度张量的元组，详见函数末尾的返回语句。
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fast_gs::rasterization::backward_wrapper(
    torch::Tensor& densification_info,              // 密集化信息（可修改）
    const torch::Tensor& grad_image,                // 图像梯度输入
    const torch::Tensor& grad_alpha,                // 透明度梯度输入
    const torch::Tensor& image,                     // 前向传播的图像输出
    const torch::Tensor& alpha,                     // 前向传播的透明度输出
    const torch::Tensor& means,                     // 高斯中心位置
    const torch::Tensor& scales_raw,                // 原始缩放参数
    const torch::Tensor& rotations_raw,             // 原始旋转参数
    const torch::Tensor& sh_coefficients_rest,      // 球谐函数高阶系数
    const torch::Tensor& per_primitive_buffers,     // 图元缓冲区
    const torch::Tensor& per_tile_buffers,          // 瓦片缓冲区
    const torch::Tensor& per_instance_buffers,      // 实例缓冲区
    const torch::Tensor& per_bucket_buffers,        // 桶缓冲区
    const torch::Tensor& w2c,                       // 世界到相机的变换矩阵
    const torch::Tensor& cam_position,              // 相机位置
    const int active_sh_bases,                      // 活跃的球谐函数基函数数量
    const int width,                                // 图像宽度
    const int height,                               // 图像高度
    const float focal_x,                            // x方向焦距
    const float focal_y,                            // y方向焦距
    const float center_x,                           // x方向主点坐标
    const float center_y,                           // y方向主点坐标
    const float near_plane,                         // 近裁剪平面
    const float far_plane,                          // 远裁剪平面
    const int n_visible_primitives,                 // 可见图元数量
    const int n_instances,                          // 实例数量
    const int n_buckets,                            // 桶数量
    const int primitive_primitive_indices_selector, // 图元索引选择器
    const int instance_primitive_indices_selector) { // 实例索引选择器
    
    // 步骤1：提取关键参数和设置张量选项
    const int n_primitives = means.size(0);                        // 高斯图元总数
    const int total_bases_sh_rest = sh_coefficients_rest.size(1);  // 高阶球谐函数的基函数总数
    
    // 创建CUDA浮点张量选项，用于梯度张量
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    
    // 步骤2：创建梯度输出张量（初始化为零）
    // 所有梯度张量都初始化为零，因为梯度是累加的
    torch::Tensor grad_means = torch::zeros({n_primitives, 3}, float_options);                    // 位置梯度：[N, 3]
    torch::Tensor grad_scales_raw = torch::zeros({n_primitives, 3}, float_options);               // 缩放梯度：[N, 3]
    torch::Tensor grad_rotations_raw = torch::zeros({n_primitives, 4}, float_options);            // 旋转梯度：[N, 4]
    torch::Tensor grad_opacities_raw = torch::zeros({n_primitives, 1}, float_options);            // 不透明度梯度：[N, 1]
    torch::Tensor grad_sh_coefficients_0 = torch::zeros({n_primitives, 1, 3}, float_options);     // 0阶球谐系数梯度：[N, 1, 3]
    torch::Tensor grad_sh_coefficients_rest = torch::zeros({n_primitives, total_bases_sh_rest, 3}, float_options); // 高阶球谐系数梯度：[N, K, 3]
    
    // 步骤3：创建辅助梯度张量
    // 这些张量用于存储中间计算结果的梯度
    torch::Tensor grad_mean2d_helper = torch::zeros({n_primitives, 2}, float_options);            // 2D投影位置梯度：[N, 2]
    torch::Tensor grad_conic_helper = torch::zeros({n_primitives, 3}, float_options);             // 圆锥体参数梯度：[N, 3]
    
    // 步骤4：条件性创建w2c梯度张量
    // 只有当w2c需要梯度时才创建对应的梯度张量
    torch::Tensor grad_w2c = torch::Tensor();  // 初始化为空张量
    if (w2c.requires_grad()) {
        // 如果w2c需要梯度，创建与w2c形状相同的零张量
        grad_w2c = torch::zeros_like(w2c, float_options);
    }

    // 步骤5：检查密集化信息更新需求
    // 根据densification_info的大小判断是否需要更新密集化信息
    const bool update_densification_info = densification_info.size(0) > 0;

    // 步骤6：调用核心反向传播函数
    // 这个函数计算所有参数的梯度并填充到梯度张量中
    backward(
        // 输入梯度数据指针
        grad_image.data_ptr<float>(),                               // 图像梯度指针
        grad_alpha.data_ptr<float>(),                               // 透明度梯度指针
        
        // 前向传播输出数据指针（用于梯度计算）
        image.data_ptr<float>(),                                    // 图像输出指针
        alpha.data_ptr<float>(),                                    // 透明度输出指针
        
        // 高斯参数数据指针（转换为CUDA类型）
        reinterpret_cast<float3*>(means.data_ptr<float>()),         // 高斯中心位置，转换为float3指针
        reinterpret_cast<float3*>(scales_raw.data_ptr<float>()),    // 缩放参数，转换为float3指针
        reinterpret_cast<float4*>(rotations_raw.data_ptr<float>()), // 旋转参数，转换为float4指针
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()), // 高阶球谐系数，转换为float3指针
        
        // 相机参数数据指针
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),     // 变换矩阵，确保连续并转换为float4指针
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()), // 相机位置，确保连续并转换为float3指针
        
        // 缓冲区数据指针（转换为char指针）
        reinterpret_cast<char*>(per_primitive_buffers.data_ptr()),   // 图元缓冲区指针
        reinterpret_cast<char*>(per_tile_buffers.data_ptr()),        // 瓦片缓冲区指针
        reinterpret_cast<char*>(per_instance_buffers.data_ptr()),    // 实例缓冲区指针
        reinterpret_cast<char*>(per_bucket_buffers.data_ptr()),      // 桶缓冲区指针
        
        // 梯度输出缓冲区指针（转换为CUDA类型）
        reinterpret_cast<float3*>(grad_means.data_ptr<float>()),           // 位置梯度输出指针
        reinterpret_cast<float3*>(grad_scales_raw.data_ptr<float>()),      // 缩放梯度输出指针
        reinterpret_cast<float4*>(grad_rotations_raw.data_ptr<float>()),   // 旋转梯度输出指针
        reinterpret_cast<float*>(grad_opacities_raw.data_ptr<float>()),    // 不透明度梯度输出指针
        reinterpret_cast<float3*>(grad_sh_coefficients_0.data_ptr<float>()),     // 0阶球谐系数梯度输出指针
        reinterpret_cast<float3*>(grad_sh_coefficients_rest.data_ptr<float>()),  // 高阶球谐系数梯度输出指针
        reinterpret_cast<float2*>(grad_mean2d_helper.data_ptr<float>()),         // 2D投影位置梯度输出指针
        grad_conic_helper.data_ptr<float>(),                                     // 圆锥体参数梯度输出指针
        
        // 条件性梯度指针
        w2c.requires_grad() ? reinterpret_cast<float4*>(grad_w2c.data_ptr<float>()) : nullptr,  // w2c梯度指针（如果需要）
        update_densification_info ? densification_info.data_ptr<float>() : nullptr,              // 密集化信息指针（如果需要更新）
        
        // 数量和配置参数
        n_primitives,                                                          // 图元总数
        n_visible_primitives,                                                  // 可见图元数量
        n_instances,                                                           // 实例数量
        n_buckets,                                                             // 桶数量
        primitive_primitive_indices_selector,                                  // 图元索引选择器
        instance_primitive_indices_selector,                                   // 实例索引选择器
        active_sh_bases,                                                       // 活跃的球谐函数基函数数量
        total_bases_sh_rest,                                                   // 高阶球谐函数的基函数总数
        width,                                                                 // 图像宽度
        height,                                                                // 图像高度
        focal_x,                                                               // x方向焦距
        focal_y,                                                               // y方向焦距
        center_x,                                                              // x方向主点坐标
        center_y);                                                             // y方向主点坐标

    // 步骤7：返回所有梯度张量
    // 返回包含7个梯度张量的元组，用于后续的参数更新
    return {
        grad_means,                    // 位置梯度：[N, 3]
        grad_scales_raw,               // 缩放梯度：[N, 3]
        grad_rotations_raw,            // 旋转梯度：[N, 4]
        grad_opacities_raw,            // 不透明度梯度：[N, 1]
        grad_sh_coefficients_0,        // 0阶球谐系数梯度：[N, 1, 3]
        grad_sh_coefficients_rest,     // 高阶球谐系数梯度：[N, K, 3]
        grad_w2c                       // w2c梯度（如果不需要则为空张量）
    };
}
