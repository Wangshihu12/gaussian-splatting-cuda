#include "bilateral_grid.hpp"           // 双边网格类的头文件声明
#include "kernels/bilateral_grid.cuh"   // CUDA内核函数，提供高效的GPU实现

namespace gs::training {

    /**
     * [功能描述]：双边网格切片函数的自动微分类。
     * 这个类继承自PyTorch的autograd::Function，实现了双边网格切片操作的前向和反向传播。
     * 双边网格切片是将网格参数根据输入图像的空间位置和颜色值进行采样的过程。
     */
    class BilateralGridSliceFunction : public torch::autograd::Function<BilateralGridSliceFunction> {
    public:
        /**
         * [功能描述]：前向传播函数，执行双边网格切片操作。
         * 根据输入图像的空间位置和颜色值，从双边网格中采样相应的变换参数，
         * 然后将这些参数应用到图像上实现后处理效果。
         * 
         * @param ctx [参数说明]：PyTorch自动微分上下文，用于管理前向和反向传播的状态。
         * @param grid [参数说明]：双边网格参数，形状为[12, L, H, W]，包含12个变换参数。
         * @param rgb [参数说明]：输入RGB图像，形状为[H, W, 3]，H和W是图像高度和宽度。
         * @return [返回值说明]：返回张量列表，包含处理后的图像。
         */
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor grid,
            torch::Tensor rgb) {
            
            // 输入验证：检查网格张量的维度和形状
            TORCH_CHECK(grid.dim() == 4 && grid.size(0) == 12,
                        "Grid must be [12, L, H, W]");
            // 检查RGB图像的维度和通道数
            TORCH_CHECK(rgb.dim() == 3 && rgb.size(2) == 3,
                        "RGB must be [H, W, 3]");
            // 确保张量在CUDA设备上
            TORCH_CHECK(grid.is_cuda() && rgb.is_cuda(),
                        "Tensors must be on CUDA");

            // 创建与输入RGB图像相同形状的输出张量
            auto output = torch::empty_like(rgb);

            // 调用CUDA内核函数执行双边网格切片操作
            bilateral_grid::slice_forward_cuda(
                grid.contiguous(),        // 确保网格张量内存连续
                rgb.contiguous(),         // 确保RGB张量内存连续
                output,                   // 输出张量
                true                      // 使用均匀坐标系统
            );

            // 保存前向传播中的张量，供反向传播使用
            ctx->save_for_backward({grid, rgb});
            return {output};
        }

        /**
         * [功能描述]：反向传播函数，计算双边网格切片操作的梯度。
         * 使用链式法则计算损失函数对网格参数和输入图像的梯度。
         * 
         * @param ctx [参数说明]：PyTorch自动微分上下文，包含前向传播中保存的信息。
         * @param grad_outputs [参数说明]：输出梯度张量列表。
         * @return [返回值说明]：返回网格参数和RGB图像的梯度。
         */
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
            
            // 从上下文中恢复前向传播中保存的张量
            auto saved = ctx->get_saved_variables();
            auto grid = saved[0];        // 网格参数
            auto rgb = saved[1];         // 输入RGB图像
            auto grad_output = grad_outputs[0];  // 输出梯度

            // 调用CUDA内核函数计算梯度
            auto [grad_grid, grad_rgb] = bilateral_grid::slice_backward_cuda(
                grid, rgb, grad_output.contiguous());

            // 返回网格参数和RGB图像的梯度
            return {grad_grid, grad_rgb};
        }
    };

    /**
     * [功能描述]：双边网格总变分损失的自动微分类。
     * 这个类实现了总变分损失的前向和反向传播，用于正则化网格参数，
     * 鼓励参数在空间上平滑变化，避免过度拟合和噪声。
     */
    class BilateralGridTVLossFunction : public torch::autograd::Function<BilateralGridTVLossFunction> {
    public:
        /**
         * [功能描述]：前向传播函数，计算总变分损失。
         * 总变分损失衡量网格参数在空间上的变化程度，鼓励平滑性。
         * 
         * @param ctx [参数说明]：PyTorch自动微分上下文。
         * @param grids [参数说明]：双边网格参数张量。
         * @return [返回值说明]：总变分损失值，是一个标量张量。
         */
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor grids) {
            
            // 保存网格参数供反向传播使用
            ctx->save_for_backward({grids});
            // 调用CUDA内核函数计算总变分损失
            return bilateral_grid::tv_loss_forward_cuda(grids.contiguous());
        }

        /**
         * [功能描述]：反向传播函数，计算总变分损失对网格参数的梯度。
         * 
         * @param ctx [参数说明]：PyTorch自动微分上下文。
         * @param grad_outputs [参数说明]：输出梯度张量列表。
         * @return [返回值说明]：返回网格参数的梯度。
         */
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
            
            // 从上下文中恢复网格参数
            auto grids = ctx->get_saved_variables()[0];
            auto grad_output = grad_outputs[0];

            // 调用CUDA内核函数计算梯度
            auto grad_grids = bilateral_grid::tv_loss_backward_cuda(
                grids, grad_output);

            return {grad_grids};
        }
    };

    /**
     * [功能描述]：双边网格类的构造函数实现。
     * 初始化双边网格，设置网格尺寸，并将网格参数初始化为单位变换。
     * 
     * @param num_images [参数说明]：图像数量，决定网格的第一个维度。
     * @param grid_W [参数说明]：网格宽度，在水平方向上的划分数量。
     * @param grid_H [参数说明]：网格高度，在垂直方向上的划分数量。
     * @param grid_L [参数说明]：网格引导维度，在颜色/亮度引导方向上的划分数量。
     */
    BilateralGrid::BilateralGrid(int num_images, int grid_W, int grid_H, int grid_L)
        : num_images_(num_images),       // 初始化图像数量
          grid_width_(grid_W),           // 初始化网格宽度
          grid_height_(grid_H),          // 初始化网格高度
          grid_guidance_(grid_L) {       // 初始化网格引导维度

        // 步骤1：创建单位变换矩阵作为基础
        auto eye = torch::eye(4, torch::kFloat32).slice(0, 0, 3);  // 取4x4单位矩阵的前3行
        // 步骤2：将单位变换重复到所有网格点
        auto grid = eye.repeat({grid_L * grid_H * grid_W, 1});      // 重复到所有网格点
        // 步骤3：重塑为5维张量 [1, L, H, W, 12]
        grid = grid.reshape({1, grid_L, grid_H, grid_W, 12});
        // 步骤4：调整维度顺序为 [1, 12, L, H, W]，符合存储格式
        grid = grid.permute({0, 4, 1, 2, 3});

        // 步骤5：为所有图像复制网格参数，并移动到CUDA设备
        grids_ = grid.repeat({num_images, 1, 1, 1, 1}).to(torch::kCUDA);
        // 步骤6：启用梯度计算，使网格参数可训练
        grids_.set_requires_grad(true);
    }

    /**
     * [功能描述]：将双边网格应用到指定的RGB图像上，实现图像后处理。
     * 这个函数处理不同的输入格式，应用双边网格变换，并返回处理后的图像。
     * 
     * @param rgb [参数说明]：输入RGB图像，支持[C, H, W]或[1, C, H, W]格式。
     * @param image_idx [参数说明]：图像索引，用于选择对应的网格参数。
     * @return [返回值说明]：处理后的图像，保持与输入相同的格式。
     */
    torch::Tensor BilateralGrid::apply(const torch::Tensor& rgb, int image_idx) {
        
        // 验证图像索引的有效性
        TORCH_CHECK(image_idx >= 0 && image_idx < num_images_,
                    "Invalid image index: ", image_idx);

        // 处理不同的输入格式
        torch::Tensor rgb_processed;
        if (rgb.dim() == 4 && rgb.size(0) == 1) {
            // 输入是[1, C, H, W]格式 - 压缩批次维度
            rgb_processed = rgb.squeeze(0); // 现在变成[C, H, W]
        } else if (rgb.dim() == 3) {
            // 输入已经是[C, H, W]格式
            rgb_processed = rgb;
        } else {
            // 输入格式无效，抛出错误
            TORCH_CHECK(false, "RGB must be [C, H, W] or [1, C, H, W], got ", rgb.sizes());
        }

        // 将像素值限制在[0, 1]范围内，确保输入有效
        rgb_processed = torch::clamp(rgb_processed, 0, 1);
        
        // 将图像从[C, H, W]格式转换为[H, W, C]格式，符合CUDA内核的期望
        auto rgb_hwc = rgb_processed.permute({1, 2, 0}).contiguous();

        // 应用双边网格：选择对应图像的网格参数
        auto grid = grids_[image_idx];
        // 调用切片函数应用网格变换
        auto output = BilateralGridSliceFunction::apply(grid, rgb_hwc)[0];

        // 将输出从[H, W, C]格式转换回[C, H, W]格式
        auto result = output.permute({2, 0, 1}).contiguous();

        // 如果输入有批次维度，将其添加回来
        if (rgb.dim() == 4) {
            result = result.unsqueeze(0);
        }

        return result;
    }

    /**
     * [功能描述]：计算双边网格的总变分损失。
     * 总变分损失用于正则化网格参数，鼓励参数在空间上平滑变化。
     * 
     * @return [返回值说明]：总变分损失值，是一个标量张量。
     */
    torch::Tensor BilateralGrid::tv_loss() const {
        // 调用总变分损失函数计算损失值
        return BilateralGridTVLossFunction::apply(grids_);
    }

} // namespace gs::training