/**
 * [文件描述]：双边网格滤波器实现文件
 * 功能：实现双边网格的前向传播、反向传播以及全变分正则化损失
 * 用途：用于高斯散点渲染中的图像后处理和色彩校正
 */

#include "core/bilateral_grid.hpp"      // 双边网格头文件
#include "kernels/bilateral_grid.cuh"   // CUDA内核函数声明

namespace gs {

    // =============================================================================
    // 双边网格切片自动微分函数类
    // =============================================================================
    
    /**
     * [类描述]：双边网格切片操作的PyTorch自动微分函数
     * 功能：实现双边网格对RGB图像的滤波处理，支持前向和反向传播
     * 继承：torch::autograd::Function，提供自动微分能力
     */
    class BilateralGridSliceFunction : public torch::autograd::Function<BilateralGridSliceFunction> {
    public:
        /**
         * [功能描述]：双边网格切片前向传播函数
         * @param ctx：自动微分上下文，用于保存反向传播需要的变量
         * @param grid：双边网格张量，形状为[12, L, H, W]，包含仿射变换参数
         * @param rgb：输入RGB图像张量，形状为[H, W, 3]
         * @return 处理后的RGB图像张量列表
         */
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor grid,
            torch::Tensor rgb) {
            
            // 输入参数验证
            TORCH_CHECK(grid.dim() == 4 && grid.size(0) == 12,
                        "Grid must be [12, L, H, W]");              // 验证网格维度和通道数
            TORCH_CHECK(rgb.dim() == 3 && rgb.size(2) == 3,
                        "RGB must be [H, W, 3]");                   // 验证RGB图像格式
            TORCH_CHECK(grid.is_cuda() && rgb.is_cuda(),
                        "Tensors must be on CUDA");                 // 确保张量在GPU上

            // 创建输出张量，形状与输入RGB相同
            auto output = torch::empty_like(rgb);

            // 调用CUDA内核执行双边网格切片操作
            bilateral_grid::slice_forward_cuda(
                grid.contiguous(),      // 确保网格内存连续
                rgb.contiguous(),       // 确保RGB内存连续
                output,                 // 输出张量
                true                    // 使用均匀坐标系
            );

            // 保存输入张量以供反向传播使用
            ctx->save_for_backward({grid, rgb});
            return {output};
        }

        /**
         * [功能描述]：双边网格切片反向传播函数
         * @param ctx：自动微分上下文，包含前向传播保存的变量
         * @param grad_outputs：输出梯度张量列表
         * @return 输入张量的梯度列表 [grad_grid, grad_rgb]
         */
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
            
            // 获取前向传播保存的变量
            auto saved = ctx->get_saved_variables();
            auto grid = saved[0];           // 双边网格
            auto rgb = saved[1];            // 输入RGB图像
            auto grad_output = grad_outputs[0];  // 输出梯度

            // 调用CUDA内核计算输入梯度
            auto [grad_grid, grad_rgb] = bilateral_grid::slice_backward_cuda(
                grid, rgb, grad_output.contiguous());

            return {grad_grid, grad_rgb};   // 返回网格和RGB的梯度
        }
    };

    // =============================================================================
    // 双边网格全变分损失自动微分函数类
    // =============================================================================
    
    /**
     * [类描述]：双边网格全变分（Total Variation）损失的自动微分函数
     * 功能：计算网格的空间平滑性损失，用于正则化训练
     * 用途：防止网格过拟合，保持滤波效果的空间连续性
     */
    class BilateralGridTVLossFunction : public torch::autograd::Function<BilateralGridTVLossFunction> {
    public:
        /**
         * [功能描述]：全变分损失前向传播函数
         * @param ctx：自动微分上下文
         * @param grids：双边网格张量，形状为[N, 12, L, H, W]
         * @return 全变分损失标量值
         */
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor grids) {
            
            // 保存网格张量以供反向传播使用
            ctx->save_for_backward({grids});
            
            // 调用CUDA内核计算全变分损失
            return bilateral_grid::tv_loss_forward_cuda(grids.contiguous());
        }

        /**
         * [功能描述]：全变分损失反向传播函数
         * @param ctx：自动微分上下文
         * @param grad_outputs：损失函数的梯度
         * @return 网格张量的梯度
         */
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
            
            // 获取保存的网格张量
            auto grids = ctx->get_saved_variables()[0];
            auto grad_output = grad_outputs[0];

            // 调用CUDA内核计算网格梯度
            auto grad_grids = bilateral_grid::tv_loss_backward_cuda(
                grids, grad_output);

            return {grad_grids};
        }
    };

    // =============================================================================
    // 双边网格类实现
    // =============================================================================
    
    /**
     * [功能描述]：双边网格构造函数
     * @param num_images：支持的图像数量
     * @param grid_W：网格宽度（空间分辨率）
     * @param grid_H：网格高度（空间分辨率）
     * @param grid_L：网格深度（颜色/亮度分辨率）
     */
    BilateralGrid::BilateralGrid(int num_images, int grid_W, int grid_H, int grid_L)
        : num_images_(num_images),      // 图像数量
        grid_width_(grid_W),          // 网格宽度
        grid_height_(grid_H),         // 网格高度
        grid_guidance_(grid_L) {      // 网格引导维度

        // 使用单位变换矩阵初始化网格
        // 创建4x4单位矩阵的前3行，用于仿射变换
        auto eye = torch::eye(4, torch::kFloat32).slice(0, 0, 3);  // [3, 4]矩阵
        
        // 将单位矩阵复制到所有网格点
        auto grid = eye.repeat({grid_L * grid_H * grid_W, 1});     // [L*H*W, 12]
        
        // 重塑为5D张量：[1, L, H, W, 12]
        grid = grid.reshape({1, grid_L, grid_H, grid_W, 12});
        
        // 调整维度顺序为：[1, 12, L, H, W]（通道优先）
        grid = grid.permute({0, 4, 1, 2, 3});

        // 为所有图像复制网格并移动到GPU
        grids_ = grid.repeat({num_images, 1, 1, 1, 1}).to(torch::kCUDA);
        grids_.set_requires_grad(true);  // 启用梯度计算
    }

    /**
     * [功能描述]：对指定图像应用双边网格滤波
     * @param rgb：输入RGB图像，支持[C, H, W]或[1, C, H, W]格式
     * @param image_idx：图像索引，指定使用哪个网格
     * @return 滤波后的RGB图像，格式与输入相同
     */
    torch::Tensor BilateralGrid::apply(const torch::Tensor& rgb, int image_idx) {
        // 验证图像索引有效性
        TORCH_CHECK(image_idx >= 0 && image_idx < num_images_,
                    "Invalid image index: ", image_idx);

        // 处理不同的输入格式
        torch::Tensor rgb_processed;
        if (rgb.dim() == 4 && rgb.size(0) == 1) {
            // 输入格式：[1, C, H, W] - 移除批次维度
            rgb_processed = rgb.squeeze(0);     // 变为[C, H, W]
        } else if (rgb.dim() == 3) {
            // 输入已经是[C, H, W]格式
            rgb_processed = rgb;
        } else {
            // 不支持的格式，抛出错误
            TORCH_CHECK(false, "RGB must be [C, H, W] or [1, C, H, W], got ", rgb.sizes());
        }

        // 将像素值限制在[0, 1]范围内
        rgb_processed = torch::clamp(rgb_processed, 0, 1);
        
        // 转换维度顺序：从[C, H, W]到[H, W, C]（CUDA内核期望的格式）
        auto rgb_hwc = rgb_processed.permute({1, 2, 0}).contiguous();

        // 获取指定图像的双边网格
        auto grid = grids_[image_idx];
        
        // 应用双边网格滤波
        auto output = BilateralGridSliceFunction::apply(grid, rgb_hwc)[0];

        // 转换回原始维度顺序：从[H, W, C]到[C, H, W]
        auto result = output.permute({2, 0, 1}).contiguous();

        // 如果输入有批次维度，则恢复批次维度
        if (rgb.dim() == 4) {
            result = result.unsqueeze(0);       // 变为[1, C, H, W]
        }

        return result;
    }

    /**
     * [功能描述]：计算双边网格的全变分正则化损失
     * @return 全变分损失标量值，用于训练时的正则化
     * 作用：鼓励网格参数的空间平滑性，防止过拟合
     */
    torch::Tensor BilateralGrid::tv_loss() const {
        return BilateralGridTVLossFunction::apply(grids_);
    }

} // namespace gs - 高斯散点项目命名空间结束