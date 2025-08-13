#include "core/fast_rasterizer.hpp"
#include "core/fast_rasterizer_autograd.hpp"
#include <torch/torch.h>

/**
 * [文件描述]：快速高斯散点光栅化器实现文件
 * 功能：实现高性能的3D高斯散点到2D图像的渲染过程
 * 用途：为训练和推理提供高效的差分渲染支持
 */

namespace gs {

    // 使用PyTorch张量索引功能的简化别名
    using torch::indexing::None;    // 空索引标记
    using torch::indexing::Slice;   // 切片索引功能

    /**
     * [功能描述]：主要的快速光栅化渲染函数
     * @param viewpoint_camera：视点相机对象，包含相机的内外参数
     * @param gaussian_model：高斯散点模型数据，包含所有3D高斯的参数
     * @param bg_color：背景颜色张量，用于填充未被高斯覆盖的像素
     * @return RenderOutput结构体，包含渲染的图像和Alpha通道
     * 
     * 渲染流程：
     * 1. 提取相机参数（内参、图像尺寸等）
     * 2. 获取高斯模型的所有参数
     * 3. 配置光栅化设置
     * 4. 执行快速光栅化
     * 5. 构造并返回渲染结果
     */
    RenderOutput fast_rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color) {

        // =============================================================================
        // 步骤1：获取相机参数
        // =============================================================================
        
        // 获取图像分辨率
        const int width = static_cast<int>(viewpoint_camera.image_width());   // 图像宽度（像素）
        const int height = static_cast<int>(viewpoint_camera.image_height()); // 图像高度（像素）
        
        // 使用结构化绑定获取相机内参
        // fx, fy: X和Y轴焦距（像素单位）
        // cx, cy: 主点坐标（像素单位）
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // =============================================================================
        // 步骤2：获取高斯模型参数
        // =============================================================================
        
        // 提取所有高斯散点的几何和外观参数
        auto means = gaussian_model.means();                // 3D位置 [N, 3]
        auto raw_opacities = gaussian_model.opacity_raw();  // 原始不透明度 [N, 1]
        auto raw_scales = gaussian_model.scaling_raw();     // 原始缩放参数 [N, 3]
        auto raw_rotations = gaussian_model.rotation_raw(); // 原始旋转四元数 [N, 4]
        auto sh0 = gaussian_model.sh0();                    // 球谐0阶系数（直流颜色）[N, 3]
        auto shN = gaussian_model.shN();                    // 球谐高阶系数 [N, (degree+1)²-1, 3]

        // 获取当前激活的球谐函数阶数和基函数数量
        const int sh_degree = gaussian_model.get_active_sh_degree();     // 当前球谐阶数
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);  // 激活的球谐基函数数量

        // =============================================================================
        // 步骤3：渲染参数配置
        // =============================================================================
        
        // 定义近远裁剪平面
        constexpr float near_plane = 0.01f;  // 近裁剪平面：1厘米
        constexpr float far_plane = 1e10f;   // 远裁剪平面：极远距离

        // 密集化信息张量（当前未使用，预留给动态高斯管理）
        auto densification_info = torch::empty({0});

        // =============================================================================
        // 步骤4：配置快速光栅化设置
        // =============================================================================
        
        // 创建光栅化设置结构体，包含所有渲染参数
        fast_gs::rasterization::FastGSSettings settings;
        settings.w2c = viewpoint_camera.world_view_transform();  // 世界到相机变换矩阵 [1, 4, 4]
        settings.cam_position = viewpoint_camera.cam_position(); // 相机在世界坐标系中的位置 [3]
        settings.active_sh_bases = active_sh_bases;              // 激活的球谐基函数数量
        settings.width = width;                                  // 输出图像宽度
        settings.height = height;                                // 输出图像高度
        settings.focal_x = fx;                                   // X轴焦距
        settings.focal_y = fy;                                   // Y轴焦距
        settings.center_x = cx;                                  // 主点X坐标
        settings.center_y = cy;                                  // 主点Y坐标
        settings.near_plane = near_plane;                        // 近裁剪平面
        settings.far_plane = far_plane;                          // 远裁剪平面

        // =============================================================================
        // 步骤5：执行快速光栅化
        // =============================================================================
        
        // 调用FastGS自定义算子执行光栅化
        // 这是一个PyTorch自动微分函数，支持前向和反向传播
        auto raster_outputs = FastGSRasterize::apply(
            means,                  // 高斯中心位置
            raw_scales,            // 高斯缩放参数
            raw_rotations,         // 高斯旋转参数
            raw_opacities,         // 高斯不透明度
            sh0,                   // 球谐0阶系数
            shN,                   // 球谐高阶系数
            densification_info,    // 密集化信息（暂未使用）
            settings               // 渲染设置
        );

        // =============================================================================
        // 步骤6：构造渲染输出
        // =============================================================================
        
        RenderOutput output;

        // TODO注释说明：背景色处理优化
        // 当前背景色总是黑色，可以节省一些计算时间
        output.image = raster_outputs[0];  // 渲染的RGB图像 [3, H, W]
        output.alpha = raster_outputs[1];  // Alpha通道（透明度掩码）[1, H, W]
        
        // 原本的背景混合代码（已注释）：
        // output.image = image + (1.0f - alpha) * bg_color.unsqueeze(-1).unsqueeze(-1);
        // 这行代码会将背景色与前景混合：前景*alpha + 背景*(1-alpha)

        // TODO注释说明：Alpha通道处理
        // 如果背景色被混合到图像中，结果图像的所有位置alpha都等于1
        // output.alpha = torch::ones_like(alpha);
        // 这表明混合后的图像是完全不透明的

        return output;  // 返回渲染结果
    }

} // namespace gs - 高斯散点项目命名空间结束