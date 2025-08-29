#include "fast_rasterizer.hpp"
#include "fast_rasterizer_autograd.hpp"

namespace gs::training {
    // 为PyTorch索引操作创建别名，简化代码
    using torch::indexing::None;
    using torch::indexing::Slice;

    /**
     * [功能描述]：快速光栅化函数，将3D高斯溅射模型渲染到2D图像平面。
     * 这是训练过程中的核心渲染函数，使用优化的光栅化算法快速生成图像和透明度通道。
     * @param viewpoint_camera [参数说明]：视点相机对象，包含相机内参、外参和图像尺寸信息。
     * @param gaussian_model [参数说明]：高斯溅射模型数据，包含位置、缩放、旋转、不透明度和球谐函数系数。
     * @param bg_color [参数说明]：背景颜色张量，用于背景混合（当前未使用）。
     * @return [返回值说明]：RenderOutput结构，包含渲染的图像和透明度通道。
     */
    RenderOutput fast_rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color) {
        
        // 步骤1：获取相机参数
        const int width = static_cast<int>(viewpoint_camera.image_width());   // 图像宽度
        const int height = static_cast<int>(viewpoint_camera.image_height()); // 图像高度
        
        // 获取相机内参：焦距和主点坐标
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();
        // fx, fy: x和y方向的焦距
        // cx, cy: 主点坐标（图像中心）

        // 步骤2：获取高斯模型的参数
        auto means = gaussian_model.means();           // 高斯中心位置 [N, 3]
        auto raw_opacities = gaussian_model.opacity_raw();     // 原始不透明度值 [N, 1]
        auto raw_scales = gaussian_model.scaling_raw();        // 原始缩放参数 [N, 3]
        auto raw_rotations = gaussian_model.rotation_raw();    // 原始旋转参数 [N, 4] (四元数)
        auto sh0 = gaussian_model.sh0();              // 球谐函数0阶系数 [N, 1]
        auto shN = gaussian_model.shN();              // 球谐函数高阶系数 [N, K]

        // 计算球谐函数的活跃基函数数量
        const int sh_degree = gaussian_model.get_active_sh_degree();  // 当前活跃的球谐函数阶数
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);  // 基函数数量 = (阶数+1)²

        // 步骤3：设置近平面和远平面参数
        constexpr float near_plane = 0.01f;   // 近裁剪平面距离，避免z-fighting
        constexpr float far_plane = 1e10f;    // 远裁剪平面距离，设置为很大的值

        // 步骤4：配置快速高斯溅射光栅化设置
        fast_gs::rasterization::FastGSSettings settings;
        
        // 获取世界坐标系到相机坐标系的变换矩阵
        auto w2c = viewpoint_camera.world_view_transform();
        
        // 设置光栅化参数
        settings.cam_position = viewpoint_camera.cam_position();  // 相机在世界坐标系中的位置
        settings.active_sh_bases = active_sh_bases;               // 活跃的球谐函数基函数数量
        settings.width = width;                                   // 图像宽度
        settings.height = height;                                 // 图像高度
        settings.focal_x = fx;                                    // x方向焦距
        settings.focal_y = fy;                                    // y方向焦距
        settings.center_x = cx;                                   // x方向主点坐标
        settings.center_y = cy;                                   // y方向主点坐标
        settings.near_plane = near_plane;                         // 近裁剪平面
        settings.far_plane = far_plane;                           // 远裁剪平面

        // 步骤5：执行快速高斯溅射光栅化
        auto raster_outputs = FastGSRasterize::apply(
            means,                           // 高斯中心位置
            raw_scales,                      // 缩放参数
            raw_rotations,                   // 旋转参数
            raw_opacities,                   // 不透明度
            sh0,                             // 球谐函数0阶系数
            shN,                             // 球谐函数高阶系数
            w2c,                             // 世界到相机的变换矩阵
            gaussian_model._densification_info,  // 密集化信息（用于动态调整高斯数量）
            settings);                        // 光栅化设置

        // 步骤6：构建输出结果
        RenderOutput output;
        output.image = raster_outputs[0];    // 第一个输出是渲染的图像
        output.alpha = raster_outputs[1];    // 第二个输出是透明度通道

        // 注意：背景颜色混合当前被注释掉
        // 如果启用背景混合，图像将包含背景，但透明度将变为全1
        // output.image = image + (1.0f - alpha) * bg_color.unsqueeze(-1).unsqueeze(-1);

        // TODO注释：如果背景颜色混合到图像中，结果图像的alpha值将处处为1
        // 这是因为混合后的图像已经包含了背景信息
        // output.alpha = torch::ones_like(alpha);
        
        return output;
    }
} // namespace gs::training
