#include "rasterizer.hpp"
#include "Ops.h"
#include "rasterizer_autograd.hpp"
#include <torch/torch.h>

namespace gs::training {
    using torch::indexing::None;
    using torch::indexing::Slice;

    inline torch::Tensor spherical_harmonics(
        int sh_degree,
        const torch::Tensor& dirs,
        const torch::Tensor& coeffs,
        const torch::Tensor& masks = {}) {
        // Validate inputs
        TORCH_CHECK((sh_degree + 1) * (sh_degree + 1) <= coeffs.size(-2),
                    "coeffs K dimension must be at least ", (sh_degree + 1) * (sh_degree + 1),
                    ", got ", coeffs.size(-2));
        TORCH_CHECK(dirs.sizes().slice(0, dirs.dim() - 1) == coeffs.sizes().slice(0, coeffs.dim() - 2),
                    "dirs and coeffs batch dimensions must match");
        TORCH_CHECK(dirs.size(-1) == 3, "dirs last dimension must be 3, got ", dirs.size(-1));
        TORCH_CHECK(coeffs.size(-1) == 3, "coeffs last dimension must be 3, got ", coeffs.size(-1));

        if (masks.defined()) {
            TORCH_CHECK(masks.sizes() == dirs.sizes().slice(0, dirs.dim() - 1),
                        "masks shape must match dirs shape without last dimension");
        }

        // Create sh_degree tensor
        auto sh_degree_tensor = torch::tensor({sh_degree},
                                              torch::TensorOptions().dtype(torch::kInt32).device(dirs.device()));

        // Call the autograd function
        return SphericalHarmonicsFunction::apply(
            sh_degree_tensor,
            dirs.contiguous(),
            coeffs.contiguous(),
            masks.defined() ? masks.contiguous() : masks)[0];
    }

    /**
     * [功能描述]：主要的渲染函数，实现3D高斯溅射模型到2D图像的光栅化过程。
     * 这是完整的渲染管线，包括投影、颜色计算、光栅化和后处理等步骤。
     * @param viewpoint_camera [参数说明]：视点相机对象，包含相机内参、外参和图像尺寸信息。
     * @param gaussian_model [参数说明]：高斯溅射模型数据，包含位置、缩放、旋转、不透明度和球谐函数系数。
     * @param bg_color [参数说明]：背景颜色张量，用于背景混合。
     * @param scaling_modifier [参数说明]：缩放修改器，用于调整高斯的大小。
     * @param packed [参数说明]：是否使用打包模式（当前实现不支持）。
     * @param antialiased [参数说明]：是否启用抗锯齿。
     * @param render_mode [参数说明]：渲染模式，决定输出图像的类型（RGB、深度、RGB+深度等）。
     * @param bounding_box [参数说明]：边界框指针，用于过滤高斯点（可选）。
     * @param gut [参数说明]：是否使用GUT（Generalized Unscented Transform）光栅化器。
     * @return [返回值说明]：RenderOutput结构，包含渲染的图像、透明度、深度等信息。
     */
    RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier,
        bool packed,
        bool antialiased,
        RenderMode render_mode,
        const gs::geometry::BoundingBox* bounding_box,
        bool gut) {
        
        // 检查不支持的功能：打包模式在此实现中不支持
        TORCH_CHECK(!packed, "Packed mode is not supported in this implementation");

        // 获取相机参数
        const int image_height = static_cast<int>(viewpoint_camera.image_height());  // 图像高度
        const int image_width = static_cast<int>(viewpoint_camera.image_width());    // 图像宽度

        // 准备视图矩阵和相机内参矩阵
        auto viewmat = viewpoint_camera.world_view_transform().to(torch::kCUDA);  // 世界到相机的变换矩阵
        TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(0) == 1 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                    "viewmat must be [1, 4, 4] after transpose and unsqueeze, got ", viewmat.sizes());
        TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA");

        const auto K = viewpoint_camera.K().to(torch::kCUDA);  // 相机内参矩阵
        TORCH_CHECK(K.is_cuda(), "K must be on CUDA");

        // 获取高斯模型的参数
        auto means3D = gaussian_model.get_means();        // 3D高斯中心位置 [N, 3]

        auto opacities = gaussian_model.get_opacity();    // 不透明度值
        if (opacities.dim() == 2 && opacities.size(1) == 1) {
            opacities = opacities.squeeze(-1);  // 如果形状是[N, 1]，压缩为[N]
        }
        auto scales = gaussian_model.get_scaling();       // 缩放参数 [N, 3]
        auto rotations = gaussian_model.get_rotation();   // 旋转参数 [N, 4]（四元数）
        auto sh_coeffs = gaussian_model.get_shs();       // 球谐函数系数 [N, K, 3]
        const int sh_degree = gaussian_model.get_active_sh_degree();  // 活跃的球谐函数阶数

        // 如果提供了边界框，应用边界框过滤
        if (bounding_box != nullptr) {
            torch::Tensor inside_indices;

            // 将GLM向量转换为PyTorch张量
            auto min_bounds = torch::tensor({bounding_box->getMinBounds().x,
                                             bounding_box->getMinBounds().y,
                                             bounding_box->getMinBounds().z},
                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto max_bounds = torch::tensor({bounding_box->getMaxBounds().x,
                                             bounding_box->getMaxBounds().y,
                                             bounding_box->getMaxBounds().z},
                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            // 获取世界坐标系到边界框坐标系的变换矩阵
            const glm::mat4 world2bbox = bounding_box->getworld2BBox().toMat4();

            // 将GLM矩阵转换为PyTorch张量 [4, 4]
            auto world2bbox_tensor = torch::zeros(
                {4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    world2bbox_tensor[i][j] = world2bbox[j][i]; // GLM是列主序！
                }
            }

            // 将点从世界空间变换到边界框空间
            // means3D: [N, 3] -> 齐次坐标: [N, 4]
            const int N = means3D.size(0);
            auto means3D_homogeneous = torch::cat({means3D, torch::ones({N, 1}, means3D.options())}, /*dim=*/1);
            // [N, 4]

            // 应用变换: [N, 4] @ [4, 4]^T = [N, 4]
            auto means3D_bbox = torch::matmul(means3D_homogeneous, world2bbox_tensor.transpose(0, 1)); // [N, 4]

            // 提取变换后的3D坐标（忽略齐次坐标）
            auto means3D_bbox_xyz = means3D_bbox.index({Slice(), Slice(None, 3)}); // [N, 3]

            // 检查哪些点在边界框空间中的轴对齐边界框内部
            // 由于点已经被变换，现在可以使用简单的轴对齐框测试
            auto greater_than_min = torch::all(means3D_bbox_xyz >= min_bounds.unsqueeze(0), /*dim=*/1); // [N]
            auto less_than_max = torch::all(means3D_bbox_xyz <= max_bounds.unsqueeze(0), /*dim=*/1);    // [N]
            auto inside_mask = greater_than_min & less_than_max;                                        // [N]

            // 获取边界框内部点的索引
            inside_indices = torch::nonzero(inside_mask).squeeze(-1); // [M] 其中 M <= N

            // 使用内部索引过滤所有高斯参数
            means3D = means3D.index({inside_indices});
            opacities = opacities.index({inside_indices});
            scales = scales.index({inside_indices});
            rotations = rotations.index({inside_indices});
            sh_coeffs = sh_coeffs.index({inside_indices});
        }

        // 验证高斯参数
        const int N = static_cast<int>(means3D.size(0));
        TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3,
                    "means3D must be [N, 3], got ", means3D.sizes());
        TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N,
                    "opacities must be [N], got ", opacities.sizes());
        TORCH_CHECK(scales.dim() == 2 && scales.size(0) == N && scales.size(1) == 3,
                    "scales must be [N, 3], got ", scales.sizes());
        TORCH_CHECK(rotations.dim() == 2 && rotations.size(0) == N && rotations.size(1) == 4,
                    "rotations must be [N, 4], got ", rotations.sizes());
        TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(0) == N && sh_coeffs.size(2) == 3,
                    "sh_coeffs must be [N, K, 3], got ", sh_coeffs.sizes());

        // 检查是否有足够的球谐函数系数来满足请求的阶数
        const int required_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
        TORCH_CHECK(sh_coeffs.size(1) >= required_sh_coeffs,
                    "Not enough SH coefficients. Expected at least ", required_sh_coeffs,
                    " but got ", sh_coeffs.size(1));

        // 检查高斯参数的设备位置
        TORCH_CHECK(means3D.is_cuda(), "means3D must be on CUDA");
        TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
        TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
        TORCH_CHECK(rotations.is_cuda(), "rotations must be on CUDA");
        TORCH_CHECK(sh_coeffs.is_cuda(), "sh_coeffs must be on CUDA");

        // 处理背景颜色 - 可能未定义
        torch::Tensor prepared_bg_color;
        if (!bg_color.defined() || bg_color.numel() == 0) {
            // 保持未定义状态
            prepared_bg_color = torch::Tensor();
        } else {
            prepared_bg_color = bg_color.view({1, -1}).to(torch::kCUDA);  // 重塑为[1, 3]并移动到CUDA
            TORCH_CHECK(prepared_bg_color.size(0) == 1 && prepared_bg_color.size(1) == 3,
                        "bg_color must be reshapeable to [1, 3], got ", prepared_bg_color.sizes());
            TORCH_CHECK(prepared_bg_color.is_cuda(), "bg_color must be on CUDA");
        }

        // 设置渲染参数
        const float eps2d = 0.3f;           // 2D投影的epsilon值
        const float near_plane = 0.01f;     // 近裁剪平面
        const float far_plane = 10000.0f;   // 远裁剪平面
        const float radius_clip = 0.0f;     // 半径裁剪值
        const int tile_size = 16;           // 瓦片大小
        const bool calc_compensations = antialiased;  // 是否计算补偿（用于抗锯齿）

        // 处理相机畸变参数
        std::optional<torch::Tensor> radial_distortion;
        if (viewpoint_camera.radial_distortion().numel() > 0) {
            radial_distortion = viewpoint_camera.radial_distortion().to(torch::kCUDA);
            TORCH_CHECK(radial_distortion->dim() == 1, "radial_distortion must be 1D, got ", radial_distortion->sizes());
        }
        std::optional<torch::Tensor> tangential_distortion;
        if (viewpoint_camera.tangential_distortion().numel() > 0) {
            tangential_distortion = viewpoint_camera.tangential_distortion().to(torch::kCUDA);
            TORCH_CHECK(tangential_distortion->dim() == 1, "tangential_distortion must be 1D, got ", tangential_distortion->sizes());
        }

        // 步骤1：投影 - 将3D高斯投影到2D图像平面
        torch::Tensor radii;           // 投影后的半径
        torch::Tensor means2d;         // 投影后的2D位置
        torch::Tensor depths;          // 深度值
        torch::Tensor conics;          // 圆锥曲线参数
        torch::Tensor compensations;   // 补偿值（用于抗锯齿）
        
        if (gut) {
            // 使用GUT投影器
            auto proj_settings = GUTProjectionSettings{
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                scaling_modifier,
                viewpoint_camera.camera_model_type()};
            auto proj_outputs = fully_fused_projection_with_ut(
                means3D,
                rotations,
                scales,
                opacities,
                viewmat,
                K,
                radial_distortion,
                tangential_distortion,
                std::nullopt,
                proj_settings,
                UnscentedTransformParameters());

            radii = proj_outputs[0];
            means2d = proj_outputs[1];
            depths = proj_outputs[2];
            conics = proj_outputs[3];
            compensations = proj_outputs[4];
        } else {
            // 使用标准投影器
            auto proj_settings = ProjectionSettings{
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                scaling_modifier};

            auto proj_outputs = ProjectionFunction::apply(
                means3D, rotations, scales, opacities, viewmat, K, proj_settings);

            radii = proj_outputs[0];
            means2d = proj_outputs[1];
            depths = proj_outputs[2];
            conics = proj_outputs[3];
            compensations = proj_outputs[4];
        }

        // 创建带梯度跟踪的means2d，用于反向传播兼容性
        auto means2d_with_grad = means2d.contiguous();
        means2d_with_grad.set_requires_grad(true);
        means2d_with_grad.retain_grad();

        // 步骤2：从球谐函数计算颜色
        // 首先，从逆视图矩阵计算相机位置
        auto viewmat_inv = torch::inverse(viewmat);
        auto campos = viewmat_inv.index({Slice(), Slice(None, 3), 3}); // [C, 3]

        // 计算从相机到每个高斯的方向向量
        auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1); // [C, N, 3]

        // 基于半径创建掩码
        auto masks = (radii > 0).all(-1); // [C, N]

        // Python代码将颜色从[N, K, 3]广播到[C, N, K, 3]（如果需要）
        auto shs = sh_coeffs.unsqueeze(0); // [1, N, K, 3]

        // 现在使用正确的方向调用球谐函数
        auto colors = spherical_harmonics(sh_degree, dirs, shs, masks); // [C, N, 3]

        // 应用球谐函数偏移和裁剪进行渲染（从[-0.5, 0.5]偏移到[0, 1]）
        colors = torch::clamp_min(colors + 0.5f, 0.0f);

        // 步骤3：根据渲染模式处理深度
        torch::Tensor render_colors;
        torch::Tensor final_bg;

        switch (render_mode) {
        case RenderMode::RGB:
            render_colors = colors;
            final_bg = prepared_bg_color;
            break;

        case RenderMode::D:
        case RenderMode::ED:
            render_colors = depths.unsqueeze(-1); // [C, N, 1]
            if (prepared_bg_color.defined()) {
                final_bg = torch::zeros({1, 1}, prepared_bg_color.options());
            } else {
                final_bg = torch::Tensor(); // 保持未定义
            }
            break;

        case RenderMode::RGB_D:
        case RenderMode::RGB_ED:
            // 连接颜色和深度
            render_colors = torch::cat({colors, depths.unsqueeze(-1)}, -1); // [C, N, 4]
            if (prepared_bg_color.defined()) {
                final_bg = torch::cat({prepared_bg_color, torch::zeros({1, 1}, prepared_bg_color.options())}, -1);
            } else {
                final_bg = torch::Tensor(); // 保持未定义
            }
            break;
        }

        if (!final_bg.defined()) {
            // 在CUDA上创建空张量 - 与投影中补偿值的模式相同
            final_bg = at::empty({0}, colors.options().dtype(torch::kFloat32));
        }

        // 步骤4：应用不透明度，包含补偿值
        torch::Tensor final_opacities;
        if (calc_compensations && compensations.defined() && compensations.numel() > 0) {
            final_opacities = opacities.unsqueeze(0) * compensations;  // 应用补偿值
        } else {
            final_opacities = opacities.unsqueeze(0);  // 不应用补偿值
        }
        TORCH_CHECK(final_opacities.is_cuda(), "final_opacities must be on CUDA");

        // 步骤5：瓦片相交测试
        const int tile_width = (image_width + tile_size - 1) / tile_size;   // 瓦片宽度数量
        const int tile_height = (image_height + tile_size - 1) / tile_size; // 瓦片高度数量

        // 计算每个高斯与哪些瓦片相交
        const auto isect_results = gsplat::intersect_tile(
            means2d_with_grad, radii, depths, {}, {},
            1, tile_size, tile_width, tile_height,
            true);

        const auto tiles_per_gauss = std::get<0>(isect_results);    // 每个高斯相交的瓦片数量
        const auto isect_ids = std::get<1>(isect_results);          // 相交ID
        const auto flatten_ids = std::get<2>(isect_results);        // 扁平化ID

        // 计算相交偏移量
        auto isect_offsets = gsplat::intersect_offset(
            isect_ids, 1, tile_width, tile_height);
        isect_offsets = isect_offsets.reshape({1, tile_height, tile_width});

        // 检查相交结果的设备位置
        TORCH_CHECK(tiles_per_gauss.is_cuda(), "tiles_per_gauss must be on CUDA");
        TORCH_CHECK(isect_ids.is_cuda(), "isect_ids must be on CUDA");
        TORCH_CHECK(flatten_ids.is_cuda(), "flatten_ids must be on CUDA");
        TORCH_CHECK(isect_offsets.is_cuda(), "isect_offsets must be on CUDA");

        // 步骤6：光栅化 - 将高斯溅射到像素网格上
        torch::Tensor rendered_image;
        torch::Tensor rendered_alpha;
        
        if (gut) {
            // 使用GUT光栅化器
            auto raster_settings = GUTRasterizationSettings{
                image_width,
                image_height,
                tile_size,
                scaling_modifier,
                viewpoint_camera.camera_model_type()};
            auto ut_params = UnscentedTransformParameters{};
            auto raster_outputs = GUTRasterizationFunction::apply(
                means3D,
                rotations,
                scales,
                render_colors,
                final_opacities,
                final_bg,
                std::nullopt,
                viewmat,
                K,
                radial_distortion,
                tangential_distortion,
                std::nullopt, // thin_prism_coeffs
                isect_offsets,
                flatten_ids,
                raster_settings,
                ut_params);
            rendered_image = raster_outputs[0];
            rendered_alpha = raster_outputs[1];
        } else {
            // 使用标准光栅化器
            auto raster_settings = RasterizationSettings{
                image_width,
                image_height,
                tile_size};

            auto raster_outputs = RasterizationFunction::apply(
                means2d_with_grad, conics, render_colors, final_opacities, final_bg,
                isect_offsets, flatten_ids, raster_settings);
            rendered_image = raster_outputs[0];
            rendered_alpha = raster_outputs[1];
        }

        // 步骤7：根据渲染模式进行后处理
        torch::Tensor final_image, final_depth;

        switch (render_mode) {
        case RenderMode::RGB:
            final_image = rendered_image;
            final_depth = torch::Tensor(); // 空
            break;

        case RenderMode::D:
            final_depth = rendered_image;  // 实际上是深度
            final_image = torch::Tensor(); // 空
            break;

        case RenderMode::ED:
            // 通过alpha归一化累积深度以获得期望深度
            final_depth = rendered_image / rendered_alpha.clamp_min(1e-10);
            final_image = torch::Tensor(); // 空
            break;

        case RenderMode::RGB_D:
            final_image = rendered_image.index({Slice(), Slice(), Slice(), Slice(None, -1)});
            final_depth = rendered_image.index({Slice(), Slice(), Slice(), Slice(-1, None)});
            break;

        case RenderMode::RGB_ED:
            final_image = rendered_image.index({Slice(), Slice(), Slice(), Slice(None, -1)});
            auto accum_depth = rendered_image.index({Slice(), Slice(), Slice(), Slice(-1, None)});
            final_depth = accum_depth / rendered_alpha.clamp_min(1e-10);
            break;
        }

        // 准备输出结果
        RenderOutput result;

        // 处理图像输出
        if (final_image.defined() && final_image.numel() > 0) {
            result.image = torch::clamp(final_image.squeeze(0).permute({2, 0, 1}), 0.0f, 1.0f);
        } else {
            result.image = torch::Tensor();
        }

        // 处理alpha输出 - 总是存在
        result.alpha = rendered_alpha.squeeze(0).permute({2, 0, 1});

        // 处理深度输出
        if (final_depth.defined() && final_depth.numel() > 0) {
            result.depth = final_depth.squeeze(0).permute({2, 0, 1});
        } else {
            result.depth = torch::Tensor();
        }

        // 设置其他输出字段
        result.means2d = means2d_with_grad;                    // 2D投影位置
        result.depths = depths.squeeze(0);                     // 深度值
        result.radii = std::get<0>(radii.squeeze(0).max(-1)); // 最大半径
        result.visibility = (result.radii > 0);                // 可见性掩码
        result.width = image_width;                            // 图像宽度
        result.height = image_height;                          // 图像高度

        // 最终检查输出结果的设备位置
        if (result.image.defined() && result.image.numel() > 0) {
            TORCH_CHECK(result.image.is_cuda(), "result.image must be on CUDA");
        }
        TORCH_CHECK(result.alpha.is_cuda(), "result.alpha must be on CUDA");
        if (result.depth.defined() && result.depth.numel() > 0) {
            TORCH_CHECK(result.depth.is_cuda(), "result.depth must be on CUDA");
        }
        TORCH_CHECK(result.means2d.is_cuda(), "result.means2d must be on CUDA");
        TORCH_CHECK(result.depths.is_cuda(), "result.depths must be on CUDA");
        TORCH_CHECK(result.radii.is_cuda(), "result.radii must be on CUDA");
        TORCH_CHECK(result.visibility.is_cuda(), "result.visibility must be on CUDA");

        return result;
    }
} // namespace gs::training
