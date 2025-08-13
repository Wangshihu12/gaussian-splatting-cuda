/**
 * [文件描述]：相机类实现文件
 * 功能：实现三维重建中的相机模型，包括相机参数、图像加载、坐标变换等功能
 * 用途：用于高斯散点渲染中的视角管理和图像处理
 */

#include "core/camera.hpp"              // 相机类头文件
#include "core/image_io.hpp"            // 图像输入输出功能
#include <c10/cuda/CUDAGuard.h>         // CUDA流管理
#include <torch/torch.h>                // PyTorch张量操作

// 使用PyTorch张量索引功能的简化别名
using torch::indexing::None;            // 空索引标记
using torch::indexing::Slice;           // 切片索引功能

namespace gs {
    
    /**
     * [功能描述]：将世界坐标系转换为相机视图坐标系的变换矩阵
     * @param R：旋转矩阵，形状为[3, 3]，描述相机的朝向
     * @param t：平移向量，形状为[3]，描述相机在世界坐标系中的位置
     * @return 世界到视图的4x4变换矩阵，形状为[1, 4, 4]，在CUDA设备上
     */
    static torch::Tensor world_to_view(const torch::Tensor& R, const torch::Tensor& t) {
        // 创建4x4单位矩阵作为基础变换矩阵
        torch::Tensor w2c = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(R.device()));
        
        // 设置旋转部分：将3x3旋转矩阵R放入左上角
        w2c.index_put_({Slice(0, 3), Slice(0, 3)}, R);

        // 设置平移部分：将3D平移向量t放入第4列的前3行
        w2c.index_put_({Slice(0, 3), 3}, t);

        // 转换到CUDA设备，添加批次维度，并确保内存连续
        return w2c.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(0).contiguous();
    }

    /**
     * [功能描述]：相机类构造函数，初始化所有相机参数
     * @param R：旋转矩阵，描述相机在世界坐标系中的朝向
     * @param T：平移向量，描述相机在世界坐标系中的位置
     * @param focal_x：X轴方向焦距（像素单位）
     * @param focal_y：Y轴方向焦距（像素单位）
     * @param center_x：主点X坐标（像素单位）
     * @param center_y：主点Y坐标（像素单位）
     * @param radial_distortion：径向畸变参数
     * @param tangential_distortion：切向畸变参数
     * @param camera_model_type：相机模型类型（如针孔、鱼眼等）
     * @param image_name：关联的图像文件名
     * @param image_path：图像文件的完整路径
     * @param camera_width：相机传感器宽度（像素）
     * @param camera_height：相机传感器高度（像素）
     * @param uid：相机的唯一标识符
     */
    Camera::Camera(const torch::Tensor& R,
                const torch::Tensor& T,
                float focal_x, float focal_y,
                float center_x, float center_y,
                const torch::Tensor radial_distortion,
                const torch::Tensor tangential_distortion,
                gsplat::CameraModelType camera_model_type,
                const std::string& image_name,
                const std::filesystem::path& image_path,
                int camera_width, int camera_height,
                int uid)
        : _uid(uid),                                    // 相机唯一标识符
        _focal_x(focal_x),                            // X轴焦距
        _focal_y(focal_y),                            // Y轴焦距
        _center_x(center_x),                          // 主点X坐标
        _center_y(center_y),                          // 主点Y坐标
        _R(R),                                        // 旋转矩阵
        _T(T),                                        // 平移向量
        _radial_distortion(radial_distortion),        // 径向畸变系数
        _tangential_distortion(tangential_distortion), // 切向畸变系数
        _camera_model_type(camera_model_type),        // 相机模型类型
        _image_name(image_name),                      // 图像文件名
        _image_path(image_path),                      // 图像路径
        _camera_width(camera_width),                  // 相机宽度
        _camera_height(camera_height),                // 相机高度
        _image_width(camera_width),                   // 图像宽度（初始与相机宽度相同）
        _image_height(camera_height),                 // 图像高度（初始与相机高度相同）
        _world_view_transform{world_to_view(R, T)} {  // 世界到视图变换矩阵

        // 计算相机在世界坐标系中的位置
        // 通过世界到视图变换矩阵的逆矩阵获得视图到世界的变换
        auto c2w = torch::inverse(_world_view_transform.squeeze());
        
        // 提取相机位置：变换矩阵的第4列前3个元素
        _cam_position = c2w.index({Slice(None, 3), 3}).contiguous().squeeze();
        
        // 计算视场角（Field of View）
        _FoVx = focal2fov(_focal_x, _camera_width);     // X轴视场角
        _FoVy = focal2fov(_focal_y, _camera_height);    // Y轴视场角
    }

    /**
     * [功能描述]：获取相机内参矩阵K
     * @return 相机内参矩阵，形状为[1, 3, 3]，包含焦距和主点信息
     * 内参矩阵格式：
     * | fx  0  cx |
     * | 0   fy cy |
     * | 0   0  1  |
     */
    torch::Tensor Camera::K() const {
        // 创建零矩阵作为内参矩阵基础
        const auto K = torch::zeros({1, 3, 3}, _world_view_transform.options());
        
        // 计算缩放因子，用于处理图像分辨率变化
        float x_scale_factor = float(_image_width) / float(_camera_width);   // X轴缩放因子
        float y_scale_factor = float(_image_height) / float(_camera_height); // Y轴缩放因子
        
        // 设置内参矩阵元素
        K[0][0][0] = _focal_x * x_scale_factor;     // fx：X轴焦距（考虑缩放）
        K[0][1][1] = _focal_y * y_scale_factor;     // fy：Y轴焦距（考虑缩放）
        K[0][0][2] = _center_x * x_scale_factor;    // cx：主点X坐标（考虑缩放）
        K[0][1][2] = _center_y * y_scale_factor;    // cy：主点Y坐标（考虑缩放）
        K[0][2][2] = 1.0f;                          // 齐次坐标标准化项
        
        return K;
    }

    /**
     * [功能描述]：加载并处理图像数据，优化GPU传输性能
     * @param resolution：目标分辨率，用于图像缩放
     * @return 处理后的图像张量，形状为[C, H, W]，数值范围[0, 1]，在CUDA设备上
     */
    torch::Tensor Camera::load_and_get_image(int resolution) {
        // 使用固定内存（Pinned Memory）选项，加速CPU到GPU的数据传输
        auto pinned_options = torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true);

        // 图像数据变量
        unsigned char* data;    // 原始图像数据指针
        int w, h, c;           // 宽度、高度、通道数

        // 同步加载图像文件
        auto result = load_image(_image_path, resolution);
        data = std::get<0>(result);     // 图像数据指针
        w = std::get<1>(result);        // 实际加载的图像宽度
        h = std::get<2>(result);        // 实际加载的图像高度
        c = std::get<3>(result);        // 图像通道数

        // 更新实际图像尺寸
        _image_width = w;
        _image_height = h;

        // 从固定内存创建张量，指定数据布局
        torch::Tensor image = torch::from_blob(
            data,                       // 数据指针
            {h, w, c},                 // 张量形状：[高度, 宽度, 通道]
            {w * c, c, 1},             // 步长：行步长, 像素步长, 通道步长
            pinned_options);           // 使用固定内存选项

        // 使用CUDA流进行异步数据传输
        at::cuda::CUDAStreamGuard guard(_stream);

        // 数据处理管道：
        // 1. 异步传输到GPU
        // 2. 调整维度顺序：[H, W, C] -> [C, H, W]
        // 3. 转换数据类型：uint8 -> float32
        // 4. 归一化：[0, 255] -> [0, 1]
        image = image.to(torch::kCUDA, /*non_blocking=*/true)
                    .permute({2, 0, 1})                 // 通道优先格式
                    .to(torch::kFloat32) /              // 转换为浮点数
                255.0f;                                 // 归一化到[0, 1]

        // 释放原始图像数据内存
        free_image(data);

        // 确保GPU传输完成后再返回
        _stream.synchronize();

        return image;
    }

    /**
     * [功能描述]：获取图像文件在内存中占用的字节数
     * @return 图像数据的字节数，用于内存管理和预分配
     */
    size_t Camera::get_num_bytes_from_file() const {
        // 获取图像文件的基本信息（宽度、高度、通道数）
        auto [w, h, c] = get_image_info(_image_path);
        
        // 计算浮点数格式下的内存占用：宽度 × 高度 × 通道数 × float大小
        size_t num_bytes = w * h * c * sizeof(float);
        
        return num_bytes;
    }
    
} // namespace gs - 高斯散点项目命名空间结束