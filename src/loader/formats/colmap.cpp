#include "colmap.hpp"
#include "core/point_cloud.hpp"
#include "core/torch_shapes.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace gs::loader {

    namespace fs = std::filesystem;
    namespace F = torch::nn::functional;

    // -----------------------------------------------------------------------------
    //  Quaternion to rotation matrix
    // -----------------------------------------------------------------------------
    inline torch::Tensor qvec2rotmat(const torch::Tensor& qraw) {
        assert_vec(qraw, 4, "qvec");

        auto q = F::normalize(qraw.to(torch::kFloat32),
                              F::NormalizeFuncOptions().dim(0));

        auto w = q[0], x = q[1], y = q[2], z = q[3];

        torch::Tensor R = torch::empty({3, 3}, torch::kFloat32);
        R[0][0] = 1 - 2 * (y * y + z * z);
        R[0][1] = 2 * (x * y - z * w);
        R[0][2] = 2 * (x * z + y * w);

        R[1][0] = 2 * (x * y + z * w);
        R[1][1] = 1 - 2 * (x * x + z * z);
        R[1][2] = 2 * (y * z - x * w);

        R[2][0] = 2 * (x * z - y * w);
        R[2][1] = 2 * (y * z + x * w);
        R[2][2] = 1 - 2 * (x * x + y * y);
        return R;
    }

    class Image {
    public:
        Image() = default;
        explicit Image(uint32_t id)
            : _image_ID(id) {}

        uint32_t _camera_id = 0;
        std::string _name;

        torch::Tensor _qvec = torch::tensor({1.f, 0.f, 0.f, 0.f}, torch::kFloat32);
        torch::Tensor _tvec = torch::zeros({3}, torch::kFloat32);

    private:
        uint32_t _image_ID = 0;
    };

    // -----------------------------------------------------------------------------
    //  Build 4x4 world-to-camera matrix
    // -----------------------------------------------------------------------------
    inline torch::Tensor getWorld2View(const torch::Tensor& R,
                                       const torch::Tensor& T) {
        assert_mat(R, 3, 3, "R");
        assert_vec(T, 3, "T");

        torch::Tensor M = torch::eye(4, torch::kFloat32);
        M.index_put_({torch::indexing::Slice(0, 3),
                      torch::indexing::Slice(0, 3)},
                     R);
        M.index_put_({torch::indexing::Slice(0, 3), 3},
                     (-torch::matmul(R, T)).reshape({3}));
        return M;
    }

    // -----------------------------------------------------------------------------
    //  POD read helpers
    // -----------------------------------------------------------------------------
    static inline uint64_t read_u64(const char*& p) {
        uint64_t v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }
    static inline uint32_t read_u32(const char*& p) {
        uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    static inline int32_t read_i32(const char*& p) {
        int32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    static inline double read_f64(const char*& p) {
        double v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }

    // -----------------------------------------------------------------------------
    //  COLMAP camera-model map
    // -----------------------------------------------------------------------------
    static const std::unordered_map<int, std::pair<CAMERA_MODEL, int32_t>> camera_model_ids = {
        {0, {CAMERA_MODEL::SIMPLE_PINHOLE, 3}},
        {1, {CAMERA_MODEL::PINHOLE, 4}},
        {2, {CAMERA_MODEL::SIMPLE_RADIAL, 4}},
        {3, {CAMERA_MODEL::RADIAL, 5}},
        {4, {CAMERA_MODEL::OPENCV, 8}},
        {5, {CAMERA_MODEL::OPENCV_FISHEYE, 8}},
        {6, {CAMERA_MODEL::FULL_OPENCV, 12}},
        {7, {CAMERA_MODEL::FOV, 5}},
        {8, {CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE, 4}},
        {9, {CAMERA_MODEL::RADIAL_FISHEYE, 5}},
        {10, {CAMERA_MODEL::THIN_PRISM_FISHEYE, 12}},
        {11, {CAMERA_MODEL::UNDEFINED, -1}}};

    // -----------------------------------------------------------------------------
    //  Binary-file loader
    // -----------------------------------------------------------------------------
    static std::unique_ptr<std::vector<char>>
    read_binary(const std::filesystem::path& p) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f)
            throw std::runtime_error("Failed to open " + p.string());

        auto sz = static_cast<std::streamsize>(f.tellg());
        auto buf = std::make_unique<std::vector<char>>(static_cast<size_t>(sz));

        f.seekg(0, std::ios::beg);
        f.read(buf->data(), sz);
        if (!f)
            throw std::runtime_error("Short read on " + p.string());
        return buf;
    }

    // -----------------------------------------------------------------------------
    //  images.bin
    // -----------------------------------------------------------------------------
    /**
     * [功能描述]：读取COLMAP二进制格式的图像文件（images.bin）
     * @param file_path：images.bin文件的路径
     * @return 图像对象向量，包含每张图像的位姿、相机参数等信息
     * 
     * COLMAP二进制格式说明：
     * - 文件开头：图像数量（uint64_t）
     * - 每张图像包含：ID、旋转四元数、平移向量、相机ID、文件名、2D特征点
     */
    std::vector<Image> read_images_binary(const std::filesystem::path& file_path) {
        // =============================================================================
        // 第一步：读取整个二进制文件到内存
        // =============================================================================
        auto buf_owner = read_binary(file_path);  // 读取文件到内存缓冲区
        const char* cur = buf_owner->data();      // 当前读取位置指针
        const char* end = cur + buf_owner->size(); // 文件结束位置指针

        // =============================================================================
        // 第二步：读取图像总数并预分配容器
        // =============================================================================
        uint64_t n_images = read_u64(cur);        // 读取图像总数（64位无符号整数）
        std::vector<Image> images;                // 创建图像容器
        images.reserve(n_images);                 // 预分配内存，避免频繁重新分配

        // =============================================================================
        // 第三步：逐个解析每张图像的数据
        // =============================================================================
        for (uint64_t i = 0; i < n_images; ++i) {
            // -----------------------------------------------------------------------------
            // 3.1 读取图像ID并创建图像对象
            // -----------------------------------------------------------------------------
            uint32_t id = read_u32(cur);           // 读取图像唯一标识符（32位）
            auto& img = images.emplace_back(id);   // 在容器中直接构造图像对象

            // -----------------------------------------------------------------------------
            // 3.2 读取相机旋转信息（四元数表示）
            // -----------------------------------------------------------------------------
            torch::Tensor q = torch::empty({4}, torch::kFloat32);  // 创建4维四元数张量
            for (int k = 0; k < 4; ++k) {
                // 从64位浮点数转换为32位浮点数存储
                // 四元数顺序：[qw, qx, qy, qz]，表示3D旋转
                q[k] = static_cast<float>(read_f64(cur));
            }
            img._qvec = q;  // 存储旋转四元数到图像对象

            // -----------------------------------------------------------------------------
            // 3.3 读取相机平移信息（3D向量）
            // -----------------------------------------------------------------------------
            torch::Tensor t = torch::empty({3}, torch::kFloat32);  // 创建3维平移向量
            for (int k = 0; k < 3; ++k) {
                // 从64位浮点数转换为32位浮点数存储
                // 平移向量：[tx, ty, tz]，表示相机在世界坐标系中的位置
                t[k] = static_cast<float>(read_f64(cur));
            }
            img._tvec = t;  // 存储平移向量到图像对象

            // -----------------------------------------------------------------------------
            // 3.4 读取关联的相机ID
            // -----------------------------------------------------------------------------
            img._camera_id = read_u32(cur);        // 读取该图像对应的相机标识符

            // -----------------------------------------------------------------------------
            // 3.5 读取图像文件名（以null结尾的字符串）
            // -----------------------------------------------------------------------------
            img._name.assign(cur);                 // 读取图像文件名字符串
            cur += img._name.size() + 1;           // 跳过字符串内容和结尾的'\0'字符

            // -----------------------------------------------------------------------------
            // 3.6 跳过2D特征点数据（当前处理中不需要）
            // -----------------------------------------------------------------------------
            uint64_t npts = read_u64(cur);         // 读取该图像的2D特征点数量
            
            // 跳过所有2D点数据，每个点包含：
            // - 2个double类型坐标值（x, y像素坐标）
            // - 1个uint64_t类型的3D点ID（对应的空间点索引）
            cur += npts * (sizeof(double) * 2 + sizeof(uint64_t));
        }

        // =============================================================================
        // 第四步：验证文件完整性
        // =============================================================================
        if (cur != end) {
            // 如果读取位置未到达文件末尾，说明文件格式有问题或解析错误
            throw std::runtime_error("images.bin: trailing bytes");
        }

        return images;  // 返回解析完成的图像数据
    }

    // -----------------------------------------------------------------------------
    //  cameras.bin
    // -----------------------------------------------------------------------------
    std::unordered_map<uint32_t, CameraData>
    read_cameras_binary(const std::filesystem::path& file_path) {
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t n_cams = read_u64(cur);
        std::unordered_map<uint32_t, CameraData> cams;
        cams.reserve(n_cams);

        for (uint64_t i = 0; i < n_cams; ++i) {
            CameraData cam;
            cam._camera_ID = read_u32(cur);

            int32_t model_id = read_i32(cur);
            cam._width = read_u64(cur);
            cam._height = read_u64(cur);

            auto it = camera_model_ids.find(model_id);
            if (it == camera_model_ids.end() || it->second.second < 0)
                throw std::runtime_error("Unsupported camera-model id " + std::to_string(model_id));

            cam._camera_model = it->second.first;
            int32_t param_cnt = it->second.second;
            cam._params = torch::from_blob(const_cast<char*>(cur),
                                           {param_cnt}, torch::kFloat64)
                              .clone()
                              .to(torch::kFloat32);
            cur += param_cnt * sizeof(double);

            cams.emplace(cam._camera_ID, std::move(cam));
        }
        if (cur != end)
            throw std::runtime_error("cameras.bin: trailing bytes");
        return cams;
    }

    // -----------------------------------------------------------------------------
    //  points3D.bin
    // -----------------------------------------------------------------------------
    PointCloud read_point3D_binary(const std::filesystem::path& file_path) {
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t N = read_u64(cur);

        // Pre-allocate tensors directly
        torch::Tensor positions = torch::empty({static_cast<int64_t>(N), 3}, torch::kFloat32);
        torch::Tensor colors = torch::empty({static_cast<int64_t>(N), 3}, torch::kUInt8);

        // Get raw pointers for efficient access
        float* pos_data = positions.data_ptr<float>();
        uint8_t* col_data = colors.data_ptr<uint8_t>();

        for (uint64_t i = 0; i < N; ++i) {
            cur += 8; // skip point ID

            // Read position directly into tensor
            pos_data[i * 3 + 0] = static_cast<float>(read_f64(cur));
            pos_data[i * 3 + 1] = static_cast<float>(read_f64(cur));
            pos_data[i * 3 + 2] = static_cast<float>(read_f64(cur));

            // Read color directly into tensor
            col_data[i * 3 + 0] = *cur++;
            col_data[i * 3 + 1] = *cur++;
            col_data[i * 3 + 2] = *cur++;

            cur += 8;                                    // skip reprojection error
            cur += read_u64(cur) * sizeof(uint32_t) * 2; // skip track
        }

        if (cur != end)
            throw std::runtime_error("points3D.bin: trailing bytes");

        return PointCloud(positions, colors);
    }

    // -----------------------------------------------------------------------------
    //  Assemble per-image camera information
    // -----------------------------------------------------------------------------
    std::tuple<std::vector<CameraData>, torch::Tensor>
    read_colmap_cameras(const std::filesystem::path base_path,
                        const std::unordered_map<uint32_t, CameraData>& cams,
                        const std::vector<Image>& images,
                        const std::string& images_folder = "images") {
        std::vector<CameraData> out(images.size());

        std::filesystem::path images_path = base_path / images_folder;

        // Prepare tensor to store all camera locations [N, 3]
        torch::Tensor camera_locations = torch::zeros({static_cast<int64_t>(images.size()), 3}, torch::kFloat32);

        // Check if the specified images folder exists
        if (!std::filesystem::exists(images_path)) {
            throw std::runtime_error("Images folder does not exist: " + images_path.string());
        }

        for (size_t i = 0; i < images.size(); ++i) {
            const Image& img = images[i];
            auto it = cams.find(img._camera_id);
            if (it == cams.end())
                throw std::runtime_error("Camera ID " + std::to_string(img._camera_id) + " not found");

            out[i] = it->second;
            out[i]._image_path = images_path / img._name;
            out[i]._image_name = img._name;

            out[i]._R = qvec2rotmat(img._qvec);
            out[i]._T = img._tvec.clone();

            // Camera location in world space = -R^T * T
            // This is equivalent to extracting camtoworlds[:, :3, 3] after inverting w2c
            camera_locations[i] = -torch::matmul(out[i]._R.t(), out[i]._T);

            switch (out[i]._camera_model) {
            // f, cx, cy
            case CAMERA_MODEL::SIMPLE_PINHOLE: {
                float fx = out[i]._params[0].item<float>();
                out[i]._focal_x = fx;
                out[i]._focal_y = fx;
                out[i]._center_x = out[i]._params[1].item<float>();
                out[i]._center_y = out[i]._params[2].item<float>();
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // fx, fy, cx, cy
            case CAMERA_MODEL::PINHOLE: {
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // f, cx, cy, k1
            case CAMERA_MODEL::SIMPLE_RADIAL: {
                float fx = out[i]._params[0].item<float>();
                out[i]._focal_x = fx;
                out[i]._focal_y = fx;
                out[i]._center_x = out[i]._params[1].item<float>();
                out[i]._center_y = out[i]._params[2].item<float>();
                float k1 = out[i]._params[3].item<float>();
                out[i]._radial_distortion = torch::tensor({k1}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // f, cx, cy, k1, k2
            case CAMERA_MODEL::RADIAL: {
                float fx = out[i]._params[0].item<float>();
                out[i]._focal_x = fx;
                out[i]._focal_y = fx;
                out[i]._center_x = out[i]._params[1].item<float>();
                out[i]._center_y = out[i]._params[2].item<float>();
                float k1 = out[i]._params[3].item<float>();
                float k2 = out[i]._params[4].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // fx, fy, cx, cy, k1, k2, p1, p2
            case CAMERA_MODEL::OPENCV: {
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();

                float k1 = out[i]._params[4].item<float>();
                float k2 = out[i]._params[5].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2}, torch::kFloat32);

                float p1 = out[i]._params[6].item<float>();
                float p2 = out[i]._params[7].item<float>();
                out[i]._tangential_distortion = torch::tensor({p1, p2}, torch::kFloat32);

                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4
            case CAMERA_MODEL::FULL_OPENCV: {
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();

                float k1 = out[i]._params[4].item<float>();
                float k2 = out[i]._params[5].item<float>();
                float k3 = out[i]._params[8].item<float>();
                float k4 = out[i]._params[9].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2, k3, k4}, torch::kFloat32);

                float p1 = out[i]._params[6].item<float>();
                float p2 = out[i]._params[7].item<float>();
                out[i]._tangential_distortion = torch::tensor({p1, p2}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // fx, fy, cx, cy, k1, k2, k3, k4
            case CAMERA_MODEL::OPENCV_FISHEYE: {
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();

                float k1 = out[i]._params[4].item<float>();
                float k2 = out[i]._params[5].item<float>();
                float k3 = out[i]._params[6].item<float>();
                float k4 = out[i]._params[7].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2, k3, k4}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::FISHEYE;
                break;
            }
            // f, cx, cy, k1, k2
            case CAMERA_MODEL::RADIAL_FISHEYE: {
                float fx = out[i]._params[0].item<float>();
                out[i]._focal_x = fx;
                out[i]._focal_y = fx;
                out[i]._center_x = out[i]._params[1].item<float>();
                out[i]._center_y = out[i]._params[2].item<float>();
                float k1 = out[i]._params[3].item<float>();
                float k2 = out[i]._params[4].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::FISHEYE;
                break;
            }
            default:
                throw std::runtime_error("Unsupported camera model");
            }

            out[i]._img_w = out[i]._img_h = out[i]._channels = 0;
            out[i]._img_data = nullptr;
        }

        std::cout << "Training with " << out.size() << " images \n";
        return {std::move(out), camera_locations.mean(0)};
    }

    // -----------------------------------------------------------------------------
    //  Public API functions
    // -----------------------------------------------------------------------------

    static fs::path get_sparse_file_path(const fs::path& base, const std::string& filename) {
        fs::path candidate0 = base / "sparse" / "0" / filename;
        if (fs::exists(candidate0))
            return candidate0;

        fs::path candidate = base / "sparse" / filename;
        if (fs::exists(candidate))
            return candidate;

        throw std::runtime_error(
            "Cannot find \"" + filename +
            "\" in \"" + candidate0.string() + "\" or \"" + candidate.string() + "\". "
                                                                                 "Expected directory structure: 'sparse/0/' or 'sparse/'.");
    }

    PointCloud read_colmap_point_cloud(const std::filesystem::path& filepath) {
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.bin");
        return read_point3D_binary(points3d_file);
    }

    std::tuple<std::vector<CameraData>, torch::Tensor> read_colmap_cameras_and_images(
        const std::filesystem::path& base,
        const std::string& images_folder) {

        fs::path cams_file = get_sparse_file_path(base, "cameras.bin");
        fs::path images_file = get_sparse_file_path(base, "images.bin");

        auto cams = read_cameras_binary(cams_file);
        auto images = read_images_binary(images_file);

        return read_colmap_cameras(base, cams, images, images_folder);
    }

} // namespace gs::loader