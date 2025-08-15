#include "config.h"

#ifdef CUDA_GL_INTEROP_ENABLED

// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

// Now include CUDA GL interop (after GLAD)
#include <cuda_gl_interop.h>

#include "rendering/cuda_gl_interop.hpp"
#include <iostream>
#include <stdexcept>

namespace gs {

    /**
     * [功能描述]：CUDA-OpenGL互操作纹理类的默认构造函数
     * 初始化所有成员变量为默认值，为后续的纹理创建和CUDA注册做准备
     */
    CudaGLInteropTexture::CudaGLInteropTexture()
        : texture_id_(0),          // OpenGL纹理ID，0表示无效纹理
        cuda_resource_(nullptr), // CUDA图形资源指针，用于与OpenGL纹理互操作
        width_(0),              // 纹理宽度
        height_(0),             // 纹理高度
        is_registered_(false) { // CUDA注册状态标志
    }

    /**
     * [功能描述]：CUDA-OpenGL互操作纹理类的析构函数
     * 确保在对象销毁时正确清理所有OpenGL和CUDA资源
     */
    CudaGLInteropTexture::~CudaGLInteropTexture() {
        cleanup(); // 调用清理函数释放所有资源
    }

    /**
     * [功能描述]：初始化CUDA-OpenGL互操作纹理
     * @param width：纹理宽度（像素）
     * @param height：纹理高度（像素）
     * 
     * 该函数执行以下关键步骤：
     * 1. 创建OpenGL纹理对象
     * 2. 配置纹理参数
     * 3. 分配纹理存储空间
     * 4. 将纹理注册到CUDA以实现互操作
     */
    void CudaGLInteropTexture::init(int width, int height) {
        // 清理任何现有资源，确保干净的初始化环境
        cleanup();

        width_ = width;   // 存储纹理尺寸
        height_ = height;

        // =============================================================================
        // 第一步：创建并配置OpenGL纹理
        // =============================================================================
        glGenTextures(1, &texture_id_);          // 生成OpenGL纹理对象
        glBindTexture(GL_TEXTURE_2D, texture_id_); // 绑定纹理以进行操作

        // 设置纹理参数（必须在分配存储之前设置）
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // 线性缩小过滤
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // 线性放大过滤
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // S轴边缘夹取
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // T轴边缘夹取

        // 分配纹理存储空间（使用RGBA格式以获得更好的内存对齐）
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // 关键步骤：在CUDA注册之前必须解绑纹理
        // 这避免了OpenGL和CUDA之间的状态冲突
        glBindTexture(GL_TEXTURE_2D, 0);

        // =============================================================================
        // 第二步：检查OpenGL错误
        // =============================================================================
        GLenum gl_err = glGetError();
        if (gl_err != GL_NO_ERROR) {
            cleanup(); // 出错时清理资源
            throw std::runtime_error("OpenGL error during texture creation: " +
                                    std::to_string(gl_err));
        }

        // =============================================================================
        // 第三步：将OpenGL纹理注册到CUDA
        // =============================================================================
        cudaGetLastError(); // 清除之前的CUDA错误状态

        // 将OpenGL纹理注册为CUDA图形资源
        // cudaGraphicsRegisterFlagsWriteDiscard: CUDA将写入但不读取纹理内容
        cudaError_t err = cudaGraphicsGLRegisterImage(
            &cuda_resource_, texture_id_, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsWriteDiscard);

        if (err != cudaSuccess) {
            cleanup(); // 注册失败时清理资源
            throw std::runtime_error("Failed to register OpenGL texture with CUDA: " +
                                    std::string(cudaGetErrorString(err)));
        }

        is_registered_ = true; // 标记注册成功
    }

    /**
     * [功能描述]：调整纹理尺寸
     * @param new_width：新的纹理宽度
     * @param new_height：新的纹理高度
     * 
     * 只有当尺寸确实发生变化时才重新初始化，避免不必要的资源重建
     */
    void CudaGLInteropTexture::resize(int new_width, int new_height) {
        if (width_ != new_width || height_ != new_height) {
            init(new_width, new_height); // 重新初始化以适应新尺寸
        }
    }

    /**
     * [功能描述]：从PyTorch张量更新纹理内容
     * @param image：CUDA上的图像张量，格式为[H, W, C]，C为3或4通道
     * 
     * 该函数实现了从CUDA张量到OpenGL纹理的零拷贝更新：
     * 1. 验证输入张量格式
     * 2. 映射CUDA图形资源
     * 3. 转换数据格式（如需要）
     * 4. 执行设备到设备的内存拷贝
     */
    void CudaGLInteropTexture::updateFromTensor(const torch::Tensor& image) {
        if (!is_registered_) {
            throw std::runtime_error("Texture not initialized");
        }

        // =============================================================================
        // 第一步：验证输入张量格式
        // =============================================================================
        TORCH_CHECK(image.is_cuda(), "Image must be on CUDA");           // 必须在CUDA设备上
        TORCH_CHECK(image.dim() == 3, "Image must be [H, W, C]");        // 三维张量
        TORCH_CHECK(image.size(2) == 3 || image.size(2) == 4,           // RGB或RGBA通道
                    "Image must have 3 or 4 channels");

        const int h = image.size(0); // 图像高度
        const int w = image.size(1); // 图像宽度
        const int c = image.size(2); // 通道数

        // 如果尺寸不匹配，自动调整纹理大小
        resize(w, h);

        // =============================================================================
        // 第二步：映射CUDA资源以便访问
        // =============================================================================
        cudaError_t err = cudaGraphicsMapResources(1, &cuda_resource_, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to map CUDA resource: " +
                                    std::string(cudaGetErrorString(err)));
        }

        try {
            // 获取映射后的CUDA数组对象
            cudaArray_t cuda_array;
            err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to get CUDA array: " +
                                        std::string(cudaGetErrorString(err)));
            }

            // =============================================================================
            // 第三步：处理颜色通道转换
            // =============================================================================
            torch::Tensor rgba_image;
            if (c == 3) {
                // RGB转RGBA：添加Alpha通道（设为1.0，表示不透明）
                rgba_image = torch::cat({image,
                                        torch::ones({h, w, 1}, image.options())},
                                        2);
            } else {
                // 已经是RGBA格式，直接使用
                rgba_image = image;
            }

            // =============================================================================
            // 第四步：数据类型转换（浮点到uint8）
            // =============================================================================
            if (rgba_image.dtype() != torch::kUInt8) {
                // 将[0,1]范围的浮点值转换为[0,255]范围的uint8值
                rgba_image = (rgba_image.clamp(0.0f, 1.0f) * 255.0f).to(torch::kUInt8);
            }

            // 确保数据在内存中连续存储，优化拷贝性能
            rgba_image = rgba_image.contiguous();

            // =============================================================================
            // 第五步：执行设备到设备的内存拷贝
            // =============================================================================
            err = cudaMemcpy2DToArray(
                cuda_array,                        // 目标CUDA数组
                0, 0,                             // 目标偏移量（x, y）
                rgba_image.data_ptr<uint8_t>(),   // 源数据指针
                w * 4,                            // 源数据行跨距（RGBA = 4字节/像素）
                w * 4,                            // 拷贝宽度（字节）
                h,                                // 拷贝高度（行数）
                cudaMemcpyDeviceToDevice);        // 设备到设备拷贝

            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy to CUDA array: " +
                                        std::string(cudaGetErrorString(err)));
            }

            // 同步CUDA设备，确保拷贝操作完成
            cudaDeviceSynchronize();

        } catch (...) {
            // 异常处理：无论如何都要解除资源映射
            cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
            throw; // 重新抛出异常
        }

        // =============================================================================
        // 第六步：解除CUDA资源映射
        // =============================================================================
        err = cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to unmap CUDA resource: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }

    /**
     * [功能描述]：清理所有OpenGL和CUDA资源
     * 按正确顺序释放资源：先解注册CUDA资源，再删除OpenGL纹理
     */
    void CudaGLInteropTexture::cleanup() {
        // 第一步：解除CUDA注册
        if (is_registered_ && cuda_resource_) {
            cudaGraphicsUnregisterResource(cuda_resource_);
            cuda_resource_ = nullptr;
            is_registered_ = false;
        }

        // 第二步：删除OpenGL纹理
        if (texture_id_ != 0) {
            glDeleteTextures(1, &texture_id_);
            texture_id_ = 0;
        }
    }

    // =============================================================================
    // InteropFrameBuffer 类实现
    // =============================================================================

    /**
     * [功能描述]：互操作帧缓冲区构造函数
     * @param use_interop：是否启用CUDA-GL互操作优化
     * 
     * 该类提供两种模式：
     * 1. 互操作模式：直接在GPU上传输数据（高性能）
     * 2. 回退模式：通过CPU中转数据（兼容性好）
     */
    InteropFrameBuffer::InteropFrameBuffer(bool use_interop)
        : FrameBuffer(),           // 调用基类构造函数
        use_interop_(use_interop) {
        if (use_interop_) {
            try {
                // 尝试初始化CUDA-GL互操作纹理
                interop_texture_.init(width, height);
            } catch (const std::exception& e) {
                // 互操作初始化失败，自动回退到CPU模式
                std::cerr << "Failed to initialize CUDA-GL interop: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU copy mode" << std::endl;
                use_interop_ = false;
            }
        }
    }

    /**
     * [功能描述]：从CUDA张量上传图像数据到帧缓冲区
     * @param cuda_image：CUDA设备上的图像张量
     * 
     * 根据当前模式选择最优的数据传输路径：
     * - 互操作模式：GPU直接传输（零拷贝）
     * - 回退模式：GPU→CPU→GPU传输（兼容性好）
     */
    void InteropFrameBuffer::uploadFromCUDA(const torch::Tensor& cuda_image) {
        if (!use_interop_) {
            // =============================================================================
            // 回退模式：通过CPU进行数据传输
            // =============================================================================
            auto cpu_image = cuda_image.to(torch::kCPU).contiguous(); // CUDA到CPU传输

            // 处理不同的张量格式
            torch::Tensor formatted;
            if (cuda_image.size(-1) == 3 || cuda_image.size(-1) == 4) {
                // 已经是[H, W, C]格式
                formatted = cpu_image;
            } else {
                // 从[C, H, W]转换为[H, W, C]格式
                formatted = cpu_image.permute({1, 2, 0}).contiguous();
            }

            // 数据类型转换：浮点到uint8
            if (formatted.dtype() != torch::kUInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(torch::kUInt8);
            }

            // 通过基类方法上传到OpenGL（CPU到GPU传输）
            uploadImage(formatted.data_ptr<unsigned char>(),
                        formatted.size(1), formatted.size(0));
            return;
        }

        // =============================================================================
        // 互操作模式：直接GPU传输（高性能路径）
        // =============================================================================
        try {
            interop_texture_.updateFromTensor(cuda_image); // 零拷贝GPU传输
        } catch (const std::exception& e) {
            // 互操作失败时自动回退到CPU模式
            std::cerr << "CUDA-GL 更新失败: " << e.what() << std::endl;
            std::cerr << "回退到 CPU 复制" << std::endl;
            use_interop_ = false;
            uploadFromCUDA(cuda_image); // 递归调用，使用CPU模式重试
        }
    }

    /**
     * [功能描述]：调整帧缓冲区尺寸
     * @param new_width：新宽度
     * @param new_height：新高度
     * 
     * 同时调整基类帧缓冲区和互操作纹理的尺寸
     */
    void InteropFrameBuffer::resize(int new_width, int new_height) {
        FrameBuffer::resize(new_width, new_height); // 调整基类帧缓冲区
        
        if (use_interop_) {
            try {
                // 调整互操作纹理尺寸
                interop_texture_.resize(new_width, new_height);
            } catch (const std::exception& e) {
                // 调整失败时禁用互操作
                std::cerr << "调整 interop 纹理失败: " << e.what() << std::endl;
                use_interop_ = false;
            }
        }
    }

} // namespace gs

#else // CUDA_GL_INTEROP_ENABLED not defined

// Stub implementation when interop is not available
namespace gs {
    // Empty implementation - all functionality handled by preprocessor in header
}

#endif // CUDA_GL_INTEROP_ENABLED
