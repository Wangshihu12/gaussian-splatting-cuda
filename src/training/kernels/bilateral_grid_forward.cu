#include "kernels/bilateral_grid.cuh"
#include <cuda_runtime.h>

namespace gs {
    namespace bilateral_grid {
        // RGB到灰度转换的常量系数
        // 使用标准的亮度转换公式：Y = 0.299*R + 0.587*G + 0.114*B
        // 这些系数基于人眼对不同颜色波长的敏感度
        __constant__ float kC2G[3] = {0.299f, 0.587f, 0.114f};

        /**
         * [功能描述]：双边网格切片前向传播的CUDA内核函数。
         * 这个内核实现了双边网格的切片操作，将3D网格参数根据输入图像的空间位置和颜色值进行采样，
         * 然后应用仿射变换来生成输出图像。这是双边网格后处理的核心计算。
         * 
         * @param grid [参数说明]：双边网格参数，形状为[12, L, H, W]，包含12个变换参数。
         * @param rgb [参数说明]：输入RGB图像，形状为[h, w, 3]。
         * @param output [参数说明]：输出图像，形状为[h, w, 3]。
         * @param L [参数说明]：网格在引导维度（颜色/亮度）上的分辨率。
         * @param H [参数说明]：网格在高度方向上的分辨率。
         * @param W [参数说明]：网格在宽度方向上的分辨率。
         * @param h [参数说明]：输入图像的高度。
         * @param w [参数说明]：输入图像的宽度。
         */
        __global__ void slice_forward_kernel(
            const float* __restrict__ grid, // [12, L, H, W] - 网格参数数组
            const float* __restrict__ rgb,  // [h, w, 3] - 输入RGB图像
            float* __restrict__ output,     // [h, w, 3] - 输出图像
            int L, int H, int W,            // 网格的三个维度
            int h, int w) {                 // 图像的高度和宽度
            
            // 步骤1：计算当前线程的全局索引
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= h * w)  // 边界检查：确保线程索引在有效范围内
                return;

            // 步骤2：将一维索引转换为二维图像坐标
            int hi = idx / w;  // 计算行索引（高度）
            int wi = idx % w;  // 计算列索引（宽度）

            // 步骤3：加载RGB颜色值
            float3 color;                    // 使用CUDA的float3类型存储RGB值
            int rgb_idx = idx * 3;           // 计算RGB数组中的起始索引
            color.x = rgb[rgb_idx + 0];      // 红色分量
            color.y = rgb[rgb_idx + 1];      // 绿色分量
            color.z = rgb[rgb_idx + 2];      // 蓝色分量

            // 步骤4：计算网格坐标（均匀采样）
            // 将图像坐标归一化到[0, 1]范围
            float gx = (float)wi / (float)(w - 1);  // x方向归一化坐标
            float gy = (float)hi / (float)(h - 1);  // y方向归一化坐标
            
            // 将RGB转换为灰度值，作为引导坐标
            // 使用预定义的转换系数计算亮度
            float gz = kC2G[0] * color.x + kC2G[1] * color.y + kC2G[2] * color.z;

            // 将归一化坐标映射到网格索引范围
            float x = gx * (W - 1);  // x方向网格索引
            float y = gy * (H - 1);  // y方向网格索引
            float z = gz * (L - 1);  // z方向网格索引

            // 步骤5：三线性插值设置
            // 计算插值的8个角点索引
            int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);  // 左下角点
            int x1 = min(x0 + 1, W - 1);                         // 右下角点
            int y1 = min(y0 + 1, H - 1);                         // 左上角点
            int z1 = min(max(z0 + 1, 0), L - 1);                 // 后上角点
            z0 = max(z0, 0);                                      // 确保z0不小于0

            // 计算插值权重（小数部分）
            float fx = x - x0, fy = y - y0, fz = z - z0;

            // 步骤6：应用仿射变换
            // 初始化输出结果为0
            float3 result = make_float3(0.0f, 0.0f, 0.0f);

            // 循环处理12个变换参数
#pragma unroll  // 编译器优化：展开循环以提高性能
            for (int ci = 0; ci < 12; ci++) {
                // 三线性插值计算
                int base = ci * L * H * W;  // 当前参数在网格中的基址
                float val = 0.0f;           // 插值结果

                // 计算8个角点的加权和（三线性插值公式）
                // 每个角点的权重 = (1-fx) * (1-fy) * (1-fz) 的相应组合
                val += grid[base + (z0 * H + y0) * W + x0] * (1 - fx) * (1 - fy) * (1 - fz);  // 000
                val += grid[base + (z0 * H + y0) * W + x1] * fx * (1 - fy) * (1 - fz);        // 100
                val += grid[base + (z0 * H + y1) * W + x0] * (1 - fx) * fy * (1 - fz);        // 010
                val += grid[base + (z0 * H + y1) * W + x1] * fx * fy * (1 - fz);               // 110
                val += grid[base + (z1 * H + y0) * W + x0] * (1 - fx) * (1 - fy) * fz;        // 001
                val += grid[base + (z1 * H + y0) * W + x1] * fx * (1 - fy) * fz;               // 101
                val += grid[base + (z1 * H + y1) * W + x0] * (1 - fx) * fy * fz;               // 011
                val += grid[base + (z1 * H + y1) * W + x1] * fx * fy * fz;                      // 111

                // 步骤7：将插值结果应用到相应的输出通道
                int si = ci % 4;  // 源索引：决定使用哪个输入值作为系数
                int di = ci / 4;  // 目标索引：决定输出到哪个通道

                // 根据源索引选择系数
                float coeff = (si == 0)
                                  ? color.x      // 红色分量
                              : (si == 1)
                                  ? color.y      // 绿色分量
                              : (si == 2)
                                  ? color.z      // 蓝色分量
                                  : 1.0f;       // 常数项

                // 根据目标索引累加到相应的输出通道
                if (di == 0)
                    result.x += val * coeff;  // 红色通道
                else if (di == 1)
                    result.y += val * coeff;  // 绿色通道
                else
                    result.z += val * coeff;  // 蓝色通道
            }

            // 步骤8：写入输出结果
            output[rgb_idx + 0] = result.x;  // 输出红色分量
            output[rgb_idx + 1] = result.y;  // 输出绿色分量
            output[rgb_idx + 2] = result.z;  // 输出蓝色分量
        }

        /**
         * [功能描述]：双边网格切片前向传播的CUDA包装函数。
         * 这个函数负责设置CUDA内核的启动参数，调用内核函数，并处理张量数据的访问。
         * 它是PyTorch张量和CUDA内核之间的桥梁。
         * 
         * @param grid [参数说明]：双边网格参数张量，形状为[12, L, H, W]。
         * @param rgb [参数说明]：输入RGB图像张量，形状为[h, w, 3]。
         * @param output [参数说明]：输出图像张量，形状为[h, w, 3]。
         * @param use_uniform_coords [参数说明]：是否使用均匀坐标（当前未使用，保留接口兼容性）。
         */
        void slice_forward_cuda(
            const torch::Tensor& grid,
            const torch::Tensor& rgb,
            torch::Tensor& output,
            bool use_uniform_coords) {
            
            // 步骤1：提取张量维度信息
            const int h = rgb.size(0);  // 图像高度
            const int w = rgb.size(1);  // 图像宽度
            const int L = grid.size(1); // 网格引导维度
            const int H = grid.size(2); // 网格高度
            const int W = grid.size(3); // 网格宽度

            // 步骤2：配置CUDA内核启动参数
            const int threads = 256;                                    // 每个块的线程数
            const int blocks = (h * w + threads - 1) / threads;        // 计算需要的块数（向上取整）

            // 步骤3：启动CUDA内核
            slice_forward_kernel<<<blocks, threads>>>(
                grid.data_ptr<float>(),    // 网格数据的浮点指针
                rgb.data_ptr<float>(),     // 输入RGB数据的浮点指针
                output.data_ptr<float>(),  // 输出数据的浮点指针
                L, H, W, h, w);           // 传递维度参数
        }
    } // namespace bilateral_grid
} // namespace gs
