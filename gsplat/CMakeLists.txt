# =============================================================================
# GSPLAT 模块 CMake 配置文件
# 功能：高斯散点渲染后端库的构建配置
# =============================================================================

cmake_minimum_required(VERSION 3.24...3.30)

# =============================================================================
# 依赖环境检查 - 确保从父项目正确继承配置
# =============================================================================

# 检查PyTorch配置是否从父项目继承
if(NOT DEFINED TORCH_INCLUDE_DIRS)
    find_package(Torch REQUIRED)    # 如果未定义，则重新查找PyTorch
endif()

# 确保CUDA运行时库目标可用（必须从父CMakeLists.txt正确设置）
if(NOT TARGET CUDA::cudart)
    message(FATAL_ERROR "CUDA::cudart not found - ensure parent CMakeLists.txt sets up CUDA properly")
endif()

# 确保CUDA随机数生成库目标可用
if(NOT TARGET CUDA::curand)
    message(FATAL_ERROR "CUDA::curand not found - ensure parent CMakeLists.txt sets up CUDA properly")
endif()

# 确保CUDA基础线性代数库目标可用
if(NOT TARGET CUDA::cublas)
    message(FATAL_ERROR "CUDA::cublas not found - ensure parent CMakeLists.txt sets up CUDA properly")
endif()

# =============================================================================
# 源文件列表定义 - 包含所有gsplat相关的C++和CUDA实现
# =============================================================================
set(GSPLAT_SOURCES
        # C++源文件 - 主机端算法实现
        Intersect.cpp                       # 相交检测算法
        Null.cpp                           # 空操作实现
        Projection.cpp                      # 投影变换
        QuatScaleToCovar.cpp               # 四元数和缩放转协方差矩阵
        QuatToRotmat.cpp                   # 四元数转旋转矩阵
        Rasterization.cpp                  # 光栅化处理
        Relocation.cpp                     # 重定位算法
        SphericalHarmonics.cpp             # 球谐函数计算

        # CUDA源文件 - GPU加速实现
        IntersectTile.cu                   # GPU瓦片相交检测
        NullCUDA.cu                        # GPU空操作
        ProjectionEWA3DGSFused.cu          # EWA 3DGS融合投影（GPU）
        ProjectionEWA3DGSPacked.cu         # EWA 3DGS打包投影（GPU）
        ProjectionEWASimple.cu             # EWA简单投影（GPU）
        ProjectionUT3DGSFused.cu           # UT 3DGS融合投影（GPU）
        QuatScaleToCovarCUDA.cu            # 四元数缩放转协方差（GPU）
        QuatToRotmatCUDA.cu                # 四元数转旋转矩阵（GPU）
        RasterizeToIndices3DGS.cu          # 3DGS索引光栅化（GPU）
        RasterizeToPixels3DGSBwd.cu        # 3DGS像素光栅化反向传播（GPU）
        RasterizeToPixels3DGSFwd.cu        # 3DGS像素光栅化前向传播（GPU）
        RasterizeToPixelsFromWorld3DGSBwd.cu   # 世界坐标到像素光栅化反向传播（GPU）
        RasterizeToPixelsFromWorld3DGSFwd.cu   # 世界坐标到像素光栅化前向传播（GPU）
        RelocationCUDA.cu                  # GPU重定位算法
        SphericalHarmonicsCUDA.cu          # GPU球谐函数计算
)

# =============================================================================
# 静态库创建 - 构建统一的gsplat后端库
# =============================================================================
add_library(gsplat_backend STATIC ${GSPLAT_SOURCES})

# =============================================================================
# 库属性配置 - 设置编译标准和CUDA特性
# =============================================================================
set_target_properties(gsplat_backend PROPERTIES
        CXX_STANDARD 23                    # 使用C++23标准
        CXX_STANDARD_REQUIRED ON           # 强制要求C++23
        CUDA_STANDARD 20                   # 使用CUDA C++20标准
        CUDA_STANDARD_REQUIRED ON          # 强制要求CUDA C++20
        CUDA_ARCHITECTURES native          # 自动检测目标GPU架构
        CUDA_SEPARABLE_COMPILATION ON      # 启用CUDA可分离编译（支持设备代码链接）
        POSITION_INDEPENDENT_CODE ON       # 生成位置无关代码（支持共享库）
        CUDA_RESOLVE_DEVICE_SYMBOLS ON     # 解析CUDA设备符号（链接时必需）
)

# =============================================================================
# 包含目录配置 - 设置头文件搜索路径
# =============================================================================
target_include_directories(gsplat_backend
        PUBLIC                              # 公共包含目录（会传播给链接此库的目标）
        ${CMAKE_CURRENT_SOURCE_DIR}         # 当前gsplat目录（本模块头文件）
        ${CMAKE_SOURCE_DIR}/include         # 项目根include目录（全局头文件）
        ${TORCH_INCLUDE_DIRS}               # PyTorch头文件目录
        ${CUDAToolkit_INCLUDE_DIRS}         # CUDA工具包头文件目录
        PRIVATE                             # 私有包含目录（仅本目标使用）
        ${Python3_INCLUDE_DIRS}             # Python开发头文件目录
)

# =============================================================================
# 链接库配置 - 设置依赖的外部库
# =============================================================================
target_link_libraries(gsplat_backend
        PUBLIC                              # 公共链接库（会传播给链接此库的目标）
        ${TORCH_LIBRARIES}                  # PyTorch库文件
        CUDA::cudart                        # CUDA运行时库
        CUDA::curand                        # CUDA随机数生成库
        CUDA::cublas                        # CUDA基础线性代数子程序库
        glm::glm                            # OpenGL数学库（向量和矩阵运算）
)

# =============================================================================
# 编译选项配置 - 针对不同语言和构建类型的优化设置
# =============================================================================
target_compile_options(gsplat_backend PRIVATE
        # CUDA Debug模式编译选项
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
        # -O0: 禁用优化，-g: 生成主机调试信息，-G: 生成设备调试信息，-lineinfo: 包含行号信息
        
        # CUDA Release模式编译选项
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress=20012,186>
        # -O3: 最高级别优化，--use_fast_math: 使用快速数学库，--expt-relaxed-constexpr: 放宽constexpr限制
        # -diag-suppress: 抑制特定警告（20012: 未使用变量，186: 比较运算符）
        
        # C++ Debug模式编译选项
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g>
        # -O0: 禁用优化，-g: 生成调试信息
        
        # C++ Release模式编译选项
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
        # -O3: 最高级别优化
)

# =============================================================================
# 构建类型宏定义 - 为代码提供编译时构建类型标识
# =============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_backend PRIVATE DEBUG DEBUG_BUILD)    # 定义DEBUG和DEBUG_BUILD宏
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(gsplat_backend PRIVATE RELEASE_BUILD)        # 定义RELEASE_BUILD宏
endif()

# =============================================================================
# 调试信息输出 - 显示gsplat后端配置详情
# =============================================================================
message(STATUS "gsplat_backend configuration:")
message(STATUS "  - CUDA targets available: ${TARGET}")                    # 可用的CUDA目标
message(STATUS "  - Sources: ${GSPLAT_SOURCES}")                          # 源文件列表
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")                     # 构建类型
message(STATUS "  - C++ Standard: 23")                                    # C++标准版本
message(STATUS "  - CUDA Standard: 20")                                   # CUDA标准版本
message(STATUS "  - CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")       # CUDA包含目录
message(STATUS "  - Torch include dirs: ${TORCH_INCLUDE_DIRS}")           # PyTorch包含目录
message(STATUS "  - Torch libraries: ${TORCH_LIBRARIES}")                # PyTorch库文件
