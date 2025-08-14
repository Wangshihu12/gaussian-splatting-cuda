# =============================================================================
# VCPKG 配置 - 跨平台包管理器设置
# =============================================================================
# 获取VCPKG环境变量并设置工具链文件，用于Windows/Linux平台的依赖包管理
set(RAW_VCPKG_PATH $ENV{VCPKG_ROOT})                    # 获取VCPKG根目录环境变量
file(TO_CMAKE_PATH "${RAW_VCPKG_PATH}" VCPKG_ROOT)       # 转换为CMake路径格式
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")  # 设置VCPKG工具链文件

# =============================================================================
# 项目基本配置
# =============================================================================
cmake_minimum_required(VERSION 3.24...3.30)                                # 要求CMake最低版本3.24，最高支持3.30
project(gaussian_splatting_cuda LANGUAGES CUDA CXX C)                      # 定义项目名称和支持的编程语言

# 获取项目根目录的绝对路径
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# 项目编译选项配置
option(ENABLE_CUDA_GL_INTEROP "Enable CUDA-OpenGL interoperability" ON)    # 启用CUDA-OpenGL互操作（默认开启）
option(BUILD_TESTS "Build tests" OFF)                                       # 构建测试程序（默认关闭）

# =============================================================================
# PYTORCH (TORCH) 路径配置 - 根据平台和构建类型设置
# =============================================================================
if (WIN32)
    # Windows平台的Torch路径配置
    if(CMAKE_CONFIGURATION_TYPES)
        # 多配置生成器（如Visual Studio）使用Release版本
        # 解决PyTorch问题：https://github.com/pytorch/pytorch/issues/155667
        set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
    else()
        # 单配置生成器根据构建类型选择相应版本
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
        elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/debug/libtorch/share/cmake/Torch")
        else()
            # Windows平台仅支持Debug和Release版本
            message(FATAL_ERROR "libtorch binaries only available for Debug and Release on Windows. Current build type: '${CMAKE_BUILD_TYPE}'")
        endif()
    endif()
else()
    # Linux/Unix平台使用统一的Torch路径
    set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch/share/cmake/Torch")
endif()

# =============================================================================
# 编译标准设置
# =============================================================================
set(CMAKE_CXX_STANDARD 23)           # 设置C++标准为C++23
set(CMAKE_CUDA_STANDARD 20)          # 设置CUDA标准为C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 强制要求C++标准
set(CMAKE_CUDA_STANDARD_REQUIRED ON) # 强制要求CUDA标准

# =============================================================================
# 并行构建配置 - 优化编译速度
# =============================================================================
include(ProcessorCount)                                    # 引入处理器核心数检测模块
ProcessorCount(total_cores)                               # 获取系统总核心数
if(total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2")            # 保留2个核心给系统，其余用于编译
    set(ENV{MAKEFLAGS} "-j${used_cores}")                 # 设置Make并行编译参数
    message(STATUS "Building with ${used_cores} cores")    # 输出使用的核心数
endif()

# =============================================================================
# CUDA 环境配置
# =============================================================================
enable_language(CUDA)                          # 启用CUDA语言支持
find_package(CUDAToolkit 12.8 REQUIRED)        # 查找并要求CUDA工具包版本12.8

# 创建nvToolsExt目标库（必须在查找Torch之前，因为PyTorch需要此目标）
if(NOT TARGET CUDA::nvToolsExt)
    # 查找nvToolsExt库文件
    find_library(NVTOOLSEXT_LIBRARY
            NAMES nvToolsExt                    # 库名称
            PATHS                               # 搜索路径
            ${CUDAToolkit_LIBRARY_ROOT}
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDAToolkit_TARGET_DIR}/lib64
            /usr/local/cuda-12.8/lib64
            PATH_SUFFIXES lib lib64             # 路径后缀
            NO_DEFAULT_PATH                     # 不使用默认搜索路径
    )

    if(NVTOOLSEXT_LIBRARY)
        # 找到nvToolsExt库，创建导入目标
        add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
        set_target_properties(CUDA::nvToolsExt PROPERTIES
                IMPORTED_LOCATION ${NVTOOLSEXT_LIBRARY}
        )
        message(STATUS "Created CUDA::nvToolsExt target: ${NVTOOLSEXT_LIBRARY}")
    else()
        # 未找到nvToolsExt库（CUDA 12+中已弃用），创建空接口库
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        message(STATUS "nvToolsExt not found (normal for CUDA 12+), created empty interface target")
    endif()
endif()

message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} found at ${CUDAToolkit_TARGET_DIR}")

# =============================================================================
# 第三方依赖包查找
# =============================================================================
find_package(Torch REQUIRED)                   # PyTorch深度学习框架
find_package(TBB REQUIRED)                     # Intel线程构建块并行库
find_package(Threads REQUIRED)                 # 线程支持
find_package(OpenGL REQUIRED)                  # OpenGL图形库
find_package(glad REQUIRED)                    # OpenGL加载库
find_package(glfw3 REQUIRED)                   # 窗口和输入处理库
find_package(glm REQUIRED)                     # OpenGL数学库
find_package(imgui REQUIRED)                   # 即时模式图形用户界面库
find_package(nlohmann_json REQUIRED)           # JSON解析库
find_package(args REQUIRED)                    # 命令行参数解析库
find_package(spdlog REQUIRED)                  # 快速日志库
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)  # Python开发环境
find_package(Freetype REQUIRED)                # 字体渲染库

# Windows平台特定配置
if(WIN32)
    # 在Windows上使用CUDA-OpenGL互操作时，需要共享CUDA运行时库
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
endif()

# =============================================================================
# CUDA-OPENGL 互操作能力检查
# =============================================================================
if(ENABLE_CUDA_GL_INTEROP)
    include(CheckCXXSourceCompiles)                                           # 引入源码编译检查模块
    set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})  # 设置编译检查需要的头文件路径
    set(CMAKE_REQUIRED_LIBRARIES ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES})      # 设置编译检查需要的库文件
    
    # 测试CUDA-OpenGL互操作头文件是否可用
    check_cxx_source_compiles("
        #ifdef _WIN32
        #include <windows.h>
        #endif
        #include <cuda_runtime.h>
        #include <cuda_gl_interop.h>
        int main() {
            cudaGraphicsResource_t resource;
            return 0;
        }
    " CUDA_GL_INTEROP_FOUND)

    if(CUDA_GL_INTEROP_FOUND)
        message(STATUS "CUDA-OpenGL interop support: ENABLED")
        set(CUDA_GL_INTEROP_ENABLED 1)
    else()
        message(WARNING "CUDA-OpenGL interop support: DISABLED (headers not found)")
        set(CUDA_GL_INTEROP_ENABLED 0)
    endif()
else()
    message(STATUS "CUDA-OpenGL interop support: DISABLED (by user option)")
    set(CUDA_GL_INTEROP_FOUND FALSE)
    set(CUDA_GL_INTEROP_ENABLED 0)
endif()

# =============================================================================
# 配置头文件生成
# =============================================================================
# 根据模板生成包含编译时配置信息的头文件
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"   # 输入模板文件
        "${CMAKE_CURRENT_BINARY_DIR}/include/config.h"      # 输出配置文件
        @ONLY                                                # 仅替换@VAR@格式的变量
)

# =============================================================================
# 子目录添加 - 引入子模块
# =============================================================================
add_subdirectory(gsplat)    # 添加gsplat子模块（高斯散点渲染核心）
add_subdirectory(fastgs)    # 添加fastgs子模块（快速高斯散点实现）

# =============================================================================
# HOST 库配置 - 核心功能实现
# =============================================================================
# 定义主机端（CPU）源文件列表
set(HOST_SOURCES
        src/external/tinyply.cpp                # PLY文件格式解析
        src/application.cpp                     # 应用程序主类
        src/argument_parser.cpp                 # 命令行参数解析
        src/scheduler.cpp                       # 任务调度器
        src/strategy.cpp                        # 训练策略基类
        src/default_strategy.cpp               # 默认训练策略
        src/mcmc.cpp                           # 马尔可夫链蒙特卡洛方法
        src/camera.cpp                         # 相机模型
        src/image_io.cpp                       # 图像输入输出
        src/parameters.cpp                     # 参数管理
        src/splat_data.cpp                     # 散点数据结构
        src/trainer.cpp                        # 训练器
        src/rasterizer.cpp                     # 光栅化器
        src/fast_rasterizer.cpp                # 快速光栅化器
        src/metrics.cpp                        # 评估指标
        src/rasterizer_autograd.cpp            # 光栅化自动微分
        src/fast_rasterizer_autograd.cpp       # 快速光栅化自动微分
        src/bilateral_grid.cpp                 # 双边网格滤波
        src/fused_adam.cpp                     # 融合Adam优化器
        src/training_setup.cpp                 # 训练环境设置
        src/geometry/bounding_box.cpp          # 包围盒几何
        src/geometry/euclidean_transform.cpp   # 欧几里得变换
)

# 创建静态库
add_library(gaussian_host STATIC ${HOST_SOURCES})

# 配置包含目录
target_include_directories(gaussian_host
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include     # 项目公共头文件
        ${CMAKE_CURRENT_BINARY_DIR}/include     # 生成的config.h文件目录
        ${CMAKE_CURRENT_SOURCE_DIR}/gsplat      # gsplat模块头文件
        ${CMAKE_CURRENT_SOURCE_DIR}/fastgs      # fastgs模块头文件
        ${CUDAToolkit_INCLUDE_DIRS}             # CUDA头文件（用于互操作）
        PRIVATE
        ${Python3_INCLUDE_DIRS}                 # Python开发头文件
        ${OPENGL_INCLUDE_DIRS}                  # OpenGL头文件
)

# 配置链接库
target_link_libraries(gaussian_host
        PUBLIC
        ${TORCH_LIBRARIES}                      # PyTorch库
        TBB::tbb                               # Intel TBB并行库
        nlohmann_json::nlohmann_json           # JSON库
        glm::glm                               # 数学库
        Threads::Threads                       # 线程库
        Python3::Python                        # Python库
        gsplat_backend                         # gsplat后端
        fastgs_backend                         # fastgs后端
        CUDA::cudart                           # CUDA运行时（用于互操作）
        spdlog::spdlog                         # 日志库
)

# Unix系统添加动态链接库支持
if(UNIX)
    target_link_libraries(gaussian_host PUBLIC dl)
endif()

# 编译选项配置 - 针对Debug和Release模式优化
target_compile_options(gaussian_host PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer -DDEBUG>    # Debug: 无优化，完整调试信息
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>              # Release: 最高优化，针对本机CPU
)

# =============================================================================
# AVX2 指令集支持检查和配置
# =============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXCompilerFlag)                           # 引入编译器标志检查模块
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)  # 检查是否支持AVX2指令集

    if(COMPILER_SUPPORTS_AVX2)
        # 启用AVX2和FMA指令集优化
        target_compile_options(gaussian_host PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
        target_compile_definitions(gaussian_host PRIVATE HAS_AVX2_SUPPORT)  # 定义AVX2支持宏
        message(STATUS "✓ AVX2 support enabled for gaussian_host")
    else()
        message(WARNING "✗ Compiler does not support AVX2")
    endif()
endif()

# Debug版本添加后缀'd'
set_target_properties(gaussian_host PROPERTIES DEBUG_POSTFIX d)

# 定义项目根路径宏
add_definitions(-DPROJECT_ROOT_PATH="${PROJ_ROOT_DIR}")

# Debug模式的额外编译标志配置
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")      # C++调试标志
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0") # CUDA调试标志
endif()

# =============================================================================
# KERNEL 库配置 - CUDA核函数实现
# =============================================================================
# 定义CUDA内核源文件列表
set(KERNEL_SOURCES
        kernels/ssim.cu                        # 结构相似性指标CUDA实现
        kernels/bilateral_grid_forward.cu      # 双边网格前向传播
        kernels/bilateral_grid_backward.cu     # 双边网格反向传播
        kernels/bilateral_grid_tv.cu           # 双边网格全变分正则化
)

# 如果支持CUDA-OpenGL互操作，添加相关内核
if(CUDA_GL_INTEROP_FOUND)
    list(APPEND KERNEL_SOURCES kernels/cuda_gl_interop.cu)
endif()

# 创建内核静态库
add_library(gaussian_kernels STATIC ${KERNEL_SOURCES})

# 设置CUDA编译属性
set_target_properties(gaussian_kernels PROPERTIES
        CUDA_ARCHITECTURES native              # 自动检测GPU架构
        CUDA_SEPARABLE_COMPILATION ON          # 启用可分离编译
        POSITION_INDEPENDENT_CODE ON           # 位置无关代码
        CUDA_RESOLVE_DEVICE_SYMBOLS ON         # 解析设备符号
)

# 配置内核库包含目录
target_include_directories(gaussian_kernels
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include     # 项目头文件
        ${CMAKE_CURRENT_BINARY_DIR}/include     # 生成的配置头文件
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels  # 内核头文件
        ${CUDAToolkit_INCLUDE_DIRS}             # CUDA头文件
        ${OPENGL_INCLUDE_DIRS}                  # OpenGL头文件
        PRIVATE
        ${Python3_INCLUDE_DIRS}                 # Python头文件
)

# 配置内核库链接依赖
target_link_libraries(gaussian_kernels
        PUBLIC
        CUDA::cudart                           # CUDA运行时
        CUDA::curand                           # CUDA随机数生成库
        CUDA::cublas                           # CUDA基础线性代数子程序库
        ${TORCH_LIBRARIES}                     # PyTorch库
        glm::glm                               # 数学库
        ${OPENGL_LIBRARIES}                    # OpenGL库
        spdlog::spdlog                         # 日志库
)

# CUDA编译选项配置
target_compile_options(gaussian_kernels PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>                    # Debug: 调试信息和行号信息
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -use_fast_math --ptxas-options=-v>  # Release: 最高优化和快速数学
)

# =============================================================================
# 模块配置 - 添加功能模块
# =============================================================================
add_subdirectory(src/loader)      # 数据加载模块
add_subdirectory(src/visualizer)  # 可视化模块

# =============================================================================
# 主可执行文件配置
# =============================================================================
# 创建主程序可执行文件
add_executable(${PROJECT_NAME} src/main.cpp)

# 设置主程序CUDA属性
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES native              # 自动检测GPU架构
        CUDA_SEPARABLE_COMPILATION ON          # 启用可分离编译
        CUDA_RESOLVE_DEVICE_SYMBOLS ON         # 解析设备符号
)

# 配置主程序包含目录
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include     # 项目头文件
        ${CMAKE_CURRENT_BINARY_DIR}/include     # 生成的配置头文件
        ${CMAKE_CURRENT_SOURCE_DIR}/gsplat      # gsplat模块头文件
        ${CMAKE_CURRENT_SOURCE_DIR}/fastgs      # fastgs模块头文件
        ${Python3_INCLUDE_DIRS}                 # Python头文件
        ${CUDAToolkit_INCLUDE_DIRS}             # CUDA头文件
        ${OPENGL_INCLUDE_DIRS}                  # OpenGL头文件
)

# 配置主程序链接库
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        gaussian_host                          # 主机端核心库
        gaussian_kernels                       # CUDA内核库
        gs_loader                              # 数据加载模块
        gs_visualizer                          # 可视化模块
        gsplat_backend                         # gsplat后端
        fastgs_backend                         # fastgs后端
        Python3::Python                        # Python库
        ${OPENGL_LIBRARIES}                    # OpenGL库
        CUDA::cudart                           # CUDA运行时
        spdlog::spdlog                         # 日志库
        taywee::args                           # 命令行参数库
)

# =============================================================================
# 平台特定配置
# =============================================================================
if(WIN32)
    # Windows: 复制PyTorch DLL文件到输出目录
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TORCH_DLL ${TORCH_DLLS})
        add_custom_command(
                TARGET ${PROJECT_NAME}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endforeach()
elseif(UNIX)
    # Unix: 链接OpenGL库并配置运行时库路径
    target_link_libraries(${PROJECT_NAME} PRIVATE GL GLU)

    # 查找PyTorch库目录
    find_path(TORCH_LIB_DIR libtorch_cpu.so
            PATHS "${Torch_DIR}/../../../lib"
            NO_DEFAULT_PATH)

    if(TORCH_LIB_DIR)
        # 设置运行时库搜索路径（RPATH）
        set_target_properties(${PROJECT_NAME} PROPERTIES
                INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${TORCH_LIB_DIR}"  # CUDA和PyTorch库路径
                BUILD_WITH_INSTALL_RPATH TRUE                               # 构建时使用RPATH
                INSTALL_RPATH_USE_LINK_PATH TRUE                           # 使用链接路径
        )
        message(STATUS "Torch library directory: ${TORCH_LIB_DIR}")
    else()
        message(WARNING "Could not find Torch library directory")
    endif()
endif()

# =============================================================================
# 构建类型配置函数
# =============================================================================
function(configure_build_type target)
    get_target_property(target_type ${target} TYPE)      # 获取目标类型

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        # 接口库使用INTERFACE作用域
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} INTERFACE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} INTERFACE RELEASE_BUILD)
        endif()
    else()
        # 其他类型库使用PRIVATE作用域
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} PRIVATE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} PRIVATE RELEASE_BUILD)
        endif()
    endif()
endfunction()

# 为所有主要目标配置构建类型
configure_build_type(gaussian_host)
configure_build_type(gaussian_kernels)
configure_build_type(gsplat_backend)
configure_build_type(fastgs_backend)
configure_build_type(${PROJECT_NAME})

# =============================================================================
# 测试配置（可选）
# =============================================================================
if(BUILD_TESTS)
    enable_testing()                                   # 启用测试支持
    find_package(GTest CONFIG REQUIRED)               # 查找Google Test框架

    # 定义测试源文件列表
    set(TEST_SOURCES
            tests/test_default_strategy.cpp           # 默认策略测试
            tests/test_mcmc.cpp                       # MCMC算法测试
            tests/test_basic.cpp                      # 基础功能测试
            tests/test_rasterization.cpp              # 光栅化测试
            tests/test_gsplat_ops.cpp                 # gsplat操作测试
            tests/test_intersect_debug.cpp            # 相交检测调试测试
            tests/test_autograd.cpp                   # 自动微分测试
            tests/test_numerical_gradients.cpp        # 数值梯度测试
            tests/test_garden_data.cpp                # 花园数据集测试
            tests/torch_impl.cpp                      # PyTorch实现测试
            tests/test_geometry.cpp                   # 几何计算测试
    )

    # 创建测试可执行文件
    add_executable(gaussian_tests ${TEST_SOURCES})

    # 设置测试程序CUDA属性
    set_target_properties(gaussian_tests PROPERTIES
            CUDA_ARCHITECTURES native              # 自动检测GPU架构
            CUDA_SEPARABLE_COMPILATION ON          # 启用可分离编译
            CUDA_RESOLVE_DEVICE_SYMBOLS ON         # 解析设备符号
    )

    # 配置测试程序包含目录
    target_include_directories(gaussian_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include     # 项目头文件
            ${CMAKE_CURRENT_BINARY_DIR}/include     # 生成的配置头文件
            ${CMAKE_CURRENT_SOURCE_DIR}/gsplat      # gsplat模块头文件
            ${CMAKE_CURRENT_SOURCE_DIR}/tests       # 测试头文件
            ${Python3_INCLUDE_DIRS}                 # Python头文件
            ${CUDAToolkit_INCLUDE_DIRS}             # CUDA头文件
            ${OPENGL_INCLUDE_DIRS}                  # OpenGL头文件
    )

    # 配置测试程序链接库
    target_link_libraries(gaussian_tests PRIVATE
            gaussian_host                          # 主机端核心库
            gaussian_kernels                       # CUDA内核库
            gs_loader                              # 数据加载模块
            gs_visualizer                          # 可视化模块
            gsplat_backend                         # gsplat后端
            GTest::gtest                           # Google Test框架
            GTest::gtest_main                      # Google Test主函数
            Python3::Python                        # Python库
            ${OPENGL_LIBRARIES}                    # OpenGL库
            CUDA::cudart                           # CUDA运行时
            spdlog::spdlog                         # 日志库
    )

    # 测试程序CUDA编译选项
    target_compile_options(gaussian_tests PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo -Xcudafe --device-debug>  # CUDA调试选项
    )

    # 复制测试数据目录（如果存在）
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_data")
        add_custom_command(TARGET gaussian_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/test_data"
                "$<TARGET_FILE_DIR:gaussian_tests>/test_data"
        )
    endif()

    # 复制特定测试数据文件（如果存在）
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/test_garden_data.pt")
        add_custom_command(TARGET gaussian_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:gaussian_tests>/tests/data"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/test_garden_data.pt"
                "$<TARGET_FILE_DIR:gaussian_tests>/tests/data/"
        )
    endif()

    # 集成Google Test发现机制
    include(GoogleTest)
    gtest_discover_tests(gaussian_tests)

    # 创建运行测试的自定义目标
    add_custom_target(run_tests
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure  # 失败时输出详细信息
            DEPENDS gaussian_tests                              # 依赖测试可执行文件
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}               # 工作目录
    )

    message(STATUS "Tests enabled. Build with 'make gaussian_tests' and run with 'make run_tests' or 'ctest'")
endif()

# =============================================================================
# 构建信息输出和优化配置
# =============================================================================
# 输出详细的构建配置信息
message(STATUS "===========================================")
message(STATUS "Build Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")              # CUDA版本
message(STATUS "  Torch Version: ${Torch_VERSION}")                   # PyTorch版本
message(STATUS "  Python Version: ${Python3_VERSION}")               # Python版本
message(STATUS "  OpenGL Found: ${OPENGL_FOUND}")                     # OpenGL是否找到
message(STATUS "  FreeType Found: ${FREETYPE_FOUND}")                 # FreeType是否找到
message(STATUS "  CUDA-GL Interop Option: ${ENABLE_CUDA_GL_INTEROP}") # CUDA-OpenGL互操作选项
message(STATUS "  CUDA-GL Interop Available: ${CUDA_GL_INTEROP_FOUND}")  # CUDA-OpenGL互操作可用性
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")                   # 构建类型
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")               # C++标准
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")             # CUDA标准
message(STATUS "  Tests: ${BUILD_TESTS}")                             # 测试是否启用
message(STATUS "===========================================")

# 启用ccache编译缓存加速（如果可用）
find_program(CCACHE_PROGRAM ccache)                    # 查找ccache程序
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")  # 设置全局编译启动规则
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")   # 输出ccache使用信息
endif()