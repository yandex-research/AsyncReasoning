cmake_minimum_required(VERSION 3.22)
project(AsyncReasoningCache CUDA CXX)
include(FetchContent)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90a;100")

find_package(CUDAToolkit REQUIRED)

option(ASYNC_REASONING_TORCH_BINDINGS "Build pytorch bindings" OFF)
option(ASYNC_REASONING_BENCHMARKS "Build the benchmark program" ON)
option(ASYNC_REASONING_TESTS "Build the test program" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()


add_subdirectory(src/tpl)

option(ASYNC_REASONING_ENABLE_FLOAT "build with float32 kernel support" ON)
option(ASYNC_REASONING_ENABLE_HALF "build with float16 kernel support" ON)
option(ASYNC_REASONING_ENABLE_BFLOAT "build with bfloat16 kernel support" ON)

add_library(async-reasoning-kernels STATIC ${ASYNC_REASONING_KERNEL_INSTANTIATIONS})
target_include_directories(async-reasoning-kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(async-reasoning-kernels PRIVATE -expt-relaxed-constexpr -lineinfo)

if(ASYNC_REASONING_ENABLE_FLOAT)
    target_compile_definitions(async-reasoning-kernels PUBLIC ASYNC_REASONING_ENABLE_FLOAT=1)
endif ()

if(ASYNC_REASONING_ENABLE_HALF)
    target_compile_definitions(async-reasoning-kernels PUBLIC ASYNC_REASONING_ENABLE_HALF=1)
endif ()

if(ASYNC_REASONING_ENABLE_BFLOAT)
    target_compile_definitions(async-reasoning-kernels PUBLIC ASYNC_REASONING_ENABLE_BFLOAT=1)
endif ()


if(ASYNC_REASONING_BENCHMARKS OR ASYNC_REASONING_TESTS)
    FetchContent_Declare(
            cli11
            QUIET
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG v2.4.2
    )
    FetchContent_MakeAvailable(cli11)
endif ()

if(ASYNC_REASONING_TORCH_BINDINGS)
    include(cmake/pytorch.cmake)
    find_package(Python3 COMPONENTS Development.SABIModule REQUIRED)
    Python3_add_library(asyncreasoningatt MODULE src/binding.cu WITH_SOABI USE_SABI 3.10)
    if(SKBUILD)
        install(TARGETS asyncreasoningatt DESTINATION ${SKBUILD_PLATLIB_DIR}/async_reasoning_inference)
    endif ()
    target_link_libraries(asyncreasoningatt PRIVATE torch)
    target_compile_options(asyncreasoningatt PUBLIC -expt-relaxed-constexpr -lineinfo)
    target_include_directories(asyncreasoningatt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif ()

if(ASYNC_REASONING_BENCHMARKS)
    add_executable(bench src/benchmark.cpp)
    target_include_directories(bench PUBLIC src)
    target_link_libraries(bench PRIVATE CLI11::CLI11 CUDA::cudart async-reasoning-kernels)
endif ()

if(ASYNC_REASONING_TESTS)
    add_executable(test src/test.cpp)
    target_include_directories(test PUBLIC src)
    target_link_libraries(test PRIVATE CLI11::CLI11 CUDA::cudart async-reasoning-kernels)
endif ()
