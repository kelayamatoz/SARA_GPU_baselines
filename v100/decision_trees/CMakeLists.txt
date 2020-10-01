cmake_minimum_required(VERSION 3.10)

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(cuda_cart LANGUAGES CXX CUDA)
project(cuda_base CXX CUDA)

find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIR})

if ($ENV{CLION_IDE})
    include_directories(
            "/usr/local/cuda/lib64/"
            "/usr/local/cuda/include/"
    )
endif ()

include_directories(
        "../common/inc"
        "../common/lib"
)
add_library(cuda_base
        lib/cuda_main.cu lib/cuda_main.h
        lib/dummy.cpp lib/cuda_headers.h
        lib/cart_inference.cu lib/cart_inference.h
        )
target_compile_features(cuda_base PUBLIC cxx_std_11)
set_target_properties( cuda_base
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")

add_executable(cuda_cart
        main.cpp
        lib/node.h
        lib/utils.h
        )

set_target_properties(
        cuda_cart
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_cart cuda_base)
