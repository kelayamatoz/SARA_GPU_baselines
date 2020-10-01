cmake_minimum_required(VERSION 3.10)

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(cuda_radix_sort LANGUAGES CXX CUDA)
project(cuda_base CXX CUDA)

option(__CUDA_RUNTIME_H__ ON)

if ($ENV{CLION_IDE})
    include_directories(
            "/usr/local/cuda/lib64/"
            "/usr/local/cuda/include/"
    )
endif ()

include_directories(
        "../common/inc/"
        "../common/lib/"
)
add_library(cuda_base
        lib/radixSortThrust.cu
        lib/radixSortThrust.h
        )
target_compile_features(cuda_base PUBLIC cxx_std_11)
set_target_properties( cuda_base
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")

add_executable(cuda_radix_sort
        main.cpp
        )

set_target_properties(
        cuda_radix_sort
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_radix_sort cuda_base)
