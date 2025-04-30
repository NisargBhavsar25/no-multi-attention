cmake_minimum_required(VERSION 3.10)

# Explicitly set CUDA compiler path
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

# Enable CUDA language before project declaration
enable_language(CUDA)

project(encrypted_transformer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures - targeting Pascal architecture (6.1) for Quadro P6000
set(CMAKE_CUDA_ARCHITECTURES "61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 --expt-relaxed-constexpr -Xcompiler -fPIC")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_61 --expt-relaxed-constexpr -Xcompiler -fPIC")

# Turn off host compiler warning propagation
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# Add CUDA include and lib paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda-12.8/include)
link_directories(/usr/local/cuda-12.8/lib64)

# Add local include directory for CUDA compatibility
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set HEonGPU directory - adjust path if needed
set(HEONGPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/HEonGPU")

# Include directories for HEonGPU
include_directories(${HEONGPU_DIR}/src)
include_directories(${HEONGPU_DIR}/src/heongpu/include)
include_directories(${HEONGPU_DIR}/src/heongpu/include/host)
include_directories(${HEONGPU_DIR}/src/heongpu/include/host/bfv)
include_directories(${HEONGPU_DIR}/src/heongpu/include/host/ckks)
include_directories(${HEONGPU_DIR}/src/heongpu/include/util)
include_directories(${HEONGPU_DIR}/src/heongpu/include/kernel)

# Include additional required dependencies
include_directories(${HEONGPU_DIR}/build/_deps/spdlog-src/include)
include_directories(${HEONGPU_DIR}/build_examples/_deps/spdlog-src/include)

# Include directories for third-party libraries
include_directories(${HEONGPU_DIR}/thirdparty/GPU-NTT/src/include)
include_directories(${HEONGPU_DIR}/thirdparty/GPU-NTT/src/include/common)
include_directories(${HEONGPU_DIR}/thirdparty/GPU-NTT/src/include/ntt_merge)
include_directories(${HEONGPU_DIR}/thirdparty/RNGonGPU/src/rngongpu/include)
include_directories(${HEONGPU_DIR}/thirdparty/RNGonGPU/src/rngongpu/include/rand_aes)
include_directories(${HEONGPU_DIR}/thirdparty/RNGonGPU/src/rngongpu/include/common)
include_directories(${HEONGPU_DIR}/thirdparty/GPU-FFT/src/include)

# Add build directories
include_directories(${HEONGPU_DIR}/build/src)
link_directories(${HEONGPU_DIR}/build/src/heongpu)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Source files
set(SOURCES
    quadratic_inhibitor_attention.cpp
    transformer.cpp
    transformer_weights.cpp
    inference_pipeline.cpp
    demo_inference.cpp
)

# Explicitly set the language to CUDA for all source files
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)

# Additional source for main executable
set(MAIN_SOURCES
    main.cpp
    quadratic_inhibitor_attention.cpp
    transformer.cpp
    transformer_weights.cpp
    inference_pipeline.cpp
)
set_source_files_properties(${MAIN_SOURCES} PROPERTIES LANGUAGE CUDA)

# Create the demo executable
add_executable(demo_inference ${SOURCES})

# Create the main executable
add_executable(main_app ${MAIN_SOURCES})

# Link libraries for demo_inference
target_link_libraries(demo_inference
    heongpu
    ${CUDA_LIBRARIES}
    cudart
    cuda
    OpenMP::OpenMP_CXX
    -lgomp
)

# Link libraries for main_app
target_link_libraries(main_app
    heongpu
    ${CUDA_LIBRARIES}
    cudart
    cuda
    OpenMP::OpenMP_CXX
    -lgomp
)

# Set target properties for demo_inference
set_target_properties(demo_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
)

# Set target properties for main_app
set_target_properties(main_app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
)

# Set CUDA compilation options for demo_inference
target_compile_options(demo_inference PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Set CUDA compilation options for main_app
target_compile_options(main_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Add custom target to create model directory
add_custom_target(create_model_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/model
)

# Installation
install(TARGETS demo_inference main_app
    RUNTIME DESTINATION bin
)

# Output configuration information
message(STATUS "Configured Encrypted Transformer with HEonGPU")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "HEonGPU directory: ${HEONGPU_DIR}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")