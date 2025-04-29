cmake_minimum_required(VERSION 3.10)
project(encrypted_transformer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set HEonGPU directory
set(HEONGPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/HEonGPU" CACHE PATH "Path to HEonGPU installation")

# Include important directories directly
include_directories(${HEONGPU_DIR}/src)
include_directories(${HEONGPU_DIR}/src/heongpu/include)
include_directories(${HEONGPU_DIR}/src/heongpu/include/host)
include_directories(${HEONGPU_DIR}/src/heongpu/include/host/bfv)
include_directories(${HEONGPU_DIR}/src/heongpu/include/host/ckks)
include_directories(${HEONGPU_DIR}/src/heongpu/include/util)
include_directories(${HEONGPU_DIR}/src/heongpu/include/kernel)
include_directories(${HEONGPU_DIR}/thirdparty/GPU-NTT/src/include)
include_directories(${HEONGPU_DIR}/thirdparty/GPU-NTT/src/include/common)
include_directories(${HEONGPU_DIR}/thirdparty/GPU-NTT/src/include/ntt_merge)
include_directories(${HEONGPU_DIR}/thirdparty/RNGonGPU/src/rngongpu/include)
include_directories(${HEONGPU_DIR}/thirdparty/RNGonGPU/src/rngongpu/include/rand_aes)
include_directories(${HEONGPU_DIR}/thirdparty/RNGonGPU/src/rngongpu/include/common)
include_directories(${HEONGPU_DIR}/thirdparty/GPU-FFT/src/include)

# Include bundled spdlog and fmt directories
include_directories(${HEONGPU_DIR}/build/_deps/fmt-src/include)
include_directories(${HEONGPU_DIR}/build/_deps/spdlog-src/include)
include_directories(${HEONGPU_DIR}/build/_deps/spdlog-build/include)

# Use our own copies of the headers to avoid conflicts
include_directories(BEFORE ${HEONGPU_DIR}/build/_deps/spdlog-src/include)
include_directories(BEFORE ${HEONGPU_DIR}/build/_deps/fmt-src/include)

# Add RMM includes
include_directories(${HEONGPU_DIR}/build/_deps/rmm-src/include)
include_directories(${HEONGPU_DIR}/build/_deps/rmm-build/include)
include_directories(${HEONGPU_DIR}/build/_deps/nvtx3-src/c/include)

# Add CCCL includes - make sure these come before other includes to prevent namespace issues
include_directories(BEFORE ${HEONGPU_DIR}/build/_deps/cccl-src/libcudacxx/include)
include_directories(BEFORE ${HEONGPU_DIR}/build/_deps/cccl-src/cuda/std/detail)
include_directories(BEFORE ${HEONGPU_DIR}/build/_deps/cccl-src)

# Add build directories
include_directories(${HEONGPU_DIR}/build/src)
link_directories(${HEONGPU_DIR}/build/src/heongpu)

# Define CUDA architecture - adjust based on your GPU
# For compatibility, let's support multiple architectures
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Add CUDA compatibility definitions
add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM)
add_definitions(-DCUDA_API_WRAPPERS_USE_THIS_THRUST_NAMESPACE=cuda::std)

# Source files
set(SOURCES
    quadratic_inhibitor_attention.cpp
    transformer.cpp
    transformer_weights.cpp
    inference_pipeline.cpp
    demo_inference.cpp
)

# Create executable
add_executable(demo_inference ${SOURCES})

# Link libraries
target_link_libraries(demo_inference
    heongpu-1.1
    ${CUDA_LIBRARIES}
    pthread
)

# Set compile definitions for RMM compatibility
target_compile_definitions(demo_inference PRIVATE 
    RMM_CUDA_INCLUDE_PATH="${CUDA_INCLUDE_DIRS}"
    LIBCUDACXX_ENABLE_CUDA_IN_CUDA_FILES
)

# Enable CUDA separable compilation
set_target_properties(demo_inference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Installation
install(TARGETS demo_inference
    RUNTIME DESTINATION bin
)

# Create directory for model weights
add_custom_target(create_model_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/model
)

# Add target to generate dummy model weights for testing
add_custom_target(generate_test_weights
    COMMAND ${CMAKE_COMMAND} -E echo "Generating test weight files in ${CMAKE_BINARY_DIR}/model"
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_test_weights.cmake
    DEPENDS create_model_dir
)

# Create the script to generate test weights
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_test_weights.cmake
"
# This script generates dummy weight files for testing the encrypted transformer
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/model)

# Function to create a dummy weight file
function(create_dummy_weight_file filename num_layers element_count)
    # Create binary file with:
    # - 4 bytes: number of layers (uint32_t)
    # - For each layer:
    #   - 4 bytes: number of elements (uint32_t)
    #   - 8 * num_elements bytes: elements (double)
    
    file(WRITE ${CMAKE_BINARY_DIR}/model/${filename} \"dummy\")
    
    # In a real script, this would write actual binary data
    # For now, we just create empty files as placeholders
    message(STATUS \"Created dummy weight file: ${CMAKE_BINARY_DIR}/model/${filename}\")
endfunction()

# Create weight files
create_dummy_weight_file(wq.bin 1 16384)
create_dummy_weight_file(wk.bin 1 16384)
create_dummy_weight_file(wv.bin 1 16384)
create_dummy_weight_file(wo.bin 1 16384)
create_dummy_weight_file(ff1.bin 1 16384)
create_dummy_weight_file(ff2.bin 1 16384)
")

message(STATUS "Configured Encrypted Transformer project with HEonGPU") 