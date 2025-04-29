cmake_minimum_required(VERSION 3.26)
project(EncryptedTransformer LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find HEonGPU
find_package(HEonGPU REQUIRED)
if(NOT HEonGPU_FOUND)
    message(FATAL_ERROR "HEonGPU not found. Please install HEonGPU first.")
endif()

# Source files
set(SOURCES
    main.cpp
    quadratic_inhibitor_attention.cpp
    transformer_weights.cpp
    transformer.cpp
    inference_pipeline.cpp
)

# Header files
set(HEADERS
    encrypted_transformer.h
)

# Add executable
add_executable(encrypted_transformer ${SOURCES} ${HEADERS})

# Set CUDA architectures based on your GPU
# For RTX 30 series, use 86
# For RTX 40 series, use 89
set_target_properties(encrypted_transformer PROPERTIES CUDA_ARCHITECTURES 89)

# Link HEonGPU and CUDA
target_link_libraries(encrypted_transformer
    PRIVATE 
    HEonGPU::heongpu
    CUDA::cudart
)

# Enable CUDA separable compilation
set_target_properties(encrypted_transformer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Installation
install(TARGETS encrypted_transformer
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