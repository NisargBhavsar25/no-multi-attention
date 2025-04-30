cmake_minimum_required(VERSION 3.12)
project(encrypted_seal_transformer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

# Find Microsoft SEAL package
find_package(SEAL 4.1 REQUIRED)

# Define the main executable - Removed main.cpp and using demo_inference.cpp as the main entry point
add_executable(encrypted_seal_transformer
  demo_inference.cpp
  inference_pipeline.cpp
  transformer_weights.cpp
  transformer.cpp
  quadratic_inhibitor_attention.cpp
)

# Include directories
target_include_directories(encrypted_seal_transformer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link against Microsoft SEAL
target_link_libraries(encrypted_seal_transformer PRIVATE SEAL::seal)

# Set output directory
set_target_properties(encrypted_seal_transformer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add compile definitions
target_compile_definitions(encrypted_seal_transformer PRIVATE
  SEAL_VERSION_MAJOR=${SEAL_VERSION_MAJOR}
  SEAL_VERSION_MINOR=${SEAL_VERSION_MINOR}
)

# Installation
install(TARGETS encrypted_seal_transformer
  RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "Configured Encrypted Transformer with Microsoft SEAL")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SEAL version: ${SEAL_VERSION}")

# Additional helper targets
add_custom_target(generate_test_weights
  COMMAND ${CMAKE_COMMAND} -E echo "Generating test weights..."
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/model
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
) 