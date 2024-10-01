cmake_minimum_required(VERSION 3.16)

# Project Declaration
if(WIN32 OR (UNIX AND NOT APPLE))
    project(Linpack LANGUAGES CXX CUDA)
else()
    project(Linpack LANGUAGES CXX)
endif()

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Qt usage options for automoc, autouic, autorcc
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Add CMake module for ExternalProject
include(ExternalProject)

# Processor count helper for parallel builds
include(ProcessorCount)
ProcessorCount(NUM_PROCESSORS)
if(NOT NUM_PROCESSORS OR NUM_PROCESSORS EQUAL 0)
    set(NUM_PROCESSORS 4)  # Fallback to a reasonable default if detection fails
endif()

# Add verbose output for debugging
# set(CMAKE_VERBOSE_MAKEFILE ON)

# Paths for building dependencies from the repository submodules
set(QtBase_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/qtbase")
set(QtBase_INSTALL_DIR "${CMAKE_BINARY_DIR}/qtbase_install")

set(OpenBLAS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/openblas")
set(OpenBLAS_INSTALL_DIR "${CMAKE_BINARY_DIR}/openblas_install")

set(OpenMP_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/llvm-project/openmp")
set(OpenMP_INSTALL_DIR "${CMAKE_BINARY_DIR}/openmp_install")

set(CudaSamples_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/cuda-samples")
set(CudaSamples_INSTALL_DIR "${CMAKE_BINARY_DIR}/cuda_samples_install")

# Add ExternalProject for QtBase
ExternalProject_Add(
    qtbase_project
    PREFIX ${CMAKE_BINARY_DIR}/qtbase_build
    SOURCE_DIR ${QtBase_SOURCE_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${QtBase_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DFEATURE_pkg_config=ON
        -DQT_BUILD_EXAMPLES=OFF                # Disable examples
        -DQT_BUILD_TESTS=OFF                   # Disable tests
    BUILD_COMMAND cmake --build . --parallel ${NUM_PROCESSORS}
    INSTALL_COMMAND cmake --install .
    BUILD_BYPRODUCTS ${QtBase_INSTALL_DIR}/lib/cmake/Qt5/Qt5Config.cmake
)

# Add ExternalProject for OpenBLAS
ExternalProject_Add(
    openblas_project
    PREFIX ${CMAKE_BINARY_DIR}/openblas_build
    SOURCE_DIR ${OpenBLAS_SOURCE_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OpenBLAS_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_TESTING=OFF                    # If applicable, disable tests
    BUILD_COMMAND make -j${NUM_PROCESSORS}
    INSTALL_COMMAND make install
)

# Add ExternalProject for OpenMP
ExternalProject_Add(
    openmp_project
    PREFIX ${CMAKE_BINARY_DIR}/openmp_build
    SOURCE_DIR ${OpenMP_SOURCE_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OpenMP_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DLLVM_ENABLE_TESTS=OFF                # Disable LLVM tests (specific to OpenMP build system)
    BUILD_COMMAND cmake --build . --parallel ${NUM_PROCESSORS}
    INSTALL_COMMAND cmake --install .
)

# Add CUDA Samples as an ExternalProject (only on supported platforms)
if(WIN32 OR (UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    ExternalProject_Add(
        cuda_samples_project
        PREFIX ${CMAKE_BINARY_DIR}/cuda_samples_build
        SOURCE_DIR ${CudaSamples_SOURCE_DIR}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${CudaSamples_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_TESTING=OFF                # Disable tests if CUDA Samples support this option
        BUILD_COMMAND cmake --build . --parallel ${NUM_PROCESSORS}
        INSTALL_COMMAND cmake --install .
    )
endif()

# Add a custom target to configure and build linpack after all dependencies are installed
add_custom_target(configure_and_build_linpack ALL
    COMMAND ${CMAKE_COMMAND} -DCMAKE_PREFIX_PATH=${QtBase_INSTALL_DIR}/lib/cmake -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR}/linpack_build
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/linpack_build --parallel ${NUM_PROCESSORS}
    DEPENDS qtbase_project openblas_project openmp_project
)

# If CUDA samples are required, add as a dependency
if(WIN32 OR (UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    add_dependencies(configure_and_build_linpack cuda_samples_project)
endif()

# Create CMakeLists.txt for Linpack in a new directory (linpack_build)
file(WRITE ${CMAKE_BINARY_DIR}/linpack_build/CMakeLists.txt "
cmake_minimum_required(VERSION 3.16)
project(Linpack LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Set CMAKE_PREFIX_PATH to QtBase install directory to find Qt5
set(CMAKE_PREFIX_PATH \"${QtBase_INSTALL_DIR}/lib/cmake\")

# Find Qt5 Components
find_package(Qt5 COMPONENTS Widgets Charts REQUIRED)

# Define linpack executable
add_executable(linpack
    \${CMAKE_SOURCE_DIR}/src/main.cpp
    \${CMAKE_SOURCE_DIR}/src/linpack.cpp
    \${CMAKE_SOURCE_DIR}/src/mainwindow.cpp
)

# Add GPU sources conditionally
if(APPLE)
    target_sources(linpack PRIVATE
        \${CMAKE_SOURCE_DIR}/src/gpu/metal_osx/linpack_gpu.mm
    )
    set_source_files_properties(\${CMAKE_SOURCE_DIR}/src/gpu/metal_osx/linpack_gpu.mm PROPERTIES COMPILE_FLAGS \"-x objective-c++\")
elseif(WIN32 OR (UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL \"x86_64\"))
    enable_language(CUDA)
    target_sources(linpack PRIVATE
        \${CMAKE_SOURCE_DIR}/src/gpu/cuda/linpack_gpu.cu
        \${CMAKE_SOURCE_DIR}/src/gpu/cuda/matrix_multiply.cu
    )
endif()

# Include directories for dependencies
include_directories(
    ${OpenBLAS_INSTALL_DIR}/include
    ${OpenMP_INSTALL_DIR}/include
)

# Link directories for dependencies
link_directories(
    ${OpenBLAS_INSTALL_DIR}/lib
    ${OpenMP_INSTALL_DIR}/lib
)

# Link the linpack executable with necessary libraries
target_link_libraries(linpack Qt5::Widgets Qt5::Charts)

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(linpack \${ACCELERATE_LIBRARY})
    endif()
else()
    target_link_libraries(linpack
        ${OpenBLAS_INSTALL_DIR}/lib/libopenblas.a
        ${OpenMP_INSTALL_DIR}/lib/libomp.so
    )
endif()
")

# This custom target ensures that the linpack_build configuration happens
add_custom_target(build_linpack
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/linpack_build --target linpack
    DEPENDS configure_and_build_linpack
)
