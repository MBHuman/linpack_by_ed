cmake_minimum_required(VERSION 3.16)
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
set(CMAKE_VERBOSE_MAKEFILE ON)

# Paths for building dependencies
set(QtBase_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/qtbase")
set(QtBase_INSTALL_DIR "${CMAKE_BINARY_DIR}/qtbase_install")

set(OpenBLAS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/openblas")
set(OpenBLAS_INSTALL_DIR "${CMAKE_BINARY_DIR}/openblas_install")

set(OpenMP_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/llvm-project/openmp")
set(OpenMP_INSTALL_DIR "${CMAKE_BINARY_DIR}/openmp_install")

# Add OpenBLAS as an ExternalProject (provides BLAS, LAPACK, and CBLAS)
ExternalProject_Add(
    openblas_project
    PREFIX ${CMAKE_BINARY_DIR}/openblas_build
    SOURCE_DIR ${OpenBLAS_SOURCE_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OpenBLAS_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
    BUILD_COMMAND make -j${NUM_PROCESSORS}
    INSTALL_COMMAND make install
)

# Add OpenMP as an ExternalProject (LLVM's OpenMP implementation)
ExternalProject_Add(
    openmp_project
    PREFIX ${CMAKE_BINARY_DIR}/openmp_build
    SOURCE_DIR ${OpenMP_SOURCE_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OpenMP_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
    BUILD_COMMAND cmake --build . --parallel ${NUM_PROCESSORS}
    INSTALL_COMMAND cmake --install .
)

# Automatically build qtbase using ExternalProject_Add
ExternalProject_Add(
    qtbase_project
    PREFIX ${CMAKE_BINARY_DIR}/qtbase_build
    SOURCE_DIR ${QtBase_SOURCE_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${QtBase_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DFEATURE_pkg_config=ON
        -DQT_BUILD_EXAMPLES=OFF
        -DQT_BUILD_TESTS=OFF
    BUILD_COMMAND cmake --build . --parallel ${NUM_PROCESSORS}
    INSTALL_COMMAND cmake --install .
    BUILD_BYPRODUCTS ${QtBase_INSTALL_DIR}/lib/cmake/Qt5/Qt5Config.cmake
)

# Set the CMAKE_PREFIX_PATH to the Qt installation directory
set(CMAKE_PREFIX_PATH "${QtBase_INSTALL_DIR}/lib/cmake")

# Debugging information about Qt installation path
message(STATUS "Looking for Qt5 packages in: ${QtBase_INSTALL_DIR}/lib/cmake/Qt5")

# Find the Qt packages after building Qt5
find_package(Qt5 COMPONENTS Widgets Charts REQUIRED)

# Platform-specific configurations
if(WIN32 OR (UNIX AND NOT APPLE))
    # Include OpenBLAS for Windows and Linux
    include_directories(${OpenBLAS_INSTALL_DIR}/include)
    link_directories(${OpenBLAS_INSTALL_DIR}/lib)
    set(BLAS_LIBRARIES ${OpenBLAS_INSTALL_DIR}/lib/libopenblas.a)
elseif(APPLE)
    # Add library Accelerate for macOS
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        message(STATUS "Found Accelerate: ${ACCELERATE_LIBRARY}")
        set(EXTRA_LIBS ${EXTRA_LIBS} ${ACCELERATE_LIBRARY})
    else()
        message(WARNING "Accelerate library not found!")
    endif()

    # Find Metal framework
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREFRAMEWORK CoreFoundation)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${COREFRAMEWORK})
    message(STATUS "Found Metal, Foundation, CoreFramework: ${METAL_FRAMEWORK}, ${FOUNDATION_FRAMEWORK}, ${COREFRAMEWORK}")
endif()

# Detect system-installed OpenMP
find_package(OpenMP)

if(OpenMP_FOUND)
    message(STATUS "Found system OpenMP: ${OpenMP_CXX_FLAGS}")
    set(OPENMP_LIBRARIES OpenMP::OpenMP_CXX)
else()
    message(WARNING "System OpenMP not found, using the submodule.")
    include_directories(${OpenMP_INSTALL_DIR}/include)
    if(WIN32)
        set(OPENMP_LIBRARIES ${OpenMP_INSTALL_DIR}/lib/libomp.dll)
    elseif(APPLE)
        set(OPENMP_LIBRARIES ${OpenMP_INSTALL_DIR}/lib/libomp.dylib)
    else()
        set(OPENMP_LIBRARIES ${OpenMP_INSTALL_DIR}/lib/libomp.so)
    endif()
endif()

# Enable CUDA only for Windows and Linux on x86_64
if(WIN32 OR (UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    # Enable CUDA language in CMake for Linux and Windows
    enable_language(CUDA)

    # Define GPU sources for CUDA
    set(GPU_SOURCES
        src/gpu/cuda/linpack_gpu.cu
        src/gpu/cuda/matrix_multiply.cu
    )
    # Set compile flag for CUDA files
    set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE CUDA)

    # Add CUDA samples as an external submodule and ensure it is built
    set(CudaSamples_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/cuda-samples")
    set(CudaSamples_BUILD_DIR "${CMAKE_BINARY_DIR}/cuda_samples_build")
    set(CudaSamples_INSTALL_DIR "${CMAKE_BINARY_DIR}/cuda_samples_install")

    ExternalProject_Add(
        cuda_samples_project
        PREFIX ${CudaSamples_BUILD_DIR}
        SOURCE_DIR ${CudaSamples_SOURCE_DIR}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${CudaSamples_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
        BUILD_COMMAND cmake --build . --parallel ${NUM_PROCESSORS}
        INSTALL_COMMAND cmake --install .
    )
    # Make sure linpack depends on the CUDA samples project
    add_dependencies(linpack cuda_samples_project)
    include_directories(${CudaSamples_INSTALL_DIR}/include)
    link_directories(${CudaSamples_INSTALL_DIR}/lib)
endif()

# Define GPU sources conditionally based on platform
if(APPLE)
    set(GPU_SOURCES
        src/gpu/metal_osx/linpack_gpu.mm
    )
    set(GPU_HEADERS
        src/gpu/metal_osx/gpu.metal
    )
endif()

# Add executable and all source files
add_executable(linpack
    src/main.cpp
    src/linpack.cpp
    src/mainwindow.cpp
    ${GPU_SOURCES}
)

# Add GUI files for the main executable
target_sources(linpack PRIVATE src/mainwindow.ui)

# Ensure linpack depends on QtBase, OpenBLAS, and OpenMP being built
add_dependencies(linpack qtbase_project openblas_project openmp_project)

# Link the linpack executable with the necessary Qt5 libraries, BLAS, and OpenMP
if(APPLE)
    target_link_libraries(linpack Qt5::Widgets Qt5::Charts ${EXTRA_LIBS} ${OPENMP_LIBRARIES})
else()
    target_link_libraries(linpack Qt5::Widgets Qt5::Charts ${BLAS_LIBRARIES} ${OPENMP_LIBRARIES})
endif()

# Add optimization flags for Release builds
if(CMAKE_BUILD_TYPE MATCHES Release)
    message("Configuring with -O3 optimization for Release build")
    target_compile_options(linpack PRIVATE -O3)
endif()

# Set Objective-C++ compile flag for the .mm file (macOS)
if(APPLE)
    set_source_files_properties(src/gpu/metal_osx/linpack_gpu.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
endif()
