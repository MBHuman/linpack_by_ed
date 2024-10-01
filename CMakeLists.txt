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

# Paths for building dependencies from the repository submodules
set(QtBase_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/qtbase")
set(QtBase_INSTALL_DIR "${CMAKE_BINARY_DIR}/qtbase_install")

set(OpenBLAS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/openblas")
set(OpenBLAS_INSTALL_DIR "${CMAKE_BINARY_DIR}/openblas_install")

set(OpenMP_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/llvm-project/openmp")
set(OpenMP_INSTALL_DIR "${CMAKE_BINARY_DIR}/openmp_install")

set(CudaSamples_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/cuda-samples")
set(CudaSamples_INSTALL_DIR "${CMAKE_BINARY_DIR}/cuda_samples_install")

# Flags to control whether we need to add dependencies as external projects
set(QtBase_NEEDS_BUILD FALSE)
set(OpenBLAS_NEEDS_BUILD FALSE)
set(OpenMP_NEEDS_BUILD FALSE)
set(CudaSamples_NEEDS_BUILD FALSE)

# Check if system Qt5 is available, if not build it from source
find_package(Qt5 COMPONENTS Widgets Charts QUIET)
if(NOT Qt5_FOUND)
    message(STATUS "Qt5 not found in the system, building from source.")
    set(QtBase_NEEDS_BUILD TRUE)
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
    )
    list(APPEND ADDITIONAL_INCLUDES "${QtBase_INSTALL_DIR}/include")
    list(APPEND ADDITIONAL_LIB_DIRS "${QtBase_INSTALL_DIR}/lib")
    set(CMAKE_PREFIX_PATH "${QtBase_INSTALL_DIR}/lib/cmake")
else()
    message(STATUS "Qt5 found in the system, using it.")
endif()

# Check if system OpenBLAS is available, if not build it from source
find_library(OPENBLAS_LIB openblas PATHS /usr/local/lib /usr/lib NO_DEFAULT_PATH)
if(NOT OPENBLAS_LIB)
    message(STATUS "OpenBLAS not found in the system, building from source.")
    set(OpenBLAS_NEEDS_BUILD TRUE)
    ExternalProject_Add(
        openblas_project
        PREFIX ${CMAKE_BINARY_DIR}/openblas_build
        SOURCE_DIR ${OpenBLAS_SOURCE_DIR}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${OpenBLAS_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_TESTING=OFF                    # Disable tests
        BUILD_COMMAND make -j${NUM_PROCESSORS}
        INSTALL_COMMAND make install
    )
    list(APPEND ADDITIONAL_INCLUDES "${OpenBLAS_INSTALL_DIR}/include")
    list(APPEND ADDITIONAL_LIB_DIRS "${OpenBLAS_INSTALL_DIR}/lib")
    set(OPENBLAS_LIB "${OpenBLAS_INSTALL_DIR}/lib/libopenblas.a")
else()
    message(STATUS "OpenBLAS found in the system, using it.")
endif()

# Check if system OpenMP is available, if not build it from source
find_package(OpenMP QUIET)
if(NOT OpenMP_FOUND)
    message(STATUS "OpenMP not found in the system, building from source.")
    set(OpenMP_NEEDS_BUILD TRUE)
    ExternalProject_Add(
        openmp_project
        PREFIX ${CMAKE_BINARY_DIR}/openmp_build
        SOURCE_DIR ${OpenMP_SOURCE_DIR}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${OpenMP_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DLLVM_ENABLE_TESTS=OFF                # Disable LLVM tests
        BUILD_COMMAND cmake --build . --parallel ${NUM_PROCESSORS}
        INSTALL_COMMAND cmake --install .
    )
    list(APPEND ADDITIONAL_INCLUDES "${OpenMP_INSTALL_DIR}/include")
    list(APPEND ADDITIONAL_LIB_DIRS "${OpenMP_INSTALL_DIR}/lib")
else()
    message(STATUS "OpenMP found in the system, using it.")
endif()

# Add the executable linpack
add_executable(linpack
    src/main.cpp
    src/linpack.cpp
    src/mainwindow.cpp
)

# Add GPU sources conditionally
if(APPLE)
    target_sources(linpack PRIVATE src/gpu/metal_osx/linpack_gpu.mm)
    set_source_files_properties(src/gpu/metal_osx/linpack_gpu.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
    # Find Metal and Foundation frameworks
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
    set(EXTRA_LIBS ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${COCOA_FRAMEWORK})
elseif(WIN32 OR (UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    enable_language(CUDA)
    target_sources(linpack PRIVATE
        src/gpu/cuda/linpack_gpu.cu
        src/gpu/cuda/matrix_multiply.cu
    )
endif()

# Add the .ui file to the target sources to trigger the Qt UIC
target_sources(linpack PRIVATE src/mainwindow.ui)

# Include directories for dependencies and generated files
include_directories(${ADDITIONAL_INCLUDES})
include_directories(${CMAKE_BINARY_DIR}) # Include the directory for generated files like ui_mainwindow.h

# Link directories for dependencies
link_directories(${ADDITIONAL_LIB_DIRS})

# Link the linpack executable with necessary libraries
target_link_libraries(linpack Qt5::Widgets Qt5::Charts)

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
    target_link_libraries(linpack ${ACCELERATE_LIBRARY} ${EXTRA_LIBS})
else()
    target_link_libraries(linpack ${OPENBLAS_LIB} OpenMP::OpenMP_CXX)
endif()

# Ensure the linpack target depends on the external projects if needed
if(QtBase_NEEDS_BUILD)
    add_dependencies(linpack qtbase_project)
endif()

if(OpenBLAS_NEEDS_BUILD)
    add_dependencies(linpack openblas_project)
endif()

if(OpenMP_NEEDS_BUILD)
    add_dependencies(linpack openmp_project)
endif()
