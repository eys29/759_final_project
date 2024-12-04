cmake_minimum_required(VERSION 3.8)

project(sift_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Globally set OpenMP flags (optional)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Optimization level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

#set(CMAKE_CXX_FLAGS "-O3")

#project(sift_cpp LANGUAGES CXX)

add_subdirectory(src)
add_subdirectory(examples)
