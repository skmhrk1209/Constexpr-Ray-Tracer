cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(
    ray_tracer
    DESCRIPTION "Ray Tracing in One Weekend"
    LANGUAGES CXX
)

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(
    ray_tracer
    ${SOURCE_DIR}/main.cpp
)
target_include_directories(
    ray_tracer PRIVATE
    ${INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
)
target_link_libraries(
    ray_tracer PRIVATE
    Boost::system
    Boost::filesystem
    Boost::program_options
    Eigen3::Eigen
    OpenCL::OpenCL
)
