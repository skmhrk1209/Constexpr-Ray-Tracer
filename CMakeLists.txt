cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og -pg")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -s -DNDEBUG -march=native")

project(
    ray_tracer
    DESCRIPTION "Ray Tracing in One Weekend"
    LANGUAGES CXX
)

find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
find_package(OpenCL REQUIRED)

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

add_executable(
    ray_tracer
    ${SOURCE_DIR}/main.cpp
    $ENV{INSTANTIATED_SOURCES}
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
    OpenCL::OpenCL
)
