cmake_minimum_required(VERSION 3.0)

project(
    ray_tracing
    DESCRIPTION "Ray Tracing in One Weekend"
    LANGUAGES CXX
)

find_package(Boost REQUIRED COMPONENTS system filesystem program_options)
find_package(OpenCL REQUIRED)

set(CONSTEXPR OFF CACHE BOOL "whether to enable compile-time ray tracing")
set(IMAGE_WIDTH 1200 CACHE STRING "width of the image")
set(IMAGE_HEIGHT 800 CACHE STRING "height of the image")
set(PATCH_WIDTH 120 CACHE STRING "width of each patch")
set(PATCH_HEIGHT 80 CACHE STRING "height of each patch")
set(PATCH_COORD_X 0 CACHE STRING "x-coordinate of the patch")
set(PATCH_COORD_Y 0 CACHE STRING "y-coordinate of the patch")
set(MAX_DEPTH 50 CACHE STRING "maximum depth for recursive ray tracing")
set(NUM_SAMPLES 500 CACHE STRING "number of samples for MSAA (Multi-Sample Anti-Aliasing)")
set(RANDOM_SEED __LINE__ CACHE STRING "random seed for Monte Carlo approximation")

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

add_executable(
    ray_tracing
    ${SOURCE_DIR}/main.cpp
)

target_include_directories(
    ray_tracing PRIVATE
    ${INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(
    ray_tracing PRIVATE
    Boost::system
    Boost::filesystem
    Boost::program_options
    OpenCL::OpenCL
)

if(CONSTEXPR)
	target_compile_definitions(
        ray_tracing PRIVATE
        CONSTEXPR=constexpr
        IS_CONSTANT_EVALUATED=true
    )
else()
    target_compile_definitions(
        ray_tracing PRIVATE
        CONSTEXPR=;
        IS_CONSTANT_EVALUATED=false
    )
endif()

target_compile_definitions(
    ray_tracing PRIVATE
    IMAGE_WIDTH=${IMAGE_WIDTH}
    IMAGE_HEIGHT=${IMAGE_HEIGHT}
    PATCH_WIDTH=${PATCH_WIDTH}
    PATCH_HEIGHT=${PATCH_HEIGHT}
    PATCH_COORD_X=${PATCH_COORD_X}
    PATCH_COORD_Y=${PATCH_COORD_Y}
    MAX_DEPTH=${MAX_DEPTH}
    NUM_SAMPLES=${NUM_SAMPLES}
    RANDOM_SEED=${RANDOM_SEED}
)

math(EXPR FCONSTEXPR_OPS_LIMIT "1 << 30")
target_compile_options(
    ray_tracing PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<BOOL:${CONSTEXPR}>:-fconstexpr-ops-limit=${FCONSTEXPR_OPS_LIMIT}>
)
target_compile_features(
    ray_tracing PRIVATE
    cxx_std_20
)
