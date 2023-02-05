#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>

#include "camera.hpp"
#include "common.hpp"
#include "geometry.hpp"
#include "image.hpp"
#include "math.hpp"
#include "reflection.hpp"
#include "rendering.hpp"
#include "tensor.hpp"

#define CONSTEXPR  // constexpr

int main() {
    using Scalar = double;
    using namespace rendex::tensor;
    using namespace rendex::reflection;
    using namespace std::literals::complex_literals;

    // object
    constexpr auto object = rendex::geometry::construct_union(
        rendex::geometry::Sphere<Scalar, Vector, Metal>(
            0.5, Vector<Scalar, 3>{1.0, 0.0, 1.0},
            Metal<Scalar, Vector>(
                Vector<std::complex<Scalar>, 3>{0.18299 + 3.4242i, 0.42108 + 2.34590i, 1.37340 + 1.77040i}, 0.0)),
        rendex::geometry::construct_union(
            rendex::geometry::Sphere<Scalar>(
                0.5, Vector<Scalar, 3>{-1.0, 0.0, 1.0},
                Dielectric<Scalar, Vector>(Vector<Scalar, 3>{0.0, 0.0, 0.0}, Vector<Scalar, 3>{1.0, 1.0, 1.0}, 1.5,
                                           0.0, 0.0)),
            rendex::geometry::construct_union(
                rendex::geometry::Sphere<Scalar>(
                    -0.4, Vector<Scalar, 3>{-1.0, 0.0, 1.0},
                    Dielectric<Scalar, Vector>(Vector<Scalar, 3>{0.0, 0.0, 0.0}, Vector<Scalar, 3>{1.0, 1.0, 1.0}, 1.5,
                                               0.0, 0.0)),
                rendex::geometry::construct_union(
                    rendex::geometry::Sphere<Scalar>(
                        0.5, Vector<Scalar, 3>{0.0, 0.0, 1.0},
                        Dielectric<Scalar>(Vector<Scalar, 3>{0.1, 0.2, 0.5}, Vector<Scalar, 3>{0.0, 0.0, 0.0}, 1.0,
                                           1.0, 0.0)),
                    rendex::geometry::Sphere<Scalar>(
                        100.0, Vector<Scalar, 3>{0.0, 100.5, 1.0},
                        Dielectric<Scalar>(Vector<Scalar, 3>{0.8, 0.8, 0.0}, Vector<Scalar, 3>{0.0, 0.0, 0.0}, 1.0,
                                           1.0, 0.0))))));

    constexpr auto object__ = rendex::geometry::construct_union(
        rendex::geometry::Sphere<Scalar, Vector, Metal>(
            0.5, Vector<Scalar, 3>{1.0, 0.0, 1.0},
            Metal<Scalar, Vector>(
                Vector<std::complex<Scalar>, 3>{0.18299 + 3.4242i, 0.42108 + 2.34590i, 1.37340 + 1.77040i}, 1.0)),
        rendex::geometry::construct_union(
            rendex::geometry::Sphere<Scalar, Vector, Metal>(
                0.5, Vector<Scalar, 3>{-1.0, 0.0, 1.0},
                Metal<Scalar, Vector>(
                    Vector<std::complex<Scalar>, 3>{0.15943 + 3.92910i, 0.14512 + 3.19000i, 0.13547 + 2.38080i}, 0.3)),
            rendex::geometry::construct_union(
                rendex::geometry::Sphere<Scalar>(0.5, Vector<Scalar, 3>{0.0, 0.0, 1.0},
                                                 Dielectric<Scalar>(Vector<Scalar, 3>{0.7, 0.3, 0.3},
                                                                    Vector<Scalar, 3>{0.0, 0.0, 0.0}, 1.0, 1.0, 0.0)),
                rendex::geometry::Sphere<Scalar>(
                    100.0, Vector<Scalar, 3>{0.0, 100.5, 1.0},
                    Dielectric<Scalar>(Vector<Scalar, 3>{0.8, 0.8, 0.0}, Vector<Scalar, 3>{0.0, 0.0, 0.0}, 1.0, 1.0,
                                       0.0)))));

    constexpr auto object_ = rendex::geometry::Sphere<Scalar>(
        0.5, Vector<Scalar, 3>{0.0, 0.0, 1.0},
        Dielectric<Scalar, Vector>(Vector<Scalar, 3>{0.0, 0.0, 0.0}, Vector<Scalar, 3>{1.0, 1.0, 1.0}, 1.5, 0.0, 1.0));

    // constexpr rendex::geometry::Sphere<Scalar> bounds(100.0, Vector<Scalar, 3>{},
    // Material<Scalar>{});

    // camera

    constexpr auto H = 400;
    constexpr auto W = 800;
    constexpr auto MSAA = 10;

    constexpr auto aspect_ratio = 1. * W / H;
    constexpr auto vertical_fov = std::numbers::pi / 2.0;

    constexpr rendex::camera::Camera<Scalar> camera(vertical_fov, aspect_ratio, Vector<Scalar, 3>{},
                                                    Matrix<Scalar, 3, 3>{});

    // background

    auto background = [](const auto &ray) {
        return rendex::math::lerp(-ray.direction()[1], -1.0, 1.0, Vector<Scalar, 3>{1.0, 1.0, 1.0},
                                  Vector<Scalar, 3>{0.5, 0.7, 1.0});
    };

    // ray marching

    constexpr auto num_iterations = 1000;
    constexpr auto convergence_threshold = 0.001;

    // rendering

    CONSTEXPR auto image = rendex::rendering::ray_casting<Tensor, Scalar, H, W, MSAA>(
        object, camera, background, 0.01, rendex::random::LCG<>(rendex::random::now()));
    // CONSTEXPR auto image = rendex::rendering::ray_marching<DynamicTensor, Scalar, H, W,
    // MSAA>(object, camera, background, bounds, num_iterations, convergence_threshold);

    rendex::image::write_ppm(elemwise(rendex::math::sqrt<Scalar>, image), "image.ppm");
}
