#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>

#include "camera.hpp"
#include "common.hpp"
#include "geometry.hpp"
#include "math.hpp"
#include "reflection.hpp"
#include "rendering.hpp"
#include "tensor.hpp"

#define CONSTEXPR  // constexpr

int main() {
    using Scalar = double;
    using Vector = rendex::tensor::Vector<Scalar, 3>;
    using Matrix = rendex::tensor::Matrix<Scalar, 3, 3>;

    // object

    CONSTEXPR rendex::geometry::Union<rendex::geometry::Sphere<Scalar>, rendex::geometry::Sphere<Scalar>> object(
        rendex::geometry::Sphere<Scalar>(
            0.5, rendex::tensor::Vector<Scalar, 3>{0.0, 0.0, 1.0},
            rendex::reflection::Material<Scalar>(rendex::tensor::Vector<Scalar, 3>{1.0, 1.0, 1.0})),
        rendex::geometry::Sphere<Scalar>(
            100.0, rendex::tensor::Vector<Scalar, 3>{0.0, 100.5, 1.0},
            rendex::reflection::Material<Scalar>(rendex::tensor::Vector<Scalar, 3>{1.0, 1.0, 1.0})));

    CONSTEXPR rendex::geometry::Sphere<Scalar> bounds(100.0, rendex::tensor::Vector<Scalar, 3>{},
                                                      rendex::reflection::Material<Scalar>{});

    // camera

    constexpr auto H = 400;
    constexpr auto W = 800;
    constexpr auto MSAA = 10;

    constexpr auto aspect_ratio = 1. * W / H;
    constexpr auto vertical_fov = std::numbers::pi / 2.0;

    CONSTEXPR rendex::camera::Camera<Scalar> camera(vertical_fov, aspect_ratio, rendex::tensor::Vector<Scalar, 3>{},
                                                    rendex::tensor::Matrix<Scalar, 3, 3>{});

    // background

    auto background = [](const auto &ray) constexpr {
        return rendex::math::lerp(-ray.direction()[1], -1.0, 1.0, 1.0,
                                  rendex::tensor::Vector<Scalar, 3>{0.5, 0.7, 1.0});
    };

    // ray marching

    constexpr auto num_iterations = 1000;
    constexpr auto convergence_threshold = 0.001;

    // rendering

    CONSTEXPR auto image = rendex::rendering::ray_casting<rendex::tensor::DynamicTensor, Scalar, H, W, MSAA>(
        object, camera, background, 50);
    // CONSTEXPR auto image = rendex::rendering::ray_marching<rendex::tensor::DynamicTensor, Scalar, H, W,
    // MSAA>(object, camera, background, bounds, num_iterations, convergence_threshold);

    std::ofstream image_stream("image.ppm");

    image_stream << "P3\n" << W << " " << H << "\n255\n";

    for (const auto &colors : image) {
        for (const auto &color : colors) {
            for (const auto &value : color) {
                image_stream << value * ((1 << 8) - 1) << " ";
            }
            image_stream << std::endl;
        }
    }
}
