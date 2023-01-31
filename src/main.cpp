#include <fstream>
#include <iostream>
#include "blas.hpp"
#include "geom.hpp"
#include "math.hpp"
#include "common.hpp"
#include "vision.hpp"
#include "graphics.hpp"
#include <cmath>
#include <numbers>

#include <boost/progress.hpp>

#define CONSTEXPR // constexpr

int main()
{
    using Scalar = double;

    // object

    CONSTEXPR rendex::geom::Union<rendex::geom::Sphere<Scalar>, rendex::geom::Sphere<Scalar>> object{
        rendex::geom::Sphere<Scalar>{rendex::blas::Vector<Scalar, 3>{0.0, 0.0, 1.0}, 0.5},
        rendex::geom::Sphere<Scalar>{rendex::blas::Vector<Scalar, 3>{0.0, 100.5, 1.0}, 100.0}};

    CONSTEXPR rendex::geom::Sphere<Scalar> bounds{rendex::blas::Vector<Scalar, 3>{}, 100.0};

    // camera

    constexpr auto H = 100;
    constexpr auto W = 200;
    constexpr auto MSAA = 1;

    constexpr auto aspect_ratio = 1. * W / H;
    constexpr auto vertical_fov = std::numbers::pi / 2.0;

    CONSTEXPR rendex::vision::Camera<Scalar> camera(
        vertical_fov,
        aspect_ratio,
        rendex::blas::Vector<Scalar, 3>{},
        rendex::blas::Matrix<Scalar, 3, 3>{});

    // background

    auto background = [](const auto &ray) constexpr
    {
        return rendex::math::lerp(ray.direction()[1], -1.0, 1.0, 1.0, rendex::blas::Vector<Scalar, 3>{0.5, 0.7, 1.0});
    };

    // ray marching

    constexpr auto num_iterations = 1000;
    constexpr auto convergence_threshold = 0.001;

    // rendering

    // CONSTEXPR auto image = rendex::graphics::ray_casting<rendex::blas::DynamicTensor, Scalar, H, W, MSAA>(object, camera, background);
    CONSTEXPR auto image = rendex::graphics::ray_marching<rendex::blas::DynamicTensor, Scalar, H, W, MSAA>(object, camera, background, bounds, num_iterations, convergence_threshold);

    std::ofstream image_stream("image.ppm");

    image_stream << "P3\n"
                 << W << " " << H << "\n255\n";

    for (const auto &colors : image)
    {
        for (const auto &color : colors)
        {
            for (const auto &value : color)
            {
                image_stream << value * ((1 << 8) - 1) << " ";
            }
            image_stream << std::endl;
        }
    }
}
