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

#define CONSTEXPR constexpr

int main()
{
    using Scalar = double;

    // camera

    constexpr auto H = 400;
    constexpr auto W = 800;
    constexpr auto MSAA = 1;

    constexpr auto aspect_ratio = 1. * W / H;
    constexpr auto vertical_fov = std::numbers::pi / 2.0;

    constexpr rendex::vision::Camera<Scalar> camera(
        vertical_fov,
        aspect_ratio,
        rendex::blas::Vector<Scalar, 3>{},
        rendex::blas::Matrix<Scalar, 3, 3>{});

    // ray marching

    constexpr auto num_iterations = 1000;
    constexpr auto convergence_threshold = 0.001;

    constexpr rendex::geom::Sphere<Scalar> bounds{rendex::blas::Vector<Scalar, 3>{}, 100.0};

    // object

    constexpr rendex::geom::Union<rendex::geom::Sphere<Scalar>, rendex::geom::Sphere<Scalar>> object{
        rendex::geom::Sphere<Scalar>{rendex::blas::Vector<Scalar, 3>{0.0, 0.0, 1.0}, 0.5},
        rendex::geom::Sphere<Scalar>{rendex::blas::Vector<Scalar, 3>{0.0, 100.5, 1.0}, 100.0}};

    auto background = [](const auto &ray) constexpr
    {
        return rendex::math::lerp(ray.direction()[1], -1.0, 1.0, 1.0, rendex::blas::Vector<Scalar, 3>{0.5, 0.7, 1.0});
    };

    // rendering

    CONSTEXPR auto image = rendex::graphics::ray_casting<Scalar, H, W, MSAA>(object, camera, background) * ((1 << 8) - 1);
    // CONSTEXPR auto image = rendex::graphics::ray_marching<Scalar, H, W, MSAA>(object, camera, background, bounds, num_iterations, convergence_threshold) * ((1 << 8) - 1);

    std::ofstream image_stream("image.ppm");

    image_stream << "P3\n"
                 << W << " " << H << "\n255\n";

    for (const auto &colors : image)
    {
        for (const auto &color : colors)
        {
            image_stream << color[0] << " " << color[1] << " " << color[2] << "\n";
        }
    }
}
