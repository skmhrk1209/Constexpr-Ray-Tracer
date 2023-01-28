#include <fstream>
#include <iostream>
#include "ray.hpp"
#include "csg.hpp"
#include "math.hpp"
#include "sphere.hpp"
#include "vector.hpp"

#include <boost/progress.hpp>

template <typename Scalar>
constexpr auto render(
    auto image_width,
    auto image_height,
    auto viewport_width,
    auto viewport_height,
    auto focal_length,
    auto num_aa_samples,
    auto num_iterations,
    auto convergence_threshold,
    const auto &object,
    const auto &bounds,
    const auto &pixel)
{
    rendex::blas::Vector<Scalar, 3> aa_color{0.0, 0.0, 0.0};

    for (auto j = 0; j < num_aa_samples; ++j)
    {
        for (auto i = 0; i < num_aa_samples; ++i)
        {
            auto u = pixel.x() - 0.5 + (i + 1.) / (num_aa_samples + 1.);
            auto v = pixel.y() - 0.5 + (j + 1.) / (num_aa_samples + 1.);

            auto x = rendex::math::linmap(u, 0., image_width - 1., -viewport_width / 2.0, viewport_width / 2.0);
            auto y = rendex::math::linmap(v, 0., image_height - 1., -viewport_height / 2.0, viewport_height / 2.0);

            rendex::blas::Vector<Scalar, 3> position{x, y, focal_length};
            auto direction = position / rendex::blas::norm(position);
            rendex::optics::Ray<Scalar> ray(position, direction);

            auto color = rendex::math::linmap(ray.direction()[1], -1.0, 1.0, rendex::blas::Vector<Scalar, 3>{1.0, 1.0, 1.0}, rendex::blas::Vector<Scalar, 3>{0.5, 0.7, 1.0});

            /*
            auto diff = ray.position() - bounds.position();
            auto a = rendex::blas::dot(ray.direction(), ray.direction());
            auto b = rendex::blas::dot(ray.direction(), diff);
            auto c = rendex::blas::dot(diff, diff) - bounds.radius() * bounds.radius();
            auto d = b * b - a * c;

            if (d >= 0)
            {
                auto distance = (-b - __builtin_sqrt(d)) / a;
                ray.advance(distance);
            }
            */

            for (auto k = 0; k < num_iterations; ++k)
            {
                auto distance = object.distance(ray.position());
                ray.advance(distance);
                if (rendex::math::abs(distance) < convergence_threshold)
                {
                    auto normal = object.normal(ray.position());
                    color = rendex::math::linmap(normal, rendex::blas::Vector<Scalar, 3>{-1.0, -1.0, -1.0}, rendex::blas::Vector<Scalar, 3>{1.0, 1.0, 1.0}, rendex::blas::Vector<Scalar, 3>{0.0, 0.0, 0.0}, rendex::blas::Vector<Scalar, 3>{1.0, 1.0, 1.0});
                    break;
                }
                else
                {
                    auto distance = bounds.distance(ray.position());
                    if (distance > 0.0)
                        break;
                }
            }

            aa_color = aa_color + color;
        }
    }

    aa_color = aa_color / static_cast<Scalar>(num_aa_samples * num_aa_samples);

    return aa_color;
}

int main()
{
    using Scalar = double;

    // image

    constexpr auto image_width = 800;
    constexpr auto image_height = 400;

    // camera

    constexpr auto viewport_width = 4.0;
    constexpr auto viewport_height = 2.0;
    constexpr auto focal_length = 1.0;

    // antialiasing

    constexpr auto num_aa_samples = 1;

    // ray marching

    constexpr auto num_iterations = 1000;
    constexpr auto convergence_threshold = 0.001;

    // object

    constexpr rendex::geom::Union<rendex::geom::Sphere<Scalar>, rendex::geom::Sphere<Scalar>> object{
        rendex::geom::Sphere<Scalar>{rendex::blas::Vector<Scalar, 3>{0.0, 0.0, 1.0}, 0.5},
        rendex::geom::Sphere<Scalar>{rendex::blas::Vector<Scalar, 3>{0.0, -100.5, 1.0}, 100.0}};

    constexpr rendex::geom::Sphere<Scalar> bounds{rendex::blas::Vector<Scalar, 3>{0.0, 0.0, 0.0}, 1000.0};

    // rendering

    boost::progress_timer progress_timer;
    boost::progress_display progress_display(image_width * image_height);

    /* constexpr */ auto colors = [&]() constexpr
    {
        std::array<rendex::blas::Vector<Scalar, 3>, image_width * image_height> colors;
        for (auto j = 0; j < image_height; ++j)
        {
            for (auto i = 0; i < image_width; ++i)
            {
                colors[image_width * j + i] = render<Scalar>(
                    image_width,
                    image_height,
                    viewport_width,
                    viewport_height,
                    focal_length,
                    num_aa_samples,
                    num_iterations,
                    convergence_threshold,
                    object,
                    bounds,
                    rendex::blas::Vector<int, 2>{i, j});

                ++progress_display;
            }
        }
        return colors;
    }();

    std::ofstream image_stream("image.ppm");

    image_stream << "P3\n"
                 << image_width << " " << image_height << "\n255\n";

    for (const auto &color : colors)
    {
        image_stream << color.x() * ((1 << 8) - 1) << " " << color.y() * ((1 << 8) - 1) << " " << color.z() * ((1 << 8) - 1) << "\n";
    }
}
