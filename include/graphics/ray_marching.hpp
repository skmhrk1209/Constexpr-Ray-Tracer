#pragma once

#include "math.hpp"
#include "blas.hpp"

namespace rendex::graphics
{
    template <template <typename, auto...> typename Image, typename Scalar, auto H, auto W, auto MSAA>
    constexpr auto ray_marching(const auto &object, const auto &camera, auto background, const auto &bounds, auto num_iterations, auto convergence_threshold)
    {
        Image<Scalar, H, W, 3> image{};

        for (auto j = 0; j < H; ++j)
        {
            for (auto i = 0; i < W; ++i)
            {
                rendex::blas::Tensor<Scalar, MSAA, MSAA, 3> subimage{};

                for (auto jj = 0; jj < MSAA; ++jj)
                {
                    for (auto ii = 0; ii < MSAA; ++ii)
                    {
                        auto u = (i + 1. * ii / MSAA) / W;
                        auto v = (j + 1. * jj / MSAA) / H;

                        auto ray = camera.ray(u, v);

                        auto render = [&](auto &ray)
                        {
                            for (auto k = 0; k < num_iterations; ++k)
                            {
                                auto [geometry, distance] = object.distance(ray.position());
                                ray.advance(distance);
                                if (std::abs(distance) < convergence_threshold)
                                {
                                    auto normal = std::visit([&](const auto &geometry)
                                                             { return geometry.normal(ray.position()); },
                                                             geometry);
                                    auto color = rendex::math::lerp(normal, -1.0, 1.0, 0.0, 1.0);
                                    return color;
                                }
                                else
                                {
                                    auto [geometry, distance] = bounds.distance(ray.position());
                                    if (distance > 0.0)
                                        break;
                                }
                            }

                            return background(ray);
                        };

                        subimage[jj][ii] = render(ray);
                    }
                }

                auto color = rendex::blas::sum(rendex::blas::sum(subimage)) / (MSAA * MSAA);
                std::move(std::begin(color), std::end(color), std::begin(image[j][i]));
            }
        }

        return image;
    }
}
