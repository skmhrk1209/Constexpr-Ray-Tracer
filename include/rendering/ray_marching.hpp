#pragma once

#include <boost/progress.hpp>
#include <execution>

#include "math.hpp"
#include "random.hpp"
#include "tensor.hpp"

namespace rendex::rendering {

template <typename Scalar, auto Width, auto Height, auto NumChunks, auto ChunkIndex,
          typename Generator = rendex::random::LCG<>>
constexpr auto ray_marching(const auto &object, const auto &camera, auto background, auto num_samples, auto max_depth,
                            const auto &bounds, auto max_step, auto epsilon) {
#ifndef CONSTEXPR
    boost::progress_timer progress_timer;
    boost::progress_display progress_display(Width * Height);
#else
#endif

    Generator generator(rendex::random::now());

    constexpr auto ChunkSize = Width * Height / NumChunks;
    std::array<rendex::tensor::Vector<Scalar, 2>, ChunkSize> coords;
    for (auto pixel_index = ChunkSize * ChunkIndex; pixel_index < ChunkSize * (ChunkIndex + 1); ++pixel_index) {
        coords[pixel_index - ChunkSize * ChunkIndex] =
            rendex::tensor::Vector<Scalar, 2>{pixel_index % Width, pixel_index / Width};
    }

    std::array<rendex::tensor::Vector<Scalar, 3>, ChunkSize> colors;
    std::transform(std::begin(coords), std::end(coords), std::begin(colors), [&](const auto &coord) {
        rendex::tensor::Vector<Scalar, 3> color{};

        for (auto sample_index = 0; sample_index < num_samples; ++sample_index) {
            auto u = (coord[0] + rendex::random::uniform(generator, -0.5, 0.5)) / Width;
            auto v = (coord[1] + rendex::random::uniform(generator, -0.5, 0.5)) / Height;

            auto ray = camera.ray(u, v, generator);

            color = color + [&]() {
                rendex::tensor::Vector<Scalar, 3> albedo{1.0, 1.0, 1.0};
                for (auto depth = 0; depth < max_depth; ++depth) {
                    auto intersected = false;
                    for (auto step = 0; step < max_step; ++step) {
                        auto [geometry, distance] = object.distance(ray.position());
                        ray.advance(distance);
                        if (std::abs(distance) < epsilon) {
                            std::visit(
                                [&](auto &geometry) {
                                    auto &material = geometry.material();
                                    auto normal = geometry.normal(ray.position());
                                    auto reflection = material(ray, normal, generator);
                                    ray = std::move(std::get<0>(reflection));
                                    albedo = albedo * std::get<1>(reflection);
                                },
                                geometry);
                            intersected = true;
                            break;
                        } else {
                            auto [geometry, distance] = bounds.distance(ray.position());
                            if (distance > 0.0) return background(ray) * albedo;
                        }
                    }
                    if (!intersected) return background(ray) * albedo;
                }

                return rendex::tensor::Vector<Scalar, 3>{};
            }();
        }

#ifndef CONSTEXPR
        ++progress_display;
#else
#endif

        return color / num_samples;
    });

    return colors;
}

}  // namespace rendex::rendering
