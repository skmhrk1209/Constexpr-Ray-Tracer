#pragma once

#include <boost/progress.hpp>
#include <execution>

#include "math.hpp"
#include "random.hpp"
#include "tensor.hpp"

namespace coex::rendering {

template <typename Scalar, auto ImageWidth, auto ImageHeight, auto PatchWidth, auto PatchHeight, auto PatchCoordX,
          auto PatchCoordY, typename Generator = coex::random::LCG<>>
constexpr auto ray_marching(const auto &object, const auto &camera, auto background, auto max_depth, auto num_samples,
                            auto random_seed, const auto &bounds, auto max_step, auto epsilon) {
#if IS_CONSTANT_EVALUATED
#else
    boost::progress_timer progress_timer;
    boost::progress_display progress_display(PatchWidth * PatchHeight);
#endif

    Generator generator(random_seed);

    std::vector<coex::tensor::Vector<Scalar, 2>> coords;
    for (auto coord_y = PatchHeight * PatchCoordY; coord_y < PatchHeight * (PatchCoordY + 1); ++coord_y) {
        for (auto coord_x = PatchWidth * PatchCoordX; coord_x < PatchWidth * (PatchCoordX + 1); ++coord_x) {
            coords.push_back(coex::tensor::Vector<Scalar, 2>{coord_x, coord_y});
        }
    }

    std::array<coex::tensor::Vector<Scalar, 3>, PatchWidth * PatchHeight> colors;
    std::transform(std::begin(coords), std::end(coords), std::begin(colors), [&](const auto &coord) {
        coex::tensor::Vector<Scalar, 3> color{};

        for (auto sample_index = 0; sample_index < num_samples; ++sample_index) {
            auto coord_u = (coord[0] + coex::random::uniform(generator, -0.5, 0.5)) / ImageWidth;
            auto coord_v = (coord[1] + coex::random::uniform(generator, -0.5, 0.5)) / ImageHeight;

            auto ray = camera.ray(coord_u, coord_v, generator);

            color = color + [&]() constexpr -> coex::tensor::Vector<Scalar, 3> {
                coex::tensor::Vector<Scalar, 3> albedo{1.0, 1.0, 1.0};
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

                return {};
            }();
        }

#if IS_CONSTANT_EVALUATED
#else
++progress_display;
#endif

        return color / num_samples;
    });

    return colors;
}

}  // namespace coex::rendering
