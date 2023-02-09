#pragma once

#include <boost/progress.hpp>
#include <execution>

#include "math.hpp"
#include "random.hpp"
#include "tensor.hpp"

namespace rendex::rendering {

template <typename Scalar, auto ImageWidth, auto ImageHeight, auto PatchWidth, auto PatchHeight, auto PatchCoordX,
          auto PatchCoordY, typename Generator = rendex::random::LCG<>>
constexpr auto ray_tracing(const auto &object, const auto &camera, auto background, auto max_depth, auto num_samples,
                           auto random_seed) {
#if IS_CONSTANT_EVALUATED
#else
    boost::progress_timer progress_timer;
    boost::progress_display progress_display(PatchWidth * PatchHeight);
#endif

    Generator generator(random_seed);

    std::vector<rendex::tensor::Vector<Scalar, 2>> coords;
    for (auto coord_y = PatchHeight * PatchCoordY; coord_y < PatchHeight * (PatchCoordY + 1); ++coord_y) {
        for (auto coord_x = PatchWidth * PatchCoordX; coord_x < PatchWidth * (PatchCoordX + 1); ++coord_x) {
            coords.push_back(rendex::tensor::Vector<Scalar, 2>{coord_x, coord_y});
        }
    }

    std::array<rendex::tensor::Vector<Scalar, 3>, PatchWidth * PatchHeight> colors;
    std::transform(std::begin(coords), std::end(coords), std::begin(colors), [&](const auto &coord) {
        rendex::tensor::Vector<Scalar, 3> color{};

        for (auto sample_index = 0; sample_index < num_samples; ++sample_index) {
            auto coord_u = (coord[0] + rendex::random::uniform(generator, -0.5, 0.5)) / ImageWidth;
            auto coord_v = (coord[1] + rendex::random::uniform(generator, -0.5, 0.5)) / ImageHeight;

            auto ray = camera.ray(coord_u, coord_v, generator);

            color = color + [&]() constexpr -> rendex::tensor::Vector<Scalar, 3> {
                rendex::tensor::Vector<Scalar, 3> albedo{1.0, 1.0, 1.0};
                for (auto depth = 0; depth < max_depth; ++depth) {
                    auto [geometry, distance] = object.intersect(ray);

                    if (!distance) return background(ray) * albedo;

                    ray.advance(distance.value());

                    std::visit(
                        [&](auto &geometry) {
                            auto &material = geometry.material();
                            auto normal = geometry.normal(ray.position());
                            auto reflection = material(ray, normal, generator);
                            ray = std::move(std::get<0>(reflection));
                            albedo = albedo * std::get<1>(reflection);
                        },
                        geometry);
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

}  // namespace rendex::rendering
