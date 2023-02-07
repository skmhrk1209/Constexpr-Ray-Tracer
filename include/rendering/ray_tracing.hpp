#pragma once

#include <execution>

#include "math.hpp"
#include "random.hpp"
#include "tensor.hpp"

namespace rendex::rendering {

template <typename Scalar, auto W, auto H, auto S, auto I = 0, auto N = W *H,
          typename Generator = rendex::random::LCG<>>
constexpr auto ray_tracing(const auto &object, const auto &camera, auto background, auto max_depth) {
    Generator generator(rendex::random::now());

    std::array<rendex::tensor::Vector<Scalar, 2>, N> coords;
    for (auto i = I; i < I + N; ++i) {
        coords[i - I] = rendex::tensor::Vector<Scalar, 2>{i % W, i / W};
    }

    std::array<rendex::tensor::Vector<Scalar, 3>, N> colors;
    std::transform(std::begin(coords), std::end(coords), std::begin(colors), [&](const auto &coord) {
        rendex::tensor::Vector<Scalar, 3> color{};

        for (auto s = 0; s < S; ++s) {
            auto u = (coord[0] + rendex::random::uniform(generator, -0.5, 0.5)) / W;
            auto v = (coord[1] + rendex::random::uniform(generator, -0.5, 0.5)) / H;

            auto ray = camera.ray(u, v, generator);

            auto render = [function = [&](auto self, auto &ray, auto depth) constexpr {
                if (!depth) return rendex::tensor::Vector<Scalar, 3>{};

                auto [geometry, distance] = object.intersect(ray);

                if (!distance) return background(ray);

                ray.advance(distance.value());

                return std::visit(
                    [&](auto &geometry) {
                        auto &material = geometry.material();
                        auto normal = geometry.normal(ray.position());
                        auto [reflected_ray, albedo] = material(ray, normal, generator);
                        return self(self, reflected_ray, depth - 1) * albedo;
                    },
                    geometry);
            }](auto &&...args) { return function(function, std::forward<decltype(args)>(args)...); };

            color = color + render(ray, max_depth) / S;
        }

        return color;
    });

    return colors;
}

}  // namespace rendex::rendering
