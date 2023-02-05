#pragma once

#include <cassert>
#include <numbers>

#include "common.hpp"
#include "math.hpp"
#include "random.hpp"
#include "tensor.hpp"

namespace rendex::rendering {

template <template <typename, auto...> typename Image, typename Scalar, auto H, auto W, auto MSAA>
constexpr auto ray_casting(const auto &object, const auto &camera, auto background, auto termination_prob,
                           auto generator) {
    Image<Scalar, H, W, 3> image{};
    rendex::random::Uniform<Scalar> uniform(0.0, 1.0);

    for (auto j = 0; j < H; ++j) {
        for (auto i = 0; i < W; ++i) {
            rendex::tensor::Tensor<Scalar, MSAA, MSAA, 3> subimage{};

            for (auto jj = 0; jj < MSAA; ++jj) {
                for (auto ii = 0; ii < MSAA; ++ii) {
                    auto u = (i + 1. * ii / MSAA) / W;
                    auto v = (j + 1. * jj / MSAA) / H;

                    auto ray = camera.ray(u, v);

                    auto render = [function = [&](auto self, auto &ray) {
                        auto [geometry, distance] = object.intersect(ray);

                        if (!distance) return background(ray);

                        if (uniform(generator) < termination_prob) return rendex::tensor::Vector<Scalar, 3>{};

                        ray.advance(distance.value());
                        return std::visit(
                            [&](auto &geometry) {
                                auto &material = geometry.material();
                                auto normal = geometry.normal(ray.position());
                                auto [reflected_ray, albedo] = material(ray, normal, generator);
                                return self(self, reflected_ray) * albedo;
                            },
                            geometry);
                    }](auto &&...args) { return function(function, std::forward<decltype(args)>(args)...); };

                    subimage[jj][ii] = render(ray);
                }
            }

            auto color = rendex::tensor::sum(rendex::tensor::sum(subimage)) / (MSAA * MSAA);
            std::move(std::begin(color), std::end(color), std::begin(image[j][i]));
        }
    }

    return image;
}

}  // namespace rendex::rendering
