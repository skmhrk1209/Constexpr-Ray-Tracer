#pragma once

#include <numbers>

#include "math.hpp"
#include "random.hpp"
#include "tensor.hpp"

namespace rendex::rendering {

template <template <typename, auto...> typename Image, typename Scalar, auto H, auto W, auto MSAA>
constexpr auto ray_casting(const auto &object, const auto &camera, auto background, auto max_depth) {
    Image<Scalar, H, W, 3> image{};

    for (auto j = 0; j < H; ++j) {
        for (auto i = 0; i < W; ++i) {
            rendex::tensor::Tensor<Scalar, MSAA, MSAA, 3> subimage{};

            for (auto jj = 0; jj < MSAA; ++jj) {
                for (auto ii = 0; ii < MSAA; ++ii) {
                    auto u = (i + 1. * ii / MSAA) / W;
                    auto v = (j + 1. * jj / MSAA) / H;

                    auto ray = camera.ray(u, v);

                    auto render = [function = [&](auto self, auto &ray, auto depth) {
                        if (depth == max_depth) return rendex::tensor::Vector<Scalar, 3>{};

                        auto [geometry, distance] = object.intersect(ray);

                        if (distance) {
                            ray.advance(distance.value());
                            return std::visit(
                                [&](auto &geometry) {
                                    auto scattered_ray = geometry.material().scatter(ray, geometry);
                                    return geometry.material().albedo() * self(self, scattered_ray, depth + 1);
                                },
                                geometry);
                        } else {
                            return background(ray);
                        }
                    }](auto &&...args) { return function(function, std::forward<decltype(args)>(args)...); };

                    subimage[jj][ii] = render(ray, 0);
                }
            }

            auto color = rendex::tensor::sum(rendex::tensor::sum(subimage)) / (MSAA * MSAA);
            std::move(std::begin(color), std::end(color), std::begin(image[j][i]));
        }
    }

    return image;
}

}  // namespace rendex::rendering
