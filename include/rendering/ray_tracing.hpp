#pragma once

#include "math.hpp"
#include "random.hpp"
#include "tensor.hpp"

#include <boost/progress.hpp>

namespace rendex::rendering {

template <template <typename, auto...> typename Image, typename Scalar, auto H, auto W, auto MSAA>
constexpr auto ray_tracing(const auto &object, const auto &camera, auto background, auto termination_prob,
                           auto generator) {

    boost::progress_timer progress_timer;
    boost::progress_display progress_display(H * W * MSAA * MSAA);

    Image<Scalar, H, W, 3> image{};

    for (auto j = 0; j < H; ++j) {
        for (auto i = 0; i < W; ++i) {
            rendex::tensor::Tensor<Scalar, MSAA, MSAA, 3> subimage{};

            for (auto jj = 0; jj < MSAA; ++jj) {
                for (auto ii = 0; ii < MSAA; ++ii) {
                    auto u = (i + 1. * ii / MSAA) / W;
                    auto v = (j + 1. * jj / MSAA) / H;

                    auto ray = camera.ray(u, v, generator);

                    auto render = [function = [&](auto self, auto &ray) constexpr {
                        if (rendex::random::uniform(generator, 0.0, 1.0) < termination_prob)
                            return rendex::tensor::Vector<Scalar, 3>{};

                        auto [geometry, distance] = object.intersect(ray);

                        if (!distance) return background(ray);

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

                    ++progress_display;
                }
            }

            auto color = rendex::tensor::sum(rendex::tensor::sum(subimage)) / (MSAA * MSAA);
            std::move(std::begin(color), std::end(color), std::begin(image[j][i]));
        }
    }

    return image;
}

}  // namespace rendex::rendering
