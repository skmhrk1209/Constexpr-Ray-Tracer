
#include <numbers>

#include "camera.hpp"
#include "geometry.hpp"
#include "math.hpp"
#include "random.hpp"
#include "reflection.hpp"
#include "tensor.hpp"

using Scalar = double;

// object
inline constexpr auto object = []() constexpr {
    using namespace std::literals::complex_literals;
    coex::random::LCG<> generator(__LINE__);
    return coex::geometry::construct_union(
        // ground sphere
        coex::geometry::Sphere<Scalar, coex::tensor::Vector, coex::reflection::Lambertian>(
            1000.0, coex::tensor::Vector<Scalar, 3>{0.0, 1000.0, 0.0},
            coex::reflection::Lambertian<Scalar, coex::tensor::Vector>(
                coex::tensor::Vector<Scalar, 3>{0.5, 0.5, 0.5})),
        coex::geometry::construct_union(
            // left sphere (gold)
            coex::geometry::Sphere<Scalar, coex::tensor::Vector, coex::reflection::Metal>(
                1.0, coex::tensor::Vector<Scalar, 3>{-4.0, -1.0, 0.0},
                coex::reflection::Metal<Scalar, coex::tensor::Vector>(
                    coex::tensor::Vector<std::complex<Scalar>, 3>{
                        0.18299 + 3.42420i,
                        0.42108 + 2.34590i,
                        1.37340 + 1.77040i,
                    },
                    0.0)),
            coex::geometry::construct_union(
                // center sphere (glass)
                coex::geometry::Sphere<Scalar, coex::tensor::Vector, coex::reflection::Dielectric>(
                    1.0, coex::tensor::Vector<Scalar, 3>{0.0, -1.0, 0.0},
                    coex::reflection::Dielectric<Scalar, coex::tensor::Vector>(
                        coex::tensor::Vector<Scalar, 3>{1.0, 1.0, 1.0}, 1.5)),
                coex::geometry::construct_union(
                    // right sphere (platinum)
                    coex::geometry::Sphere<Scalar, coex::tensor::Vector, coex::reflection::Metal>(
                        1.0, coex::tensor::Vector<Scalar, 3>{4.0, -1.0, 0.0},
                        coex::reflection::Metal<Scalar, coex::tensor::Vector>(
                            coex::tensor::Vector<std::complex<Scalar>, 3>{
                                2.37570 + 4.26550i,
                                2.08470 + 3.71530i,
                                1.84530 + 3.13650i,
                            },
                            0.0)),
                    coex::geometry::construct_union(
                        // tiny sphere (scatteing only)
                        [function =
                             [&]<auto I, auto... Is>(auto self, std::index_sequence<I, Is...>) constexpr {
                                 auto center =
                                     coex::random::uniform_in_unit_sphere<Scalar, coex::tensor::Vector>(generator) *
                                     12.0;
                                 auto position = coex::tensor::Vector<Scalar, 3>{center[0], -0.2, center[1]};

                                 auto albedo = coex::tensor::elemwise(
                                     coex::math::square<Scalar>,
                                     coex::tensor::Vector<Scalar, 3>{coex::random::uniform(generator, 0.0, 1.0),
                                                                     coex::random::uniform(generator, 0.0, 1.0),
                                                                     coex::random::uniform(generator, 0.0, 1.0)});
                                 auto refractive_index = coex::random::uniform(generator, 1.0, 2.0);
                                 coex::geometry::Sphere<Scalar, coex::tensor::Vector, coex::reflection::Lambertian>
                                     sphere(0.2, std::move(position),
                                            coex::reflection::Lambertian<Scalar, coex::tensor::Vector>(
                                                std::move(albedo)));

                                 if constexpr (sizeof...(Is))
                                     return coex::geometry::construct_union(std::move(sphere),
                                                                            self(self, std::index_sequence<Is...>{}));
                                 else
                                     return sphere;
                             }](auto &&...args) { return function(function, std::forward<decltype(args)>(args)...); }(
                            std::make_index_sequence<400>{}),
                        coex::geometry::construct_union(
                            // tiny sphere (transmission only)
                            [function =
                                 [&]<auto I, auto... Is>(auto self, std::index_sequence<I, Is...>) constexpr {
                                     auto center = coex::random::uniform_in_unit_sphere<Scalar, coex::tensor::Vector>(
                                                       generator) *
                                                   12.0;
                                     auto position = coex::tensor::Vector<Scalar, 3>{center[0], -0.2, center[1]};

                                     auto albedo = coex::tensor::elemwise(
                                         coex::math::sqrt<Scalar>,
                                         coex::tensor::Vector<Scalar, 3>{coex::random::uniform(generator, 0.5, 1.0),
                                                                         coex::random::uniform(generator, 0.5, 1.0),
                                                                         coex::random::uniform(generator, 0.5, 1.0)});
                                     auto refractive_index = coex::random::uniform(generator, 1.0, 2.0);
                                     coex::geometry::Sphere<Scalar, coex::tensor::Vector, coex::reflection::Dielectric>
                                         sphere(0.2, std::move(position),
                                                coex::reflection::Dielectric<Scalar, coex::tensor::Vector>(
                                                    std::move(albedo), refractive_index));

                                     if constexpr (sizeof...(Is))
                                         return coex::geometry::construct_union(
                                             std::move(sphere), self(self, std::index_sequence<Is...>{}));
                                     else
                                         return sphere;
                                 }](
                                auto &&...args) { return function(function, std::forward<decltype(args)>(args)...); }(
                                std::make_index_sequence<200>{}),
                            // tiny sphere (reflection only)
                            [function = [&]<auto I, auto... Is>(auto self, std::index_sequence<I, Is...>) constexpr {
                                auto center =
                                    coex::random::uniform_in_unit_sphere<Scalar, coex::tensor::Vector>(generator) *
                                    12.0;
                                auto position = coex::tensor::Vector<Scalar, 3>{center[0], -0.2, center[1]};

                                coex::tensor::Vector<std::complex<Scalar>, 3> refractive_index{
                                    coex::random::uniform(generator, 0.0, 5.0) +
                                        coex::random::uniform(generator, 0.0, 5.0) * 1i,
                                    coex::random::uniform(generator, 0.0, 5.0) +
                                        coex::random::uniform(generator, 0.0, 5.0) * 1i,
                                    coex::random::uniform(generator, 0.0, 5.0) +
                                        coex::random::uniform(generator, 0.0, 5.0) * 1i,
                                };
                                auto fuzziness = coex::random::uniform(generator, 0.0, 0.5);
                                coex::geometry::Sphere<Scalar, coex::tensor::Vector, coex::reflection::Metal> sphere(
                                    0.2, std::move(position),
                                    coex::reflection::Metal<Scalar, coex::tensor::Vector>(std::move(refractive_index),
                                                                                          fuzziness));

                                if constexpr (sizeof...(Is))
                                    return coex::geometry::construct_union(std::move(sphere),
                                                                           self(self, std::index_sequence<Is...>{}));
                                else
                                    return sphere;
                            }](auto &&...args) {
                                return function(function, std::forward<decltype(args)>(args)...);
                            }(std::make_index_sequence<100>{})))))));
}();

inline constexpr auto camera = []() constexpr {
    auto vertical_fov = 20.0 / 180.0 * std::numbers::pi;
    auto aspect_ratio = 1.5;
    auto focus_distance = 10.0;
    auto aperture_radius = 0.1;

    coex::tensor::Vector<Scalar, 3> position{12.0, -2.0, -4.0};
    coex::tensor::Vector<Scalar, 3> target{0.0, 0.0, 0.0};
    coex::tensor::Vector<Scalar, 3> down{0.0, 1.0, 0.0};

    auto w = coex::tensor::normalized(target - position);
    auto u = coex::tensor::normalized(coex::tensor::cross(down, w));
    auto v = coex::tensor::cross(w, u);
    auto rotation = coex::tensor::transposed(coex::tensor::Matrix<Scalar, 3, 3>{u, v, w});

    return coex::camera::Camera<Scalar>(vertical_fov, aspect_ratio, focus_distance, aperture_radius, position,
                                        rotation);
}();

inline constexpr auto background = [](const auto &ray) constexpr {
    return coex::math::lerp(ray.direction()[1], -1.0, 1.0, coex::tensor::Vector<Scalar, 3>{0.5, 0.7, 1.0},
                            coex::tensor::Vector<Scalar, 3>{1.0, 1.0, 1.0});
};
