
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
    rendex::random::LCG<> generator(rendex::random::now());
    return rendex::geometry::construct_union(
        // ground sphere
        rendex::geometry::Sphere<Scalar, rendex::tensor::Vector, rendex::reflection::Dielectric>(
            1000.0, rendex::tensor::Vector<Scalar, 3>{0.0, 1000.0, 0.0},
            rendex::reflection::Dielectric<Scalar, rendex::tensor::Vector>(
                rendex::tensor::Vector<Scalar, 3>{0.5, 0.5, 0.5}, {}, 1.0, 1.0, 0.0)),
        rendex::geometry::construct_union(
            // left sphere (gold)
            rendex::geometry::Sphere<Scalar, rendex::tensor::Vector, rendex::reflection::Metal>(
                1.0, rendex::tensor::Vector<Scalar, 3>{-3.0, -1.0, 0.0},
                rendex::reflection::Metal<Scalar, rendex::tensor::Vector>(
                    rendex::tensor::Vector<std::complex<Scalar>, 3>{
                        0.18299 + 3.42420i,
                        0.42108 + 2.34590i,
                        1.37340 + 1.77040i,
                    },
                    rendex::random::uniform(generator, 0.0, 0.5))),
            rendex::geometry::construct_union(
                // center sphere (glass)
                rendex::geometry::Sphere<Scalar, rendex::tensor::Vector, rendex::reflection::Dielectric>(
                    1.0, rendex::tensor::Vector<Scalar, 3>{0.0, -1.0, 0.0},
                    rendex::reflection::Dielectric<Scalar, rendex::tensor::Vector>(
                        {}, rendex::tensor::Vector<Scalar, 3>{1.0, 1.0, 1.0},
                        rendex::random::uniform(generator, 1.0, 2.0), 0.0,
                        rendex::random::uniform(generator, 0.0, 0.5))),
                rendex::geometry::construct_union(
                    // right sphere (platinum)
                    rendex::geometry::Sphere<Scalar, rendex::tensor::Vector, rendex::reflection::Metal>(
                        1.0, rendex::tensor::Vector<Scalar, 3>{3.0, -1.0, 0.0},
                        rendex::reflection::Metal<Scalar, rendex::tensor::Vector>(
                            rendex::tensor::Vector<std::complex<Scalar>, 3>{
                                2.37570 + 4.26550i,
                                2.08470 + 3.71530i,
                                1.84530 + 3.13650i,
                            },
                            rendex::random::uniform(generator, 0.0, 0.5))),
                    rendex::geometry::construct_union(
                        // tiny sphere (scatteing only)
                        [function =
                             [&]<auto I, auto... Is>(auto self, std::index_sequence<I, Is...>) constexpr {
                                 auto center = rendex::random::uniform_in_unit_sphere<Scalar, rendex::tensor::Vector>(
                                                   generator) *
                                               10.0;
                                 auto radius = rendex::random::uniform(generator, 0.1, 0.3);
                                 auto position = rendex::tensor::Vector<Scalar, 3>{center[0], -radius, center[1]};

                                 auto albedo = rendex::tensor::elemwise(
                                     rendex::math::square<Scalar>,
                                     rendex::tensor::Vector<Scalar, 3>{rendex::random::uniform(generator, 0.0, 1.0),
                                                                       rendex::random::uniform(generator, 0.0, 1.0),
                                                                       rendex::random::uniform(generator, 0.0, 1.0)});
                                 auto refractive_index = rendex::random::uniform(generator, 1.0, 2.0);
                                 auto fuzziness = rendex::random::uniform(generator, 0.0, 0.5);
                                 rendex::geometry::Sphere<Scalar, rendex::tensor::Vector,
                                                          rendex::reflection::Dielectric>
                                     sphere(radius, position,
                                            rendex::reflection::Dielectric<Scalar, rendex::tensor::Vector>(
                                                albedo, {}, refractive_index, 1.0, fuzziness));

                                 if constexpr (sizeof...(Is))
                                     return rendex::geometry::construct_union(
                                         std::move(sphere), self(self, std::index_sequence<Is...>{}));
                                 else
                                     return sphere;
                             }](auto &&...args) { return function(function, std::forward<decltype(args)>(args)...); }(
                            std::make_index_sequence<200>{}),
                        rendex::geometry::construct_union(
                            // tiny sphere (transmission only)
                            [function =
                                 [&]<auto I, auto... Is>(auto self, std::index_sequence<I, Is...>) constexpr {
                                     auto center =
                                         rendex::random::uniform_in_unit_sphere<Scalar, rendex::tensor::Vector>(
                                             generator) *
                                         10.0;
                                     auto radius = rendex::random::uniform(generator, 0.1, 0.3);
                                     auto position = rendex::tensor::Vector<Scalar, 3>{center[0], -radius, center[1]};

                                     auto transmittance =
                                         rendex::tensor::elemwise(rendex::math::sqrt<Scalar>,
                                                                  rendex::tensor::Vector<Scalar, 3>{
                                                                      rendex::random::uniform(generator, 0.5, 1.0),
                                                                      rendex::random::uniform(generator, 0.5, 1.0),
                                                                      rendex::random::uniform(generator, 0.5, 1.0)});
                                     auto refractive_index = rendex::random::uniform(generator, 1.0, 2.0);
                                     auto fuzziness = rendex::random::uniform(generator, 0.0, 0.5);
                                     rendex::geometry::Sphere<Scalar, rendex::tensor::Vector,
                                                              rendex::reflection::Dielectric>
                                         sphere(radius, position,
                                                rendex::reflection::Dielectric<Scalar, rendex::tensor::Vector>(
                                                    {}, transmittance, refractive_index, 0.0, fuzziness));

                                     if constexpr (sizeof...(Is))
                                         return rendex::geometry::construct_union(
                                             std::move(sphere), self(self, std::index_sequence<Is...>{}));
                                     else
                                         return sphere;
                                 }](
                                auto &&...args) { return function(function, std::forward<decltype(args)>(args)...); }(
                                std::make_index_sequence<100>{}),
                            // tiny sphere (reflection only)
                            [function = [&]<auto I, auto... Is>(auto self, std::index_sequence<I, Is...>) constexpr {
                                auto center =
                                    rendex::random::uniform_in_unit_sphere<Scalar, rendex::tensor::Vector>(generator) *
                                    10.0;
                                auto radius = rendex::random::uniform(generator, 0.1, 0.3);
                                auto position = rendex::tensor::Vector<Scalar, 3>{center[0], -radius, center[1]};

                                rendex::tensor::Vector<std::complex<Scalar>, 3> refractive_index{
                                    rendex::random::uniform(generator, 0.0, 5.0) +
                                        rendex::random::uniform(generator, 0.0, 5.0) * 1i,
                                    rendex::random::uniform(generator, 0.0, 5.0) +
                                        rendex::random::uniform(generator, 0.0, 5.0) * 1i,
                                    rendex::random::uniform(generator, 0.0, 5.0) +
                                        rendex::random::uniform(generator, 0.0, 5.0) * 1i,
                                };
                                auto fuzziness = rendex::random::uniform(generator, 0.0, 0.5);
                                rendex::geometry::Sphere<Scalar, rendex::tensor::Vector, rendex::reflection::Metal>
                                    sphere(radius, position,
                                           rendex::reflection::Metal<Scalar, rendex::tensor::Vector>(refractive_index,
                                                                                                     fuzziness));

                                if constexpr (sizeof...(Is))
                                    return rendex::geometry::construct_union(std::move(sphere),
                                                                             self(self, std::index_sequence<Is...>{}));
                                else
                                    return sphere;
                            }](auto &&...args) {
                                return function(function, std::forward<decltype(args)>(args)...);
                            }(std::make_index_sequence<50>{})))))));
}();

inline constexpr auto camera = []() constexpr {
    auto vertical_fov = 20.0 / 180.0 * std::numbers::pi;
    auto aspect_ratio = 1.5;
    auto focus_distance = 10.0;
    auto aperture_radius = 0.1;

    rendex::tensor::Vector<Scalar, 3> position{10.0, -2.0, -5.0};
    rendex::tensor::Vector<Scalar, 3> target{0.0, 0.0, 0.0};
    rendex::tensor::Vector<Scalar, 3> down{0.0, 1.0, 0.0};

    auto w = rendex::tensor::normalized(target - position);
    auto u = rendex::tensor::normalized(rendex::tensor::cross(down, w));
    auto v = rendex::tensor::cross(w, u);
    auto rotation = rendex::tensor::transposed(rendex::tensor::Matrix<Scalar, 3, 3>{u, v, w});

    return rendex::camera::Camera<Scalar>(vertical_fov, aspect_ratio, focus_distance, aperture_radius, position,
                                          rotation);
}();

inline constexpr auto background = [](const auto &ray) constexpr {
    return rendex::math::lerp(ray.direction()[1], -1.0, 1.0, rendex::tensor::Vector<Scalar, 3>{0.5, 0.7, 1.0},
                              rendex::tensor::Vector<Scalar, 3>{1.0, 1.0, 1.0});
};
