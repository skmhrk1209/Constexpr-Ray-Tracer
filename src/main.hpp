
#include <execution>
#include <numbers>

#include "camera.hpp"
#include "common.hpp"
#include "geometry.hpp"
#include "image.hpp"
#include "math.hpp"
#include "random.hpp"
#include "reflection.hpp"
#include "rendering.hpp"
#include "tensor.hpp"

template <typename T, auto W, auto H, auto S, auto I, auto N>
constexpr auto render() {
    using namespace std::literals::complex_literals;

    // object
    auto object = []() constexpr {
        rendex::random::LCG<> generator(rendex::random::now());
        return rendex::geometry::construct_union(
            // ground sphere
            rendex::geometry::Sphere<T, rendex::tensor::Vector, rendex::reflection::Dielectric>(
                1000.0, rendex::tensor::Vector<T, 3>{0.0, 1000.0, 0.0},
                rendex::reflection::Dielectric<T>(rendex::tensor::Vector<T, 3>{0.5, 0.5, 0.5}, {}, 1.0, 1.0, 0.0)),
            rendex::geometry::construct_union(
                // left sphere (hollow Glass)
                // right sphere (gold)
                rendex::geometry::Sphere<T, rendex::tensor::Vector, rendex::reflection::Metal>(
                    1.0, rendex::tensor::Vector<T, 3>{-3.0, -1.0, 0.0},
                    rendex::reflection::Metal<T, rendex::tensor::Vector>(
                        rendex::tensor::Vector<std::complex<T>, 3>{
                            0.18299 + 3.42420i,
                            0.42108 + 2.34590i,
                            1.37340 + 1.77040i,
                        },
                        rendex::random::uniform(generator, 0.0, 0.5))),
                rendex::geometry::construct_union(
                    // center sphere (glass)
                    rendex::geometry::Sphere<T, rendex::tensor::Vector, rendex::reflection::Dielectric>(
                        1.0, rendex::tensor::Vector<T, 3>{0.0, -1.0, 0.0},
                        rendex::reflection::Dielectric<T, rendex::tensor::Vector>(
                            {}, rendex::tensor::Vector<T, 3>{1.0, 1.0, 1.0},
                            rendex::random::uniform(generator, 1.0, 2.0), 0.0,
                            rendex::random::uniform(generator, 0.0, 0.5))),
                    rendex::geometry::construct_union(
                        // right sphere (platinum)
                        rendex::geometry::Sphere<T, rendex::tensor::Vector, rendex::reflection::Metal>(
                            1.0, rendex::tensor::Vector<T, 3>{3.0, -1.0, 0.0},
                            rendex::reflection::Metal<T, rendex::tensor::Vector>(
                                rendex::tensor::Vector<std::complex<T>, 3>{
                                    2.37570 + 4.26550i,
                                    2.08470 + 3.71530i,
                                    1.84530 + 3.13650i,
                                },
                                rendex::random::uniform(generator, 0.0, 0.5))),
                        rendex::geometry::construct_union(
                            // tiny sphere (scatteing only)
                            [function =
                                 [&]<auto J, auto... Js>(auto self, std::index_sequence<J, Js...>) constexpr {
                                     auto center =
                                         rendex::random::uniform_in_unit_sphere<T, rendex::tensor::Vector>(generator) *
                                         10.0;
                                     auto radius = rendex::random::uniform(generator, 0.1, 0.3);
                                     auto position = rendex::tensor::Vector<T, 3>{center[0], -radius, center[1]};

                                     auto albedo = rendex::tensor::elemwise(
                                         rendex::math::square<T>,
                                         rendex::tensor::Vector<T, 3>{rendex::random::uniform(generator, 0.0, 1.0),
                                                                      rendex::random::uniform(generator, 0.0, 1.0),
                                                                      rendex::random::uniform(generator, 0.0, 1.0)});
                                     auto refractive_index = rendex::random::uniform(generator, 1.0, 2.0);
                                     auto fuzziness = rendex::random::uniform(generator, 0.0, 0.5);
                                     rendex::geometry::Sphere<T, rendex::tensor::Vector,
                                                              rendex::reflection::Dielectric>
                                         sphere(radius, position,
                                                rendex::reflection::Dielectric<T, rendex::tensor::Vector>(
                                                    albedo, {}, refractive_index, 1.0, fuzziness));

                                     if constexpr (sizeof...(Js))
                                         return rendex::geometry::construct_union(
                                             std::move(sphere), self(self, std::index_sequence<Js...>{}));
                                     else
                                         return sphere;
                                 }](
                                auto &&...args) { return function(function, std::forward<decltype(args)>(args)...); }(
                                std::make_index_sequence<240>{}),
                            rendex::geometry::construct_union(
                                // tiny sphere (transmission only)
                                [function =
                                     [&]<auto J, auto... Js>(auto self, std::index_sequence<J, Js...>) constexpr {
                                         auto center =
                                             rendex::random::uniform_in_unit_sphere<T, rendex::tensor::Vector>(
                                                 generator) *
                                             10.0;
                                         auto radius = rendex::random::uniform(generator, 0.1, 0.3);
                                         auto position = rendex::tensor::Vector<T, 3>{center[0], -radius, center[1]};

                                         auto transmittance = rendex::tensor::elemwise(
                                             rendex::math::sqrt<T>, rendex::tensor::Vector<T, 3>{
                                                                        rendex::random::uniform(generator, 0.5, 1.0),
                                                                        rendex::random::uniform(generator, 0.5, 1.0),
                                                                        rendex::random::uniform(generator, 0.5, 1.0)});
                                         auto refractive_index = rendex::random::uniform(generator, 1.0, 2.0);
                                         auto fuzziness = rendex::random::uniform(generator, 0.0, 0.5);
                                         rendex::geometry::Sphere<T, rendex::tensor::Vector,
                                                                  rendex::reflection::Dielectric>
                                             sphere(radius, position,
                                                    rendex::reflection::Dielectric<T, rendex::tensor::Vector>(
                                                        {}, transmittance, refractive_index, 0.0, fuzziness));

                                         if constexpr (sizeof...(Js))
                                             return rendex::geometry::construct_union(
                                                 std::move(sphere), self(self, std::index_sequence<Js...>{}));
                                         else
                                             return sphere;
                                     }](
                                    auto
                                        &&...args) { return function(function, std::forward<decltype(args)>(args)...); }(
                                    std::make_index_sequence<120>{}),
                                // tiny sphere (reflection only)
                                [function = [&]<auto J, auto... Js>(auto self,
                                                                    std::index_sequence<J, Js...>) constexpr {
                                    auto center =
                                        rendex::random::uniform_in_unit_sphere<T, rendex::tensor::Vector>(generator) *
                                        10.0;
                                    auto radius = rendex::random::uniform(generator, 0.1, 0.3);
                                    auto position = rendex::tensor::Vector<T, 3>{center[0], -radius, center[1]};

                                    rendex::tensor::Vector<std::complex<T>, 3> refractive_index{
                                        rendex::random::uniform(generator, 0.0, 5.0) +
                                            rendex::random::uniform(generator, 0.0, 5.0) * 1i,
                                        rendex::random::uniform(generator, 0.0, 5.0) +
                                            rendex::random::uniform(generator, 0.0, 5.0) * 1i,
                                        rendex::random::uniform(generator, 0.0, 5.0) +
                                            rendex::random::uniform(generator, 0.0, 5.0) * 1i,
                                    };
                                    auto fuzziness = rendex::random::uniform(generator, 0.0, 0.5);
                                    rendex::geometry::Sphere<T, rendex::tensor::Vector, rendex::reflection::Metal>
                                        sphere(radius, position,
                                               rendex::reflection::Metal<T, rendex::tensor::Vector>(refractive_index,
                                                                                                    fuzziness));

                                    if constexpr (sizeof...(Js))
                                        return rendex::geometry::construct_union(
                                            std::move(sphere), self(self, std::index_sequence<Js...>{}));
                                    else
                                        return sphere;
                                }](auto &&...args) {
                                    return function(function, std::forward<decltype(args)>(args)...);
                                }(std::make_index_sequence<40>{})))))));
    }();

    rendex::geometry::Sphere<T> bounds(1000.0, rendex::tensor::Vector<T, 3>{}, rendex::reflection::Dielectric<T>{});

    // camera

    auto aspect_ratio = 1. * W / H;
    auto vertical_fov = 20.0 / 180.0 * std::numbers::pi;

    rendex::tensor::Vector<T, 3> position{10.0, -2.0, -5.0};
    rendex::tensor::Vector<T, 3> target{0.0, 0.0, 0.0};
    constexpr rendex::tensor::Vector<T, 3> down{0.0, 1.0, 0.0};

    auto focus_distance = rendex::tensor::norm(rendex::tensor::Vector<T, 3>{3.0, -1.0, 0.0} - position);
    auto aperture_radius = 0.1;

    auto w = rendex::tensor::normalized(target - position);
    auto u = rendex::tensor::normalized(rendex::tensor::cross(down, w));
    auto v = rendex::tensor::cross(w, u);
    auto rotation = rendex::tensor::transposed(rendex::tensor::Matrix<T, 3, 3>{u, v, w});

    rendex::camera::Camera<T> camera(vertical_fov, aspect_ratio, focus_distance, aperture_radius, position, rotation);

    // background

    auto background = [](const auto &ray) constexpr {
        return rendex::math::lerp(ray.direction()[1], -1.0, 1.0, rendex::tensor::Vector<T, 3>{0.5, 0.7, 1.0},
                                  rendex::tensor::Vector<T, 3>{1.0, 1.0, 1.0});
    };

    // parameters

    auto max_depth = 50;
    auto max_steps = 1000;
    auto epsilon = 1e-3;

    // rendering

    auto colors = rendex::rendering::ray_tracing<T, W, H, S, I, N>(object, camera, background, max_depth);

    // gamma correction
    std::transform(std::begin(colors), std::end(colors), std::begin(colors),
                   [](const auto &color) { return rendex::tensor::elemwise(rendex::math::sqrt<T>, color); });

    return colors;
}
