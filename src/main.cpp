#include <cmath>
#include <fstream>
#include <iostream>
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

#define CONSTEXPR  // constexpr

int main() {
    using Scalar = double;
    using namespace std::literals::complex_literals;

    // object

    rendex::random::LCG<> generator(rendex::random::now());

    constexpr auto object = []() constexpr {
        rendex::random::LCG<> generator(rendex::random::now());
        return rendex::geometry::construct_union(
            // ground sphere
            rendex::geometry::Sphere<Scalar, rendex::tensor::Vector, rendex::reflection::Dielectric>(
                1000.0, rendex::tensor::Vector<Scalar, 3>{0.0, 1000.0, 0.0},
                rendex::reflection::Dielectric<Scalar>(rendex::tensor::Vector<Scalar, 3>{0.5, 0.5, 0.5}, {}, 1.0, 1.0,
                                                       0.0)),
            rendex::geometry::construct_union(
                // left sphere (hollow Glass)
                rendex::geometry::construct_union(
                    rendex::geometry::Sphere<Scalar, rendex::tensor::Vector, rendex::reflection::Dielectric>(
                        1.0, rendex::tensor::Vector<Scalar, 3>{-3.0, -1.0, 0.0},
                        rendex::reflection::Dielectric<Scalar, rendex::tensor::Vector>(
                            {}, rendex::tensor::Vector<Scalar, 3>{1.0, 1.0, 1.0},
                            rendex::random::uniform(generator, 1.0, 2.0), 0.0,
                            rendex::random::uniform(generator, 0.0, 0.5))),
                    rendex::geometry::Sphere<Scalar, rendex::tensor::Vector, rendex::reflection::Dielectric>(
                        -0.5, rendex::tensor::Vector<Scalar, 3>{-3.0, -1.0, 0.0},
                        rendex::reflection::Dielectric<Scalar, rendex::tensor::Vector>(
                            {}, rendex::tensor::Vector<Scalar, 3>{1.0, 1.0, 1.0},
                            rendex::random::uniform(generator, 1.0, 2.0), 0.0,
                            rendex::random::uniform(generator, 0.0, 0.5)))),
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
                                     auto center =
                                         rendex::random::uniform_in_unit_sphere<Scalar, rendex::tensor::Vector>(
                                             generator) *
                                         10.0;
                                     auto radius = rendex::random::uniform(generator, 0.1, 0.3);
                                     auto position = rendex::tensor::Vector<Scalar, 3>{center[0], -radius, center[1]};

                                     auto albedo =
                                         rendex::tensor::elemwise(rendex::math::square<Scalar>,
                                                                  rendex::tensor::Vector<Scalar, 3>{
                                                                      rendex::random::uniform(generator, 0.0, 1.0),
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
                                 }](
                                auto &&...args) { return function(function, std::forward<decltype(args)>(args)...); }(
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
                                         auto position =
                                             rendex::tensor::Vector<Scalar, 3>{center[0], -radius, center[1]};

                                         auto transmittance = rendex::tensor::elemwise(
                                             rendex::math::sqrt<Scalar>,
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
                                    auto
                                        &&...args) { return function(function, std::forward<decltype(args)>(args)...); }(
                                    std::make_index_sequence<100>{}),
                                // tiny sphere (reflection only)
                                [function = [&]<auto I, auto... Is>(auto self,
                                                                    std::index_sequence<I, Is...>) constexpr {
                                    auto center =
                                        rendex::random::uniform_in_unit_sphere<Scalar, rendex::tensor::Vector>(
                                            generator) *
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
                                               rendex::reflection::Metal<Scalar, rendex::tensor::Vector>(
                                                   refractive_index, fuzziness));

                                    if constexpr (sizeof...(Is))
                                        return rendex::geometry::construct_union(
                                            std::move(sphere), self(self, std::index_sequence<Is...>{}));
                                    else
                                        return sphere;
                                }](auto &&...args) {
                                    return function(function, std::forward<decltype(args)>(args)...);
                                }(std::make_index_sequence<50>{})))))));
    }();

    // camera

    constexpr auto H = 400;
    constexpr auto W = 800;
    constexpr auto MSAA = 10;

    constexpr auto aspect_ratio = 1. * W / H;
    constexpr auto vertical_fov = 20.0 / 180.0 * std::numbers::pi;

    constexpr rendex::tensor::Vector<Scalar, 3> position{12.0, -2.0, -4.0};
    constexpr rendex::tensor::Vector<Scalar, 3> target{0.0, 0.0, 0.0};
    constexpr rendex::tensor::Vector<Scalar, 3> down{0.0, 1.0, 0.0};

    constexpr auto focus_distance = rendex::tensor::norm(target - position);
    constexpr auto aperture_radius = 0.1;

    constexpr auto w = rendex::tensor::normalized(target - position);
    constexpr auto u = rendex::tensor::normalized(rendex::tensor::cross(down, w));
    constexpr auto v = rendex::tensor::cross(w, u);
    constexpr auto rotation = rendex::tensor::transposed(rendex::tensor::Matrix<Scalar, 3, 3>{u, v, w});

    constexpr rendex::camera::Camera<Scalar> camera(vertical_fov, aspect_ratio, focus_distance, aperture_radius,
                                                    position, rotation);

    // background

    auto background = [](const auto &ray) constexpr {
        return rendex::math::lerp(ray.direction()[1], -1.0, 1.0, rendex::tensor::Vector<Scalar, 3>{0.5, 0.7, 1.0},
                                  rendex::tensor::Vector<Scalar, 3>{1.0, 1.0, 1.0});
    };

    // parameters

    constexpr auto termination_prob = 0.01;

    constexpr rendex::geometry::Sphere<Scalar> bounding_sphere(1000.0, rendex::tensor::Vector<Scalar, 3>{},
                                                               rendex::reflection::Dielectric<Scalar>{});
    constexpr auto num_iterations = 1000;
    constexpr auto convergence_threshold = 1e-3;

    // rendering

    CONSTEXPR auto image = rendex::rendering::ray_tracing<rendex::tensor::DynamicTensor, Scalar, H, W, MSAA>(
        object, camera, background, termination_prob, generator);
    // CONSTEXPR auto image = rendex::rendering::ray_marching<DynamicTensor, Scalar, H, W, MSAA>(object, camera,
    // background, termination_prob, generator, bounding_sphere, num_iterations, convergence_threshold);

    // rendex::image::write_ppm(elemwise(rendex::math::sqrt<Scalar>, image), "image.ppm");
    rendex::image::write_ppm(image, "image.ppm");
}
