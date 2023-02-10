#pragma once

#include <numbers>

#include "distributions.hpp"
#include "math.hpp"

namespace coex::random {

template <typename Scalar, template <typename, auto> typename Vector>
constexpr auto uniform_on_unit_circle(auto &generator) {
    auto theta = coex::random::uniform(generator, -std::numbers::pi, std::numbers::pi);
    return Vector<Scalar, 3>{std::cos(theta), std::sin(theta), 0.0};
}

template <typename Scalar, template <typename, auto> typename Vector>
constexpr auto uniform_in_unit_circle(auto &generator) {
    auto radius = coex::math::sqrt(coex::random::uniform(generator, 0.0, 1.0));
    return uniform_on_unit_circle<Scalar, Vector>(generator) * radius;
}

template <typename Scalar, template <typename, auto> typename Vector>
constexpr auto uniform_on_unit_sphere(auto &generator) {
    auto cosine = coex::random::uniform(generator, -1.0, 1.0);
    auto sine = coex::math::sqrt(1.0 - cosine * cosine);
    auto phi = coex::random::uniform(generator, -std::numbers::pi, std::numbers::pi);
    return Vector<Scalar, 3>{sine * std::cos(phi), sine * std::sin(phi), cosine};
}

template <typename Scalar, template <typename, auto> typename Vector>
constexpr auto uniform_in_unit_sphere(auto &generator) {
    auto radius = coex::math::cbrt(coex::random::uniform(generator, 0.0, 1.0));
    return uniform_on_unit_sphere<Scalar, Vector>(generator) * radius;
}

}  // namespace coex::random
