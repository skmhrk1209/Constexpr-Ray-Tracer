#pragma once

#include <cmath>
#include <numbers>

#include "math.hpp"
#include "tensor.hpp"

namespace rendex::reflection {

constexpr auto reflect(const auto &incident, const auto &normal) {
    return incident - 2.0 * rendex::tensor::dot(incident, normal) * normal;
}

constexpr auto refract(const auto &incident, const auto &normal, auto refractive_index) {
    auto parl = (incident - rendex::tensor::dot(incident, normal) * normal) / refractive_index;
    auto perp = -rendex::math::sqrt(1.0 - rendex::tensor::dot(parl, parl)) * normal;
    return parl + perp;
}

constexpr auto schlick_approx(auto specular_reflectance, auto cosine) {
    return specular_reflectance + (1.0 - specular_reflectance) * rendex::math::pow(1.0 - cosine, 5);
}

template <typename Scalar, template <typename, auto> typename Vector>
constexpr auto random_on_unit_sphere(const auto &uniform, auto &generator) {
    auto cosine = -2.0 * uniform(generator) + 1.0;
    auto sine = rendex::math::sqrt(1.0 - cosine * cosine);
    auto phi = 2.0 * std::numbers::pi * uniform(generator);
    return Vector<Scalar, 3>{sine * std::cos(phi), sine * std::sin(phi), cosine};
}

template <typename Scalar, template <typename, auto> typename Vector>
constexpr auto random_in_unit_sphere(const auto &uniform, auto &generator) {
    return random_on_unit_sphere<Scalar, Vector>(uniform, generator) * std::cbrt(uniform(generator));
}

}  // namespace rendex::reflection
