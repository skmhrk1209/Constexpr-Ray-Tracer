#pragma once

#include <cmath>
#include <numbers>

#include "math.hpp"
#include "tensor.hpp"

namespace coex::reflection {

constexpr auto reflect(const auto &incident, const auto &normal) {
    return incident - 2.0 * coex::tensor::dot(incident, normal) * normal;
}

constexpr auto refract(const auto &incident, const auto &normal, auto refractive_index) {
    auto parl = (incident - coex::tensor::dot(incident, normal) * normal) / refractive_index;
    auto perp = -coex::math::sqrt(1.0 - coex::tensor::dot(parl, parl)) * normal;
    return parl + perp;
}

constexpr auto schlick_approx(auto specular_reflectance, auto cosine) {
    return specular_reflectance + (1.0 - specular_reflectance) * coex::math::pow(1.0 - cosine, 5);
}

}  // namespace coex::reflection
