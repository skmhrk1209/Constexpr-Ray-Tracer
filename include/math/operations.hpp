#pragma once

#include <concepts>
#include <type_traits>

namespace rendex::math {

constexpr auto sqrt_impl(std::floating_point auto x, std::floating_point auto y, std::floating_point auto z)
    -> decltype(x) {
    return z - y < std::numeric_limits<decltype(x)>::epsilon() ? y : sqrt_impl(x, (y + x / y) / 2, y);
}

constexpr auto sqrt_impl(std::floating_point auto x, std::floating_point auto y) {
    return sqrt_impl(x, (y + x / y) / 2, y);
}

constexpr auto sqrt(std::floating_point auto x) { return sqrt_impl(x, std::max<decltype(x)>(x, 1)); }

constexpr auto pow(auto x, std::integral auto n) -> decltype(x) { return n == 1 ? x : x * pow(x, n - 1); }

constexpr auto lerp(const auto &in_val, const auto &in_min, const auto &in_max, const auto &out_min,
                    const auto &out_max) {
    return out_min + (out_max - out_min) * (in_val - in_min) / (in_max - in_min);
}

}  // namespace rendex::math
