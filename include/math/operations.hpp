#pragma once

#include <type_traits>

namespace rendex::math {

constexpr auto sqrt_impl(auto x, auto y, auto z) -> decltype(x)
    requires std::is_floating_point_v<decltype(x)>
{
    return z - y < std::numeric_limits<decltype(x)>::epsilon() ? y : sqrt_impl(x, (y + x / y) / 2, y);
}

constexpr auto sqrt_impl(auto x, auto y)
    requires std::is_floating_point_v<decltype(x)>
{
    return sqrt_impl(x, (y + x / y) / 2, y);
}

constexpr auto sqrt(auto x)
    requires std::is_floating_point_v<decltype(x)>
{
    return sqrt_impl(x, std::max<decltype(x)>(x, 1));
}

constexpr auto lerp(const auto &in_val, const auto &in_min, const auto &in_max, const auto &out_min,
                    const auto &out_max) {
    return out_min + (out_max - out_min) * (in_val - in_min) / (in_max - in_min);
}

}  // namespace rendex::math
