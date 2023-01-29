#pragma once

#include <type_traits>

namespace rendex::math
{
    constexpr auto abs(auto x)
        -> std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(x)>>, std::decay_t<decltype(x)>>
    {
        return x > static_cast<std::decay_t<decltype(x)>>(0) ? x : -x;
    }

    constexpr auto sqrt_impl(auto x, auto y, auto z)
        -> std::enable_if_t<std::is_floating_point_v<std::decay_t<decltype(x)>>, std::decay_t<decltype(x)>>
    {
        return /* abs(y - z) */ z - y < std::numeric_limits<std::decay_t<decltype(x)>>::epsilon() ? y : sqrt_impl(x, (y + x / y) / static_cast<std::decay_t<decltype(x)>>(2.0), y);
    }

    constexpr auto sqrt_impl(auto x, auto y)
        -> std::enable_if_t<std::is_floating_point_v<std::decay_t<decltype(x)>>, std::decay_t<decltype(x)>>
    {
        return sqrt_impl(x, (y + x / y) / static_cast<std::decay_t<decltype(x)>>(2), y);
    }

    constexpr auto sqrt(auto x)
        -> std::enable_if_t<std::is_floating_point_v<std::decay_t<decltype(x)>>, std::decay_t<decltype(x)>>
    {
        return sqrt_impl(x, std::max(x, static_cast<std::decay_t<decltype(x)>>(1)));
    }

    constexpr auto lerp(const auto &in_val, const auto &in_min, const auto &in_max, const auto &out_min, const auto &out_max)
    {
        return out_min + (out_max - out_min) * (in_val - in_min) / (in_max - in_min);
    }
}
