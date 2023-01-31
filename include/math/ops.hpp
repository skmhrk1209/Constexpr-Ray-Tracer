#pragma once

#include <type_traits>

namespace rendex::math
{
    constexpr auto lerp(const auto &in_val, const auto &in_min, const auto &in_max, const auto &out_min, const auto &out_max)
    {
        return out_min + (out_max - out_min) * (in_val - in_min) / (in_max - in_min);
    }
}
