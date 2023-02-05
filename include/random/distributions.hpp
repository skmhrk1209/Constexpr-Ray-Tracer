#pragma once

#include "generators.hpp"
#include "math.hpp"

namespace rendex::random {

template <typename T>
class Uniform {
   public:
    constexpr Uniform() = default;

    constexpr Uniform(T min, T max) : m_min(min), m_max(max) {}

    template <typename G>
    constexpr auto operator()(G& generator) const {
        return rendex::math::lerp(generator(), G::min, G::max, m_min, m_max);
    }

   private:
    T m_min;
    T m_max;
};

}  // namespace rendex::random
