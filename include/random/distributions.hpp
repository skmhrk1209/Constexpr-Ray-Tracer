#pragma once

#include "generators.hpp"
#include "math.hpp"

namespace rendex::random {

template <typename Type, typename Generator = LCG<>>
class Uniform {
   public:
    using Float = Type;
    using UInt = typename Generator::Type;

    constexpr Uniform() = default;

    constexpr Uniform(Float min, Float max, UInt seed) : m_min(min), m_max(max), m_random(seed) {}

    constexpr auto operator()() {
        m_random = Generator()(m_random);
        return rendex::math::lerp(m_random, Generator::min, Generator::max, m_min, m_max);
    }

   private:
    Float m_min;
    Float m_max;
    UInt m_random;
};

}  // namespace rendex::random
