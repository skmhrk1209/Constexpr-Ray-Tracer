#pragma once

#include <cstdint>

namespace rendex::random {

template <typename T = std::uint_fast32_t>
constexpr auto now() {
    return static_cast<T>(__TIME__[0] - '0') * 10 + static_cast<T>(__TIME__[1] - '0') * 60 * 60 +
           static_cast<T>(__TIME__[3] - '0') * 10 + static_cast<T>(__TIME__[4] - '0') * 60 +
           static_cast<T>(__TIME__[6] - '0') * 10 + static_cast<T>(__TIME__[7] - '0');
}

template <typename T = std::uint_fast32_t, T A = 48271u, T B = 0u, T M = (1u << 31) - 1>
class LCG {
   public:
    static constexpr auto min = 0;
    static constexpr auto max = M;

    constexpr LCG() = default;
    constexpr LCG(T random) : m_random(random){};

    constexpr auto operator()() {
        m_random = (A * m_random + B) % M;
        return m_random;
    }

   private:
    T m_random;
};

}  // namespace rendex::random
