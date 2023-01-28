#pragma once

#include "vector.hpp"

namespace rendex::optics
{
    template <typename Scalar>
    class Ray
    {
    public:
        using Vector = rendex::blas::Vector<Scalar, 3>;

        constexpr Ray(const auto &position, const auto &direction)
            : m_position(position),
              m_direction(direction) {}

        constexpr Ray(auto &&position, auto &&direction)
            : m_position(std::forward<std::decay_t<decltype(position)>>(position)),
              m_direction(std::forward<std::decay_t<decltype(direction)>>(direction)) {}

        constexpr auto &position() { return m_position; }
        constexpr const auto &position() const { return m_position; }

        constexpr auto &direction() { return m_direction; }
        constexpr const auto &direction() const { return m_direction; }

        constexpr void advance(auto distance) { m_position = m_position + m_direction * distance; }

    private:
        Vector m_position;
        Vector m_direction;
    };
}
