#pragma once

#include "blas.hpp"

namespace rendex::vision
{
    template <typename Scalar>
    class Ray
    {
    public:
        using Vector = rendex::blas::Vector<Scalar, 3>;

        constexpr Ray() = default;
        constexpr Ray(const Ray &) = default;
        constexpr Ray(Ray &&) = default;

        constexpr Ray(const auto &position, const auto &direction)
            : m_position(position),
              m_direction(direction) {}

        constexpr Ray(const auto &&position, const auto &&direction)
            : m_position(std::move(position)),
              m_direction(std::move(direction)) {}

        constexpr auto &position() { return m_position; }
        constexpr const auto &position() const { return m_position; }

        constexpr auto &direction() { return m_direction; }
        constexpr const auto &direction() const { return m_direction; }

        constexpr auto advance(auto distance) { m_position = m_position + m_direction * distance; }
        constexpr auto advanced(auto distance) const { return m_position + m_direction * distance; }

    private:
        Vector m_position;
        Vector m_direction;
    };
}
