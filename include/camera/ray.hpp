#pragma once

#include "tensor.hpp"

namespace rendex::camera {
template <typename Scalar, template <typename, auto> typename Vector = rendex::tensor::Vector>
class Ray {
   public:
    constexpr Ray() = default;

    constexpr Ray(const Vector<Scalar, 3> &position, const Vector<Scalar, 3> &direction)
        : m_position(position), m_direction(direction) {}

    constexpr Ray(Vector<Scalar, 3> &&position, Vector<Scalar, 3> &&direction)
        : m_position(std::move(position)), m_direction(std::move(direction)) {}

    constexpr auto &position() { return m_position; }
    constexpr const auto &position() const { return m_position; }

    constexpr auto &direction() { return m_direction; }
    constexpr const auto &direction() const { return m_direction; }

    constexpr auto advance(auto distance) { m_position = m_position + m_direction * distance; }
    constexpr auto advanced(auto distance) const { return m_position + m_direction * distance; }

   private:
    Vector<Scalar, 3> m_position;
    Vector<Scalar, 3> m_direction;
};
}  // namespace rendex::camera
