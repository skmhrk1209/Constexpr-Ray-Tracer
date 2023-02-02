#pragma once

#include "ray.hpp"
#include "tensor.hpp"

namespace rendex::camera {
template <typename Scalar, typename Vector = rendex::tensor::Vector<Scalar, 3>,
          typename Matrix = rendex::tensor::Matrix<Scalar, 3, 3>>
class Camera {
   public:
    constexpr Camera() = default;

    constexpr Camera(Scalar vertical_fov, Scalar aspect_ratio, const Vector &position, const Matrix &orientation)
        : m_vertical_fov(vertical_fov),
          m_aspect_ratio(aspect_ratio),
          m_position(position),
          m_orientation(orientation) {}

    constexpr Camera(Scalar vertical_fov, Scalar aspect_ratio, const Vector &&position, const Matrix &&orientation)
        : m_vertical_fov(vertical_fov),
          m_aspect_ratio(aspect_ratio),
          m_position(std::move(position)),
          m_orientation(std::move(orientation)) {}

    constexpr auto &vertical_fov() { return m_vertical_fov; }
    constexpr const auto &vertical_fov() const { return m_vertical_fov; }

    constexpr auto &aspect_ratio() { return m_aspect_ratio; }
    constexpr const auto &aspect_ratio() const { return m_aspect_ratio; }

    constexpr auto &position() { return m_position; }
    constexpr const auto &position() const { return m_position; }

    constexpr auto &orientation() { return m_orientation; }
    constexpr const auto &orientation() const { return m_orientation; }

    constexpr auto ray(auto u, auto v) const {
        auto viewport_height = 2.0 * std::tan(m_vertical_fov / 2.0);
        auto viewport_width = viewport_height * m_aspect_ratio;
        auto x = rendex::math::lerp(u, 0.0, 1.0, -viewport_width / 2.0, viewport_width / 2.0);
        auto y = rendex::math::lerp(v, 0.0, 1.0, -viewport_height / 2.0, viewport_height / 2.0);
        return Ray<Scalar, Vector>{Vector{}, rendex::tensor::normalized(Vector{x, y, 1.0})};
    }

   private:
    Scalar m_vertical_fov;
    Scalar m_aspect_ratio;
    Vector m_position;
    Matrix m_orientation;
};
}  // namespace rendex::camera
