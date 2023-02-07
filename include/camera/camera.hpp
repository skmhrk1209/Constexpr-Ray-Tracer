#pragma once

#include <numbers>

#include "common.hpp"
#include "random.hpp"
#include "ray.hpp"
#include "tensor.hpp"

namespace rendex::camera {
template <typename Scalar, template <typename, auto> typename Vector = rendex::tensor::Vector,
          template <typename, auto, auto> typename Matrix = rendex::tensor::Matrix>
class Camera {
   public:
    constexpr Camera() = default;

    constexpr Camera(Scalar vertical_fov, Scalar aspect_ratio, Scalar focus_distance, Scalar aperture_radius,
                     const Vector<Scalar, 3> &position, const Matrix<Scalar, 3, 3> &orientation)
        : m_vertical_fov(vertical_fov),
          m_aspect_ratio(aspect_ratio),
          m_focus_distance(focus_distance),
          m_aperture_radius(aperture_radius),
          m_position(position),
          m_orientation(orientation) {}

    constexpr Camera(Scalar vertical_fov, Scalar aspect_ratio, Scalar focus_distance, Scalar aperture_radius,
                     Vector<Scalar, 3> &&position, Matrix<Scalar, 3, 3> &&orientation)
        : m_vertical_fov(vertical_fov),
          m_aspect_ratio(aspect_ratio),
          m_focus_distance(focus_distance),
          m_aperture_radius(aperture_radius),
          m_position(std::move(position)),
          m_orientation(std::move(orientation)) {}

    constexpr auto &vertical_fov() { return m_vertical_fov; }
    constexpr const auto &vertical_fov() const { return m_vertical_fov; }

    constexpr auto &aspect_ratio() { return m_aspect_ratio; }
    constexpr const auto &aspect_ratio() const { return m_aspect_ratio; }

    constexpr auto &focus_distance() { return m_focus_distance; }
    constexpr const auto &focus_distance() const { return m_focus_distance; }

    constexpr auto &aperture_radius() { return m_aperture_radius; }
    constexpr const auto &aperture_radius() const { return m_aperture_radius; }

    constexpr auto &position() { return m_position; }
    constexpr const auto &position() const { return m_position; }

    constexpr auto &orientation() { return m_orientation; }
    constexpr const auto &orientation() const { return m_orientation; }

    constexpr auto ray(auto u, auto v, auto &generator) const {
        auto viewport_height = 2.0 * std::tan(m_vertical_fov / 2.0);
        auto viewport_width = viewport_height * m_aspect_ratio;
        auto x = rendex::math::lerp(u, 0.0, 1.0, -viewport_width / 2.0, viewport_width / 2.0);
        auto y = rendex::math::lerp(v, 0.0, 1.0, -viewport_height / 2.0, viewport_height / 2.0);
        auto target = m_position + m_orientation % Vector<Scalar, 3>{x, y, 1.0} * m_focus_distance;
        auto defocus = rendex::random::uniform_in_unit_circle<Scalar, Vector>(generator) * m_aperture_radius;
        auto position = m_position + m_orientation % defocus;
        auto direction = rendex::tensor::normalized(target - position);
        return Ray<Scalar, Vector>{position, direction};
    }

   private:
    Scalar m_vertical_fov;
    Scalar m_aspect_ratio;
    Scalar m_focus_distance;
    Scalar m_aperture_radius;
    Vector<Scalar, 3> m_position;
    Matrix<Scalar, 3, 3> m_orientation;
};
}  // namespace rendex::camera
