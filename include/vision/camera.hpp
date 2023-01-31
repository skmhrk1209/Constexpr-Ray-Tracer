#pragma once

#include "ray.hpp"
#include "../blas.hpp"

namespace rendex::vision
{
    template <typename Scalar>
    class Camera
    {
    public:
        using Vector = rendex::blas::Vector<Scalar, 3>;
        using Matrix = rendex::blas::Matrix<Scalar, 3, 3>;

        constexpr Camera() = default;
        constexpr Camera(const Camera &) = default;
        constexpr Camera(Camera &&) = default;

        constexpr Camera(auto vertical_fov,
                         auto aspect_ratio,
                         const auto &position,
                         const auto &orientation)
            : m_vertical_fov(vertical_fov),
              m_aspect_ratio(aspect_ratio),
              m_position(position),
              m_orientation(orientation)
        {
        }

        constexpr Camera(auto vertical_fov,
                         auto aspect_ratio,
                         const auto &&position,
                         const auto &&orientation)
            : m_vertical_fov(vertical_fov),
              m_aspect_ratio(aspect_ratio),
              m_position(std::move(position)),
              m_orientation(std::move(orientation))
        {
        }

        constexpr auto &vertical_fov() { return m_vertical_fov; }
        constexpr const auto &vertical_fov() const { return m_vertical_fov; }

        constexpr auto &aspect_ratio() { return m_aspect_ratio; }
        constexpr const auto &aspect_ratio() const { return m_aspect_ratio; }

        constexpr auto &position() { return m_position; }
        constexpr const auto &position() const { return m_position; }

        constexpr auto &orientation() { return m_orientation; }
        constexpr const auto &orientation() const { return m_orientation; }

        constexpr auto ray(auto u, auto v) const
        {
            auto viewport_height = 2.0 * std::tan(m_vertical_fov / 2.0);
            auto viewport_width = viewport_height * m_aspect_ratio;
            auto x = rendex::math::lerp(u, 0.0, 1.0, -viewport_width / 2.0, viewport_width / 2.0);
            auto y = rendex::math::lerp(v, 0.0, 1.0, -viewport_height / 2.0, viewport_height / 2.0);
            return Ray<Scalar>(Vector{}, rendex::blas::normalized(Vector{x, y, 1.0}));
        }

    private:
        Scalar m_vertical_fov;
        Scalar m_aspect_ratio;
        Vector m_position;
        Matrix m_orientation;
    };
}
