#pragma once

#include <optional>
#include "math.hpp"
#include "blas.hpp"

namespace rendex::geom
{
    template <typename Scalar>
    class Sphere
    {
    public:
        using Vector = rendex::blas::Vector<Scalar, 3>;

        constexpr Sphere() = default;
        constexpr Sphere(const Sphere &) = default;
        constexpr Sphere(Sphere &&) = default;

        constexpr Sphere(const auto &position, const auto &radius)
            : m_position(position),
              m_radius(radius) {}

        constexpr Sphere(const auto &&position, const auto &&radius)
            : m_position(std::move(position)),
              m_radius(std::move(radius)) {}

        constexpr auto &position() { return m_position; }
        constexpr auto &radius() { return m_radius; }

        constexpr const auto &position() const { return m_position; }
        constexpr const auto &radius() const { return m_radius; }

        constexpr auto intersect(const auto &ray) const
        {
            auto direction = ray.position() - m_position;
            auto a = rendex::blas::dot(ray.direction(), ray.direction());
            auto b = rendex::blas::dot(ray.direction(), direction);
            auto c = rendex::blas::dot(direction, direction) - m_radius * m_radius;
            auto d = b * b - a * c;

            std::optional<decltype(d)> distance;
            if (d >= 0)
            {
                auto distance_1 = (-b - std::sqrt(d)) / a;
                auto distance_2 = (-b + std::sqrt(d)) / a;

                if (distance_1 >= 0 || distance_2 >= 0)
                {
                    distance = distance_1 >= 0 ? distance_2 >= 0 ? std::min(distance_1, distance_2) : distance_1 : distance_2;
                }
            }
            return std::make_tuple(std::cref(*this), std::move(distance));
        }

        constexpr auto distance(const auto &position) const
        {
            auto distance = rendex::blas::norm(position - m_position) - m_radius;
            return std::make_tuple(std::cref(*this), std::move(distance));
        }

        constexpr auto normal(const auto &position) const
        {
            return (position - m_position) / rendex::blas::norm(position - m_position);
        }

    private:
        Vector m_position;
        Scalar m_radius;
    };
}
