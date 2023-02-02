#pragma once

#include <optional>
#include <variant>

#include "geometry.hpp"
#include "math.hpp"
#include "reflection.hpp"
#include "tensor.hpp"

namespace rendex::geometry {

template <typename Scalar, typename Vector = rendex::tensor::Vector<Scalar, 3>,
          typename Material = rendex::reflection::Material<Scalar, Vector>>
class Sphere {
   public:
    constexpr Sphere() = default;

    constexpr Sphere(Scalar radius, const Vector &position, const Material &material)
        : m_radius(radius), m_position(position), m_material(material) {}

    constexpr Sphere(Scalar radius, const Vector &&position, const Material &&material)
        : m_radius(radius), m_position(std::move(position)), m_material(std::move(material)) {}

    constexpr auto &radius() { return m_radius; }
    constexpr const auto &radius() const { return m_radius; }

    constexpr auto &position() { return m_position; }
    constexpr const auto &position() const { return m_position; }

    constexpr auto &material() { return m_material; }
    constexpr const auto &material() const { return m_material; }

    constexpr auto intersect(const auto &ray) const {
        auto direction = ray.position() - m_position;
        auto a = rendex::tensor::dot(ray.direction(), ray.direction());
        auto b = rendex::tensor::dot(ray.direction(), direction);
        auto c = rendex::tensor::dot(direction, direction) - m_radius * m_radius;
        auto d = b * b - a * c;

        if (d >= 0) {
            auto distance_1 = (-b - std::sqrt(d)) / a;
            auto distance_2 = (-b + std::sqrt(d)) / a;

            if (distance_1 >= 0.0 || distance_2 >= 0.0) {
                auto distance =
                    distance_1 >= 0.0 ? distance_2 >= 0.0 ? std::min(distance_1, distance_2) : distance_1 : distance_2;
                return std::make_tuple(Geometry<Scalar, Vector, Material>(*this), std::optional<Scalar>(distance));
            } else {
                return std::make_tuple(Geometry<Scalar, Vector, Material>(*this), std::optional<Scalar>{});
            }
        } else {
            return std::make_tuple(Geometry<Scalar, Vector, Material>(*this), std::optional<Scalar>{});
        }
    }

    constexpr auto distance(const auto &position) const {
        auto distance = rendex::tensor::norm(position - m_position) - m_radius;
        return std::make_tuple(Geometry<Scalar, Vector, Material>(*this), distance);
    }

    constexpr auto normal(const auto &position) const {
        return (position - m_position) / rendex::tensor::norm(position - m_position);
    }

   private:
    Scalar m_radius;
    Vector m_position;
    Material m_material;
};

}  // namespace rendex::geometry
