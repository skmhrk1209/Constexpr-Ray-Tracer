#pragma once

#include <complex>

#include "camera.hpp"
#include "random.hpp"
#include "tensor.hpp"
#include "utilities.hpp"

namespace coex::reflection {

template <typename Scalar, template <typename, auto> typename Vector = coex::tensor::Vector>
class Lambertian {
   public:
    constexpr Lambertian() = default;

    constexpr Lambertian(const Vector<Scalar, 3> &albedo) : m_albedo(albedo) {}

    constexpr Lambertian(Vector<Scalar, 3> &&albedo) : m_albedo(std::move(albedo)) {}

    constexpr auto &albedo() { return m_albedo; }
    constexpr const auto &albedo() const { return m_albedo; }

    constexpr auto operator()(const auto &ray, const auto &normal, auto &generator) const {
        auto scattered_position = ray.position() + 1e-6 * normal;
        auto random_direction = coex::random::uniform_on_unit_sphere<Scalar, Vector>(generator);
        auto scattered_direction = coex::tensor::normalized(normal + random_direction);
        coex::camera::Ray<Scalar, Vector> scattered_ray(std::move(scattered_position), std::move(scattered_direction));
        return std::make_tuple(std::move(scattered_ray), m_albedo);
    }

   private:
    Vector<Scalar, 3> m_albedo;
};

}  // namespace coex::reflection
