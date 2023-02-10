#pragma once

#include <complex>

#include "camera.hpp"
#include "random.hpp"
#include "tensor.hpp"
#include "utilities.hpp"

namespace coex::reflection {

template <typename Scalar, template <typename, auto> typename Vector = coex::tensor::Vector>
class Dielectric {
   public:
    constexpr Dielectric() = default;

    constexpr Dielectric(const Vector<Scalar, 3> &albedo, auto refractive_index)
        : m_albedo(albedo), m_refractive_index(refractive_index) {}

    constexpr Dielectric(Vector<Scalar, 3> &&albedo, auto refractive_index)
        : m_albedo(std::move(albedo)), m_refractive_index(refractive_index) {}

    constexpr auto &albedo() { return m_albedo; }
    constexpr const auto &albedo() const { return m_albedo; }

    constexpr auto &refractive_index() { return m_refractive_index; }
    constexpr const auto &refractive_index() const { return m_refractive_index; }

    constexpr auto operator()(const auto &ray, const auto &normal, auto &generator) const {
        auto cosine = -coex::tensor::dot(ray.direction(), normal);
        auto sine = coex::math::sqrt(1.0 - cosine * cosine);
        auto inout_normal = cosine > 0 ? normal : -normal;
        auto refractive_index = cosine > 0 ? m_refractive_index : 1.0 / m_refractive_index;
        auto specular_reflectance = coex::math::square((1.0 - refractive_index) / (1.0 + refractive_index));
        auto fresnel_reflectance = schlick_approx(specular_reflectance, std::abs(cosine));
        if (sine > refractive_index || coex::random::uniform(generator, 0.0, 1.0) < fresnel_reflectance) {
            auto reflected_position = ray.position() + 1e-6 * inout_normal;
            auto reflected_direction = reflect(ray.direction(), inout_normal);
            coex::camera::Ray<Scalar, Vector> reflected_ray(std::move(reflected_position),
                                                            std::move(reflected_direction));
            return std::make_tuple(std::move(reflected_ray), Vector<Scalar, 3>{1.0, 1.0, 1.0});
        } else {
            auto refracted_position = ray.position() - 1e-6 * inout_normal;
            auto refracted_direction = refract(ray.direction(), inout_normal, refractive_index);
            coex::camera::Ray<Scalar, Vector> refracted_ray(std::move(refracted_position),
                                                            std::move(refracted_direction));
            return std::make_tuple(std::move(refracted_ray), m_albedo);
        }
    }

   private:
    Vector<Scalar, 3> m_albedo;
    Scalar m_refractive_index;
};

}  // namespace coex::reflection
