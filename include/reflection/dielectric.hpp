#pragma once

#include <complex>

#include "camera.hpp"
#include "random.hpp"
#include "tensor.hpp"
#include "utilities.hpp"

namespace rendex::reflection {

template <typename Scalar, template <typename, auto> typename Vector = rendex::tensor::Vector>
class Dielectric {
   public:
    constexpr Dielectric() = default;

    constexpr Dielectric(const Vector<Scalar, 3> &albedo, const Vector<Scalar, 3> &transmittance,
                         auto refractive_index, auto scatterness, auto fuzziness)
        : m_albedo(albedo),
          m_transmittance(transmittance),
          m_refractive_index(refractive_index),
          m_scatterness(scatterness),
          m_fuzziness(fuzziness) {}

    constexpr Dielectric(Vector<Scalar, 3> &&albedo, Vector<Scalar, 3> &&transmittance, auto refractive_index,
                         auto scatterness, auto fuzziness)
        : m_albedo(std::move(albedo)),
          m_transmittance(std::move(transmittance)),
          m_refractive_index(refractive_index),
          m_scatterness(scatterness),
          m_fuzziness(fuzziness) {}

    constexpr auto &albedo() { return m_albedo; }
    constexpr const auto &albedo() const { return m_albedo; }

    constexpr auto &transmittance() { return m_transmittance; }
    constexpr const auto &transmittance() const { return m_transmittance; }

    constexpr auto &refractive_index() { return m_refractive_index; }
    constexpr const auto &refractive_index() const { return m_refractive_index; }

    constexpr auto &scatterness() { return m_scatterness; }
    constexpr const auto &scatterness() const { return m_scatterness; }

    constexpr auto &fuzziness() { return m_fuzziness; }
    constexpr const auto &fuzziness() const { return m_fuzziness; }

    constexpr auto operator()(const auto &ray, const auto &normal, auto &generator) const {
        auto cosine = -rendex::tensor::dot(ray.direction(), normal);
        auto sine = rendex::math::sqrt(1.0 - cosine * cosine);
        auto local_normal = cosine > 0 ? normal : -normal;
        auto refractive_index = cosine > 0 ? m_refractive_index : 1.0 / m_refractive_index;
        auto specular_reflectance = rendex::math::square((1.0 - refractive_index) / (1.0 + refractive_index));
        auto fresnel_reflectance = schlick_approx(specular_reflectance, std::abs(cosine));
        if (sine > refractive_index || rendex::random::uniform(generator, 0.0, 1.0) < fresnel_reflectance) {
            auto reflected_position = ray.position() + 1e-6 * local_normal;
            auto reflected_direction = reflect(ray.direction(), local_normal);
            auto random_direction = rendex::random::uniform_in_unit_sphere<Scalar, Vector>(generator) * m_fuzziness;
            auto fuzzy_reflected_direction = rendex::tensor::normalized(reflected_direction + random_direction);
            rendex::camera::Ray<Scalar, Vector> reflected_ray(reflected_position, fuzzy_reflected_direction);
            return std::make_tuple(reflected_ray, Vector<Scalar, 3>{1.0, 1.0, 1.0});
        } else {
            if (rendex::random::uniform(generator, 0.0, 1.0) < m_scatterness) {
                auto scattered_position = ray.position() + 1e-6 * local_normal;
                auto random_direction = rendex::random::uniform_on_unit_sphere<Scalar, Vector>(generator);
                auto scattered_direction = rendex::tensor::normalized(local_normal + random_direction);
                rendex::camera::Ray<Scalar, Vector> scattered_ray(scattered_position, scattered_direction);
                return std::make_tuple(scattered_ray, m_albedo);
            } else {
                auto transmitted_position = ray.position() - 1e-6 * local_normal;
                auto transmitted_direction = refract(ray.direction(), local_normal, refractive_index);
                rendex::camera::Ray<Scalar, Vector> transmitted_ray(transmitted_position, transmitted_direction);
                return std::make_tuple(transmitted_ray, m_transmittance);
            }
        }
    }

   private:
    Vector<Scalar, 3> m_albedo;
    Vector<Scalar, 3> m_transmittance;
    Scalar m_refractive_index;
    Scalar m_scatterness;
    Scalar m_fuzziness;
};

}  // namespace rendex::reflection
