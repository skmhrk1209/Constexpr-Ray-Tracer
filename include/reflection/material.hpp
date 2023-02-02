#pragma once

#include "camera.hpp"
#include "random.hpp"
#include "tensor.hpp"

namespace rendex::reflection {

template <typename Scalar, typename Vector = rendex::tensor::Vector<Scalar, 3>,
          typename Generator = rendex::random::LCG<>, auto Seed = rendex::random::now()>
class Material {
   public:
    constexpr Material() = default;

    constexpr Material(const Vector &albedo) : m_albedo(albedo) {}

    constexpr Material(const Vector &&albedo) : m_albedo(std::move(albedo)) {}

    constexpr auto &albedo() { return m_albedo; }
    constexpr const auto &albedo() const { return m_albedo; }

    constexpr auto scatter(const auto &ray, const auto &geometry) {
        auto normal = geometry.normal(ray.position());

        auto theta = s_uniform() * std::numbers::pi;
        auto phi = s_uniform() * std::numbers::pi;
        rendex::tensor::Vector<Scalar, 3> random{std::cos(phi) * std::cos(theta), std::cos(phi) * std::sin(theta),
                                                 std::sin(phi)};

        auto direction = rendex::tensor::normalized(normal + random);
        rendex::camera::Ray<Scalar, Vector> scattered_ray(ray.position(), direction);

        return scattered_ray;
    }

   private:
    Vector m_albedo;
    static inline rendex::random::Uniform<Scalar, Generator> s_uniform{-1.0, 1.0, Seed};
};

}  // namespace rendex::reflection
