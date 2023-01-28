#pragma once

namespace rendex::geom
{
    template <typename Scalar>
    class Sphere
    {
    public:
        using Vector = rendex::blas::Vector<Scalar, 3>;

        constexpr Sphere(const auto &position, const auto &radius)
            : m_position(position),
              m_radius(radius) {}

        constexpr Sphere(auto &&position, auto &&radius)
            : m_position(std::forward<std::decay_t<decltype(position)>>(position)),
              m_radius(std::forward<std::decay_t<decltype(radius)>>(radius)) {}

        constexpr auto &position() { return m_position; }
        constexpr auto &radius() { return m_radius; }

        constexpr const auto &position() const { return m_position; }
        constexpr const auto &radius() const { return m_radius; }

        constexpr auto distance(const auto &position) const
        {
            return rendex::blas::norm(position - m_position) - m_radius;
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
