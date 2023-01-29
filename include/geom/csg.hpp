#pragma once

namespace rendex::geom
{
    template <typename Geometry1, typename Geometry2, typename Compare>
    class CSG
    {
    public:
        constexpr CSG(const auto &geometry_1, const auto &geometry_2)
            : m_geometry_1(geometry_1),
              m_geometry_2(geometry_2) {}

        constexpr CSG(auto &&geometry_1, auto &&geometry_2)
            : m_geometry_1(std::forward<std::decay_t<decltype(geometry_1)>>(geometry_1)),
              m_geometry_2(std::forward<std::decay_t<decltype(geometry_2)>>(geometry_2)) {}

        constexpr auto distance(const auto &position) const
        {
            auto distance_1 = m_geometry_1.distance(position);
            auto distance_2 = m_geometry_2.distance(position);
            return Compare()(distance_1, distance_2) ? distance_1 : distance_2;
        }

        constexpr auto normal(const auto &position) const
        {
            auto distance_1 = m_geometry_1.distance(position);
            auto distance_2 = m_geometry_2.distance(position);
            return Compare()(distance_1, distance_2) ? m_geometry_1.normal(position) : m_geometry_2.normal(position);
        }

    private:
        Geometry1 m_geometry_1;
        Geometry2 m_geometry_2;
    };

    struct UnionOp
    {
        constexpr auto operator()(const auto &x, const auto &y) const
        {
            return x < y;
        }
    };

    template <typename Geometry1, typename Geometry2>
    using Union = CSG<Geometry1, Geometry2, UnionOp>;

    struct SubtractionOp
    {
        constexpr auto operator()(const auto &x, const auto &y) const
        {
            return x > -y;
        }
    };

    template <typename Geometry1, typename Geometry2>
    using Subtraction = CSG<Geometry1, Geometry2, SubtractionOp>;

    struct IntersectionOp
    {
        constexpr auto operator()(const auto &x, const auto &y) const
        {
            return x > y;
        }
    };

    template <typename Geometry1, typename Geometry2>
    using Intersection = CSG<Geometry1, Geometry2, IntersectionOp>;
}
