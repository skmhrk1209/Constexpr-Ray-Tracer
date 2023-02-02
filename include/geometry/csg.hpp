#pragma once

#include <concepts>
#include <tuple>
#include <variant>

#include "camera.hpp"
#include "common.hpp"

namespace rendex::geometry {

template <typename Geometry1, typename Geometry2, typename Op>
class CSG {
   public:
    constexpr CSG() = default;

    constexpr CSG(const Geometry1 &geometry_1, const Geometry2 &geometry_2)
        : m_geometry_1(geometry_1), m_geometry_2(geometry_2) {}

    constexpr CSG(const Geometry1 &&geometry_1, const Geometry2 &&geometry_2)
        : m_geometry_1(std::move(geometry_1)), m_geometry_2(std::move(geometry_2)) {}

    constexpr auto distance(const auto &position) const {
        auto [geometry_1, distance_1] = m_geometry_1.distance(position);
        auto [geometry_2, distance_2] = m_geometry_2.distance(position);
        if (Op()(distance_1, distance_2)) {
            return std::make_tuple(std::move(geometry_1), std::move(distance_1));
        } else {
            return std::make_tuple(std::move(geometry_2), std::move(distance_2));
        }
    }

    constexpr auto intersect(const auto &ray) const {
        auto [geometry_1, distance_1] = m_geometry_1.intersect(ray);
        auto [geometry_2, distance_2] = m_geometry_2.intersect(ray);
        if (Op()(distance_1, distance_2)) {
            return std::make_tuple(std::move(geometry_1), std::move(distance_1));
        } else {
            return std::make_tuple(std::move(geometry_2), std::move(distance_2));
        }
    }

   private:
    Geometry1 m_geometry_1;
    Geometry2 m_geometry_2;
};

struct UnionOp {
    template <typename T>
    constexpr auto operator()(const T &x, const T &y) const {
        if constexpr (rendex::is_optional_v<T>)
            return x ? y ? x.value() < y.value() : true : false;
        else
            return x < y;
    }
};

template <typename Geometry1, typename Geometry2>
using Union = CSG<Geometry1, Geometry2, UnionOp>;

struct SubtractionOp {
    template <typename T>
    constexpr auto operator()(const T &x, const T &y) const {
        if constexpr (rendex::is_optional_v<T>)
            return x ? y ? x.value() > -y.value() : true : false;
        else
            return x > -y;
    }
};

template <typename Geometry1, typename Geometry2>
using Subtraction = CSG<Geometry1, Geometry2, SubtractionOp>;

struct IntersectionOp {
    template <typename T>
    constexpr auto operator()(const T &x, const T &y) const {
        if constexpr (rendex::is_optional_v<T>)
            return x ? y ? x.value() > y.value() : true : false;
        else
            return x > y;
    }
};

template <typename Geometry1, typename Geometry2>
using Intersection = CSG<Geometry1, Geometry2, IntersectionOp>;

}  // namespace rendex::geometry
