#pragma once

#include <array>
#include <tuple>
#include <numeric>
#include <type_traits>
#include "math.hpp"
#include "traits.hpp"

namespace rendex::blas
{
    template <typename Scalar, auto N>
    class Vector : public std::array<Scalar, N>
    {
    };

    template <typename Scalar>
    struct Vector<Scalar, 1> : public std::array<Scalar, 1>
    {
        constexpr auto &x() { return (*this)[0]; }
        constexpr const auto &x() const { return (*this)[0]; }
    };

    template <typename Scalar>
    struct Vector<Scalar, 2> : public std::array<Scalar, 2>
    {
        constexpr auto &x() { return (*this)[0]; }
        constexpr const auto &x() const { return (*this)[0]; }

        constexpr auto &y() { return (*this)[1]; }
        constexpr const auto &y() const { return (*this)[1]; }
    };

    template <typename Scalar>
    struct Vector<Scalar, 3> : public std::array<Scalar, 3>
    {
        constexpr auto &x() { return (*this)[0]; }
        constexpr const auto &x() const { return (*this)[0]; }

        constexpr auto &y() { return (*this)[1]; }
        constexpr const auto &y() const { return (*this)[1]; }

        constexpr auto &z() { return (*this)[2]; }
        constexpr const auto &z() const { return (*this)[2]; }
    };

    template <typename Scalar>
    struct Vector<Scalar, 4> : public std::array<Scalar, 4>
    {
        constexpr auto &x() { return (*this)[0]; }
        constexpr const auto &x() const { return (*this)[0]; }

        constexpr auto &y() { return (*this)[1]; }
        constexpr const auto &y() const { return (*this)[1]; }

        constexpr auto &z() { return (*this)[2]; }
        constexpr const auto &z() const { return (*this)[2]; }

        constexpr auto &w() { return (*this)[3]; }
        constexpr const auto &w() const { return (*this)[3]; }
    };

    // ================================================================
    // concepts

    template <auto N>
    struct is_dimensional;

    template <>
    struct is_dimensional<1>
    {
        template <typename Type>
        using Check = decltype(std::declval<Type>().x());
    };

    template <>
    struct is_dimensional<2>
    {
        template <typename Type>
        using Check = decltype(std::declval<Type>().x(),
                               std::declval<Type>().y());
    };

    template <>
    struct is_dimensional<3>
    {
        template <typename Type>
        using Check = decltype(std::declval<Type>().x(),
                               std::declval<Type>().y(),
                               std::declval<Type>().z());
    };

    template <>
    struct is_dimensional<4>
    {
        template <typename Type>
        using Check = decltype(std::declval<Type>().x(),
                               std::declval<Type>().y(),
                               std::declval<Type>().z(),
                               std::declval<Type>().w());
    };

    template <typename Type, auto N>
    concept Dimensional = (rendex::traits::is_detected<is_dimensional<N>::template Check, Type>::value &&
                           (N == 4 || !rendex::traits::is_detected<is_dimensional<N + 1>::template Check, Type>::value));

    template <typename Type>
    concept AnyDimensional = Dimensional<Type, 1> || Dimensional<Type, 2> || Dimensional<Type, 3> || Dimensional<Type, 4>;

    template <typename Type>
    concept NonDimensional = std::is_arithmetic_v<Type>;

    // ================================================================
    // addition

    // ----------------------------------------------------------------
    // vector + vector

    // 4-dimantional

    constexpr auto operator+(const Dimensional<4> auto &v1, const Dimensional<4> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z(), v1.w() + v2.w()};
    }

    // 3-dimantional

    constexpr auto operator+(const Dimensional<3> auto &v1, const Dimensional<3> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z()};
    }

    // 2-dimantional

    constexpr auto operator+(const Dimensional<2> auto &v1, const Dimensional<2> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() + v2.x(), v1.y() + v2.y()};
    }

    // 1-dimantional

    constexpr auto operator+(const Dimensional<1> auto &v1, const Dimensional<1> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() + v2.x()};
    }

    // ----------------------------------------------------------------
    // vector + scalar

    // 4-dimantional

    constexpr auto operator+(const Dimensional<4> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v + std::decay_t<decltype(v)>{s, s, s, s};
    }

    // 3-dimantional

    constexpr auto operator+(const Dimensional<3> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v + std::decay_t<decltype(v)>{s, s, s};
    }

    // 2-dimantional

    constexpr auto operator+(const Dimensional<2> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v + std::decay_t<decltype(v)>{s, s};
    }

    // 1-dimantional

    constexpr auto operator+(const Dimensional<1> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v + std::decay_t<decltype(v)>{s};
    }

    // ----------------------------------------------------------------
    // scalar + vector

    // 4-dimantional

    constexpr auto operator+(NonDimensional auto s, const Dimensional<4> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s, s, s} + v;
    }

    // 3-dimantional

    constexpr auto operator+(NonDimensional auto s, const Dimensional<3> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s, s} + v;
    }

    // 2-dimantional

    constexpr auto operator+(NonDimensional auto s, const Dimensional<2> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s} + v;
    }

    // 1-dimantional

    constexpr auto operator+(NonDimensional auto s, const Dimensional<1> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s} + v;
    }

    // ----------------------------------------------------------------
    // + vector

    constexpr auto operator+(const auto &v) -> std::decay_t<decltype(v)>
    {
        return 0 + v;
    }

    // ================================================================
    // subtraction

    // ----------------------------------------------------------------
    // vector - vector

    // 4-dimantional

    constexpr auto operator-(const Dimensional<4> auto &v1, const Dimensional<4> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z(), v1.w() - v2.w()};
    }

    // 3-dimantional

    constexpr auto operator-(const Dimensional<3> auto &v1, const Dimensional<3> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z()};
    }

    // 2-dimantional

    constexpr auto operator-(const Dimensional<2> auto &v1, const Dimensional<2> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() - v2.x(), v1.y() - v2.y()};
    }

    // 1-dimantional

    constexpr auto operator-(const Dimensional<1> auto &v1, const Dimensional<1> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() - v2.x()};
    }

    // ----------------------------------------------------------------
    // vector - scalar

    // 4-dimantional

    constexpr auto operator-(const Dimensional<4> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v - std::decay_t<decltype(v)>{s, s, s, s};
    }

    // 3-dimantional

    constexpr auto operator-(const Dimensional<3> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v - std::decay_t<decltype(v)>{s, s, s};
    }

    // 2-dimantional

    constexpr auto operator-(const Dimensional<2> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v - std::decay_t<decltype(v)>{s, s};
    }

    // 1-dimantional

    constexpr auto operator-(const Dimensional<1> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v - std::decay_t<decltype(v)>{s};
    }

    // ----------------------------------------------------------------
    // scalar - vector

    // 4-dimantional

    constexpr auto operator-(NonDimensional auto s, const Dimensional<4> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s, s, s} - v;
    }

    // 3-dimantional

    constexpr auto operator-(NonDimensional auto s, const Dimensional<3> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s, s} - v;
    }

    // 2-dimantional

    constexpr auto operator-(NonDimensional auto s, const Dimensional<2> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s} - v;
    }

    // 1-dimantional

    constexpr auto operator-(NonDimensional auto s, const Dimensional<1> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s} - v;
    }

    // ----------------------------------------------------------------
    // - vector

    constexpr auto operator-(const auto &v) -> std::decay_t<decltype(v)>
    {
        return 0 - v;
    }

    // ================================================================
    // multiplication

    // ----------------------------------------------------------------
    // vector * vector

    // 4-dimantional

    constexpr auto operator*(const Dimensional<4> auto &v1, const Dimensional<4> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z(), v1.w() * v2.w()};
    }

    // 3-dimantional

    constexpr auto operator*(const Dimensional<3> auto &v1, const Dimensional<3> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z()};
    }

    // 2-dimantional

    constexpr auto operator*(const Dimensional<2> auto &v1, const Dimensional<2> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() * v2.x(), v1.y() * v2.y()};
    }

    // 1-dimantional

    constexpr auto operator*(const Dimensional<1> auto &v1, const Dimensional<1> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() * v2.x()};
    }

    // ----------------------------------------------------------------
    // vector * scalar

    // 4-dimantional

    constexpr auto operator*(const Dimensional<4> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v * std::decay_t<decltype(v)>{s, s, s, s};
    }

    // 3-dimantional

    constexpr auto operator*(const Dimensional<3> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v * std::decay_t<decltype(v)>{s, s, s};
    }

    // 2-dimantional

    constexpr auto operator*(const Dimensional<2> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v * std::decay_t<decltype(v)>{s, s};
    }

    // 1-dimantional

    constexpr auto operator*(const Dimensional<1> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v * std::decay_t<decltype(v)>{s};
    }

    // ----------------------------------------------------------------
    // scalar * vector

    // 4-dimantional

    constexpr auto operator*(NonDimensional auto s, const Dimensional<4> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s, s, s} * v;
    }

    // 3-dimantional

    constexpr auto operator*(NonDimensional auto s, const Dimensional<3> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s, s} * v;
    }

    // 2-dimantional

    constexpr auto operator*(NonDimensional auto s, const Dimensional<2> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s} * v;
    }

    // 1-dimantional

    constexpr auto operator*(NonDimensional auto s, const Dimensional<1> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s} * v;
    }

    // ================================================================
    // division

    // ----------------------------------------------------------------
    // vector / vector

    // 4-dimantional

    constexpr auto operator/(const Dimensional<4> auto &v1, const Dimensional<4> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() / v2.x(), v1.y() / v2.y(), v1.z() / v2.z(), v1.w() / v2.w()};
    }

    // 3-dimantional

    constexpr auto operator/(const Dimensional<3> auto &v1, const Dimensional<3> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() / v2.x(), v1.y() / v2.y(), v1.z() / v2.z()};
    }

    // 2-dimantional

    constexpr auto operator/(const Dimensional<2> auto &v1, const Dimensional<2> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() / v2.x(), v1.y() / v2.y()};
    }

    // 1-dimantional

    constexpr auto operator/(const Dimensional<1> auto &v1, const Dimensional<1> auto &v2) -> std::decay_t<decltype(v1)>
    {
        return {v1.x() / v2.x()};
    }

    // ----------------------------------------------------------------
    // vector / scalar

    // 4-dimantional

    constexpr auto operator/(const Dimensional<4> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v / std::decay_t<decltype(v)>{s, s, s, s};
    }

    // 3-dimantional

    constexpr auto operator/(const Dimensional<3> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v / std::decay_t<decltype(v)>{s, s, s};
    }

    // 2-dimantional

    constexpr auto operator/(const Dimensional<2> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v / std::decay_t<decltype(v)>{s, s};
    }

    // 1-dimantional

    constexpr auto operator/(const Dimensional<1> auto &v, NonDimensional auto s) -> std::decay_t<decltype(v)>
    {
        return v / std::decay_t<decltype(v)>{s};
    }

    // ----------------------------------------------------------------
    // scalar / vector

    // 4-dimantional

    constexpr auto operator/(NonDimensional auto s, const Dimensional<4> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s, s, s} / v;
    }

    // 3-dimantional

    constexpr auto operator/(NonDimensional auto s, const Dimensional<3> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s, s} / v;
    }

    // 2-dimantional

    constexpr auto operator/(NonDimensional auto s, const Dimensional<2> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s, s} / v;
    }

    // 1-dimantional

    constexpr auto operator/(NonDimensional auto s, const Dimensional<1> auto &v) -> std::decay_t<decltype(v)>
    {
        return std::decay_t<decltype(v)>{s} / v;
    }

    // ================================================================
    // dot

    constexpr auto dot(const AnyDimensional auto &v1, const AnyDimensional auto &v2) -> std::decay_t<decltype(v1.x())>
    {
        return rendex::math::sum(v1 * v2);
    }

    // ================================================================
    // norm

    constexpr auto norm(const AnyDimensional auto &v) -> std::decay_t<decltype(v.x())>
    {
        return rendex::math::sqrt(dot(v, v));
    }
}
