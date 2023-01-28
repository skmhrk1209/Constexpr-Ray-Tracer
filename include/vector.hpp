#pragma once

#include <array>
#include <numeric>
#include <type_traits>
#include "math.hpp"

namespace rendex::blas
{
    template <typename Scalar, auto N>
    class Vector : public std::array<Scalar, N>
    {
    };

    template <typename Scalar>
    class Vector<Scalar, 1> : public std::array<Scalar, 1>
    {
    public:
        constexpr auto &x() { return (*this)[0]; }
        constexpr const auto &x() const { return (*this)[0]; }
    };

    template <typename Scalar>
    class Vector<Scalar, 2> : public std::array<Scalar, 2>
    {
    public:
        constexpr auto &x() { return (*this)[0]; }
        constexpr const auto &x() const { return (*this)[0]; }

        constexpr auto &y() { return (*this)[1]; }
        constexpr const auto &y() const { return (*this)[1]; }
    };

    template <typename Scalar>
    class Vector<Scalar, 3> : public std::array<Scalar, 3>
    {
    public:
        constexpr auto &x() { return (*this)[0]; }
        constexpr const auto &x() const { return (*this)[0]; }

        constexpr auto &y() { return (*this)[1]; }
        constexpr const auto &y() const { return (*this)[1]; }

        constexpr auto &z() { return (*this)[2]; }
        constexpr const auto &z() const { return (*this)[2]; }
    };

    template <typename Scalar>
    class Vector<Scalar, 4> : public std::array<Scalar, 4>
    {
    public:
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
    // SFINAE priority

    template <auto N>
    struct priority : priority<N - 1>
    {
    };
    template <>
    struct priority<0>
    {
    };

    // ================================================================
    // addition

    // ----------------------------------------------------------------
    // vector + vector

    // 4-dimantional

    constexpr auto add_impl(const auto &v1, const auto &v2, priority<4>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v1.z(), v1.w(), v2.x(), v2.y(), v2.z(), v2.w(), v1)>
    {
        return {v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z(), v1.w() + v2.w()};
    }

    // 3-dimantional

    constexpr auto add_impl(const auto &v1, const auto &v2, priority<3>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v1.z(), v2.x(), v2.y(), v2.z(), v1)>
    {
        return {v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z()};
    }

    // 2-dimantional

    constexpr auto add_impl(const auto &v1, const auto &v2, priority<2>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v2.x(), v2.y(), v1)>
    {
        return {v1.x() + v2.x(), v1.y() + v2.y()};
    }

    // 1-dimantional

    constexpr auto add_impl(const auto &v1, const auto &v2, priority<1>)
        -> std::decay_t<decltype(v1.x(), v2.x(), v1)>
    {
        return {v1.x() + v2.x()};
    }

    // N-dimantional

    constexpr auto operator+(const auto &v1, const auto &v2)
        -> std::decay_t<decltype(v1.x(), v2.x(), v1)>
    {
        return add_impl(v1, v2, priority<4>{});
    }

    // ----------------------------------------------------------------
    // vector + scalar

    // 4-dimantional

    constexpr auto add_impl(const auto &v, auto s, priority<4>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v.w(), v)>
    {
        return v + std::decay_t<decltype(v)>{s, s, s, s};
    }

    // 3-dimantional

    constexpr auto add_impl(const auto &v, auto s, priority<3>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v)>
    {
        return v + std::decay_t<decltype(v)>{s, s, s};
    }

    // 2-dimantional

    constexpr auto add_impl(const auto &v, auto s, priority<2>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v)>
    {
        return v + std::decay_t<decltype(v)>{s, s};
    }

    // 1-dimantional

    constexpr auto add_impl(const auto &v, auto s, priority<1>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return v + std::decay_t<decltype(v)>{s};
    }

    // N-dimantional

    constexpr auto operator+(const auto &v, auto s)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return add_impl(v, s, priority<4>{});
    }

    // ----------------------------------------------------------------
    // scalar + vector

    // 4-dimantional

    constexpr auto add_impl(auto s, const auto &v, priority<4>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v.w(), v)>
    {
        return std::decay_t<decltype(v)>{s, s, s, s} + v;
    }

    // 3-dimantional

    constexpr auto add_impl(auto s, const auto &v, priority<3>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v)>
    {
        return std::decay_t<decltype(v)>{s, s, s} + v;
    }

    // 2-dimantional

    constexpr auto add_impl(auto s, const auto &v, priority<2>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v)>
    {
        return std::decay_t<decltype(v)>{s, s} + v;
    }

    // 1-dimantional

    constexpr auto add_impl(auto s, const auto &v, priority<1>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return std::decay_t<decltype(v)>{s} + v;
    }

    // N-dimantional

    constexpr auto operator+(auto s, const auto &v)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return add_impl(s, v, priority<4>{});
    }

    // ----------------------------------------------------------------
    // + vector

    constexpr auto operator+(const auto &v)
        -> std::decay_t<decltype(v.x(), v)>
    {
        return 0 + v;
    }

    // ================================================================
    // subtraction

    // ----------------------------------------------------------------
    // vector - vector

    // 4-dimantional

    constexpr auto sub_impl(const auto &v1, const auto &v2, priority<4>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v1.z(), v1.w(), v2.x(), v2.y(), v2.z(), v2.w(), v1)>
    {
        return {v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z(), v1.w() - v2.w()};
    }

    // 3-dimantional

    constexpr auto sub_impl(const auto &v1, const auto &v2, priority<3>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v1.z(), v2.x(), v2.y(), v2.z(), v1)>
    {
        return {v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z()};
    }

    // 2-dimantional

    constexpr auto sub_impl(const auto &v1, const auto &v2, priority<2>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v2.x(), v2.y(), v1)>
    {
        return {v1.x() - v2.x(), v1.y() - v2.y()};
    }

    // 1-dimantional

    constexpr auto sub_impl(const auto &v1, const auto &v2, priority<1>)
        -> std::decay_t<decltype(v1.x(), v2.x(), v1)>
    {
        return {v1.x() - v2.x()};
    }

    // N-dimantional

    constexpr auto operator-(const auto &v1, const auto &v2)
        -> std::decay_t<decltype(v1.x(), v2.x(), v1)>
    {
        return sub_impl(v1, v2, priority<4>{});
    }

    // ----------------------------------------------------------------
    // vector - scalar

    // 4-dimantional

    constexpr auto sub_impl(const auto &v, auto s, priority<4>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v.w(), v)>
    {
        return v - std::decay_t<decltype(v)>{s, s, s, s};
    }

    // 3-dimantional

    constexpr auto sub_impl(const auto &v, auto s, priority<3>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v)>
    {
        return v - std::decay_t<decltype(v)>{s, s, s};
    }

    // 2-dimantional

    constexpr auto sub_impl(const auto &v, auto s, priority<2>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v)>
    {
        return v - std::decay_t<decltype(v)>{s, s};
    }

    // 1-dimantional

    constexpr auto sub_impl(const auto &v, auto s, priority<1>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return v - std::decay_t<decltype(v)>{s};
    }

    // N-dimantional

    constexpr auto operator-(const auto &v, auto s)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return sub_impl(v, s, priority<4>{});
    }

    // ----------------------------------------------------------------
    // scalar - vector

    // 4-dimantional

    constexpr auto sub_impl(auto s, const auto &v, priority<4>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v.w(), v)>
    {
        return std::decay_t<decltype(v)>{s, s, s, s} - v;
    }

    // 3-dimantional

    constexpr auto sub_impl(auto s, const auto &v, priority<3>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v)>
    {
        return std::decay_t<decltype(v)>{s, s, s} - v;
    }

    // 2-dimantional

    constexpr auto sub_impl(auto s, const auto &v, priority<2>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v)>
    {
        return std::decay_t<decltype(v)>{s, s} - v;
    }

    // 1-dimantional

    constexpr auto sub_impl(auto s, const auto &v, priority<1>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return std::decay_t<decltype(v)>{s} - v;
    }

    // N-dimantional

    constexpr auto operator-(auto s, const auto &v)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return sub_impl(s, v, priority<4>{});
    }

    // ----------------------------------------------------------------
    // - vector

    constexpr auto operator-(const auto &v)
        -> std::decay_t<decltype(v.x(), v)>
    {
        return 0 - v;
    }

    // ================================================================
    // multiplication

    // ----------------------------------------------------------------
    // vector * vector

    // 4-dimantional

    constexpr auto mul_impl(const auto &v1, const auto &v2, priority<4>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v1.z(), v1.w(), v2.x(), v2.y(), v2.z(), v2.w(), v1)>
    {
        return {v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z(), v1.w() * v2.w()};
    }

    // 3-dimantional

    constexpr auto mul_impl(const auto &v1, const auto &v2, priority<3>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v1.z(), v2.x(), v2.y(), v2.z(), v1)>
    {
        return {v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z()};
    }

    // 2-dimantional

    constexpr auto mul_impl(const auto &v1, const auto &v2, priority<2>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v2.x(), v2.y(), v1)>
    {
        return {v1.x() * v2.x(), v1.y() * v2.y()};
    }

    // 1-dimantional

    constexpr auto mul_impl(const auto &v1, const auto &v2, priority<1>)
        -> std::decay_t<decltype(v1.x(), v2.x(), v1)>
    {
        return {v1.x() * v2.x()};
    }

    // N-dimantional

    constexpr auto operator*(const auto &v1, const auto &v2)
        -> std::decay_t<decltype(v1.x(), v2.x(), v1)>
    {
        return mul_impl(v1, v2, priority<4>{});
    }

    // ----------------------------------------------------------------
    // vector * scalar

    // 4-dimantional

    constexpr auto mul_impl(const auto &v, auto s, priority<4>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v.w(), v)>
    {
        return v * std::decay_t<decltype(v)>{s, s, s, s};
    }

    // 3-dimantional

    constexpr auto mul_impl(const auto &v, auto s, priority<3>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v)>
    {
        return v * std::decay_t<decltype(v)>{s, s, s};
    }

    // 2-dimantional

    constexpr auto mul_impl(const auto &v, auto s, priority<2>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v)>
    {
        return v * std::decay_t<decltype(v)>{s, s};
    }

    // 1-dimantional

    constexpr auto mul_impl(const auto &v, auto s, priority<1>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return v * std::decay_t<decltype(v)>{s};
    }

    // N-dimantional

    constexpr auto operator*(const auto &v, auto s)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return mul_impl(v, s, priority<4>{});
    }

    // ----------------------------------------------------------------
    // scalar * vector

    // 4-dimantional

    constexpr auto mul_impl(auto s, const auto &v, priority<4>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v.w(), v)>
    {
        return std::decay_t<decltype(v)>{s, s, s, s} * v;
    }

    // 3-dimantional

    constexpr auto mul_impl(auto s, const auto &v, priority<3>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v)>
    {
        return std::decay_t<decltype(v)>{s, s, s} * v;
    }

    // 2-dimantional

    constexpr auto mul_impl(auto s, const auto &v, priority<2>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v)>
    {
        return std::decay_t<decltype(v)>{s, s} * v;
    }

    // 1-dimantional

    constexpr auto mul_impl(auto s, const auto &v, priority<1>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return std::decay_t<decltype(v)>{s} * v;
    }

    // N-dimantional

    constexpr auto operator*(auto s, const auto &v)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return mul_impl(s, v, priority<4>{});
    }

    // ================================================================
    // division

    // ----------------------------------------------------------------
    // vector / vector

    // 4-dimantional

    constexpr auto div_impl(const auto &v1, const auto &v2, priority<4>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v1.z(), v1.w(), v2.x(), v2.y(), v2.z(), v2.w(), v1)>
    {
        return {v1.x() / v2.x(), v1.y() / v2.y(), v1.z() / v2.z(), v1.w() / v2.w()};
    }

    // 3-dimantional

    constexpr auto div_impl(const auto &v1, const auto &v2, priority<3>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v1.z(), v2.x(), v2.y(), v2.z(), v1)>
    {
        return {v1.x() / v2.x(), v1.y() / v2.y(), v1.z() / v2.z()};
    }

    // 2-dimantional

    constexpr auto div_impl(const auto &v1, const auto &v2, priority<2>)
        -> std::decay_t<decltype(v1.x(), v1.y(), v2.x(), v2.y(), v1)>
    {
        return {v1.x() / v2.x(), v1.y() / v2.y()};
    }

    // 1-dimantional

    constexpr auto div_impl(const auto &v1, const auto &v2, priority<1>)
        -> std::decay_t<decltype(v1.x(), v2.x(), v1)>
    {
        return {v1.x() / v2.x()};
    }

    // N-dimantional

    constexpr auto operator/(const auto &v1, const auto &v2)
        -> std::decay_t<decltype(v1.x(), v2.x(), v1)>
    {
        return div_impl(v1, v2, priority<4>{});
    }

    // ----------------------------------------------------------------
    // vector / scalar

    // 4-dimantional

    constexpr auto div_impl(const auto &v, auto s, priority<4>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v.w(), v)>
    {
        return v / std::decay_t<decltype(v)>{s, s, s, s};
    }

    // 3-dimantional

    constexpr auto div_impl(const auto &v, auto s, priority<3>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v)>
    {
        return v / std::decay_t<decltype(v)>{s, s, s};
    }

    // 2-dimantional

    constexpr auto div_impl(const auto &v, auto s, priority<2>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v)>
    {
        return v / std::decay_t<decltype(v)>{s, s};
    }

    // 1-dimantional

    constexpr auto div_impl(const auto &v, auto s, priority<1>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return v / std::decay_t<decltype(v)>{s};
    }

    // N-dimantional

    constexpr auto operator/(const auto &v, auto s)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return div_impl(v, s, priority<4>{});
    }

    // ----------------------------------------------------------------
    // scalar / vector

    // 4-dimantional

    constexpr auto div_impl(auto s, const auto &v, priority<4>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v.w(), v)>
    {
        return std::decay_t<decltype(v)>{s, s, s, s} / v;
    }

    // 3-dimantional

    constexpr auto div_impl(auto s, const auto &v, priority<3>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v.z(), v)>
    {
        return std::decay_t<decltype(v)>{s, s, s} / v;
    }

    // 2-dimantional

    constexpr auto div_impl(auto s, const auto &v, priority<2>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v.y(), v)>
    {
        return std::decay_t<decltype(v)>{s, s} / v;
    }

    // 1-dimantional

    constexpr auto div_impl(auto s, const auto &v, priority<1>)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return std::decay_t<decltype(v)>{s} / v;
    }

    // N-dimantional

    constexpr auto operator/(auto s, const auto &v)
        -> std::decay_t<decltype(std::declval<std::enable_if_t<std::is_arithmetic_v<std::decay_t<decltype(s)>>, std::nullptr_t>>(), v.x(), v)>
    {
        return div_impl(s, v, priority<4>{});
    }

    // ================================================================
    // dot

    constexpr auto dot(const auto &v1, const auto &v2)
        -> std::decay_t<decltype(v1.x())>
    {
        return rendex::math::sum(v1 * v2);
    }

    // ================================================================
    // norm

    constexpr auto norm(const auto &v)
        -> std::decay_t<decltype(v.x())>
    {
        return rendex::math::sqrt(dot(v, v));
    }
}
