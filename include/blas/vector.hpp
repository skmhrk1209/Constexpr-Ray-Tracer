#pragma once

#include <array>
#include <tuple>
#include <numeric>
#include <type_traits>
#include "common.hpp"
#include "../math.hpp"
#include "../common.hpp"

namespace rendex::blas
{
    // ================================================================
    // class

    template <typename Scalar, auto Dimension>
    class Vector : public std::array<Scalar, Dimension>
    {
    };

    template <typename Scalar>
    struct Vector<Scalar, 1> : public std::array<Scalar, 1>
    {
        constexpr auto &x() { return element<0>(*this); }
        constexpr const auto &x() const { return element<0>(*this); }
    };

    template <typename Scalar>
    struct Vector<Scalar, 2> : public std::array<Scalar, 2>
    {
        constexpr auto &x() { return element<0>(*this); }
        constexpr const auto &x() const { return element<0>(*this); }

        constexpr auto &y() { return element<1>(*this); }
        constexpr const auto &y() const { return element<1>(*this); }
    };

    template <typename Scalar>
    struct Vector<Scalar, 3> : public std::array<Scalar, 3>
    {
        constexpr auto &x() { return element<0>(*this); }
        constexpr const auto &x() const { return element<0>(*this); }

        constexpr auto &y() { return element<1>(*this); }
        constexpr const auto &y() const { return element<1>(*this); }

        constexpr auto &z() { return element<2>(*this); }
        constexpr const auto &z() const { return element<2>(*this); }
    };

    template <typename Scalar>
    struct Vector<Scalar, 4> : public std::array<Scalar, 4>
    {
        constexpr auto &x() { return element<0>(*this); }
        constexpr const auto &x() const { return element<0>(*this); }

        constexpr auto &y() { return element<1>(*this); }
        constexpr const auto &y() const { return element<1>(*this); }

        constexpr auto &z() { return element<2>(*this); }
        constexpr const auto &z() const { return element<2>(*this); }

        constexpr auto &w() { return element<3>(*this); }
        constexpr const auto &w() const { return element<3>(*this); }
    };

    // ================================================================
    // interface

    template <typename Type, auto Dimension>
    struct dimension<Vector<Type, Dimension>> : std::integral_constant<std::size_t, Dimension>
    {
    };

    template <auto Index, typename Type, auto Dimension>
    constexpr decltype(auto) element(Vector<Type, Dimension> &vector)
    {
        return vector[Index];
    };

    template <auto Index, typename Type, auto Dimension>
    constexpr decltype(auto) element(const Vector<Type, Dimension> &vector)
    {
        return vector[Index];
    };

    // ================================================================
    // addition

    constexpr auto operator+(const auto &vector1, const auto &vector2)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector1)>> &&
            FiniteDimensional<std::decay_t<decltype(vector2)>> &&
            HaveSameDimension<std::decay_t<decltype(vector1)>, std::decay_t<decltype(vector2)>>)
    {
        std::decay_t<decltype(vector1)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector1)>>)
            {
                element<Index>(output) = element<Index>(vector1) + element<Index>(vector2);
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator+(const auto &vector, auto scalar)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector)>> &&
            NonDimensional<std::decay_t<decltype(scalar)>>)
    {
        std::decay_t<decltype(vector)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector)>>)
            {
                element<Index>(output) = element<Index>(vector) + scalar;
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator+(auto scalar, const auto &vector)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector)>> &&
            NonDimensional<std::decay_t<decltype(scalar)>>)
    {
        std::decay_t<decltype(vector)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector)>>)
            {
                element<Index>(output) = scalar + element<Index>(vector);
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator+(const auto &vector)
    {
        return 0 + vector;
    }

    // ================================================================
    // subtraction

    constexpr auto operator-(const auto &vector1, const auto &vector2)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector1)>> &&
            FiniteDimensional<std::decay_t<decltype(vector2)>> &&
            HaveSameDimension<std::decay_t<decltype(vector1)>, std::decay_t<decltype(vector2)>>)
    {
        std::decay_t<decltype(vector1)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector1)>>)
            {
                element<Index>(output) = element<Index>(vector1) - element<Index>(vector2);
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator-(const auto &vector, auto scalar)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector)>> &&
            NonDimensional<std::decay_t<decltype(scalar)>>)
    {
        std::decay_t<decltype(vector)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector)>>)
            {
                element<Index>(output) = element<Index>(vector) - scalar;
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator-(auto scalar, const auto &vector)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector)>> &&
            NonDimensional<std::decay_t<decltype(scalar)>>)
    {
        std::decay_t<decltype(vector)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector)>>)
            {
                element<Index>(output) = scalar - element<Index>(vector);
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator-(const auto &vector)
    {
        return 0 - vector;
    }

    // ================================================================
    // multiplication

    constexpr auto operator*(const auto &vector1, const auto &vector2)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector1)>> &&
            FiniteDimensional<std::decay_t<decltype(vector2)>> &&
            HaveSameDimension<std::decay_t<decltype(vector1)>, std::decay_t<decltype(vector2)>>)
    {
        std::decay_t<decltype(vector1)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector1)>>)
            {
                element<Index>(output) = element<Index>(vector1) * element<Index>(vector2);
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator*(const auto &vector, auto scalar)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector)>> &&
            NonDimensional<std::decay_t<decltype(scalar)>>)
    {
        std::decay_t<decltype(vector)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector)>>)
            {
                element<Index>(output) = element<Index>(vector) * scalar;
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator*(auto scalar, const auto &vector)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector)>> &&
            NonDimensional<std::decay_t<decltype(scalar)>>)
    {
        std::decay_t<decltype(vector)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector)>>)
            {
                element<Index>(output) = scalar * element<Index>(vector);
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    // ================================================================
    // division

    constexpr auto operator/(const auto &vector1, const auto &vector2)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector1)>> &&
            FiniteDimensional<std::decay_t<decltype(vector2)>> &&
            HaveSameDimension<std::decay_t<decltype(vector1)>, std::decay_t<decltype(vector2)>>)
    {
        std::decay_t<decltype(vector1)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector1)>>)
            {
                element<Index>(output) = element<Index>(vector1) / element<Index>(vector2);
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator/(const auto &vector, auto scalar)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector)>> &&
            NonDimensional<std::decay_t<decltype(scalar)>>)
    {
        std::decay_t<decltype(vector)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector)>>)
            {
                element<Index>(output) = element<Index>(vector) / scalar;
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    constexpr auto operator/(auto scalar, const auto &vector)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector)>> &&
            NonDimensional<std::decay_t<decltype(scalar)>>)
    {
        std::decay_t<decltype(vector)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(vector)>>)
            {
                element<Index>(output) = scalar / element<Index>(vector);
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }

    // ================================================================
    // dot

    constexpr auto dot(const auto &vector1, const auto &vector2)
        requires(
            FiniteDimensional<std::decay_t<decltype(vector1)>> &&
            FiniteDimensional<std::decay_t<decltype(vector2)>> &&
            HaveSameDimension<std::decay_t<decltype(vector1)>, std::decay_t<decltype(vector2)>>)
    {
        return sum(vector1 * vector2);
    }

    constexpr auto sum(const auto &vector)
        requires FiniteDimensional<std::decay_t<decltype(vector)>>
    {
        return std::accumulate(std::begin(vector), std::end(vector), static_cast<std::decay_t<decltype(element<0>(vector))>>(0));
    }

    // ================================================================
    // norm

    constexpr auto norm(const auto &vector)
        requires FiniteDimensional<std::decay_t<decltype(vector)>>
    {
        return rendex::math::sqrt(dot(vector, vector));
    }
}
