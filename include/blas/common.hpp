#pragma once

#include <array>
#include <tuple>
#include <type_traits>
#include "../common.hpp"

namespace rendex::blas
{
    // ================================================================
    // traits

    template <typename Type>
    struct dimension : std::integral_constant<std::size_t, 0>
    {
    };

    template <typename... Types>
    struct dimension<std::tuple<Types...>> : std::tuple_size<std::tuple<Types...>>
    {
    };

    template <typename Type, auto Dimension>
    struct dimension<std::array<Type, Dimension>> : std::tuple_size<std::array<Type, Dimension>>
    {
    };

    template <typename Type>
    inline constexpr auto dimension_v = dimension<Type>::value;

    // ================================================================
    // interfaces

    template <auto Index, typename... Types>
    constexpr decltype(auto) element(std::tuple<Types...> &tuple)
    {
        return std::get<Index>(tuple);
    };

    template <auto Index, typename... Types>
    constexpr decltype(auto) element(const std::tuple<Types...> &tuple)
    {
        return std::get<Index>(tuple);
    };

    template <auto Index, typename Type, auto Dimension>
    constexpr decltype(auto) element(std::array<Type, Dimension> &array)
    {
        return std::get<Index>(array);
    };

    template <auto Index, typename Type, auto Dimension>
    constexpr decltype(auto) element(const std::array<Type, Dimension> &array)
    {
        return std::get<Index>(array);
    };

    // ================================================================
    // concepts

    template <typename Type>
    concept NonDimensional = std::is_arithmetic_v<Type>;

    template <typename Type>
    concept FiniteDimensional = dimension_v<Type> > 0;

    template <typename Type1,typename Type2>
    concept HaveSameDimension = dimension_v<Type1> == dimension_v<Type2>;
}
