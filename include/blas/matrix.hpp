#pragma once

#include <tuple>
#include <type_traits>
#include "tensor.hpp"

namespace rendex::blas
{
    // ================================================================
    // lazy evaluation

    template <typename Matrix1, typename Matrix2>
    struct LazyMatMul : std::tuple<Matrix1, Matrix2>
    {
        constexpr auto operator%(auto vector) const
        {
            return std::get<0>(*this) % (std::get<1>(*this) % vector);
        }
    };

    // ================================================================
    // multiplication

    constexpr auto operator%(const auto &matrix1, const auto &matrix2)
        requires(
            MatrixShaped<std::decay_t<decltype(matrix1)>> &&
            MatrixShaped<std::decay_t<decltype(matrix2)>> &&
            dimension_v<std::decay_t<decltype(matrix1)>, 1> == dimension_v<std::decay_t<decltype(matrix2)>, 0>)
    {
        return LazyMatMul<decltype(matrix1), decltype(matrix2)>(std::tie(matrix1, matrix2));
    }

    constexpr auto operator%(const auto &matrix, const auto &vector)
        requires(
            MatrixShaped<std::decay_t<decltype(matrix)>> &&
            VectorShaped<std::decay_t<decltype(vector)>> &&
            dimension_v<std::decay_t<decltype(matrix)>, 0> == dimension_v<std::decay_t<decltype(vector)>, 0> &&
            dimension_v<std::decay_t<decltype(matrix)>, 1> == dimension_v<std::decay_t<decltype(vector)>, 0>)
    {
        std::decay_t<decltype(vector)> output;
        [function = [&]<auto Index>(auto function, std::integral_constant<std::size_t, Index>) {
            if constexpr (Index < dimension_v<std::decay_t<decltype(matrix)>, 0>)
            {
                output[Index] = dot(matrix[Index], vector);
                function(function, std::integral_constant<std::size_t, Index + 1>{});
            }
        }](auto &&...args)
        {
            return function(function, std::forward<decltype(args)>(args)...);
        }(std::integral_constant<std::size_t, 0>{});
        return output;
    }
}
