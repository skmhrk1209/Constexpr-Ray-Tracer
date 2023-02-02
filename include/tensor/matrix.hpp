#pragma once

#include <tuple>
#include <type_traits>

#include "tensor.hpp"

namespace rendex::tensor {

// ================================================================
// lazy evaluation

template <typename Matrix1, typename Matrix2>
struct LazyMatMul : std::tuple<Matrix1, Matrix2> {
    constexpr auto operator%(auto vector) const { return std::get<0>(*this) % (std::get<1>(*this) % vector); }
};

// ================================================================
// multiplication

constexpr auto operator%(const auto &matrix1, const auto &matrix2)
    requires(MatrixShaped<std::decay_t<decltype(matrix1)>> && MatrixShaped<std::decay_t<decltype(matrix2)>> &&
             dimension_v<std::decay_t<decltype(matrix1)>, 1> == dimension_v<std::decay_t<decltype(matrix2)>, 0>)
{
    return LazyMatMul<decltype(matrix1), decltype(matrix2)>(std::tie(matrix1, matrix2));
}

constexpr auto operator%(const auto &matrix, const auto &vector)
    requires(MatrixShaped<std::decay_t<decltype(matrix)>> && VectorShaped<std::decay_t<decltype(vector)>> &&
             dimension_v<std::decay_t<decltype(matrix)>, 0> == dimension_v<std::decay_t<decltype(vector)>, 0> &&
             dimension_v<std::decay_t<decltype(matrix)>, 1> == dimension_v<std::decay_t<decltype(vector)>, 0>)
{
    return [&]<auto... Is>(std::index_sequence<Is...>) {
        return std::decay_t<decltype(vector)>{dot(matrix[Is], vector)...};
    }
    (std::make_index_sequence<dimension_v<std::decay_t<decltype(matrix)>, 0>>{});
}

}  // namespace rendex::tensor
