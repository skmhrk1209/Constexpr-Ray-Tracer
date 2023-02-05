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

template <typename Matrix1, typename Matrix2>
constexpr auto operator%(const Matrix1 &matrix_1, const Matrix2 &matrix_2)
    requires MatrixShaped<Matrix1> && MatrixShaped<Matrix2> && (dimension_v<Matrix1, 1> == dimension_v<Matrix2, 0>)
{
    return LazyMatMul<Matrix1, Matrix2>(std::tie(matrix_1, matrix_2));
}

template <typename Matrix, typename Vector>
constexpr auto operator%(const Matrix &matrix, const Vector &vector)
    requires MatrixShaped<Matrix> && VectorShaped<Vector> && (dimension_v<Matrix, 0> == dimension_v<Vector, 0>) &&
             (dimension_v<Matrix, 1> == dimension_v<Vector, 0>)
{
    return [&]<auto... Is>(std::index_sequence<Is...>) { return Vector{dot(matrix[Is], vector)...}; }
    (std::make_index_sequence<dimension_v<std::decay_t<decltype(matrix)>, 0>>{});
}

}  // namespace rendex::tensor
