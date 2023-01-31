#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <numeric>
#include <type_traits>
#include "math.hpp"
#include "common.hpp"

namespace rendex::blas
{
    // ================================================================
    // class

    template <typename T>
    using IndexedType = decltype(std::declval<T>()[std::declval<std::size_t>()]);

    template <template <typename, auto> typename Array, typename T, auto N, auto... Ns>
    struct GenericTensorImpl : Array<GenericTensorImpl<Array, T, Ns...>, N>
    {
    };

    template <template <typename, auto> typename Array, typename T, auto N>
        requires rendex::is_detected_v<IndexedType, Array<T, N>>
    struct GenericTensorImpl<Array, T, N> : Array<T, N>
    {
    };

    template <template <typename, auto> typename Array, typename T, auto... Ns>
    using GenericTensor = GenericTensorImpl<Array, T, Ns...>;

    // ----------------------------------------------------------------

    template <typename T, auto... Ns>
    using Tensor = GenericTensor<std::array, T, Ns...>;

    template <typename T, auto N>
    using Vector = Tensor<T, N>;

    template <typename T, auto M, auto N>
    using Matrix = Tensor<T, M, N>;

    // ----------------------------------------------------------------

    template <typename T, auto N>
    struct vector : std::vector<T>
    {
        using std::vector<T>::vector;
        constexpr vector() : vector(N) {}
        constexpr vector(const std::allocator<T> &allocator) : vector(N, allocator) {}
    };

    template <typename T, auto... Ns>
    using DynamicTensor = GenericTensor<vector, T, Ns...>;

    // ================================================================
    // rank

    template <typename T>
    struct rank : std::integral_constant<std::size_t, 0>
    {
    };

    template <template <typename, auto> typename Array, typename T, auto... Ns>
    struct rank<GenericTensor<Array, T, Ns...>> : std::integral_constant<std::size_t, sizeof...(Ns)>
    {
    };

    template <typename T>
    inline constexpr auto rank_v = rank<T>::value;

    // ================================================================
    // dimension

    template <typename T, auto I>
    struct dimension : std::integral_constant<std::size_t, 0>
    {
    };

    template <template <typename, auto> typename Array, typename T, auto... Ns, auto I>
    struct dimension<GenericTensor<Array, T, Ns...>, I> : std::integral_constant<std::size_t, std::get<I>(std::make_tuple(Ns...))>
    {
    };

    template <typename T, auto I>
    inline constexpr auto dimension_v = dimension<T, I>::value;

    // ================================================================
    // concept

    template <typename T>
    concept ScalarShaped = (rank_v<T> == 0);

    template <typename T>
    concept VectorShaped = (rank_v<T> == 1);

    template <typename T>
    concept MatrixShaped = (rank_v<T> == 2);

    template <typename T>
    concept TensorShaped = (rank_v<T> > 0);

    template <typename T, typename U>
    concept Broadcastable = (dimension_v<T, 0> == dimension_v<U, 0>);

    // ================================================================
    // addition

    constexpr auto operator+(const auto &tensor1, const auto &tensor2)
        requires(
            TensorShaped<std::decay_t<decltype(tensor1)>> &&
            TensorShaped<std::decay_t<decltype(tensor2)>> &&
            Broadcastable<std::decay_t<decltype(tensor1)>, std::decay_t<decltype(tensor2)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor1)>{(tensor1[Is] + tensor2[Is])...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor1)>, 0>>{});
    }

    constexpr auto operator+(const auto &tensor, auto scalar)
        requires(
            TensorShaped<std::decay_t<decltype(tensor)>> &&
            ScalarShaped<std::decay_t<decltype(scalar)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor)>{(tensor[Is] + scalar)...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor)>, 0>>{});
    }

    constexpr auto operator+(auto scalar, const auto &tensor)
        requires(
            TensorShaped<std::decay_t<decltype(tensor)>> &&
            ScalarShaped<std::decay_t<decltype(scalar)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor)>{(scalar + tensor[Is])...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor)>, 0>>{});
    }

    constexpr auto operator+(const auto &tensor)
    {
        return 0 + tensor;
    }

    // ================================================================
    // subtraction

    constexpr auto operator-(const auto &tensor1, const auto &tensor2)
        requires(
            TensorShaped<std::decay_t<decltype(tensor1)>> &&
            TensorShaped<std::decay_t<decltype(tensor2)>> &&
            Broadcastable<std::decay_t<decltype(tensor1)>, std::decay_t<decltype(tensor2)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor1)>{(tensor1[Is] - tensor2[Is])...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor1)>, 0>>{});
    }

    constexpr auto operator-(const auto &tensor, auto scalar)
        requires(
            TensorShaped<std::decay_t<decltype(tensor)>> &&
            ScalarShaped<std::decay_t<decltype(scalar)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor)>{(tensor[Is] - scalar)...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor)>, 0>>{});
    }

    constexpr auto operator-(auto scalar, const auto &tensor)
        requires(
            TensorShaped<std::decay_t<decltype(tensor)>> &&
            ScalarShaped<std::decay_t<decltype(scalar)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor)>{(scalar - tensor[Is])...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor)>, 0>>{});
    }

    constexpr auto operator-(const auto &tensor)
    {
        return 0 - tensor;
    }

    // ================================================================
    // multiplication

    constexpr auto operator*(const auto &tensor1, const auto &tensor2)
        requires(
            TensorShaped<std::decay_t<decltype(tensor1)>> &&
            TensorShaped<std::decay_t<decltype(tensor2)>> &&
            Broadcastable<std::decay_t<decltype(tensor1)>, std::decay_t<decltype(tensor2)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor1)>{(tensor1[Is] * tensor2[Is])...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor1)>, 0>>{});
    }

    constexpr auto operator*(const auto &tensor, auto scalar)
        requires(
            TensorShaped<std::decay_t<decltype(tensor)>> &&
            ScalarShaped<std::decay_t<decltype(scalar)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor)>{(tensor[Is] * scalar)...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor)>, 0>>{});
    }

    constexpr auto operator*(auto scalar, const auto &tensor)
        requires(
            TensorShaped<std::decay_t<decltype(tensor)>> &&
            ScalarShaped<std::decay_t<decltype(scalar)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor)>{(scalar * tensor[Is])...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor)>, 0>>{});
    }

    // ================================================================
    // division

    constexpr auto operator/(const auto &tensor1, const auto &tensor2)
        requires(
            TensorShaped<std::decay_t<decltype(tensor1)>> &&
            TensorShaped<std::decay_t<decltype(tensor2)>> &&
            Broadcastable<std::decay_t<decltype(tensor1)>, std::decay_t<decltype(tensor2)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor1)>{(tensor1[Is] / tensor2[Is])...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor1)>, 0>>{});
    }

    constexpr auto operator/(const auto &tensor, auto scalar)
        requires(
            TensorShaped<std::decay_t<decltype(tensor)>> &&
            ScalarShaped<std::decay_t<decltype(scalar)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor)>{(tensor[Is] / scalar)...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor)>, 0>>{});
    }

    constexpr auto operator/(auto scalar, const auto &tensor)
        requires(
            TensorShaped<std::decay_t<decltype(tensor)>> &&
            ScalarShaped<std::decay_t<decltype(scalar)>>)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return std::decay_t<decltype(tensor)>{(scalar / tensor[Is])...};
        }
        (std::make_index_sequence<dimension_v<std::decay_t<decltype(tensor)>, 0>>{});
    }

    // ================================================================
    // dot

    constexpr auto sum(const auto &tensor)
    {
        return std::accumulate(std::begin(tensor), std::end(tensor), std::decay_t<decltype(tensor[0])>{});
    }

    constexpr auto dot(const auto &tensor1, const auto &tensor2)
    {
        return sum(tensor1 * tensor2);
    }

    // ================================================================
    // norm

    constexpr auto norm(const auto &tensor)
    {
        return rendex::math::sqrt(dot(tensor, tensor));
    }

    constexpr auto normalized(const auto &tensor)
    {
        return tensor / norm(tensor);
    }
}
