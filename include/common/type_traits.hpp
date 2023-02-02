#pragma once

#include <optional>
#include <type_traits>

namespace rendex {

template <typename, template <typename...> typename, typename...>
struct is_detected_impl : std::false_type {};

template <template <typename...> typename Op, typename... Ts>
struct is_detected_impl<std::void_t<Op<Ts...>>, Op, Ts...> : std::true_type {};

template <template <typename...> typename Op, typename... Ts>
using is_detected = is_detected_impl<std::void_t<>, Op, Ts...>;

template <template <typename...> typename Op, typename... Ts>
constexpr auto is_detected_v = is_detected<Op, Ts...>::value;

template <typename>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
constexpr auto is_optional_v = is_optional<T>::value;

}  // namespace rendex
