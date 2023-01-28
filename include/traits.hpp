#pragma once

#include <type_traits>

namespace rendex::traits
{
    template <typename, template <typename...> typename, typename...>
    struct is_detected_impl : std::false_type
    {
    };

    template <template <typename...> typename Check, typename... Types>
    struct is_detected_impl<std::void_t<Check<Types...>>, Check, Types...> : std::true_type
    {
    };

    template <template <typename...> typename Check, typename... Types>
    using is_detected = is_detected_impl<std::void_t<>, Check, Types...>;
}
