#pragma once

#include <functional>

namespace rendex {

template <typename... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};

// additional deduction guide (CTAD)
template <typename... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

// currying & partial application
constexpr decltype(auto) curry(auto &&function) {
    return [function = std::forward<decltype(function)>(function)](auto &&...args1) constexpr {
        if constexpr (std::is_invocable_v<decltype(function), decltype(args1)...>) {
            return std::invoke(function, std::forward<decltype(args1)>(args1)...);
        } else {
            return curry([function = std::move(function),
                          ... args1 = std::forward<decltype(args1)>(args1)](auto &&...args2) constexpr {
                return std::invoke(function, args1..., std::forward<decltype(args2)>(args2)...);
            });
        }
    };
}

// fixed point combinator for anonymous recursive function
constexpr decltype(auto) fix(auto &&function) { return curry(function)(function); }

}  // namespace rendex
