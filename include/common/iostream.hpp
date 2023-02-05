#pragma once

#include <iostream>
#include <tuple>
#include <variant>

#include "algorithm.hpp"

template <typename T>
concept Iterable = requires(T x) { std::begin(x), std::end(x); };

template <typename... Ts>
decltype(auto) operator>>(std::istream &istream, std::tuple<Ts...> &tuple) {
    for_each(tuple, [&istream](auto &element) { istream >> element; });
    return istream;
}

template <typename... Ts>
decltype(auto) operator<<(std::ostream &ostream, const std::tuple<Ts...> &tuple) {
    ostream << "( ";
    for_each(tuple, [&ostream](const auto &element) { ostream << element << " "; });
    ostream << ")";
    return ostream;
}

template <typename T, typename U>
decltype(auto) operator>>(std::istream &istream, std::pair<T, U> &pair) {
    istream >> pair.first >> pair.second;
    return istream;
}

template <typename T, typename U>
decltype(auto) operator<<(std::ostream &ostream, const std::pair<T, U> &pair) {
    ostream << "( " << pair.first << " " << pair.second << " )";
    return ostream;
}

template <typename T, typename... Ts>
decltype(auto) operator>>(std::istream &istream, std::variant<T, Ts...> &variant) {
    std::visit([&istream](auto &value) { istream >> value; }, variant);
    return istream;
}

template <typename T, typename... Ts>
decltype(auto) operator<<(std::ostream &ostream, const std::variant<T, Ts...> &variant) {
    std::visit([&ostream](const auto &value) { ostream << value; }, variant);
    return ostream;
}

decltype(auto) operator>>(std::istream &istream, Iterable auto &container) {
    for (auto &element : container) {
        istream >> element;
    }
    return istream;
}

decltype(auto) operator<<(std::ostream &ostream, const Iterable auto &container) {
    ostream << "[ ";
    for (const auto &element : container) {
        ostream << element << " ";
    }
    ostream << "]";
    return ostream;
}
