#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "meta.h"

namespace micro_nn {
template <class T, class U>
constexpr T narrow_cast(U&& u) noexcept {
    return static_cast<T>(std::forward<U>(u));
}

template <class T, class U>
constexpr T narrow(U&& u) {
    T t = static_cast<T>(std::forward<U>(u));
    if (static_cast<U>(t) != u) {
        throw std::runtime_error("Narrowing conversion out of range.");
    }
    return t;
}

template <class T>
    requires std::is_arithmetic_v<std::remove_reference_t<T>>
constexpr T abs(T t) {
    if constexpr (meta::is_constexpr([&] { return std::abs(T{}); })) {
        return std::abs(t);
    } else {
        return t < T{} ? -t : t;
    }
}
}  // namespace micro_nn