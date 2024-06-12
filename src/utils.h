#pragma once

#include <stdexcept>

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
}