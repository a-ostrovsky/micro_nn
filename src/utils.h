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

constexpr const float kSqrtEpsilon{1e-4f};

template <class T>
    requires std::is_arithmetic_v<std::remove_reference_t<T>>
constexpr T sqrt(T t) {
    if constexpr (meta::is_constexpr([&] { return std::sqrt(T{}); })) {
        return std::sqrt(t);
    } else {
        if (t < T{}) {
            return std::numeric_limits<T>::quiet_NaN();
        }
        T x{t / 2};
        // Tolerance for convergence with an empirical value.
        const auto epsilon{narrow_cast<T>(kSqrtEpsilon)};
        T prev_x{};
        while (epsilon < abs(x * x - t)) {
            prev_x = x;
            x = (x + t / x) / 2;
            if (x == prev_x) {
                break;  // Break if x does not change anymore
            }
        }
        return x;
    }
}

template <class T, class Exponent>
    requires std::is_arithmetic_v<std::remove_reference_t<T>> &&
             std::is_arithmetic_v<std::remove_reference_t<Exponent>>
constexpr T pow(T base, Exponent exp) {
    if constexpr (meta::is_constexpr(
                      [&] { return std::pow(T{}, Exponent{}); })) {
        return std::pow(base, exp);
    } else {
        if (exp == 0) {
            return T{1};
        } else if (exp < 0) {
            return T{1} / pow(base, -exp);
        } else {
            T result = T{1};
            for (Exponent i = 0; i < exp; ++i) {
                result *= base;
            }
            return result;
        }
    }
}
}  // namespace micro_nn