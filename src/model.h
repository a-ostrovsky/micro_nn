#pragma once

#include <concepts>
#include <type_traits>

#include "config.h"
#include "linalg.h"

namespace micro_nn {
template <class NumT, class T>
concept Model = requires(T model, const micro_nn::linalg::Matrix<NumT>& m) {
    { model.forward(m) } -> std::convertible_to<micro_nn::linalg::Matrix<NumT>>;
    {
        model.backward(m)
    } -> std::convertible_to<micro_nn::linalg::Matrix<NumT>>;
};

}  // namespace micro_nn