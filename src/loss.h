#pragma once

#include "linalg.h"

namespace micro_nn::loss {

template <class NumT = config::kFloat>
class MSE {
public:
    constexpr NumT forward(const linalg::Matrix<NumT>& y_true,
                           const linalg::Matrix<NumT>& y_pred) {
        const auto diff = y_true - y_pred;
        return (diff.elementwise_multiply(diff)).sum() / diff.size();
    }

    constexpr linalg::Matrix<NumT> backward(
        const linalg::Matrix<NumT>& y_true,
        const linalg::Matrix<NumT>& y_pred) {
        return (y_pred - y_true).unary_expr([](NumT x) { return 2 * x; });
    }
};
}  // namespace micro_nn::loss