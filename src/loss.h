#pragma once

#include "linalg/matrix.h"

namespace micro_nn::loss {

template <class NumT, class T>
concept Loss = requires(T loss, const linalg::Matrix<NumT>& y_true,
                        const linalg::Matrix<NumT>& y_pred) {
    { loss.forward(y_true, y_pred) } -> std::convertible_to<NumT>;
    {
        loss.backward(y_true, y_pred)
    } -> std::convertible_to<linalg::Matrix<NumT>>;
};

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

template <typename NumT = config::kFloat>
class CrossEntropy {
public:
    constexpr NumT forward(const linalg::Matrix<NumT>& y_true,
                           const linalg::Matrix<NumT>& y_pred) {
        // 1e-15 and 1e15 are used to avoid log(0) and log(inf)
        const auto clamped_y_pred{clamp(y_pred, 1e-15f, 1e15f)};
        return -(y_true.elementwise_multiply(log(clamped_y_pred))).sum();
    }

    constexpr linalg::Matrix<NumT> backward(
        const linalg::Matrix<NumT>& y_true,
        const linalg::Matrix<NumT>& y_pred) {
        return y_pred - y_true;
    }
};
}  // namespace micro_nn::loss