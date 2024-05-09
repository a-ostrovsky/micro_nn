#pragma once
#include "config.h"
#include "linalg.h"

namespace micro_nn::layers {
template <class NumT = config::kFloat>
class Sigmoid {
public:
    constexpr micro_nn::linalg::Matrix<NumT> forward(
        const micro_nn::linalg::Matrix<NumT>& x) {
        auto output{x.unary_expr([](NumT x) { return sigmoid(x); })};
        return output;
    }

    micro_nn::linalg::Matrix<NumT> backward(
        const micro_nn::linalg::Matrix<NumT>& d_out) {
        auto d_sigmoid =
            d_out.unary_expr([](NumT x) { return sigmoid_derivative(x); });
        auto d_input = d_out * d_sigmoid;
        return d_input;
    }

private:
    constexpr static NumT sigmoid(NumT x) { return 1 / (1 + std::exp(-x)); }
    constexpr static NumT sigmoid_derivative(NumT x) { return x * (1 - x); }
};

template <class NumT = config::kFloat>
class Tanh {
public:
    constexpr micro_nn::linalg::Matrix<NumT> forward(
        const micro_nn::linalg::Matrix<NumT>& x) {
        auto output{x.unary_expr([](NumT x) { return std::tanh(x); })};
        return output;
    }

    micro_nn::linalg::Matrix<NumT> backward(
        const micro_nn::linalg::Matrix<NumT>& d_out) {
        auto d_tanh =
            d_out.unary_expr([](NumT x) { return tanh_derivative(x); });
        auto d_input = d_out * d_tanh;
        return d_input;
    }

private:
    NumT tanh_derivative(NumT x) const {
        NumT tanh_x = std::tanh(x);
        return 1 - tanh_x * tanh_x;
    }
};

template <class NumT = config::kFloat>
class ReLU {
public:
    constexpr micro_nn::linalg::Matrix<NumT> forward(
        const micro_nn::linalg::Matrix<NumT>& x) {
        x_ = x;
        auto output{x.unary_expr([](NumT x) { return std::max(NumT(0), x); })};
        return output;
    }

    constexpr micro_nn::linalg::Matrix<NumT> backward(
        const micro_nn::linalg::Matrix<NumT>& d_out) {
        auto ret{d_out};
        for (auto row{0}; row < ret.rows(); ++row) {
            for (auto col{0}; col < ret.cols(); ++col) {
                if (x_.at(row, col) <= 0) {
                    ret.at(row, col) = 0;
                }
            }
        }
        return ret;
    }

private:
    micro_nn::linalg::Matrix<NumT> x_{};
};

template <class NumT = config::kFloat>
struct LinearBackwardResult {
    micro_nn::linalg::Matrix<NumT> d_input{};
    micro_nn::linalg::Matrix<NumT> d_weights{};
    micro_nn::linalg::Matrix<NumT> d_bias{};
};

template <class NumT = config::kFloat>
class Linear {
public:
    constexpr Linear(int input_features, int output_features)
        : weights_(input_features, output_features),
          bias_(1, output_features) {}

    constexpr micro_nn::linalg::Matrix<NumT> forward(
        const micro_nn::linalg::Matrix<NumT>& x) {
        auto output{x * weights_ + bias_};
        x_ = x;
        return output;
    }

    constexpr LinearBackwardResult<NumT> backward(
        const micro_nn::linalg::Matrix<NumT>& d_out) {
        auto d_weights{x_.transpose() * d_out};
        auto d_bias{d_out.rowwise_sum()};
        auto d_input{d_out * weights_.transpose()};
        return {.d_input = std::move(d_input),
                .d_weights = std::move(d_weights),
                .d_bias = std::move(d_bias)};
    }

    constexpr void set_weights(const micro_nn::linalg::Matrix<NumT> weights,
                               const micro_nn::linalg::Matrix<NumT> bias) {
        weights_ = std::move(weights);
        bias_ = std::move(bias);
    }

private:
    micro_nn::linalg::Matrix<NumT> weights_{};
    micro_nn::linalg::Matrix<NumT> bias_{};
    micro_nn::linalg::Matrix<NumT> x_{};
};
}  // namespace micro_nn::layers