#pragma once

#include "config.h"
#include "linalg/matrix.h"
#include "unique_id.h"

namespace micro_nn::layers {

using LayerId = std::uint32_t;

template <class NumT, class T>
concept Layer = requires(T layer, const linalg::Matrix<NumT>& m) {
    { layer.forward(m) } -> std::convertible_to<linalg::Matrix<NumT>>;
    { layer.backward(m) } -> std::convertible_to<linalg::Matrix<NumT>>;
    { layer.id() } -> std::convertible_to<LayerId>;
};

template <class NumT, class T>
concept WeightedLayer = requires(T a, const linalg::Matrix<NumT>& m) {
    { a.d_bias() } -> std::convertible_to<linalg::Matrix<NumT>>;
    { a.d_weights() } -> std::convertible_to<linalg::Matrix<NumT>>;
    { a.bias() } -> std::convertible_to<linalg::Matrix<NumT>>;
    { a.weights() } -> std::convertible_to<linalg::Matrix<NumT>>;
    { a.set_weights(m, m) };
};

template <class NumT = config::kFloat>
class Sigmoid {
public:
    constexpr micro_nn::linalg::Matrix<NumT> forward(
        const micro_nn::linalg::Matrix<NumT>& x) {
        auto output{x.unary_expr([](NumT x) { return sigmoid(x); })};
        return output;
    }

    constexpr micro_nn::linalg::Matrix<NumT> backward(
        const micro_nn::linalg::Matrix<NumT>& d_out) {
        auto d_sigmoid =
            d_out.unary_expr([](NumT x) { return sigmoid_derivative(x); });
        auto d_input = d_out * d_sigmoid;
        return d_input;
    }

    constexpr LayerId id() const { return id_; }

private:
    constexpr static NumT sigmoid(NumT x) { return 1 / (1 + std::exp(-x)); }
    constexpr static NumT sigmoid_derivative(NumT x) { return x * (1 - x); }
    constexpr static std::uint32_t id_{NEXT_UNIQUE_ID()};
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
        micro_nn::linalg::Matrix<NumT> ret{d_out};
        for (auto row{0}; row < ret.rows(); ++row) {
            for (auto col{0}; col < ret.cols(); ++col) {
                if (x_.at(row, col) <= 0) {
                    ret.at(row, col) = 0;
                }
            }
        }
        return ret;
    }

    constexpr LayerId id() const { return id_; }

private:
    micro_nn::linalg::Matrix<NumT> x_{};
    constexpr static std::uint32_t id_{NEXT_UNIQUE_ID()};
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

    constexpr micro_nn::linalg::Matrix<NumT> backward(
        const micro_nn::linalg::Matrix<NumT>& d_out) {
        d_weights_ = x_.transpose() * d_out;
        d_bias_ = d_out.rowwise_sum();
        auto d_input{d_out * weights_.transpose()};
        return d_input;
    }

    constexpr void set_weights(micro_nn::linalg::Matrix<NumT> weights,
                               micro_nn::linalg::Matrix<NumT> bias) {
        weights_ = std::move(weights);
        bias_ = std::move(bias);
    }

    constexpr const micro_nn::linalg::Matrix<NumT>& weights() const {
        return weights_;
    }

    constexpr const micro_nn::linalg::Matrix<NumT>& bias() const {
        return bias_;
    }

    constexpr const micro_nn::linalg::Matrix<NumT>& d_weights() const {
        return d_weights_;
    }

    constexpr const micro_nn::linalg::Matrix<NumT>& d_bias() const {
        return d_bias_;
    }

    constexpr LayerId id() const { return id_; }

private:
    micro_nn::linalg::Matrix<NumT> weights_{};
    micro_nn::linalg::Matrix<NumT> bias_{};
    micro_nn::linalg::Matrix<NumT> x_{};
    micro_nn::linalg::Matrix<NumT> d_weights_{};
    micro_nn::linalg::Matrix<NumT> d_bias_{};
    constexpr static std::uint32_t id_{NEXT_UNIQUE_ID()};
};
}  // namespace micro_nn::layers