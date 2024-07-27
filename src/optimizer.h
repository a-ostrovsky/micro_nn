#pragma once
#include <tuple>

#include "config.h"
#include "data_structures/hash_map.h"
#include "model.h"
#include "utils.h"

namespace micro_nn::optimizer {

template <class T>
concept Optimizer = requires(T opt) {
    { opt.step() };
};

template <class NumT = config::kFloat>
struct SGDOptimizerConfig {
    NumT learning_rate_{static_cast<NumT>(0.01f)};
    NumT weight_decay_{static_cast<NumT>(0.0f)};
    NumT momentum_{static_cast<NumT>(0.0f)};
};

template <class NumT = config::kFloat, class... Layers>
class SGDOptimizer {
    struct Velocities {
        linalg::Matrix<NumT> weights_velocity_{{{0}}};
        linalg::Matrix<NumT> bias_velocity{{{0}}};
    };
    using VelocitiesMap = data_structures::HashMap<layers::LayerId, Velocities>;

public:
    explicit constexpr SGDOptimizer(
        model::SequentialModel<NumT, Layers...>& model,
        SGDOptimizerConfig<NumT> config = {})
        : model_{model},
          learning_rate_{config.learning_rate_},
          weight_decay_{config.weight_decay_},
          momentum_{config.momentum_} {}

    constexpr void step() {
        std::apply(
            [&](auto&... layer) {
                ((apply_step<decltype(layer)>(layer)), ...);
            },
            model_.layers());
    }

private:
    template <class Layer>
    constexpr void apply_step(Layer& layer) {
        if constexpr (layers::WeightedLayer<NumT, Layer>) {
            const auto d_weights_adjusted{layer.d_weights() +
                                          weight_decay_ * layer.weights()};
            auto [weights_velocity, bias_velocity] = velocity_[layer.id()];
            weights_velocity = momentum_ * weights_velocity -
                               learning_rate_ * d_weights_adjusted;
            bias_velocity =
                momentum_ * bias_velocity - learning_rate_ * layer.d_bias();
            velocity_[layer.id()] = Velocities{weights_velocity, bias_velocity};
            layer.set_weights(layer.weights() + weights_velocity,
                              layer.bias() + bias_velocity);
        }
    }

    model::SequentialModel<NumT, Layers...>& model_;
    VelocitiesMap velocity_;
    NumT learning_rate_;
    NumT weight_decay_;
    NumT momentum_;
};

template <class NumT = config::kFloat>
struct AdamOptimizerConfig {
    NumT learning_rate_{static_cast<NumT>(0.001f)};
    // Beta1 and beta2 are the exponential decay rates for the first and second
    // moments respectively.
    NumT beta1_{static_cast<NumT>(0.9f)};
    NumT beta2_{static_cast<NumT>(0.999f)};
    NumT epsilon_{static_cast<NumT>(1e-8f)};
};

template <class NumT = config::kFloat, class... Layers>
class AdamOptimizer {
    struct LayerParams {
        linalg::Matrix<NumT> m_{{{0}}};  // First moment estimate.
        linalg::Matrix<NumT> v_{{{0}}};  // Second moment estimate.
    };
    using LayerParamsMap =
        data_structures::HashMap<layers::LayerId, LayerParams>;

public:
    AdamOptimizer(model::SequentialModel<NumT, Layers...>& model,
                  AdamOptimizerConfig<NumT> config = {})
        : model_(model),
          beta1_(config.beta1_),
          beta2_(config.beta2_),
          epsilon_(config.epsilon_),
          learning_rate_(config.learning_rate_) {}

    void step() {
        std::apply(
            [&](auto&... layer) {
                ((apply_step<decltype(layer)>(layer)), ...);
            },
            model_.layers());
    }

private:
    template <class Layer>
    constexpr void apply_step(Layer& layer) {
        if constexpr (layers::WeightedLayer<NumT, Layer>) {
            const auto new_weights{calc_params(layer_params_weights_,
                                               layer.id(), layer.weights(),
                                               layer.d_weights())};
            const auto new_bias{calc_params(layer_params_bias_, layer.id(),
                                            layer.bias(), layer.d_bias())};
            layer.set_weights(new_weights, new_bias);
        }
    }

    constexpr linalg::Matrix<NumT> calc_params(
        LayerParamsMap& params_map, layers::LayerId id,
        const linalg::Matrix<NumT>& params, const linalg::Matrix<NumT>& grads) {
        auto& m{params_map[id].m_};
        auto& v{params_map[id].v_};
        // Initialize m and v to zeros at the first step.
        if (m.rows() == 0 && m.cols() == 0) {
            m = linalg::Matrix<NumT>::zeros(params.rows(), params.cols());
            v = linalg::Matrix<NumT>::zeros(params.rows(), params.cols());
        }
        ++t_;

        // Update the first moment (mean). This tries to accumulate the
        // momentum.
        m = beta1_ * m + (1 - beta1_) * grads;

        // Update the second moment (variance). This tries to restrict the
        // fluctuations in the dimensions which are fluctuating a lot.
        v = beta2_ * v +
            (1 - beta2_) * grads.unary_expr([](NumT x) { return x * x; });

        // m_hat is the bias-corrected first moment estimate.
        // Without this, m would be too small in the beginning of the training
        // leading to unstable training.
        auto m_hat = m / (1 - pow(beta1_, t_));

        // v_hat is the bias-corrected second moment estimate.
        auto v_hat = v / (1 - pow(beta2_, t_));

        return params -
               learning_rate_ *
                   m_hat.elementwise_multiply(v_hat.unary_expr([this](NumT x) {
                       return static_cast<NumT>(1.0) / (sqrt(x) + epsilon_);
                   }));
    }

    model::SequentialModel<NumT, Layers...>& model_;
    LayerParamsMap layer_params_weights_;
    LayerParamsMap layer_params_bias_;
    NumT beta1_;
    NumT beta2_;
    NumT epsilon_;
    NumT learning_rate_;
    int t_{};  // Time step.
};
}  // namespace micro_nn::optimizer