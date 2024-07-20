#pragma once
#include <tuple>

#include "config.h"
#include "data_structures/hash_map.h"
#include "model.h"

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
            weights_velocity = momentum_ * weights_velocity +
                               -learning_rate_ * d_weights_adjusted;
            bias_velocity =
                momentum_ * bias_velocity + -learning_rate_ * layer.d_bias();
            velocity_[layer.id()] = Velocities{weights_velocity, bias_velocity};
            layer.set_weights(layer.weights() + weights_velocity,
                              layer.bias() + bias_velocity);
        }
    }

    model::SequentialModel<NumT, Layers...>& model_;
    data_structures::HashMap<layers::LayerId, Velocities> velocity_;
    NumT learning_rate_;
    NumT weight_decay_;
    NumT momentum_;
};

}  // namespace micro_nn::optimizer