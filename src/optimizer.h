#pragma once
#include "config.h"
#include "sequential_model.h"

namespace micro_nn::optimizer {

template <typename T>
concept Optimizer = requires(T opt) {
    { opt.step() };
};

template <class NumT = config::kFloat>
struct SGDOptimizerConfig {
    float learning_rate_{static_cast<NumT>(0.01f)};
    float weight_decay_{static_cast<NumT>(0.0f)};
};

template <class NumT = config::kFloat, class... Layers>
class SGDOptimizer {
public:
    constexpr SGDOptimizer(SequentialModel<NumT, Layers...>& model,
                           SGDOptimizerConfig<NumT> config = {})
        : model_(model),
          learning_rate_(config.learning_rate_),
          weight_decay_(config.weight_decay_) {}

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
            layer.set_weights(
                layer.weights() -
                    learning_rate_ *
                        (layer.d_weights() + weight_decay_ * layer.weights()),
                layer.bias() - learning_rate_ * layer.d_bias());
        }
    }

    SequentialModel<NumT, Layers...>& model_;
    NumT learning_rate_;
    NumT weight_decay_;
};
}  // namespace micro_nn::optimizer