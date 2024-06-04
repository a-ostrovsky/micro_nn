#pragma once
#include "config.h"
#include "sequential_model.h"

namespace micro_nn::optimizer {
template <class NumT = config::kFloat, class... Layers>
class SGDOptimizer {
public:
    constexpr SGDOptimizer(SequentialModel<NumT, Layers...>& model,
                           NumT learning_rate)
        : model_(model), learning_rate_(learning_rate) {}

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
                layer.weights() - learning_rate_ * layer.d_weights(),
                layer.bias() - learning_rate_ * layer.d_bias());
        }
    }

    SequentialModel<NumT, Layers...>& model_;
    NumT learning_rate_;
};
}  // namespace micro_nn::optimizer