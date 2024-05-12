#pragma once

#include <memory>

#include "layers.h"
#include "model.h"

namespace micro_nn {
template <class NumT = config::kFloat, class... Layers>
    requires(layers::Layer<NumT, Layers> && ...)
class SequentialModel {
public:
    constexpr explicit SequentialModel(Layers&&... layers)
        : layers_{std::forward<Layers>(layers)...} {}

    constexpr micro_nn::linalg::Matrix<NumT> forward(
        const micro_nn::linalg::Matrix<NumT>& x) {
        micro_nn::linalg::Matrix<NumT> y = x;
        std::apply([&](auto&... layer) { ((y = layer.forward(y)), ...); },
                   layers_);
        return y;
    }
    micro_nn::linalg::Matrix<NumT> backward(
        const micro_nn::linalg::Matrix<NumT>& d_out) {
        throw std::exception("not implemented");
    }

private:
    std::tuple<Layers...> layers_;
};
}  // namespace micro_nn