#pragma once

#include <optional>
#include <random>
#include <tuple>

#include "config.h"
#include "linalg.h"
#include "model.h"
#include "rand.h"
#include "utils.h"

namespace micro_nn::init {

template <class NumT, typename T>
concept Initializer = requires(T a, linalg::Matrix<NumT>& matrix) {
    { a.init(matrix) };
};

template <class NumT = config::kFloat, class InitializerT, class ModelT>
    requires(Initializer<NumT, InitializerT> && model::Model<NumT, ModelT>)
constexpr void init_model(InitializerT& initializer, ModelT& model) {
    auto apply_initializer{[&initializer](auto& layer) {
        if constexpr (layers::WeightedLayer<NumT, decltype(layer)>) {
            // TODO: Inplace modification
            auto matrix{layer.weights()};
            initializer.init(matrix);
            layer.set_weights(matrix, layer.bias());
        }
    }};

    std::apply([&](auto&... layers) { (apply_initializer(layers), ...); },
               model.layers());
}

struct KaimingNormalSettings {
    std::optional<rand::SimpleLCG::result_type> seed_{};
};

template <class NumT = config::kFloat>
class KaimingNormal {
public:
    constexpr explicit KaimingNormal(KaimingNormalSettings settings)
        : seed_(settings.seed_) {}

    constexpr void init(linalg::Matrix<NumT>& matrix) {
        auto rng{seed_ ? rand::SimpleLCG{*seed_} : rand::SimpleLCG{}};
        const auto fan_in{matrix.cols()};
        const NumT std_dev{
            sqrt(narrow_cast<NumT>(2) / narrow_cast<NumT>(fan_in))};
        std::normal_distribution<NumT> dist(NumT{}, std_dev);
        matrix.unary_expr_inplace([&](auto) { return dist(rng); });
    }

private:
    std::optional<rand::SimpleLCG::result_type> seed_;
};

}  // namespace micro_nn::init