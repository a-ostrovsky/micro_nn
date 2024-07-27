#include "../src/optimizer.h"

#include <gtest/gtest.h>

#include "../src/model.h"

namespace micro_nn::optimizer {

TEST(OptimizerTest, SGDOptimizer_WeightsAreUpdated) {
    layers::Linear linear{2, 2};
    linear.set_weights(/*weights*/ linalg::Matrix<>::identity(2),
                       /*bias*/ linalg::Matrix<>{{{0}, {0}}});
    auto initial_weights{linear.weights()};
    auto initial_bias{linear.bias()};

    model::SequentialModel model{std::move(linear), layers::ReLU()};
    optimizer::SGDOptimizer optimizer{model};

    model.forward(linalg::Matrix<>{{{1, -1}, {-1, 1}}});
    model.backward(linalg::Matrix<>::identity(2));

    optimizer.step();

    auto& linearInSequential{std::get<layers::Linear<>>(model.layers())};
    EXPECT_NE(linearInSequential.weights(), initial_weights);
    EXPECT_NE(linearInSequential.bias(), initial_bias);
}

TEST(OptimizerTest, SGDOptimizer_WeightDecayReducesWeights) {
    auto runTestAndReturnWeight{[](float weight_decay) {
        layers::Linear<float> linear{1, 1};
        linear.set_weights(linalg::Matrix<>{{{1.0f}}},
                           linalg::Matrix<>{{{0.0f}}});

        model::SequentialModel model{std::move(linear)};
        optimizer::SGDOptimizer optimizer(model,
                                          {.weight_decay_ = weight_decay});

        model.forward(linalg::Matrix<>{{{1.0f}}});
        model.backward(linalg::Matrix<>{{{1.0f}}});
        optimizer.step();

        return std::get<0>(model.layers()).weights().at(0, 0);
    }};

    auto withoutWeightDecay{runTestAndReturnWeight(0.0f)};
    auto withWeightDecay{runTestAndReturnWeight(0.1f)};
    EXPECT_NE(withWeightDecay, withoutWeightDecay);
}

TEST(OptimizerTest, AdamOptimizer_WeightsAreUpdated) {
    layers::Linear linear{2, 2};
    linear.set_weights(/*weights*/ linalg::Matrix<>::identity(2),
                       /*bias*/ linalg::Matrix<>{{{0}, {0}}});
    auto initial_weights{linear.weights()};
    auto initial_bias{linear.bias()};

    model::SequentialModel model{std::move(linear), layers::ReLU()};
    optimizer::AdamOptimizer optimizer{model};

    model.forward(linalg::Matrix<>{{{1, -1}, {-1, 1}}});
    model.backward(linalg::Matrix<>::identity(2));

    optimizer.step();

    auto& linearInSequential{std::get<layers::Linear<>>(model.layers())};
    EXPECT_NE(linearInSequential.weights(), initial_weights);
    EXPECT_NE(linearInSequential.bias(), initial_bias);
}

}  // namespace micro_nn::optimizer