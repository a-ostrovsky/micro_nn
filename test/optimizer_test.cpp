#include "../src/optimizer.h"

#include <gtest/gtest.h>

#include "../src/sequential_model.h"

namespace micro_nn::optimizer {

TEST(OptimizerTest, SGDOptimizer_WeightsAreUpdated) {
    layers::Linear linear{2, 2};
    linear.set_weights(/*weights*/ linalg::Matrix<>::unity(2),
                       /*bias*/ linalg::Matrix<>{{{0}, {0}}});
    auto initial_weights{linear.weights()};
    auto initial_bias{linear.bias()};

    SequentialModel model{std::move(linear), layers::ReLU()};
    optimizer::SGDOptimizer optimizer{model, 0.01f};

    model.forward(linalg::Matrix<>{{{1, -1}, {-1, 1}}});
    model.backward(linalg::Matrix<>::unity(2));

    optimizer.step();

    auto& linearInSequential{std::get<layers::Linear<>>(model.layers())};
    EXPECT_NE(linearInSequential.weights(), initial_weights);
    EXPECT_NE(linearInSequential.bias(), initial_bias);
}
}  // namespace micro_nn::optimizer