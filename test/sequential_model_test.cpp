#include <gtest/gtest.h>

#include "../src/layers.h"
#include "../src/model.h"

namespace micro_nn::model {

TEST(SequentialModelTest, Forward) {
    layers::Linear linear{2, 2};
    linear.set_weights(/*weights*/ linalg::Matrix<>::identity(2),
                       /*bias*/ linalg::Matrix<>{{{0}, {0}}});
    SequentialModel model{std::move(linear), layers::ReLU()};
    auto output{model.forward(linalg::Matrix<>{{{1, -1}, {-1, 1}}})};
    linalg::Matrix<> expected_output{{{1, 0}, {0, 1}}};
    EXPECT_EQ(output, expected_output);
}

TEST(SequentialModelTest, Backward) {
    layers::Linear linear{2, 2};
    linear.set_weights(/*weights*/ linalg::Matrix<>::identity(2),
                       /*bias*/ linalg::Matrix<>{{{0}, {0}}});
    SequentialModel model{std::move(linear), layers::ReLU()};
    auto output{model.forward(linalg::Matrix<>{{{1, -1}, {-1, 1}}})};
    auto d_output{model.backward(output)};

    // Input and output should be the same here because we have the identity
    // matrix in the linear layer with zero bias
    linalg::Matrix<> expected_output{linalg::Matrix<>::identity(2)};
    EXPECT_EQ(output, expected_output);
}

}  // namespace micro_nn::model