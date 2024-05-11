#include "../src/layers.h"

#include <gtest/gtest.h>

#include "../src/linalg.h"

namespace micro_nn::layers {
namespace {
template <class NumT = float>
using Matrix = micro_nn::linalg::Matrix<NumT>;
}

TEST(ReLULayerTest, ForwardAndBackward) {
    micro_nn::layers::ReLU<float> relu;
    Matrix<float> input{{{1.0, -2.0}, {3.0, -4.0}}};
    Matrix<float> expected_output{{{1.0, 0.0}, {3.0, 0.0}}};

    // Test forward pass
    auto output{relu.forward(input)};
    EXPECT_EQ(output, expected_output);

    // Test backward pass
    Matrix<float> d_out{{{1.0, 1.0}, {1.0, 1.0}}};
    Matrix<float> expected_d_input{{{1.0, 0.0}, {1.0, 0.0}}};
    auto d_input{relu.backward(d_out)};
    EXPECT_EQ(d_input, expected_d_input);
}

TEST(LinearLayerTest, ForwardAndBackward) {
    micro_nn::layers::Linear<float> linear(2, 2);
    Matrix<float> weights{{{5, 6}, {7, 8}}};
    Matrix<float> bias{{{9}, {10}}};
    linear.set_weights(weights, bias);

    Matrix<float> x{{{1, 2}, {3, 4}}};

    // Test forward method
    auto output{linear.forward(x)};
    Matrix<float> expected_output{{{28, 31}, {53, 60}}};
    EXPECT_EQ(output, expected_output);

    // Test backward method
    auto d_input{linear.backward(output)};
    EXPECT_EQ(d_input.shape(), x.shape());
    EXPECT_EQ(linear.weights().shape(), weights.shape());
    EXPECT_EQ(linear.bias().shape(), bias.shape());
}

}  // namespace micro_nn::layers