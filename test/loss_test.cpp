#include "../src/loss.h"

#include <gtest/gtest.h>

#include "../src/layers.h"

namespace micro_nn::loss {

TEST(MSETest, ForwardAndBackward_NoLoss) {
    MSE<> mse;
    linalg::Matrix<> y_true{{{1.0}}};
    auto y_pred{y_true};
    EXPECT_FLOAT_EQ(mse.forward(y_true, y_pred), 0.0);  // No error

    linalg::Matrix<> expected_grad{{{0.0}}};
    EXPECT_FLOAT_EQ(mse.backward(y_true, y_pred).at(0, 0),
                    expected_grad.at(0, 0));
}

TEST(MSETest, ForwardAndBackward_SomeLoss) {
    MSE<> mse;
    linalg::Matrix<> y_true{{{1.0, 3.0}}};
    linalg::Matrix<> y_pred{{{3.0, 1.0}}};
    const auto y{mse.forward(y_true, y_pred)};
    EXPECT_FLOAT_EQ(y, 4.0);  // 0.5 * ((1 - 3)^2 + (3 - 1)^2)

    // 2 * (y_pred - y_true) = [[4.0, -4.0]]
    linalg::Matrix<> expected_grad{{{4.0}, {-4.0}}};
    const auto grad{mse.backward(y_true, y_pred)};
    EXPECT_FLOAT_EQ(grad.at(0, 0), expected_grad.at(0, 0));
}

TEST(CrossEntropyTest, Forward) {
    CrossEntropy<> cross_entropy{};
    linalg::Matrix<> y_true{{{1.0f}}};
    linalg::Matrix<> y_pred{{{0.9f}}};
    float expected_loss{-std::log(0.9f)};
    EXPECT_NEAR(cross_entropy.forward(y_true, y_pred), expected_loss, 1e-6);
}

TEST(CrossEntropyTest, Backward) {
    micro_nn::loss::CrossEntropy<float> cross_entropy{};
    linalg::Matrix<float> y_true{{{1.0f}}};
    linalg::Matrix<float> y_pred{{{0.9f}}};
    linalg::Matrix<float> expected_grad{{{-0.1f}}};
    const auto grad{cross_entropy.backward(y_true, y_pred)};
    EXPECT_NEAR(grad.at(0, 0), expected_grad.at(0, 0), 1e-6);
}

// TODO: Probably this test should be moved out or replaced by the
// implementation of the optimizer.
TEST(MSETest, LinearRegressionSample) {
    constexpr const auto kLearningRate{0.01f};
    constexpr const auto kEpochs{1'000};

    layers::Linear<> input{1, 1};
    input.set_weights(linalg::Matrix<>{{{0.5}}}, linalg::Matrix<>{{{0.5}}});
    MSE<> mse{};
    auto x{linalg::Matrix<>{{{1.0}, {2.0}, {3.0}}}};
    auto y_true{linalg::Matrix<>{{{3.0}, {5.0}, {7.0}}}};  // x*2+1

    // Actually only the last value is needed. The history is stored for
    // debugging purposes.
    std::vector<float> lossHistory{};
    for (int epoch{0}; epoch < kEpochs; ++epoch) {
        auto y_pred{input.forward(x)};
        const auto loss{mse.forward(y_true, y_pred)};
        lossHistory.push_back(loss);
        auto d_out{mse.backward(y_true, y_pred)};
        input.backward(d_out);

        const auto updatedWeights{
            input.weights() -
            input.d_weights().unary_expr(
                [kLearningRate](auto x) { return kLearningRate * x; })};
        const auto updatedBias{
            input.bias() - input.d_bias().unary_expr([kLearningRate](auto x) {
                return kLearningRate * x;
            })};
        input.set_weights(std::move(updatedWeights), std::move(updatedBias));
    }

    EXPECT_LT(lossHistory.back(), 1e-5);  // Loss should be low
    const auto y_pred{input.forward(x)};
    EXPECT_NEAR(y_pred.at(0, 0), 3.0, 1e-3);  // 1*2+1
}

}  // namespace micro_nn::loss