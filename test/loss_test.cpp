#include "../src/loss.h"

#include <gtest/gtest.h>

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

}  // namespace micro_nn::loss