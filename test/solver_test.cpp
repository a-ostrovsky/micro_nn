#include "../src/solver.h"

#include <gtest/gtest.h>

#include "../src/init.h"
#include "../src/layers.h"
#include "../src/loss.h"

namespace micro_nn::solver {

TEST(SolverTest, LinearRegression) {
    layers::Linear linear{1, 1};
    linear.set_weights(linalg::Matrix<>{{{0.5}}}, linalg::Matrix<>{{{0.5}}});

    auto initial_weights{linear.weights()};
    auto initial_bias{linear.bias()};

    model::SequentialModel model{std::move(linear)};
    optimizer::SGDOptimizer optimizer{model};
    loss::MSE<> mse{};

    // y = x * 2 + 1
    auto x1{linalg::Matrix<>{{{1.0}}}};
    auto x2{linalg::Matrix<>{{{2.0}}}};
    auto x3{linalg::Matrix<>{{{3.0}}}};

    auto y1{linalg::Matrix<>{{{3.0}}}};
    auto y2{linalg::Matrix<>{{{5.0}}}};
    auto y3{linalg::Matrix<>{{{7.0}}}};

    data::SimpleDataLoader dataloader(
        std::vector{x1, x2, x3}, std::vector{y1, y2, y3}, {.batch_size_ = 3});

    // Consistency check before training.
    const auto y_pred_before_test{model.forward(linalg::Matrix<>{{{5.0}}})};
    const bool y_pred_before_test_is_close{
        std::abs(y_pred_before_test.at(0, 0) - 11.0f) < 1e-3f};
    EXPECT_FALSE(y_pred_before_test_is_close);

    Solver solver{model, optimizer, mse, dataloader};
    solver.train(1'000);
    auto y_pred{model.forward(linalg::Matrix<>{{{5.0}}})};
    EXPECT_NEAR(y_pred.at(0, 0), 11.0, 1e-3);  // 5*2+1
}

TEST(SolverTest, MultiLayerPerceptron) {
    layers::Linear layer1{3, 4};
    layers::ReLU activation1;
    layers::Linear layer2{4, 3};
    layers::ReLU activation2;

    model::SequentialModel model(std::move(layer1), std::move(activation1),
                                 std::move(layer2), std::move(activation2));

    auto initializer{init::KaimingNormal<>({.seed_ = 42})};
    init::init_model<>(initializer, model);

    optimizer::SGDOptimizer optimizer{
        model, {.learning_rate_ = 0.005f, .weight_decay_ = 0.01f}};
    loss::MSE<> mse{};

    auto x1{linalg::Matrix<>{{{1.0f, 0.0f, 0.0f}}}};
    auto x2{linalg::Matrix<>{{{0.0f, 1.0f, 0.0f}}}};
    auto x3{linalg::Matrix<>{{{0.0f, 0.0f, 1.0f}}}};

    // Logic:
    // x[0] mapped to y[2], x[1] mapped to y[0], x[2] mapped to y[1]
    auto y1{linalg::Matrix<>{{{0.0f, 0.0f, 1.0f}}}};
    auto y2{linalg::Matrix<>{{{1.0f, 0.0f, 0.0f}}}};
    auto y3{linalg::Matrix<>{{{0.0f, 1.0f, 0.0f}}}};

    data::SimpleDataLoader dataloader(
        std::vector{x1, x2, x3}, std::vector{y1, y2, y3}, {.batch_size_ = 3});

    Solver solver{model, optimizer, mse, dataloader};
    solver.train(1'000);

    auto y_pred{model.forward(linalg::Matrix<>{{{0.9f, 0.0f, 0.1f}}})};

    EXPECT_GT(y_pred.at(0, 2), y_pred.at(0, 1));
    EXPECT_GT(y_pred.at(0, 2), y_pred.at(0, 0));
}

}  // namespace micro_nn::solver