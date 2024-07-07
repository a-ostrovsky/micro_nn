#include "../src/solver.h"

#include <gtest/gtest.h>

#include <random>

#include "../src/init.h"
#include "../src/layers.h"
#include "../src/loss.h"
#include "../src/rand.h"

namespace micro_nn::solver {

namespace {
struct LinearCombinationWithWeights {
    std::vector<micro_nn::linalg::Matrix<>> combinations_{};
    micro_nn::linalg::Matrix<> weights_{};
};

std::vector<micro_nn::linalg::Matrix<>> generate_random_train_data(
    int features = 3, int samples = 100) {
    rand::SimpleLCG rng{42};
    std::uniform_real_distribution<float> distribution{0.0f, 10.0f};
    std::vector<linalg::Matrix<>> test_data{};
    for (int i = 0; i < samples; ++i) {
        linalg::Matrix<> matrix{1, features};
        for (int j = 0; j < features; ++j) {
            matrix.at(0, j) = distribution(rng);
        }
        test_data.push_back(std::move(matrix));
    }
    return test_data;
}

LinearCombinationWithWeights create_linear_combination_with_noise(
    const std::vector<micro_nn::linalg::Matrix<>>& input) {
    if (input.empty()) {
        return {};
    }
    const auto cols{input.front().cols()};
    rand::SimpleLCG rng{4242};
    std::uniform_real_distribution<float> noise_distribution{-0.001f, 0.001f};
    std::uniform_real_distribution<float> weight_distribution{-1.0f, 1.0f};
    std::vector<linalg::Matrix<>> output{};
    // Create random weights.
    linalg::Matrix<> weights{1, cols};
    weights.unary_expr_inplace([&](auto) { return weight_distribution(rng); });

    // Create linear combination of each row with the weight defined above and
    // with noise.
    for (const auto& matrix : input) {
        const auto ret{matrix.elementwise_multiply(weights).sum()};
        const auto ret_with_noise{ret + noise_distribution(rng)};
        output.push_back(linalg::Matrix<>{{{ret_with_noise}}});
    }

    return {std::move(output), std::move(weights)};
}
}  // namespace

TEST(SolverTest, LearnSimpleLinearRegression) {
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

TEST(SolverTest, LearnIdentity) {
    constexpr int num_features{3};

    model::SequentialModel model(layers::Linear{3, 3});

    optimizer::SGDOptimizer optimizer{
        model, {.learning_rate_ = .001f, .weight_decay_ = 0.01f}};
    loss::MSE<> mse{};

    const auto train_data{generate_random_train_data(num_features)};
    data::SimpleDataLoader dataloader(train_data, train_data,
                                      {.batch_size_ = 10});

    Solver solver{model, optimizer, mse, dataloader};
    solver.train(100);

    auto y_pred{model.forward(linalg::Matrix<>{{{1.0f, 1.0f, 1.0f}}})};

    EXPECT_NEAR(y_pred.at(0, 0), 1.0f, 1e-1f);
}

TEST(SolverTest, LearnLinearTransformation) {
    constexpr int num_features{3};

    model::SequentialModel model(layers::Linear{3, 1}, layers::Linear{1, 1});

    auto initializer{init::KaimingNormal<>({.seed_ = 42})};
    init::init_model<>(initializer, model);

    optimizer::SGDOptimizer optimizer{model, {.learning_rate_ = 0.0001f}};
    lr_scheduler::StepDecay<> lr_scheduler{
        {.epoch_count_ = 100, .drop_factor_ = 0.5f}};
    loss::MSE<> mse{};

    const auto x{generate_random_train_data(num_features)};
    const auto [y, weights]{create_linear_combination_with_noise(x)};

    data::SimpleDataLoader dataloader(x, y, {.batch_size_ = 50});

    Solver solver{model, optimizer, mse, dataloader, lr_scheduler};
    solver.train(500);

    // Verify learning result with the test data. This is of course not good for
    // testing the model but this test verifies only the solver functionality.
    auto y_pred{model.forward(x.at(0))};
    EXPECT_NEAR(y_pred.at(0, 0), y.at(0).at(0, 0), 0.1f);
}

}  // namespace micro_nn::solver