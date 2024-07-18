#include "../src/config.h"
#include "../src/layers.h"
#include "../src/loss.h"
#include "../src/solver.h"
#include "../src/utils.h"

namespace {
using namespace micro_nn;
using namespace micro_nn::solver;

// Formula: y = 2x + 1
constexpr float create_model_and_solve(float input) {
    layers::Linear linear{1, 1};
    linear.set_weights(linalg::Matrix<>{{{0.9f}}}, linalg::Matrix<>{{{0.9f}}});

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

    Solver solver{model, optimizer, mse, dataloader};
    solver.train(50);

    auto y_pred{model.forward(linalg::Matrix<>{{{input}}})};
    return y_pred.at(0, 0);
}

static_assert(micro_nn::abs(create_model_and_solve(5.0f) - 11.0f) <= 0.5f,
              "solve_linear_regression() test failed!");
}  // namespace