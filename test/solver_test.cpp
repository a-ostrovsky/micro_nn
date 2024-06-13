#include "../src/solver.h"

#include <gtest/gtest.h>

#include "../src/layers.h"
#include "../src/loss.h"

namespace micro_nn::solver {

TEST(SolverTest, LinearRegression) {
    layers::Linear linear{1, 1};
    linear.set_weights(linalg::Matrix<>{{{0.5}}}, linalg::Matrix<>{{{0.5}}});

    auto initial_weights{linear.weights()};
    auto initial_bias{linear.bias()};

    SequentialModel model{std::move(linear)};
    optimizer::SGDOptimizer optimizer{model, 0.01f};
    loss::MSE<> mse{};

    // y = x * 2 + 1
    auto x1{linalg::Matrix<>{{{1.0}}}};
    auto x2{linalg::Matrix<>{{{2.0}}}};
    auto x3{linalg::Matrix<>{{{3.0}}}};

    auto y1{linalg::Matrix<>{{{3.0}}}};
    auto y2{linalg::Matrix<>{{{5.0}}}};
    auto y3{linalg::Matrix<>{{{7.0}}}};

    data::SimpleDataLoader dataloader(std::vector{x1, x2, x3},
                                      std::vector{y1, y2, y3}, 3);

    Solver solver{model, optimizer, mse, dataloader};
    solver.train(1'000);
    auto y_pred = model.forward(linalg::Matrix<>{{{5.0}}});
    EXPECT_NEAR(y_pred.at(0, 0), 11.0, 1e-3);  // 5*2+1
}

}  // namespace micro_nn::solver