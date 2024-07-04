#include "../src/init.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

namespace micro_nn::init {
TEST(InitTest, KaimingNormal) {
    const int fan_in{100};
    KaimingNormal<> kaiming_init({.seed_ = 0});
    linalg::Matrix<> matrix(100, 100);
    kaiming_init.init(matrix);

    // Calculate the empirical standard deviation of the initialized matrix
    float mean{matrix.sum() / matrix.size()};
    float accum{};
    for (int row{}; row < matrix.rows(); ++row) {
        for (int col{}; col < matrix.cols(); ++col) {
            accum +=
                (matrix.at(row, col) - mean) * (matrix.at(row, col) - mean);
        }
    }
    float std_dev{std::sqrt(accum / (matrix.size() - 1))};

    float expected_std_dev{std::sqrt(2.0f / narrow<float>(fan_in))};

    EXPECT_NEAR(std_dev, expected_std_dev, 0.1);
}
}  // namespace micro_nn::init