#pragma once
#include <gtest/gtest.h>

#include "../../src/config.h"
#include "../../src/linalg/matrix.h"
#include "../../src/utils.h"

namespace micro_nn {
template <class NumT = config::kFloat>
void expect_near(const linalg::Matrix<NumT>& lhs,
                 const linalg::Matrix<NumT>& rhs, NumT tolerance = 0.01) {
    EXPECT_EQ(lhs.rows(), rhs.rows());
    EXPECT_EQ(lhs.cols(), rhs.cols());
    for (int i{0}; i < lhs.rows(); ++i) {
        for (int j{0}; j < lhs.cols(); ++j) {
            EXPECT_NEAR(lhs.at(i, j), rhs.at(i, j), narrow<NumT>(tolerance));
        }
    }
}
}  // namespace micro_nn