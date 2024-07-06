#include "../src/linalg/lu_decomp.h"

#include <gtest/gtest.h>

#include "../src/linalg/matrix.h"
#include "utils/custom_assert.h"

namespace micro_nn::linalg {
TEST(LUDecompositionTest, NonSquareMatrixThrows) {
    Matrix<> nonSquareMatrix(2, 3);
    EXPECT_THROW(lu_factor(nonSquareMatrix), std::invalid_argument);
}

TEST(LUDecompositionTest, CorrectnessTest) {
    Matrix<> a{{{4.0f, 3.0f}, {6.0f, 3.0f}}};
    Matrix<> l_expected{{{1.0f, 0.0f}, {1.5f, 1.0f}}};
    Matrix<> u_expected{{{4.0f, 3.0f}, {0.0f, -1.5f}}};
    const auto [l, u]{lu_factor(a)};
    // Assuming Matrix class has an equality operator or a similar method to
    // compare matrices
    expect_near(l, l_expected, 0.01f);
    expect_near(u, u_expected, 0.01f);
}
}  // namespace micro_nn::linalg