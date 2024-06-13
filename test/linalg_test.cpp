#include "../src/linalg.h"

#include <gtest/gtest.h>

namespace micro_nn::linalg {

TEST(MatrixTest, Constructor) {
    Matrix<int> m(10, 20);
    EXPECT_EQ(m.rows(), 10);
    EXPECT_EQ(m.cols(), 20);
}

TEST(MatrixTest, SetValues) {
    auto m{Matrix<int>::zeros(1, 1)};
    m.at(0, 0) = 10;
    EXPECT_EQ(m.at(0, 0), 10);
}

TEST(MatrixTest, Multiplication) {
    Matrix<int> m1{{{1, 2, 3}, {4, 5, 6}}};
    Matrix<int> m2{{{7, 8}, {9, 10}, {11, 12}}};

    Matrix<int> result{m1 * m2};
    Matrix<int> expected{{{58, 64}, {139, 154}}};

    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, Addition) {
    Matrix<int> m1{{{1, 2, 3}, {4, 5, 6}}};
    Matrix<int> m2{{{7, 8, 9}, {10, 11, 12}}};

    Matrix<int> result{m1 + m2};
    Matrix<int> expected{{{8, 10, 12}, {14, 16, 18}}};

    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, Transpose) {
    Matrix<int> m1{{{1, 2, 3}, {4, 5, 6}}};

    Matrix<int> result{m1.transpose()};
    Matrix<int> expected{{{1, 4}, {2, 5}, {3, 6}}};

    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, Subtraction) {
    Matrix<int> m1{{{8, 10, 12}, {14, 16, 18}}};
    Matrix<int> m2{{{1, 2, 3}, {4, 5, 6}}};

    Matrix<int> result{m1 - m2};
    Matrix<int> expected{{{7, 8, 9}, {10, 11, 12}}};

    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, ElementwiseMultiply) {
    Matrix<int> m1{{{1, 2, 3}, {4, 5, 6}}};
    Matrix<int> m2{{{7, 8, 9}, {10, 11, 12}}};

    Matrix<int> result{m1.elementwise_multiply(m2)};
    Matrix<int> expected{{{7, 16, 27}, {40, 55, 72}}};

    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, BroadcastingAddition) {
    Matrix<int> m1{{{1, 2, 3}, {4, 5, 6}}};
    Matrix<int> m2{{{1}, {2}}};

    Matrix<int> result{m1 + m2};
    Matrix<int> expected{{{2, 3, 4}, {6, 7, 8}}};

    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, BroadcastingSubtraction) {
    Matrix<int> m1{{{2, 3, 4}, {6, 7, 8}}};
    Matrix<int> m2{{{1}, {2}}};

    Matrix<int> result{m1 - m2};
    Matrix<int> expected{{{1, 2, 3}, {4, 5, 6}}};

    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, UnaryExpr) {
    Matrix<int> m1{{{1, 2, 3}, {4, 5, 6}}};

    Matrix<int> result{m1.unary_expr([](int x) { return x * x; })};
    Matrix<int> expected{{{1, 4, 9}, {16, 25, 36}}};

    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, Sum) {
    Matrix<int> m{{{1, 2}, {3, 4}}};
    EXPECT_EQ(m.sum(), 10);
}

TEST(MatrixTest, FromRowVectors) {
    std::vector<Matrix<int>> row_vectors = {Matrix<int>{{{1, 2, 3}}},
                                            Matrix<int>{{{4, 5, 6}}},
                                            Matrix<int>{{{7, 8, 9}}}};

    auto result{Matrix<int>::from_row_vectors(row_vectors)};
    Matrix<int> expected{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
    EXPECT_EQ(result, expected);
}

}  // namespace micro_nn::linalg