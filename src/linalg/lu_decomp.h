#pragma once
#include <cassert>
#include <stdexcept>

#include "../config.h"
#include "matrix.h"

namespace micro_nn::linalg {
template <class NumT = config::kFloat>
struct LUDecomposition {
    Matrix<NumT> l_;
    Matrix<NumT> u_;
};
template <class NumT = config::kFloat>
constexpr LUDecomposition<NumT> lu_factor(const Matrix<NumT>& m) {
    const auto rows{m.rows()};
    const auto cols{m.cols()};
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square.");
    }
    const auto n = rows;
    Matrix<NumT> l{Matrix<NumT>::identity(n)};
    Matrix<NumT> u{m};
    for (int row_from_u{0}; row_from_u < n; ++row_from_u) {
        for (int row_to_u{row_from_u + 1}; row_to_u < n; ++row_to_u) {
            auto col_u{row_from_u};
            const auto col_l{row_from_u};
            const auto row_l{row_to_u};
            assert(u.at(row_from_u, col_u) != NumT{});
            const auto factor{u.at(row_to_u, col_u) / u.at(row_from_u, col_u)};
            l.at(row_l, col_l) = factor;
            for (; col_u < n; ++col_u) {
                u.at(row_to_u, col_u) -=
                    l.at(row_l, col_l) * u.at(row_from_u, col_u);
            }
        }
    }
    return {l, u};
}
}  // namespace micro_nn::linalg