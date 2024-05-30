#pragma once

#include <array>
#include <cassert>
#include <format>
#include <mdspan>
#include <numeric>
#include <vector>

#include "config.h"
#include "utils.h"

namespace micro_nn::linalg {
template <class NumT = config::kFloat>
class Matrix {
    using MdSpanT = std::mdspan<
        NumT, std::extents<int, std::dynamic_extent, std::dynamic_extent>>;

public:
    constexpr Matrix() = default;

    constexpr Matrix(int rows, int cols) {
        data_.resize(rows * cols);
        dataSpan_ = MdSpanT(data_.data(), rows, cols);
    }

    explicit Matrix(const std::vector<std::vector<NumT>>& data) {
        const int rows{narrow_cast<int>(data.size())};
        const int cols{narrow_cast<int>(data[0].size())};
        data_.resize(rows * cols);
        dataSpan_ = MdSpanT(data_.data(), rows, cols);

        for (int row = 0; row < rows; ++row) {
            if (data[row].size() != cols) {
                throw std::invalid_argument(
                    "All rows must have the same number of columns");
            }

            for (int col = 0; col < cols; ++col) {
                dataSpan_[std::array{row, col}] = data[row][col];
            }
        }
    }

    constexpr static Matrix<NumT> unity(int size) {
        assert(size >= 0 && "Size must be non-negative");
        Matrix<NumT> matrix(size, size);
        for (auto i = 0; i < size; ++i) {
            matrix.dataSpan_[std::array{i, i}] = 1;
        }
        return matrix;
    }

    constexpr static Matrix<NumT> zeros(int rows, int cols) {
        assert(rows >= 0 && cols >= 0 &&
               "Rows and columns must be non-negative");
        Matrix<NumT> matrix(rows, cols);
        for (auto row = 0; row < rows; ++row) {
            for (auto col = 0; col < cols; ++col) {
                matrix.dataSpan_[std::array{row, col}] = 0;
            }
        }
        return matrix;
    }

    constexpr Matrix operator*(const Matrix& other) const {
        if (cols() != other.rows()) {
            throw std::invalid_argument(
                std::format("Matrix dimensions do not match for "
                            "multiplication. {}<->{}",
                            cols(), other.rows()));
        }

        Matrix result{rows(), other.cols()};
        for (auto row{0}; row < rows(); ++row) {
            for (auto col{0}; col < other.cols(); ++col) {
                NumT sum = 0;
                for (auto k = 0; k < cols(); ++k) {
                    sum += dataSpan_[std::array{row, k}] *
                           other.dataSpan_[std::array{k, col}];
                }
                result.dataSpan_[std::array{row, col}] = sum;
            }
        }
        return result;
    }

    constexpr Matrix& operator*=(Matrix other) {
        *this = std::move(*this * other);
        return *this;
    }

    constexpr Matrix operator+(const Matrix& other) const {
        if (rows() != other.rows() && other.rows() != 1 &&
            cols() != other.cols() && other.cols() != 1) {
            throw std::invalid_argument(
                "Matrix dimensions do not match for addition and cannot be "
                "broadcasted");
        }

        Matrix result(rows(), cols());
        for (int row = 0; row < rows(); ++row) {
            for (int col = 0; col < cols(); ++col) {
                result.dataSpan_[std::array{row, col}] =
                    dataSpan_[std::array{row, col}] +
                    other
                        .dataSpan_[std::array{std::min(row, other.rows() - 1),
                                              std::min(col, other.cols() - 1)}];
            }
        }
        return result;
    }

    constexpr Matrix operator-(const Matrix& other) const {
        if (rows() != other.rows() && other.rows() != 1 &&
            cols() != other.cols() && other.cols() != 1) {
            throw std::invalid_argument(
                "Matrix dimensions do not match for subtraction and cannot be "
                "broadcasted");
        }

        Matrix result(rows(), cols());
        for (int row = 0; row < rows(); ++row) {
            for (int col = 0; col < cols(); ++col) {
                result.dataSpan_[std::array{row, col}] =
                    dataSpan_[std::array{row, col}] -
                    other
                        .dataSpan_[std::array{std::min(row, other.rows() - 1),
                                              std::min(col, other.cols() - 1)}];
            }
        }
        return result;
    }

    constexpr Matrix& operator+=(Matrix other) {
        *this = std::move(*this + other);
        return *this;
    }

    constexpr Matrix& operator-=(Matrix other) {
        *this = std::move(*this - other);
        return *this;
    }

    constexpr Matrix elementwise_multiply(const Matrix& other) const {
        if (rows() != other.rows() || cols() != other.cols()) {
            throw std::invalid_argument(
                std::format("Matrix dimensions do not match for "
                            "element-wise multiplication. {}x{}<->{}x{}",
                            rows(), cols(), other.rows(), other.cols()));
        }

        Matrix result{rows(), cols()};
        for (int row = 0; row < rows(); ++row) {
            for (int col = 0; col < cols(); ++col) {
                result.dataSpan_[std::array{row, col}] =
                    dataSpan_[std::array{row, col}] *
                    other.dataSpan_[std::array{row, col}];
            }
        }
        return result;
    }

    template <typename FuncT>
    constexpr Matrix unary_expr(const FuncT& func) const {
        Matrix result{rows(), cols()};
        for (int row = 0; row < rows(); ++row) {
            for (int col = 0; col < cols(); ++col) {
                result.dataSpan_[std::array{row, col}] =
                    func(dataSpan_[std::array{row, col}]);
            }
        }
        return result;
    }

    constexpr const NumT& at(int row, int col) const {
        if (!is_valid_index(row, col)) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return dataSpan_[std::array{row, col}];
    }

    constexpr NumT& at(int row, int col) {
        if (!is_valid_index(row, col)) {
            throw std::out_of_range("Matrix indices out of range");
        }
        const auto idx{row * cols() + col};
        return data_[idx];
        // TODO: Why this doesn't work?
        // return dataSpan_[std::array{row, col}];
    }

    constexpr Matrix transpose() const {
        Matrix result(cols(), rows());
        for (int row = 0; row < rows(); ++row) {
            for (int col = 0; col < cols(); ++col) {
                result.dataSpan_[std::array{col, row}] =
                    dataSpan_[std::array{row, col}];
            }
        }
        return result;
    }

    constexpr Matrix<NumT> rowwise(int row) const {
        Matrix<NumT> row_data{1, cols()};
        for (int col = 0; col < cols(); ++col) {
            row_data(0, col) = dataSpan_[std::array{row, col}];
        }
        return row_data;
    }

    constexpr Matrix<NumT> colwise(int col) const {
        Matrix<NumT> col_data{rows(), 1};
        for (int row = 0; row < rows(); ++row) {
            col_data(row, 0) = dataSpan_[std::array{row, col}];
        }
        return col_data;
    }

    constexpr NumT sum() const {
        NumT sum{std::accumulate(data_.begin(), data_.end(), NumT{0})};
        return sum;
    }

    constexpr Matrix<NumT> rowwise_sum() const {
        Matrix<NumT> row_sums{rows(), 1};
        for (int row = 0; row < rows(); ++row) {
            NumT sum = 0;
            for (int col = 0; col < cols(); ++col) {
                sum += dataSpan_[std::array{row, col}];
            }
            row_sums.dataSpan_[std::array{row, 0}] = sum;
        }
        return row_sums;
    }

    constexpr Matrix<NumT> colwise_sum() const {
        Matrix<NumT> col_sums{1, cols()};
        for (int col = 0; col < cols(); ++col) {
            NumT sum = 0;
            for (int row = 0; row < rows(); ++row) {
                sum += dataSpan_[std::array{row, col}];
            }
            col_sums.dataSpan_[std::array{0, col}] = sum;
        }
        return col_sums;
    }

    constexpr int rows() const { return dataSpan_.extent(0); }

    constexpr int cols() const { return dataSpan_.extent(1); }

    constexpr size_t size() const { return data_.size(); }

    constexpr std::pair<int, int> shape() const { return {rows(), cols()}; }

    constexpr bool operator==(const Matrix& other) const {
        if (rows() != other.rows() || cols() != other.cols()) {
            return false;
        }

        return other.data_ == data_;
    }

    constexpr bool operator!=(const Matrix& other) const {
        return !(*this == other);
    }

private:
    constexpr bool is_valid_index(int row, int col) const {
        return row >= 0 && row < rows() && col >= 0 && col < cols();
    }

    std::vector<NumT> data_{};
    MdSpanT dataSpan_{};
};
}  // namespace micro_nn::linalg