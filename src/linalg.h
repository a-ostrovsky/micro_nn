#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numeric>
#include <vector>

#include "config.h"
#include "utils.h"

namespace micro_nn::linalg {
template <class NumT = config::kFloat>
class Matrix {
public:
    constexpr Matrix() = default;

    constexpr Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
        data_.resize(rows * cols);
    }

    constexpr explicit Matrix(NumT scalar) : Matrix(1, 1) { data_[0] = scalar; }

    constexpr explicit Matrix(const std::vector<std::vector<NumT>>& data) {
        rows_ = narrow_cast<int>(data.size());
        cols_ = narrow_cast<int>(data[0].size());
        data_.resize(rows_ * cols_);

        for (int row = 0; row < rows_; ++row) {
            if (data[row].size() != cols_) {
                throw std::invalid_argument(
                    "All rows must have the same number of columns");
            }

            for (int col = 0; col < cols_; ++col) {
                data_[index(row, col)] = data[row][col];
            }
        }
    }

    constexpr static Matrix<NumT> zeros(int rows, int cols) {
        assert(rows >= 0 && cols >= 0 &&
               "Rows and columns must be non-negative");
        Matrix<NumT> matrix(rows, cols);
        for (auto row = 0; row < rows; ++row) {
            for (auto col = 0; col < cols; ++col) {
                matrix.data_[matrix.index(row, col)] = 0;
            }
        }
        return matrix;
    }

    constexpr static Matrix<NumT> unity(int size) {
        assert(size >= 0 && "Size must be non-negative");
        Matrix<NumT> matrix(size, size);
        for (auto i = 0; i < size; ++i) {
            matrix.data_[matrix.index(i, i)] = 1;
        }
        return matrix;
    }

    // TODO: Use range for the input type
    static constexpr Matrix<NumT> from_row_vectors(
        const std::vector<Matrix<NumT>>& row_vectors) {
        if (std::ranges::empty(row_vectors)) {
            return Matrix<NumT>{0, 0};
        }
        int cols = std::ranges::begin(row_vectors)->cols();
        int total_rows = narrow<int>(
            std::ranges::distance(row_vectors.begin(), row_vectors.end()));

        Matrix<NumT> result(total_rows, cols);

        int row = 0;
        for (auto& vec : row_vectors) {
            if (vec.rows() != 1 || vec.cols() != cols) {
                throw std::invalid_argument(
                    "All row vectors must have the same number of columns and "
                    "a single row");
            }
            for (auto col{0}; col < cols; ++col) {
                result.at(row, col) = vec.at(0, col);
            }
            ++row;
        }

        return result;
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
                    sum +=
                        data_[index(row, k)] * other.data_[other.index(k, col)];
                }
                result.data_[result.index(row, col)] = sum;
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
                result.data_[index(row, col)] =
                    data_[index(row, col)] +
                    other.data_[other.index(std::min(row, other.rows() - 1),
                                            std::min(col, other.cols() - 1))];
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
                result.data_[index(row, col)] =
                    data_[index(row, col)] -
                    other.data_[other.index(std::min(row, other.rows() - 1),
                                            std::min(col, other.cols() - 1))];
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

    constexpr Matrix operator+(NumT scalar) const {
        return unary_expr([scalar](NumT x) { return x + scalar; });
    }

    constexpr Matrix& operator+=(NumT scalar) {
        *this = std::move(*this + scalar);
        return *this;
    }
    constexpr Matrix operator-(NumT scalar) const { return operator+(-scalar); }

    constexpr Matrix& operator-=(NumT scalar) {
        *this = std::move(*this - scalar);
        return *this;
    }

    constexpr Matrix operator*(NumT scalar) const {
        return unary_expr([scalar](NumT x) { return x * scalar; });
    }

    constexpr Matrix& operator*=(NumT scalar) {
        *this = std::move(*this * scalar);
        return *this;
    }

    constexpr Matrix operator/(NumT scalar) const {
        return operator*(1 / scalar);
    }

    constexpr Matrix& operator/=(NumT scalar) {
        *this = std::move(*this / scalar);
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
                result.data_[result.index(row, col)] =
                    data_[index(row, col)] * other.data_[other.index(row, col)];
            }
        }
        return result;
    }

    template <typename FuncT>
    constexpr Matrix unary_expr(const FuncT& func) const {
        Matrix result{rows(), cols()};
        for (int row = 0; row < rows(); ++row) {
            for (int col = 0; col < cols(); ++col) {
                result.data_[result.index(row, col)] =
                    func(data_[index(row, col)]);
            }
        }
        return result;
    }

    constexpr Matrix operator-() const {
        return unary_expr([](NumT x) { return -x; });
    }

    constexpr const NumT& at(int row, int col) const {
        return data_[index_checked(row, col)];
    }

    constexpr NumT& at(int row, int col) {
        return data_[index_checked(row, col)];
    }

    constexpr Matrix transpose() const {
        Matrix result(cols(), rows());
        for (int row = 0; row < rows(); ++row) {
            for (int col = 0; col < cols(); ++col) {
                result.data_[result.index(col, row)] = data_[index(row, col)];
            }
        }
        return result;
    }

    constexpr Matrix<NumT> rowwise(int row) const {
        Matrix<NumT> row_data{1, cols()};
        for (int col = 0; col < cols(); ++col) {
            row_data(0, col) = data_[index(row, col)];
        }
        return row_data;
    }

    constexpr Matrix<NumT> colwise(int col) const {
        Matrix<NumT> col_data{rows(), 1};
        for (int row = 0; row < rows(); ++row) {
            col_data(row, 0) = data_[index(row, col)];
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
                sum += data_[index(row, col)];
            }
            row_sums.data_[row_sums.index(row, 0)] = sum;
        }
        return row_sums;
    }

    constexpr Matrix<NumT> colwise_sum() const {
        Matrix<NumT> col_sums{1, cols()};
        for (int col = 0; col < cols(); ++col) {
            NumT sum = 0;
            for (int row = 0; row < rows(); ++row) {
                sum += data_[index(row, col)];
            }
            col_sums.data_[col_sums.index(0, col)] = sum;
        }
        return col_sums;
    }

    constexpr int rows() const { return rows_; }

    constexpr int cols() const { return cols_; }

    constexpr std::pair<int, int> shape() const { return {rows(), cols()}; }

    constexpr size_t size() const { return data_.size(); }

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
    constexpr int index(int row, int col) const { return row * cols() + col; }

    constexpr int index_checked(int row, int col) const {
        if (row >= rows() || col >= cols()) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return index(row, col);
    }

    std::vector<NumT> data_{};
    int rows_{};
    int cols_{};
};

template <typename NumT>
constexpr Matrix<NumT> operator+(NumT scalar, const Matrix<NumT>& matrix) {
    return matrix + scalar;
}

template <typename NumT>
constexpr Matrix<NumT> operator-(NumT scalar, const Matrix<NumT>& matrix) {
    return matrix - scalar;
}

template <typename NumT>
constexpr Matrix<NumT> operator*(NumT scalar, const Matrix<NumT>& matrix) {
    return matrix * scalar;
}

template <typename NumT>
constexpr Matrix<NumT> operator/(NumT scalar, const Matrix<NumT>& matrix) {
    return matrix / scalar;
}

template <typename NumT>
constexpr Matrix<NumT> log(const Matrix<NumT>& matrix) {
    return matrix.unary_expr([](NumT x) { return std::log(x); });
}

template <typename NumT>
constexpr Matrix<NumT> clamp(const Matrix<NumT>& matrix, NumT min, NumT max) {
    return matrix.unary_expr(
        [min, max](NumT x) { return std::clamp(x, min, max); });
}

}  // namespace micro_nn::linalg