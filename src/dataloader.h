#pragma once
#include <ranges>

#include "config.h"
#include "linalg.h"
#include "utils.h"

namespace micro_nn::data {
template <typename NumT = config::kFloat>
struct Data {
    linalg::Matrix<NumT> x;
    linalg::Matrix<NumT> y;
};

template <class NumT, typename T>
concept DataLoader = requires(T a) {
    { a.has_next() } -> std::convertible_to<bool>;
    { a.next() } -> std::convertible_to<Data<NumT>>;
    { a.reset() };
};

template <typename NumT = config::kFloat>
class SimpleDataLoader {
public:
    using ElementT = NumT;

    // TODO: Use ranges
    constexpr SimpleDataLoader(std::vector<linalg::Matrix<NumT>> x,
                               std::vector<linalg::Matrix<NumT>> y,
                               std::size_t batch_size)
        : x_(std::move(x)),
          y_(std::move(y)),
          batch_size_(batch_size),
          current_(0) {
        if (x_.size() != y_.size()) {
            throw std::invalid_argument(
                "x and y must have the same number of elements");
        }
    }

    constexpr bool has_next() const { return current_ < x_.size(); }

    constexpr void reset() { current_ = 0; }

    constexpr Data<NumT> next() {
        if (!has_next()) {
            throw std::out_of_range("No more batches available");
        }

        std::size_t batch_end = std::min(current_ + batch_size_, x_.size());
        std::vector<linalg::Matrix<NumT>> batch_x(x_.begin() + current_,
                                                  x_.begin() + batch_end);
        std::vector<linalg::Matrix<NumT>> batch_y(y_.begin() + current_,
                                                  y_.begin() + batch_end);

        current_ = batch_end;

        return {linalg::Matrix<NumT>::from_row_vectors(batch_x),
                linalg::Matrix<NumT>::from_row_vectors(batch_y)};
    }

private:
    std::vector<linalg::Matrix<NumT>> x_;
    std::vector<linalg::Matrix<NumT>> y_;
    std::size_t batch_size_;
    std::size_t current_;
};
}  // namespace micro_nn::data