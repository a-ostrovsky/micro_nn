#pragma once
#include <optional>
#include <random>
#include <ranges>

#include "config.h"
#include "linalg/matrix.h"
#include "rand.h"
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

struct SimpleDataLoaderSettings {
    size_t batch_size_{std::numeric_limits<std::size_t>::max()};
    bool shuffle_indices_{false};
    std::optional<rand::SimpleLCG::result_type> seed_{};
};

template <typename NumT = config::kFloat>
class SimpleDataLoader {
public:
    using ElementT = NumT;

    // TODO: Use ranges
    constexpr SimpleDataLoader(std::vector<linalg::Matrix<NumT>> x,
                               std::vector<linalg::Matrix<NumT>> y,
                               SimpleDataLoaderSettings settings)
        : x_(std::move(x)),
          y_(std::move(y)),
          batch_size_(settings.batch_size_),
          shuffle_indices_(settings.shuffle_indices_),
          seed_(settings.seed_),
          current_(0) {
        if (x_.size() != y_.size()) {
            throw std::invalid_argument(
                "x and y must have the same number of elements");
        }
        if (shuffle_indices_) {
            shuffle_indices(x_, y_);
        }
    }

    constexpr bool has_next() const { return current_ < x_.size(); }

    constexpr void reset() {
        current_ = 0;
        if (shuffle_indices_) {
            shuffle_indices(x_, y_);
        }
    }

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
    constexpr void shuffle_indices(std::vector<linalg::Matrix<NumT>>& x,
                                   std::vector<linalg::Matrix<NumT>>& y) {
        assert(x.size() == y.size());
        auto rng{seed_ ? rand::SimpleLCG{*seed_} : rand::SimpleLCG{}};
        auto dist{std::uniform_int_distribution<std::size_t>{0, x.size() - 1}};
        for (auto i{x.size() - 1}; i > 0; --i) {
            auto j{dist(rng)};
            std::swap(x[i], x[j]);
            std::swap(y[i], y[j]);
        }
    }

    std::vector<linalg::Matrix<NumT>> x_;
    std::vector<linalg::Matrix<NumT>> y_;
    std::size_t batch_size_;
    bool shuffle_indices_;
    std::optional<rand::SimpleLCG::result_type> seed_;
    std::size_t current_;
};
}  // namespace micro_nn::data