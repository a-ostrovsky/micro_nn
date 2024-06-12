#pragma once
#include <ranges>

#include "config.h"
#include "utils.h"

namespace micro_nn::data {
template <std::ranges::input_range RangeT>
class DataLoader {
    using IteratorT = std::ranges::iterator_t<RangeT>;

public:
    constexpr DataLoader(std::ranges::input_range auto&& dataset,
                         std::size_t batch_size)
        : dataset_(std::forward<decltype(dataset)>(dataset)),
          batch_size_(batch_size),
          current_(begin(dataset_)) {}

    constexpr bool has_next() const { return current_ != dataset_.end(); }

    constexpr auto next() {
        auto start{current_};
        size_t max_possible_advance{
            narrow<size_t>(std::ranges::distance(current_, dataset_.end()))};
        auto advance_by{std::min(batch_size_, max_possible_advance)};
        auto end{std::next(current_, advance_by)};
        current_ = end;
        return std::ranges::subrange(start, end);
    }

private:
    RangeT dataset_;
    std::size_t batch_size_;
    IteratorT current_;
};

// Deduction guide
template <std::ranges::input_range RangeT>
DataLoader(RangeT&&, std::size_t) -> DataLoader<RangeT>;
}  // namespace micro_nn::data