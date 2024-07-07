#pragma once
#include <stdexcept>
#include <type_traits>

#include "config.h"

namespace micro_nn::lr_scheduler {
template <class NumT = config::kFloat>
struct GetLrParams {
    int epoch_{};
    NumT lr_{};
};

template <class NumT, class T>
concept LRScheduler = requires(T sched, GetLrParams<NumT> params) {
    { sched.get_lr(params) } -> std::convertible_to<NumT>;
};

template <class NumT = config::kFloat>
struct ConstantRateScheduler {
    constexpr NumT get_lr(GetLrParams<NumT> params) { return params.lr_; }
};

template <class NumT = config::kFloat>
struct StepDecayConfig {
    int epoch_count_{};
    NumT drop_factor_{};
};

template <class NumT = config::kFloat>
class StepDecay {
public:
    explicit constexpr StepDecay(StepDecayConfig<NumT> config)
        : epoch_count_{config.epoch_count_}, drop_factor_{config.drop_factor_} {
        if (epoch_count_ < 1) {
            throw std::invalid_argument("epoch_count must be greater than 0");
        }
    }

    constexpr NumT get_lr(GetLrParams<NumT> params) {
        return params.epoch_ % epoch_count_ == 0 ? params.lr_ * drop_factor_
                                                 : params.lr_;
    }

private:
    int epoch_count_;
    NumT drop_factor_;
};
}  // namespace micro_nn::lr_scheduler