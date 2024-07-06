#pragma once
#include <cstdint>
#include <limits>
#include <random>

#include "config.h"

// Provides utilities for compile-time random number generation. It relies on
// the __TIME__ macro, which provides the current time at the point of
// compilation.
namespace micro_nn::rand {

class SimpleLCG {
public:
    using result_type = std::uint32_t;

    explicit constexpr SimpleLCG(std::uint32_t seed = time_to_int(__TIME__))
        : state_(seed) {}

    constexpr std::uint32_t operator()() {
        state_ = (a_ * state_ + c_) % m_;
        return state_;
    }

    constexpr std::uint32_t get_seed() const { return state_; }

    constexpr void seed(std::uint32_t new_seed) { state_ = new_seed; }

    constexpr static std::uint32_t min() { return 0; }

    constexpr static std::uint32_t max() { return m_ - 1; }

private:
    constexpr static std::uint32_t time_to_int(const char* time) {
        return (static_cast<std::uint32_t>(time[0] - '0') * 10 +
                (time[1] - '0')) *
                   3600 +
               (static_cast<std::uint32_t>(time[3] - '0') * 10 +
                (time[4] - '0')) *
                   60 +
               (static_cast<std::uint32_t>(time[6] - '0') * 10 +
                (time[7] - '0'));
    }

    std::uint32_t state_;
    // See https://en.wikipedia.org/wiki/Linear_congruential_generator for
    // explanation of a (multiplier), c (increment) and m (modulus)
    static constexpr uint32_t a_{1664525};
    static constexpr uint32_t c_{1013904223};
    static constexpr uint32_t m_{std::numeric_limits<uint32_t>::max()};
};

template <typename NumT = config::kFloat>
class UniformRealDistribution {
public:
    explicit constexpr UniformRealDistribution(NumT min = 0.0, NumT max = 1.0)
        : min_(min), max_(max) {}

    template <std::uniform_random_bit_generator EngineT>
    constexpr NumT operator()(EngineT& eng) {
        return min_ + (max_ - min_) * eng() / eng.max();
    }

    constexpr NumT min() const { return min_; }

    constexpr NumT max() const { return max_; }

private:
    NumT min_;
    NumT max_;
};

}  // namespace micro_nn::rand