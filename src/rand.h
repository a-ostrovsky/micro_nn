#pragma once
#include <cassert>
#include <cstdint>
#include <limits>

#include "config.h"

namespace micro_nn::rand {

template <typename NumT = config::kFloat>
class SimpleLCG {
public:
    explicit constexpr SimpleLCG(std::uint32_t seed = time_to_int(__TIME__))
        : state_(seed) {}

    constexpr NumT next(NumT minInclusive, NumT maxExclusive) {
        assert(minInclusive < maxExclusive && "Invalid range");
        state_ = (a_ * state_ + c_) % m_;
        return static_cast<NumT>(state_) / m_ * (maxExclusive - minInclusive) +
               minInclusive;
    }

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
}  // namespace micro_nn::rand