#pragma once
#include <atomic>
#include <cstdint>
#include <source_location>
#include <type_traits>

namespace micro_nn {
namespace detail_ {
template <int T>
struct Identity {
    static constexpr int value = T;
};
std::uint32_t get_next_at_runtime();
template <int T>
struct UniqueId {
    static constexpr std::uint32_t get_next() {
        return std::is_constant_evaluated() ? T : get_next_at_runtime();
    }
};
}  // namespace detail_
}  // namespace micro_nn

// Defined as a macro because I didn't found how to define it in constexpr
// method such that it is expanded at every call.
#define NEXT_UNIQUE_ID() micro_nn::detail_::UniqueId<__COUNTER__>::get_next()