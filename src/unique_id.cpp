#include "unique_id.h"

#include <type_traits>
namespace micro_nn::detail_ {
namespace {
std::atomic<std::uint32_t> counter_{};
}  // namespace
std::uint32_t get_next_id_at_runtime() {
    // Relaxed memory order is sufficient for this use case because
    // we just need the next value of the counter. We don't care about
    // the order of other memory operations.
    return counter_.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace micro_nn::detail_