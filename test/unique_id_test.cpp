#include "../src/unique_id.h"

#include <gtest/gtest.h>

#include <unordered_set>

namespace micro_nn {
namespace {
// Subsequent calls to NEXT_UNIQUE_ID() should return different values.
static_assert(NEXT_UNIQUE_ID() != NEXT_UNIQUE_ID());
}  // namespace

TEST(UniqueIdTest, ReturnsUniqueAndSequentialIds) {
    std::unordered_set<std::uint32_t> ids;
    const size_t num_ids{10};  // Number of IDs to generate and test
    for (size_t i{0}; i < num_ids; ++i) {
        auto id{NEXT_UNIQUE_ID()};
        ASSERT_TRUE(ids.insert(id).second);
    }
}
}  // namespace micro_nn