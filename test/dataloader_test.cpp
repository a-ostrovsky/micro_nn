#include "../src/dataloader.h"

#include <gtest/gtest.h>

#include <ranges>
#include <vector>

namespace micro_nn::data {

TEST(SimpleDataLoaderTest, TestNext) {
    using namespace linalg;
    std::vector x = {Matrix{1}, Matrix{2}, Matrix{3}, Matrix{4}, Matrix{5}};
    auto y = x;
    SimpleDataLoader dataloader(x, y, 3);

    auto batch1 = dataloader.next();
    ASSERT_EQ(batch1.x.rows(), 3);
    ASSERT_EQ(batch1.y.rows(), 3);
    ASSERT_EQ(batch1.x.at(0, 0), 1);
    ASSERT_EQ(batch1.x.at(1, 0), 2);
    ASSERT_EQ(batch1.x.at(2, 0), 3);

    auto batch2 = dataloader.next();
    ASSERT_EQ(batch2.x.rows(), 2);
    ASSERT_EQ(batch2.y.rows(), 2);
    ASSERT_EQ(batch2.x.at(0, 0), 4);
    ASSERT_EQ(batch2.x.at(1, 0), 5);

    ASSERT_FALSE(dataloader.has_next());
}
}  // namespace micro_nn::data