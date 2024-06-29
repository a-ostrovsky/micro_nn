#include "../src/dataloader.h"

#include <gtest/gtest.h>

#include <ranges>
#include <vector>

namespace micro_nn::data {

TEST(SimpleDataLoaderTest, TestNext) {
    using namespace linalg;
    std::vector x = {Matrix{1}, Matrix{2}, Matrix{3}, Matrix{4}, Matrix{5}};
    auto y = x;
    SimpleDataLoader dataloader(x, y, {.batch_size_ = 3});

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

TEST(SimpleDataLoaderTest, TestShuffleIndices) {
    using namespace linalg;
    std::vector x = {Matrix{1}, Matrix{2}, Matrix{3}, Matrix{4}, Matrix{5}};
    auto y = x;
    SimpleDataLoader dataloader(
        x, y, {.batch_size_ = 3, .shuffle_indices_ = true, .seed_ = 1});

    auto batch1 = dataloader.next();

    // Without shuffling the indices of the data, the first batch are 1, 2, 3.
    // With shuffling it is usually something else. With the fixed seed it will
    // be different.
    ASSERT_TRUE(batch1.x.at(0, 0) != 1 && batch1.x.at(1, 0) != 2 &&
                batch1.x.at(2, 0) != 3);
}

}  // namespace micro_nn::data