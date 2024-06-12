#include "../src/dataloader.h"

#include <gtest/gtest.h>

#include <ranges>
#include <vector>

namespace micro_nn::data {

TEST(DataLoaderTest, TestNext) {
    std::vector data = {1, 2, 3, 4, 5};
    DataLoader dataloader(data, 3);

    auto batch1 = dataloader.next();
    ASSERT_EQ(batch1.size(), 3);
    ASSERT_EQ(batch1[0], 1);
    ASSERT_EQ(batch1[1], 2);
    ASSERT_EQ(batch1[2], 3);

    auto batch2 = dataloader.next();
    ASSERT_EQ(batch2.size(), 2);
    ASSERT_EQ(batch2[0], 4);
    ASSERT_EQ(batch2[1], 5);

    ASSERT_FALSE(dataloader.has_next());
}
}