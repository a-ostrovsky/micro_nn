#include "../src/data_structures/hash_map.h"

#include <gtest/gtest.h>

namespace micro_nn::data_structures {
TEST(HashMapTest, InsertAndRetrieve) {
    micro_nn::data_structures::HashMap<int, std::string> map{};

    // Test insertion
    map[1] = "one";
    map[2] = "two";

    ASSERT_EQ(map.size(), 2);
    EXPECT_EQ(map[1], "one");
    EXPECT_EQ(map[2], "two");

    // Test retrieval of a non-existent key
    EXPECT_EQ(map[3], std::string{});
}

TEST(HashMapTest, Overwrite) {
    micro_nn::data_structures::HashMap<int, std::string> map{};

    map[1] = "X";
    map[1] = "Y";

    EXPECT_EQ(map[1], "Y");
}

TEST(HashMapTest, Collision) {
    // Create a hash map with a size of 1 to force a collision
    micro_nn::data_structures::HashMap<int, std::string> map{1};

    map[1] = "one";
    map[2] = "two";
    map[3] = "three";
    map[4] = "four";

    ASSERT_EQ(map.size(), 4);
    EXPECT_EQ(map[1], "one");
    EXPECT_EQ(map[2], "two");
    EXPECT_EQ(map[3], "three");
    EXPECT_EQ(map[4], "four");
}
}  // namespace micro_nn::data_structures