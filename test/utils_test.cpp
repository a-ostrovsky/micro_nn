#include "../src/utils.h"  // Include the header file where sqrt is defined

#include <gtest/gtest.h>

namespace micro_nn {

TEST(UtilsTest, Sqrt) {
    EXPECT_NEAR(micro_nn::sqrt(4.0), 2.0, kSqrtEpsilon);
    EXPECT_NEAR(micro_nn::sqrt(9.0), 3.0, kSqrtEpsilon);
    EXPECT_NEAR(micro_nn::sqrt(0.0), 0.0, kSqrtEpsilon);
    EXPECT_TRUE(std::isnan(micro_nn::sqrt(-1.0)));

    EXPECT_EQ(micro_nn::sqrt<int>(9), 3);
    EXPECT_EQ(micro_nn::sqrt<int>(-9), std::numeric_limits<int>::quiet_NaN());
}
}  // namespace micro_nn