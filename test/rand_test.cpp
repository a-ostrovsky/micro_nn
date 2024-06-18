#include "../src/rand.h"

#include <gtest/gtest.h>

namespace micro_nn::rand {

class SimpleLCGTest : public ::testing::Test {
protected:
    SimpleLCG<> lcg{42};
};

TEST_F(SimpleLCGTest, NextGeneratesWithinRange) {
    const float iterationCnt{1000};
    const float min{0.0f};
    const float max{1.0f};
    for (int i = 0; i < iterationCnt; ++i) {
        const float generated{lcg.next(min, max)};
        EXPECT_GE(generated, min);
        EXPECT_LT(generated, max);
    }
}

TEST_F(SimpleLCGTest, NextGeneratesDifferentValues) {
    const float iterationCnt{1000};
    const float min{0.0f};
    const float max{1.0f};
    float prev{lcg.next(min, max)};
    for (int i = 0; i < iterationCnt; ++i) {
        const float next{lcg.next(min, max)};
        EXPECT_NE(prev, next);
        prev = next;
    }
}
}  // namespace micro_nn::rand