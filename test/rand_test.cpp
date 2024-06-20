#include "../src/rand.h"

#include <gtest/gtest.h>

#include <random>

namespace micro_nn::rand {

TEST(RandTest, NextGeneratesWithinRange) {
    SimpleLCG lcg{};
    auto runTest = [&](auto dist) {
        const float iterationCnt{1000};
        const float min{0.0f};
        const float max{1.0f};
        for (int i = 0; i < iterationCnt; ++i) {
            const float generated{dist(lcg)};
            EXPECT_GE(generated, min);
            EXPECT_LT(generated, max);
        }
    };

    runTest(std::uniform_real_distribution<float>(0.0f, 1.0f));
    runTest(UniformRealDistribution<float>(0.0f, 1.0f));
}

TEST(RandTest, NextGeneratesDifferentValues) {
    SimpleLCG lcg{};
    auto runTest = [&](auto dist) {
        const float iterationCnt{1000};
        const float min{0.0f};
        const float max{1.0f};
        float prev{dist(lcg)};
        for (int i = 0; i < iterationCnt; ++i) {
            const float next{dist(lcg)};
            EXPECT_NE(prev, next);
            prev = next;
        }
    };

    runTest(std::uniform_real_distribution<float>(0.0f, 1.0f));
    runTest(UniformRealDistribution<float>(0.0f, 1.0f));
}
}  // namespace micro_nn::rand