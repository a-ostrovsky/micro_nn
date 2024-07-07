#include "../src/lr_scheduler.h"

#include <gtest/gtest.h>

namespace micro_nn::lr_scheduler {
TEST(LRSchedulerTest, StepDecay) {
    StepDecay<> step_decay({.epoch_count_ = 2, .drop_factor_ = 0.1f});
    auto lr1{step_decay.get_lr({.epoch_ = 1, .lr_ = 1.0f})};
    EXPECT_FLOAT_EQ(lr1, 1.0f);  // Learning rate should remain unchanged
    auto lr2{step_decay.get_lr({.epoch_ = 2, .lr_ = 1.0f})};
    EXPECT_FLOAT_EQ(lr2, 0.1f);  // Learning rate should drop by 10%
    auto lr3{step_decay.get_lr({.epoch_ = 3, .lr_ = 1.0f})};
    EXPECT_FLOAT_EQ(lr3, 1.0f);  // Learning rate should remain unchanged
    auto lr4{step_decay.get_lr({.epoch_ = 4, .lr_ = 1.0f})};
    EXPECT_FLOAT_EQ(lr4, 0.1f);  // Learning rate should drop by 10%
}

TEST(ConstantRateSchedulerTest, NoDecay) {
    ConstantRateScheduler<> constant_rate_scheduler;
    auto lr{constant_rate_scheduler.get_lr({.epoch_ = 1, .lr_ = 1.0f})};
    EXPECT_FLOAT_EQ(lr, 1.0f);  // Learning rate should remain unchanged
}
}  // namespace micro_nn::lr_scheduler