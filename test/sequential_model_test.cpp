#include "../src/sequential_model.h"

#include <gtest/gtest.h>

#include "../src/layers.h"

namespace micro_nn {

TEST(SequentialModelTest, Creation) {
    // TODO: This is currently just a scaffold
    SequentialModel model{layers::Linear(2, 2), layers::ReLU()};
}

}  // namespace micro_nn