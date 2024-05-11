#pragma once

#include <memory>

#include "model.h"

namespace micro_nn {
template <class NumT = config::kFloat, class... Models>
    requires(Model<NumT, Models> && ...)
class SequentialModel {
public:
    constexpr explicit SequentialModel(Models... models) : models_{models...} {}

    constexpr micro_nn::linalg::Matrix<NumT> forward(
        const micro_nn::linalg::Matrix<NumT>& x) const {
        throw std::exception("not implemented");
    }
    micro_nn::linalg::Matrix<NumT> backward(
        const micro_nn::linalg::Matrix<NumT>& d_out) const {
        throw std::exception("not implemented");
    }

private:
    std::tuple<Models...> models_;
};
}  // namespace micro_nn