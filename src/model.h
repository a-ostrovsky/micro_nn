#pragma once

#include <concepts>
#include <type_traits>

#include "config.h"
#include "layers.h"
#include "linalg/matrix.h"

namespace micro_nn::model {
template <class NumT, class T>
concept Model = requires(T model, const micro_nn::linalg::Matrix<NumT>& m) {
    { model.forward(m) } -> std::convertible_to<micro_nn::linalg::Matrix<NumT>>;
    {
        model.backward(m)
    } -> std::convertible_to<micro_nn::linalg::Matrix<NumT>>;
    { model.layers() };
};

template <class NumT = config::kFloat, class... Layers>
    requires(layers::Layer<NumT, Layers> && ...)
class SequentialModel {
public:
    constexpr explicit SequentialModel(Layers&&... layers)
        : layers_{std::move(layers)...} {}

    constexpr micro_nn::linalg::Matrix<NumT> forward(
        const micro_nn::linalg::Matrix<NumT>& x) {
        micro_nn::linalg::Matrix<NumT> y = x;
        std::apply([&](auto&... layer) { ((y = layer.forward(y)), ...); },
                   layers_);
        return y;
    }

    constexpr micro_nn::linalg::Matrix<NumT> backward(
        const micro_nn::linalg::Matrix<NumT>& d_out) {
        micro_nn::linalg::Matrix<NumT> d_y = d_out;
        auto reversed_layers{reverse_tuple(layers_)};
        std::apply([&](auto&... layer) { ((d_y = layer->backward(d_y)), ...); },
                   reversed_layers);
        return d_y;
    }

    constexpr auto& layers() { return layers_; }

private:
    // Inspired by:
    // https://www.reddit.com/r/cpp/comments/gu29m9/reversing_tuples_with_c_20/
    // Reverse the order of the tuple elements and returns pointer to the
    // original elements.
    template <class Tp, class TpNoRef = std::remove_reference_t<Tp>,
              std::size_t N = std::tuple_size<TpNoRef>::value,
              class Seq = std::make_index_sequence<N>>
    static constexpr auto reverse_tuple(Tp&& tp) {
        auto impl{[&tp]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(
                &std::get<N - 1 - I>(std::forward<Tp>(tp))...);
        }};
        return impl(Seq());
    }

    std::tuple<Layers...> layers_;
};

}  // namespace micro_nn::model