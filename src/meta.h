#pragma once

// Meta-programming utilities
namespace micro_nn::meta {

// Checks if the given expression can be evaluated at compile-time. This works
// by using the fact that FuncT{}() can only be used as template argument if
// it can be evaluated at compile-time.
template <class FuncT, void* = FuncT{}()>
constexpr bool is_constexpr(FuncT) {
    return true;
}
constexpr bool is_constexpr(...) { return false; }
}  // namespace micro_nn::meta