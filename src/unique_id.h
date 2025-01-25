#pragma once
#include <atomic>
#include <cstdint>
#include <source_location>
#include <type_traits>

// Inspired by
// https://mc-deltat.github.io/articles/stateful-metaprogramming-cpp20
// Probably a non standard compliant hack which shouldn't be used. Used only for
// experemental purposes.

namespace micro_nn {
namespace detail_ {
std::uint32_t get_next_id_at_runtime();

// Reader creates a method in the micro_nn::detail_ namespace and only
// accessible via ADL because it is a friend of Reader. It is later
// defined in the Writer class. This method can be considered as a flag, either
// defined or not defined. It will only be defined when a template
// specialization of Writer is instantiated. Note that the template parameter is
// an unsigned. It will be used as a counter.
template <unsigned N>
struct Reader {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif
    friend auto counted_flag(Reader<N>);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
};

// When instantiating a specialization for Writer with a given N, the function
// counted_flag will be defined. For example, when we have instantiated
// Writer<0>, Writer<1> and Writer<2> then the counted_flag<0>, counted_flag<1>
// and counted_flag<2> is defined while counted_flag<3> etc are not yet defined.
template <unsigned N>
struct Writer {
    friend auto counted_flag(Reader<N>) {}

    static constexpr unsigned n = N;
};

// Here we will use search for the first value of N for which counted_flag
// is defined in the corresponding Reader<N> class.
// This is done by recursively calling counter_impl until the counted_flag is
// not defined. Tag should be set to an empty lambda. This forces to
// re-instantiate the template every time this method is called because every
// time a new lambda is created.
template <auto Tag, unsigned N = 0>
[[nodiscard]]
consteval auto counter_impl() {
    // Check if the counted_flag is defined for the current value of N.
    constexpr bool value_already_counted =
        requires(Reader<N> r) { counted_flag(r); };

    if constexpr (value_already_counted) {
        // If the counted_flag is defined for the current value of N, then
        // we recursively call counter_impl with the next value of N.
        return counter_impl<Tag, N + 1>();
    } else {
        Writer<N> s;
        return s.n;
    }
}

template <auto Tag = [] {}, auto Val = counter_impl<Tag>()>
constexpr auto get_next_id_at_compile_time = Val;

}  // namespace detail_

// Returns a unique ID. For constexpr contexts, a new ID is generated every time
// the method is called. So, it means that it shouldn't be called from within
// loops. In case of non constexpr evaluation the ID is unique regardless how
// the method is called.
template <auto Tag = [] {}>
static constexpr std::uint32_t get_next_id() {
    return std::is_constant_evaluated()
               ? detail_::get_next_id_at_compile_time<Tag>
               : detail_::get_next_id_at_runtime();
}
}  // namespace micro_nn
