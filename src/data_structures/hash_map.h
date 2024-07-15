#pragma once

#include <cassert>
#include <optional>
#include <ranges>
#include <vector>

namespace micro_nn::data_structures {
// a constexpr version of a hash map
template <std::integral Key, typename Value>
class HashMap {
public:
    constexpr HashMap(size_t initial_size = 32)
        : table1_{initial_size}, table2_{initial_size} {
        assert(initial_size > 0);
    }

    constexpr Value& operator[](const Key& key) {
        size_t hash1{key % table1_.size()};
        if (!table1_[hash1]) {
            table1_[hash1] = {key, Value()};
            ++size_;
            return table1_[hash1]->second;
        }
        if (table1_[hash1]->first == key) {
            return table1_[hash1]->second;
        }

        size_t hash2{key % table2_.size()};
        if (!table2_[hash2]) {
            table2_[hash2] = {key, Value()};
            ++size_;
            return table2_[hash2]->second;
        }
        if (table2_[hash2]->first == key) {
            return table2_[hash2]->second;
        }
        rehash();
        return operator[](key);
    }

    constexpr bool contains(const Key& key) const {
        size_t hash1{key % table1_.size()};
        if (table1_[hash1] && table1_[hash1]->first == key) {
            return true;
        }
        size_t hash2{key % table2_.size()};
        if (table2_[hash2] && table2_[hash2]->first == key) {
            return true;
        }
        return false;
    }

    constexpr size_t size() const { return size_; }

private:
    constexpr void rehash() {
        auto old_table1{std::move(table1_)};
        auto old_table2{std::move(table2_)};
        table1_.resize(2 * old_table1.size());
        table2_.resize(2 * old_table2.size());
        size_ = 0;

        auto reinsert{[this](auto& table) {
            for (auto& entry : table) {
                if (entry) {
                    operator[](std::move(entry->first)) =
                        std::move(entry->second);
                }
            }
        }};

        reinsert(old_table1);
        reinsert(old_table2);
    }

    std::vector<std::optional<std::pair<Key, Value>>> table1_;
    std::vector<std::optional<std::pair<Key, Value>>> table2_;
    size_t size_{};
};
}  // namespace micro_nn::data_structures