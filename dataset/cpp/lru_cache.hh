#pragma once

#include <deque>
#include <unordered_map>

template <typename K, typename V>
struct lru_cache {

  lru_cache(int max_size) : max_size_(max_size) {}

  template <typename F = int>
  auto operator()(K key, F fallback = 0)
  {
    auto it = entries_.find(key);
    if (it != entries_.end())
      return it->second;

    if (entries_.size() >= max_size_)
    {
      K k = queue_.front();
      queue_.pop_front();
      entries_.erase(k);
    }
    assert(entries_.size() < max_size_);
    if constexpr (std::is_same_v<F, int>)
    {
      throw std::runtime_error("Error: lru_cache miss with no fallback");
    }
    else
    {
      V res = fallback();
      entries_.emplace(key, res);
      queue_.push_back(key);
      return res;
    }
  }

  int size() { return queue_.size(); }
  void clear() { queue_.clear(); entries_.clear(); }
  
  int max_size_;
  std::deque<K> queue_;
  std::unordered_map<K, V> entries_;
};
