#pragma once

#include <ostream>
#include <memory>

#include <jank/runtime/hash.hpp>
#include <jank/runtime/detail/object_util.hpp>

namespace jank::runtime::detail
{
  struct in_place_unique
  { };

  /* TODO: Int sequence to clean this up? */
  template <typename KV>
  KV* make_next_array(KV const * const prev, size_t const length, KV const key, KV const value)
  {
    switch(length)
    {
      case 0:
      {
        auto const ret(new (GC) KV[2]{ key, value });
        return ret;
      }
      case 2:
      {
        auto const ret(new (GC) KV[4]{ prev[0], prev[1], key, value });
        return ret;
      }
      case 4:
      {
        auto const ret
        (
          new (GC) KV[6]
          {
            prev[0], prev[1],
            prev[2], prev[3],
            key, value
          }
        );
        return ret;
      }
      case 6:
      {
        auto const ret
        (
          new (GC) KV[8]
          {
            prev[0], prev[1],
            prev[2], prev[3],
            prev[4], prev[5],
            key, value
          }
        );
        return ret;
      }
      case 8:
      {
        auto const ret
        (
          new (GC) KV[10]
          {
            prev[0], prev[1],
            prev[2], prev[3],
            prev[4], prev[5],
            prev[6], prev[7],
            key, value
          }
        );
        return ret;
      }
      case 10:
      {
        auto const ret
        (
          new (GC) KV[12]
          {
            prev[0], prev[1],
            prev[2], prev[3],
            prev[4], prev[5],
            prev[6], prev[7],
            prev[7], prev[8],
            key, value
          }
        );
        return ret;
      }
      case 12:
      {
        auto const ret
        (
          new (GC) KV[14]
          {
            prev[0], prev[1],
            prev[2], prev[3],
            prev[4], prev[5],
            prev[6], prev[7],
            prev[8], prev[9],
            prev[10], prev[11],
            key, value
          }
        );
        return ret;
      }
      case 14:
      {
        auto const ret
        (
          new (GC) KV[16]
          {
            prev[0], prev[1],
            prev[2], prev[3],
            prev[4], prev[5],
            prev[6], prev[7],
            prev[8], prev[9],
            prev[10], prev[11],
            prev[12], prev[13],
            key, value
          }
        );
        return ret;
      }
      // TODO: Convert to hash map.
      default:
        throw std::runtime_error{ fmt::format("unsupported array size: {}", length + 2) };
    }
  }

  /* TODO: Rename to native_array_map. */
  /* This is a short map, storing a vector of pairs. This is only until immer has proper
   * support for short maps and map transients. */
  /* TODO: Just use one template param. */
  template <typename K, typename V>
  struct map_type_impl
  {
    using value_type = V*;

    map_type_impl() = default;
    map_type_impl(map_type_impl const &s) = default;
    map_type_impl(map_type_impl &&s) noexcept = default;
    template <typename L, typename E = std::enable_if_t<std::is_integral_v<L>>>
    map_type_impl(in_place_unique, value_type kvs, L const l)
      : data{ std::move(kvs) }, length{ static_cast<decltype(length)>(l) }
    { }
    ~map_type_impl() = default;

    void insert_unique(K const key, V const val)
    {
      data = make_next_array(data, length, key, val);
      length += 2;
      hash = 0;
    }
    void insert_or_assign(K const key, V const val)
    {
      if(key->type == runtime::object_type::keyword)
      {
        for(size_t i{}; i < length; i += 2)
        {
          if(data[i] == key)
          {
            data[i + 1] = val;
            hash = 0;
            return;
          }
        }
      }
      else
      {
        for(size_t i{}; i < length; i += 2)
        {
          if(detail::equal(data[i], key))
          {
            data[i + 1] = val;
            hash = 0;
            return;
          }
        }
      }
      insert_unique(key, val);
    }

    V find(K const key) const
    {
      if(key->type == runtime::object_type::keyword)
      {
        for(size_t i{}; i < length; i += 2)
        {
          if(data[i] == key)
          { return data[i + 1]; }
        }
      }
      else
      {
        for(size_t i{}; i < length; i += 2)
        {
          if(detail::equal(data[i], key))
          { return data[i + 1]; }
        }
      }
      return nullptr;
    }

    size_t to_hash() const
    {
      if(hash != 0)
      { return hash; }

      hash = length;
      for(size_t i{}; i < length; i += 2)
      {
        hash = jank::runtime::detail::hash_combine(hash, *data[i]);
        hash = jank::runtime::detail::hash_combine(hash, *data[i + 1]);
      }
      return hash;
    }

    struct iterator
    {
      using iterator_category = std::input_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = std::pair<V, V>;
      using pointer = value_type*;
      using reference = value_type&;

      value_type operator *() const
      {
        return { data[index], data[index + 1] };
      }
      iterator& operator ++()
      {
        index += 2;
        return *this;
      }
      bool operator !=(iterator const &rhs) const
      { return data != rhs.data || index != rhs.index; }
      bool operator ==(iterator const &rhs) const
      { return !(*this != rhs); }
      iterator& operator=(iterator const &rhs)
      {
        if(this == &rhs)
        { return *this; }

        data = rhs.data;
        index = rhs.index;
        return *this;
      }

      V const* data{};
      size_t index{};
    };
    using const_iterator = iterator;

    iterator begin()
    { return iterator{ data, 0 }; }
    iterator begin() const
    { return iterator{ data, 0 }; }
    iterator end()
    { return iterator{ data, length }; }
    iterator end() const
    { return iterator{ data, length }; }

    size_t size() const
    { return length / 2; }

    map_type_impl<K, V> clone() const
    {
      map_type_impl<K, V> ret{ *this };
      ret.data = new (GC) V[length];
      memcpy(ret.data, data, length * sizeof(V));
      return ret;
    }

    value_type data{};
    size_t length{};
    mutable size_t hash{};
  };

  using persistent_map = map_type_impl<object_ptr, object_ptr>;
}
