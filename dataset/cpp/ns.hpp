#pragma once

#include <functional>
#include <unordered_map>
#include <mutex>

#include <folly/Synchronized.h>

#include <jank/runtime/obj/symbol.hpp>
#include <jank/runtime/var.hpp>

namespace jank::runtime
{
  struct context;

  template <>
  struct static_object<object_type::ns> : gc
  {
    static constexpr bool pointer_free{ false };

    static_object() = delete;
    static_object(static_object &&) = default;
    static_object(static_object const &) = default;
    static_object(obj::symbol_ptr const &name, context const &c)
      : name{ name }, rt_ctx{ c }
    { }

    /* behavior::objectable */
    native_bool equal(object const &) const;
    native_string to_string() const;
    void to_string(fmt::memory_buffer &buff) const;
    native_integer to_hash() const;

    bool operator ==(static_object const &rhs) const;

    native_box<static_object> clone() const;

    object base{ object_type::ns };
    obj::symbol_ptr name{};
    folly::Synchronized<native_unordered_map<obj::symbol_ptr, var_ptr>> vars;
    context const &rt_ctx;
  };

  using ns = static_object<object_type::ns>;
  using ns_ptr = native_box<ns>;
}
