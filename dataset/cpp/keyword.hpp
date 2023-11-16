#pragma once

#include <jank/runtime/object.hpp>
#include <jank/runtime/obj/symbol.hpp>

namespace jank::runtime
{
  namespace obj
  {
    using map = static_object<object_type::map>;
    using map_ptr = native_box<map>;
  }

  /* The correct way to create a keyword for normal use is through interning via the RT context. */
  template <>
  struct static_object<object_type::keyword> : gc
  {
    static constexpr bool pointer_free{ true };
    /* Clojure uses this. No idea. https://github.com/clojure/clojure/blob/master/src/jvm/clojure/lang/Keyword.java */
    static constexpr size_t hash_magic{ 0x9e3779b9 };

    static_object() = default;
    static_object(static_object &&) = default;
    static_object(static_object const &) = default;
    static_object(object &&base);
    static_object(obj::symbol const &s, bool resolved);
    static_object(obj::symbol &&s, bool resolved);

    /* behavior::objectable */
    native_bool equal(object const &) const;
    native_string to_string() const;
    void to_string(fmt::memory_buffer &buff) const;
    native_integer to_hash() const;

    /* behavior::metadatable */
    object_ptr with_meta(object_ptr m) const;

    bool operator ==(static_object const &rhs) const;

    object base{ object_type::keyword };
    obj::symbol sym;
    option<obj::map_ptr> meta;
    /* TODO: Remove this state and always use the RT context to resolve upon creation. */
    /* Not resolved means this is a :: keyword. If ns is set, when this is true, it's an ns alias.
     * Upon interning, this will be resolved. */
    bool resolved{ true };
  };

  namespace obj
  {
    using keyword = static_object<object_type::keyword>;
    using keyword_ptr = native_box<keyword>;
  }
}

namespace std
{
  template <>
  struct hash<jank::runtime::obj::keyword_ptr>
  {
    size_t operator()(jank::runtime::obj::keyword_ptr const o) const noexcept
    { return o->to_hash(); }
  };

  template <>
  struct hash<jank::runtime::obj::keyword>
  {
    size_t operator()(jank::runtime::obj::keyword const &o) const noexcept
    {
      static auto hasher(std::hash<jank::runtime::obj::keyword_ptr>{});
      return hasher(const_cast<jank::runtime::obj::keyword*>(&o));
    }
  };

  template <>
  struct equal_to<jank::runtime::obj::keyword_ptr>
  {
    bool operator()(jank::runtime::obj::keyword_ptr const &lhs, jank::runtime::obj::keyword_ptr const &rhs) const noexcept
    { return lhs == rhs; }
  };
}
