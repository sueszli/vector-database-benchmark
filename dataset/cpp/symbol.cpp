#include <iostream>
#include <sstream>

#include <fmt/compile.h>

#include <jank/runtime/obj/symbol.hpp>

namespace jank::runtime
{
  template <typename S>
  void separate(obj::symbol &sym, S &&s)
  {
    auto const found(s.find('/'));
    if(found != native_string::npos && s.size() > 1)
    {
      sym.ns = s.substr(0, found);
      sym.name = s.substr(found + 1);
    }
    else
    { sym.name = std::forward<S>(s); }
  }

  obj::symbol::static_object(native_string const &d)
  { separate(*this, d); }
  obj::symbol::static_object(native_string &&d)
  { separate(*this, std::move(d)); }

  obj::symbol::static_object(native_string const &ns, native_string const &n)
    : ns{ ns }, name{ n }
  { }
  obj::symbol::static_object(native_string &&ns, native_string &&n)
    : ns{ std::move(ns) }, name{ std::move(n) }
  { }

  native_bool obj::symbol::equal(object const &o) const
  {
    if(o.type != object_type::symbol)
    { return false; }

    auto const s(expect_object<obj::symbol>(&o));
    return ns == s->ns && name == s->name;
  }

  native_bool obj::symbol::equal(obj::symbol const &s) const
  { return ns == s.ns && name == s.name; }

  void to_string_impl
  (
    native_string const &ns,
    native_string const &name,
    fmt::memory_buffer &buff
  )
  {
    if(!ns.empty())
    { format_to(std::back_inserter(buff), FMT_COMPILE("{}/{}"), ns, name); }
    else
    { format_to(std::back_inserter(buff), FMT_COMPILE("{}"), name); }
  }
  void obj::symbol::to_string(fmt::memory_buffer &buff) const
  { to_string_impl(ns, name, buff); }
  native_string obj::symbol::to_string() const
  {
    fmt::memory_buffer buff;
    to_string_impl(ns, name, buff);
    return native_string{ buff.data(), buff.size() };
  }
  native_integer obj::symbol::to_hash() const
  /* TODO: Cache this. */
  { return runtime::detail::hash_combine(ns.to_hash(), name.to_hash()); }

  object_ptr obj::symbol::with_meta(object_ptr const m) const
  {
    auto const meta(behavior::detail::validate_meta(m));
    auto ret(jank::make_box<obj::symbol>(ns, name));
    ret->meta = meta;
    return ret;
  }

  bool obj::symbol::operator ==(obj::symbol const &rhs) const
  { return ns == rhs.ns && name == rhs.name; }

  bool obj::symbol::operator <(obj::symbol const &rhs) const
  { return to_hash() < rhs.to_hash(); }
}
