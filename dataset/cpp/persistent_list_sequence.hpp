#pragma once

#include <jank/runtime/object.hpp>
#include <jank/runtime/behavior/seqable.hpp>
#include <jank/runtime/obj/detail/iterator_sequence.hpp>

namespace jank::runtime
{
  namespace obj
  {
    using list = static_object<object_type::list>;
    using list_ptr = native_box<list>;
  }

  template <>
  struct static_object<object_type::persistent_list_sequence>
    : gc,
      obj::detail::iterator_sequence<static_object<object_type::persistent_list_sequence>, runtime::detail::persistent_list::iterator>
  {
    static constexpr bool pointer_free{ false };

    static_object() = default;
    static_object(static_object &&) = default;
    static_object(static_object const &) = default;
    using obj::detail::iterator_sequence<static_object<object_type::persistent_list_sequence>, runtime::detail::persistent_list::iterator>::iterator_sequence;

    object base{ object_type::persistent_list_sequence };
  };

  namespace obj
  {
    using persistent_list_sequence = static_object<object_type::persistent_list_sequence>;
    using persistent_list_sequence_ptr = native_box<persistent_list_sequence>;
  }
}
