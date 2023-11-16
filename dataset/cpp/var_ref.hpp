#pragma once

#include <jank/runtime/var.hpp>
#include <jank/analyze/local_frame.hpp>
#include <jank/analyze/expression_base.hpp>
#include <jank/detail/to_runtime_data.hpp>

namespace jank::analyze::expr
{
  template <typename E>
  struct var_ref : expression_base
  {
    runtime::obj::symbol_ptr qualified_name{};
    runtime::var_ptr var{};

    runtime::object_ptr to_runtime_data() const
    {
      return runtime::obj::map::create_unique
      (
        make_box("__type"), make_box("expr::var_ref"),
        make_box("qualified_name"), qualified_name,
        make_box("var"), var
      );
    }
  };
}
