#pragma once

#include <memory>

#include <jank/runtime/obj/list.hpp>
#include <jank/analyze/expression_base.hpp>
#include <jank/runtime/seq.hpp>

namespace jank::analyze::expr
{
  template <typename E>
  struct call : expression_base
  {
    /* Var, local, or callable. */
    native_box<E> source_expr{};
    runtime::obj::list_ptr args{};
    native_vector<native_box<E>> arg_exprs;

    runtime::object_ptr to_runtime_data() const
    {
      runtime::object_ptr arg_expr_maps(make_box<runtime::obj::vector>());
      for(auto const &e : arg_exprs)
      { arg_expr_maps = runtime::conj(arg_expr_maps, e->to_runtime_data()); }

      return runtime::obj::map::create_unique
      (
        make_box("__type"), make_box("expr::call"),
        make_box("source_expr"), source_expr->to_runtime_data(),
        make_box("args"), args,
        make_box("arg_exprs"), arg_expr_maps
      );
    }
  };
}
