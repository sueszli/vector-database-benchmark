#include <iostream>

#include <jank/runtime/context.hpp>
#include <jank/runtime/ns.hpp>
#include <jank/runtime/obj/vector.hpp>
#include <jank/runtime/obj/number.hpp>
#include <jank/runtime/util.hpp>
#include <jank/codegen/processor.hpp>
#include <jank/jit/processor.hpp>
#include <jank/evaluate.hpp>

namespace jank::evaluate
{
  template <typename T>
  concept has_frame = requires(T const *t){ t->frame; };

  /* Some expressions don't make sense to eval outright and aren't fns that can be JIT compiled.
   * For those, we wrap them in a fn expression and then JIT compile and call them.
   *
   * There's an oddity here, since that expr wouldn't've been analyzed within a fn frame, so
   * its lifted vars/constants, for example, aren't in a fn frame. Instead, they're put in the
   * root frame. So, when wrapping this expr, we give the fn the root frame, but change its
   * type to a fn frame. */
  template <typename E>
  analyze::expr::function<analyze::expression> wrap_expression(E expr)
  {
    analyze::expr::function<analyze::expression> wrapper;
    analyze::expr::function_arity<analyze::expression> arity;

    arity.frame = expr.frame;
    while(arity.frame->parent.is_some())
    { arity.frame = arity.frame->parent.unwrap(); }
    arity.frame->type = analyze::local_frame::frame_type::fn;
    expr.expr_type = analyze::expression_type::return_statement;
    /* TODO: Avoid allocation by using existing ptr. */
    arity.body.body.push_back(make_box<analyze::expression>(expr));
    arity.fn_ctx = make_box<analyze::expr::function_context>();
    arity.body.frame = arity.frame;

    wrapper.arities.emplace_back(std::move(arity));
    wrapper.frame = expr.frame;

    return wrapper;
  }

  analyze::expr::function<analyze::expression> wrap_expression(analyze::expression_ptr const expr)
  {
    return boost::apply_visitor
    (
      [](auto const &typed_expr)
      { return wrap_expression(typed_expr); },
      expr->data
    );
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expression_ptr const &ex
  )
  {
    runtime::object_ptr ret{};
    boost::apply_visitor
    (
      [&rt_ctx, &jit_prc, &ret](auto const &typed_ex)
      { ret = eval(rt_ctx, jit_prc, typed_ex); },
      ex->data
    );

    assert(ret);
    return ret;
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expr::def<analyze::expression> const &expr
  )
  {
    auto var(rt_ctx.intern_var(expr.name).expect_ok());
    if(expr.value.is_none())
    { return var; }

    auto const evaluated_value(eval(rt_ctx, jit_prc, expr.value.unwrap()));
    var->set_root(evaluated_value);
    return var;
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &,
    analyze::expr::var_deref<analyze::expression> const &expr
  )
  {
    auto const var(rt_ctx.find_var(expr.qualified_name));
    return var.unwrap()->get_root();
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &,
    analyze::expr::var_ref<analyze::expression> const &expr
  )
  {
    auto const var(rt_ctx.find_var(expr.qualified_name));
    return var.unwrap();
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expr::call<analyze::expression> const &expr
  )
  {
    auto const source(eval(rt_ctx, jit_prc, expr.source_expr));
    return runtime::visit_object
    (
      [&](auto const typed_source) -> runtime::object_ptr
      {
        using T = typename decltype(typed_source)::value_type;

        if constexpr(std::is_base_of_v<runtime::behavior::callable, T>)
        {
          native_vector<runtime::object_ptr> arg_vals;
          arg_vals.reserve(expr.arg_exprs.size());
          for(auto const &arg_expr: expr.arg_exprs)
          { arg_vals.emplace_back(eval(rt_ctx, jit_prc, arg_expr)); }

          /* TODO: Use apply_to */
          switch(arg_vals.size())
          {
            case 0:
              return runtime::dynamic_call(source);
            case 1:
              return runtime::dynamic_call(source, arg_vals[0]);
            case 2:
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1]);
            case 3:
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1], arg_vals[2]);
            case 4:
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1], arg_vals[2], arg_vals[3]);
            case 5:
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1], arg_vals[2], arg_vals[3], arg_vals[4]);
            case 6:
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1], arg_vals[2], arg_vals[3], arg_vals[4], arg_vals[5]);
            case 7:
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1], arg_vals[2], arg_vals[3], arg_vals[4], arg_vals[5], arg_vals[6]);
            case 8:
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1], arg_vals[2], arg_vals[3], arg_vals[4], arg_vals[5], arg_vals[6], arg_vals[7]);
            case 9:
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1], arg_vals[2], arg_vals[3], arg_vals[4], arg_vals[5], arg_vals[6], arg_vals[7], arg_vals[8]);
            case 10:
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1], arg_vals[2], arg_vals[3], arg_vals[4], arg_vals[5], arg_vals[6], arg_vals[7], arg_vals[8], arg_vals[9]);
            default:
            {
              /* TODO: This could be optimized; making lists sucks right now. */
              runtime::detail::persistent_list all{ arg_vals.rbegin(), arg_vals.rend() };
              for(size_t i{}; i < 10; ++i)
              { all = all.rest(); }
              return runtime::dynamic_call(source, arg_vals[0], arg_vals[1], arg_vals[2], arg_vals[3], arg_vals[4], arg_vals[5], arg_vals[6], arg_vals[7], arg_vals[8], arg_vals[9], make_box<runtime::obj::list>(all));
            }
          }
        }
        else
        { throw std::runtime_error{ fmt::format("not callable: {}", typed_source->to_string()) }; }
      },
      source
    );
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &,
    analyze::expr::primitive_literal<analyze::expression> const &expr
  )
  {
    if(expr.data->type == runtime::object_type::keyword)
    {
      auto const d(runtime::expect_object<runtime::obj::keyword>(expr.data));
      return rt_ctx.intern_keyword(d->sym, d->resolved);
    }
    return expr.data;
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expr::vector<analyze::expression> const &expr
  )
  {
    runtime::detail::transient_vector ret;
    for(auto const &e : expr.data_exprs)
    { ret.push_back(eval(rt_ctx, jit_prc, e)); }
    return make_box<runtime::obj::vector>(ret.persistent());
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expr::map<analyze::expression> const &expr
  )
  {
    /* TODO: Optimize with a transient or something. */
    runtime::detail::persistent_map ret;
    for(auto const &e : expr.data_exprs)
    {
      ret.insert_or_assign
      (
        eval(rt_ctx, jit_prc, e.first),
        eval(rt_ctx, jit_prc, e.second)
      );
    }
    return make_box<runtime::obj::map>(std::move(ret));
  }

  runtime::object_ptr eval
  (
    runtime::context &,
    jit::processor const &,
    analyze::expr::local_reference const &
  )
  /* Doesn't make sense to eval these, since let is wrapped in a fn and JIT compiled. */
  { throw "unsupported eval: local_reference"; }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expr::function<analyze::expression> const &expr
  )
  {
    jank::codegen::processor cg_prc{ rt_ctx, expr };
    return jit_prc.eval(rt_ctx, cg_prc).expect_ok().unwrap();
  }

  runtime::object_ptr eval
  (
    runtime::context &,
    jit::processor const &,
    analyze::expr::recur<analyze::expression> const &
  )
  /* Recur will always be in a fn or loop, which will be JIT compiled. */
  { throw "unsupported eval: recur"; }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expr::do_<analyze::expression> const &expr
  )
  {
    runtime::object_ptr ret{};
    for(auto const &form : expr.body)
    { ret = eval(rt_ctx, jit_prc, form); }
    return ret;
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expr::let<analyze::expression> const &expr
  )
  { return runtime::dynamic_call(eval(rt_ctx, jit_prc, wrap_expression(expr))); }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expr::if_<analyze::expression> const &expr
  )
  {
    auto const condition(eval(rt_ctx, jit_prc, expr.condition));
    if(runtime::detail::truthy(condition))
    { return eval(rt_ctx, jit_prc, expr.then); }
    else if(expr.else_.is_some())
    { return eval(rt_ctx, jit_prc, expr.else_.unwrap()); }
    return runtime::obj::nil::nil_const();
  }

  runtime::object_ptr eval
  (
    runtime::context &rt_ctx,
    jit::processor const &jit_prc,
    analyze::expr::native_raw<analyze::expression> const &expr
  )
  { return runtime::dynamic_call(eval(rt_ctx, jit_prc, wrap_expression(expr))); }
}

