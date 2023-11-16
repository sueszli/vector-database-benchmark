"""Special case IR generation of calls to specific builtin functions.

Most special cases should be handled using the data driven "primitive
ops" system, but certain operations require special handling that has
access to the AST/IR directly and can make decisions/optimizations
based on it. These special cases can be implemented here.

For example, we use specializers to statically emit the length of a
fixed length tuple and to emit optimized code for any()/all() calls with
generator comprehensions as the argument.

See comment below for more documentation.
"""
from __future__ import annotations
from typing import Callable, Optional
from mypy.nodes import ARG_NAMED, ARG_POS, CallExpr, DictExpr, Expression, GeneratorExpr, IntExpr, ListExpr, MemberExpr, NameExpr, RefExpr, StrExpr, TupleExpr
from mypy.types import AnyType, TypeOfAny
from mypyc.ir.ops import BasicBlock, Extend, Integer, RaiseStandardError, Register, Truncate, Unreachable, Value
from mypyc.ir.rtypes import RInstance, RPrimitive, RTuple, RType, bool_rprimitive, c_int_rprimitive, dict_rprimitive, int16_rprimitive, int32_rprimitive, int64_rprimitive, int_rprimitive, is_bool_rprimitive, is_dict_rprimitive, is_fixed_width_rtype, is_float_rprimitive, is_int16_rprimitive, is_int32_rprimitive, is_int64_rprimitive, is_int_rprimitive, is_list_rprimitive, is_uint8_rprimitive, list_rprimitive, set_rprimitive, str_rprimitive, uint8_rprimitive
from mypyc.irbuild.builder import IRBuilder
from mypyc.irbuild.for_helpers import comprehension_helper, sequence_from_generator_preallocate_helper, translate_list_comprehension, translate_set_comprehension
from mypyc.irbuild.format_str_tokenizer import FormatOp, convert_format_expr_to_str, join_formatted_strings, tokenizer_format_call
from mypyc.primitives.dict_ops import dict_items_op, dict_keys_op, dict_setdefault_spec_init_op, dict_values_op
from mypyc.primitives.list_ops import new_list_set_item_op
from mypyc.primitives.tuple_ops import new_tuple_set_item_op
Specializer = Callable[['IRBuilder', CallExpr, RefExpr], Optional[Value]]
specializers: dict[tuple[str, RType | None], list[Specializer]] = {}

def _apply_specialization(builder: IRBuilder, expr: CallExpr, callee: RefExpr, name: str | None, typ: RType | None=None) -> Value | None:
    if False:
        print('Hello World!')
    if name and (name, typ) in specializers:
        for specializer in specializers[name, typ]:
            val = specializer(builder, expr, callee)
            if val is not None:
                return val
    return None

def apply_function_specialization(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        return 10
    'Invoke the Specializer callback for a function if one has been registered'
    return _apply_specialization(builder, expr, callee, callee.fullname)

def apply_method_specialization(builder: IRBuilder, expr: CallExpr, callee: MemberExpr, typ: RType | None=None) -> Value | None:
    if False:
        print('Hello World!')
    'Invoke the Specializer callback for a method if one has been registered'
    name = callee.fullname if typ is None else callee.name
    return _apply_specialization(builder, expr, callee, name, typ)

def specialize_function(name: str, typ: RType | None=None) -> Callable[[Specializer], Specializer]:
    if False:
        return 10
    'Decorator to register a function as being a specializer.\n\n    There may exist multiple specializers for one function. When\n    translating method calls, the earlier appended specializer has\n    higher priority.\n    '

    def wrapper(f: Specializer) -> Specializer:
        if False:
            while True:
                i = 10
        specializers.setdefault((name, typ), []).append(f)
        return f
    return wrapper

@specialize_function('builtins.globals')
def translate_globals(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        for i in range(10):
            print('nop')
    if len(expr.args) == 0:
        return builder.load_globals_dict()
    return None

@specialize_function('builtins.abs')
@specialize_function('builtins.int')
@specialize_function('builtins.float')
@specialize_function('builtins.complex')
@specialize_function('mypy_extensions.i64')
@specialize_function('mypy_extensions.i32')
@specialize_function('mypy_extensions.i16')
@specialize_function('mypy_extensions.u8')
def translate_builtins_with_unary_dunder(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        print('Hello World!')
    'Specialize calls on native classes that implement the associated dunder.\n\n    E.g. i64(x) gets specialized to x.__int__() if x is a native instance.\n    '
    if len(expr.args) == 1 and expr.arg_kinds == [ARG_POS] and isinstance(callee, NameExpr):
        arg = expr.args[0]
        arg_typ = builder.node_type(arg)
        shortname = callee.fullname.split('.')[1]
        if shortname in ('i64', 'i32', 'i16', 'u8'):
            method = '__int__'
        else:
            method = f'__{shortname}__'
        if isinstance(arg_typ, RInstance) and arg_typ.class_ir.has_method(method):
            obj = builder.accept(arg)
            return builder.gen_method_call(obj, method, [], None, expr.line)
    return None

@specialize_function('builtins.len')
def translate_len(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        while True:
            i = 10
    if len(expr.args) == 1 and expr.arg_kinds == [ARG_POS]:
        arg = expr.args[0]
        expr_rtype = builder.node_type(arg)
        if isinstance(expr_rtype, RTuple):
            builder.accept(arg)
            return Integer(len(expr_rtype.types))
        else:
            if is_list_rprimitive(builder.node_type(arg)):
                borrow = True
            else:
                borrow = False
            obj = builder.accept(arg, can_borrow=borrow)
            return builder.builtin_len(obj, expr.line)
    return None

@specialize_function('builtins.list')
def dict_methods_fast_path(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        for i in range(10):
            print('nop')
    'Specialize a common case when list() is called on a dictionary\n    view method call.\n\n    For example:\n        foo = list(bar.keys())\n    '
    if not (len(expr.args) == 1 and expr.arg_kinds == [ARG_POS]):
        return None
    arg = expr.args[0]
    if not (isinstance(arg, CallExpr) and (not arg.args) and isinstance(arg.callee, MemberExpr)):
        return None
    base = arg.callee.expr
    attr = arg.callee.name
    rtype = builder.node_type(base)
    if not (is_dict_rprimitive(rtype) and attr in ('keys', 'values', 'items')):
        return None
    obj = builder.accept(base)
    if attr == 'keys':
        return builder.call_c(dict_keys_op, [obj], expr.line)
    elif attr == 'values':
        return builder.call_c(dict_values_op, [obj], expr.line)
    else:
        return builder.call_c(dict_items_op, [obj], expr.line)

@specialize_function('builtins.list')
def translate_list_from_generator_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        print('Hello World!')
    "Special case for simplest list comprehension.\n\n    For example:\n        list(f(x) for x in some_list/some_tuple/some_str)\n    'translate_list_comprehension()' would take care of other cases\n    if this fails.\n    "
    if len(expr.args) == 1 and expr.arg_kinds[0] == ARG_POS and isinstance(expr.args[0], GeneratorExpr):
        return sequence_from_generator_preallocate_helper(builder, expr.args[0], empty_op_llbuilder=builder.builder.new_list_op_with_length, set_item_op=new_list_set_item_op)
    return None

@specialize_function('builtins.tuple')
def translate_tuple_from_generator_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        print('Hello World!')
    "Special case for simplest tuple creation from a generator.\n\n    For example:\n        tuple(f(x) for x in some_list/some_tuple/some_str)\n    'translate_safe_generator_call()' would take care of other cases\n    if this fails.\n    "
    if len(expr.args) == 1 and expr.arg_kinds[0] == ARG_POS and isinstance(expr.args[0], GeneratorExpr):
        return sequence_from_generator_preallocate_helper(builder, expr.args[0], empty_op_llbuilder=builder.builder.new_tuple_with_length, set_item_op=new_tuple_set_item_op)
    return None

@specialize_function('builtins.set')
def translate_set_from_generator_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        i = 10
        return i + 15
    'Special case for set creation from a generator.\n\n    For example:\n        set(f(...) for ... in iterator/nested_generators...)\n    '
    if len(expr.args) == 1 and expr.arg_kinds[0] == ARG_POS and isinstance(expr.args[0], GeneratorExpr):
        return translate_set_comprehension(builder, expr.args[0])
    return None

@specialize_function('builtins.min')
@specialize_function('builtins.max')
def faster_min_max(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        i = 10
        return i + 15
    if expr.arg_kinds == [ARG_POS, ARG_POS]:
        (x, y) = (builder.accept(expr.args[0]), builder.accept(expr.args[1]))
        result = Register(builder.node_type(expr))
        if callee.fullname == 'builtins.min':
            comparison = builder.binary_op(y, x, '<', expr.line)
        else:
            comparison = builder.binary_op(y, x, '>', expr.line)
        (true_block, false_block, next_block) = (BasicBlock(), BasicBlock(), BasicBlock())
        builder.add_bool_branch(comparison, true_block, false_block)
        builder.activate_block(true_block)
        builder.assign(result, builder.coerce(y, result.type, expr.line), expr.line)
        builder.goto(next_block)
        builder.activate_block(false_block)
        builder.assign(result, builder.coerce(x, result.type, expr.line), expr.line)
        builder.goto(next_block)
        builder.activate_block(next_block)
        return result
    return None

@specialize_function('builtins.tuple')
@specialize_function('builtins.frozenset')
@specialize_function('builtins.dict')
@specialize_function('builtins.min')
@specialize_function('builtins.max')
@specialize_function('builtins.sorted')
@specialize_function('collections.OrderedDict')
@specialize_function('join', str_rprimitive)
@specialize_function('extend', list_rprimitive)
@specialize_function('update', dict_rprimitive)
@specialize_function('update', set_rprimitive)
def translate_safe_generator_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        return 10
    'Special cases for things that consume iterators where we know we\n    can safely compile a generator into a list.\n    '
    if len(expr.args) > 0 and expr.arg_kinds[0] == ARG_POS and isinstance(expr.args[0], GeneratorExpr):
        if isinstance(callee, MemberExpr):
            return builder.gen_method_call(builder.accept(callee.expr), callee.name, [translate_list_comprehension(builder, expr.args[0])] + [builder.accept(arg) for arg in expr.args[1:]], builder.node_type(expr), expr.line, expr.arg_kinds, expr.arg_names)
        else:
            return builder.call_refexpr_with_args(expr, callee, [translate_list_comprehension(builder, expr.args[0])] + [builder.accept(arg) for arg in expr.args[1:]])
    return None

@specialize_function('builtins.any')
def translate_any_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        while True:
            i = 10
    if len(expr.args) == 1 and expr.arg_kinds == [ARG_POS] and isinstance(expr.args[0], GeneratorExpr):
        return any_all_helper(builder, expr.args[0], builder.false, lambda x: x, builder.true)
    return None

@specialize_function('builtins.all')
def translate_all_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        for i in range(10):
            print('nop')
    if len(expr.args) == 1 and expr.arg_kinds == [ARG_POS] and isinstance(expr.args[0], GeneratorExpr):
        return any_all_helper(builder, expr.args[0], builder.true, lambda x: builder.unary_op(x, 'not', expr.line), builder.false)
    return None

def any_all_helper(builder: IRBuilder, gen: GeneratorExpr, initial_value: Callable[[], Value], modify: Callable[[Value], Value], new_value: Callable[[], Value]) -> Value:
    if False:
        i = 10
        return i + 15
    retval = Register(bool_rprimitive)
    builder.assign(retval, initial_value(), -1)
    loop_params = list(zip(gen.indices, gen.sequences, gen.condlists, gen.is_async))
    (true_block, false_block, exit_block) = (BasicBlock(), BasicBlock(), BasicBlock())

    def gen_inner_stmts() -> None:
        if False:
            print('Hello World!')
        comparison = modify(builder.accept(gen.left_expr))
        builder.add_bool_branch(comparison, true_block, false_block)
        builder.activate_block(true_block)
        builder.assign(retval, new_value(), -1)
        builder.goto(exit_block)
        builder.activate_block(false_block)
    comprehension_helper(builder, loop_params, gen_inner_stmts, gen.line)
    builder.goto_and_activate(exit_block)
    return retval

@specialize_function('builtins.sum')
def translate_sum_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        for i in range(10):
            print('nop')
    if not (len(expr.args) in (1, 2) and expr.arg_kinds[0] == ARG_POS and isinstance(expr.args[0], GeneratorExpr)):
        return None
    if len(expr.args) == 2:
        if expr.arg_kinds[1] not in (ARG_POS, ARG_NAMED):
            return None
        start_expr = expr.args[1]
    else:
        start_expr = IntExpr(0)
    gen_expr = expr.args[0]
    target_type = builder.node_type(expr)
    retval = Register(target_type)
    builder.assign(retval, builder.coerce(builder.accept(start_expr), target_type, -1), -1)

    def gen_inner_stmts() -> None:
        if False:
            return 10
        call_expr = builder.accept(gen_expr.left_expr)
        builder.assign(retval, builder.binary_op(retval, call_expr, '+', -1), -1)
    loop_params = list(zip(gen_expr.indices, gen_expr.sequences, gen_expr.condlists, gen_expr.is_async))
    comprehension_helper(builder, loop_params, gen_inner_stmts, gen_expr.line)
    return retval

@specialize_function('dataclasses.field')
@specialize_function('attr.ib')
@specialize_function('attr.attrib')
@specialize_function('attr.Factory')
def translate_dataclasses_field_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        while True:
            i = 10
    "Special case for 'dataclasses.field', 'attr.attrib', and 'attr.Factory'\n    function calls because the results of such calls are type-checked\n    by mypy using the types of the arguments to their respective\n    functions, resulting in attempted coercions by mypyc that throw a\n    runtime error.\n    "
    builder.types[expr] = AnyType(TypeOfAny.from_error)
    return None

@specialize_function('builtins.next')
def translate_next_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        return 10
    'Special case for calling next() on a generator expression, an\n    idiom that shows up some in mypy.\n\n    For example, next(x for x in l if x.id == 12, None) will\n    generate code that searches l for an element where x.id == 12\n    and produce the first such object, or None if no such element\n    exists.\n    '
    if not (expr.arg_kinds in ([ARG_POS], [ARG_POS, ARG_POS]) and isinstance(expr.args[0], GeneratorExpr)):
        return None
    gen = expr.args[0]
    retval = Register(builder.node_type(expr))
    default_val = builder.accept(expr.args[1]) if len(expr.args) > 1 else None
    exit_block = BasicBlock()

    def gen_inner_stmts() -> None:
        if False:
            print('Hello World!')
        builder.assign(retval, builder.accept(gen.left_expr), gen.left_expr.line)
        builder.goto(exit_block)
    loop_params = list(zip(gen.indices, gen.sequences, gen.condlists, gen.is_async))
    comprehension_helper(builder, loop_params, gen_inner_stmts, gen.line)
    if default_val:
        builder.assign(retval, default_val, gen.left_expr.line)
        builder.goto(exit_block)
    else:
        builder.add(RaiseStandardError(RaiseStandardError.STOP_ITERATION, None, expr.line))
        builder.add(Unreachable())
    builder.activate_block(exit_block)
    return retval

@specialize_function('builtins.isinstance')
def translate_isinstance(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        return 10
    'Special case for builtins.isinstance.\n\n    Prevent coercions on the thing we are checking the instance of -\n    there is no need to coerce something to a new type before checking\n    what type it is, and the coercion could lead to bugs.\n    '
    if len(expr.args) == 2 and expr.arg_kinds == [ARG_POS, ARG_POS] and isinstance(expr.args[1], (RefExpr, TupleExpr)):
        builder.types[expr.args[0]] = AnyType(TypeOfAny.from_error)
        irs = builder.flatten_classes(expr.args[1])
        if irs is not None:
            can_borrow = all((ir.is_ext_class and (not ir.inherits_python) and (not ir.allow_interpreted_subclasses) for ir in irs))
            obj = builder.accept(expr.args[0], can_borrow=can_borrow)
            return builder.builder.isinstance_helper(obj, irs, expr.line)
    return None

@specialize_function('setdefault', dict_rprimitive)
def translate_dict_setdefault(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        i = 10
        return i + 15
    "Special case for 'dict.setdefault' which would only construct\n    default empty collection when needed.\n\n    The dict_setdefault_spec_init_op checks whether the dict contains\n    the key and would construct the empty collection only once.\n\n    For example, this specializer works for the following cases:\n         d.setdefault(key, set()).add(value)\n         d.setdefault(key, []).append(value)\n         d.setdefault(key, {})[inner_key] = inner_val\n    "
    if len(expr.args) == 2 and expr.arg_kinds == [ARG_POS, ARG_POS] and isinstance(callee, MemberExpr):
        arg = expr.args[1]
        if isinstance(arg, ListExpr):
            if len(arg.items):
                return None
            data_type = Integer(1, c_int_rprimitive, expr.line)
        elif isinstance(arg, DictExpr):
            if len(arg.items):
                return None
            data_type = Integer(2, c_int_rprimitive, expr.line)
        elif isinstance(arg, CallExpr) and isinstance(arg.callee, NameExpr) and (arg.callee.fullname == 'builtins.set'):
            if len(arg.args):
                return None
            data_type = Integer(3, c_int_rprimitive, expr.line)
        else:
            return None
        callee_dict = builder.accept(callee.expr)
        key_val = builder.accept(expr.args[0])
        return builder.call_c(dict_setdefault_spec_init_op, [callee_dict, key_val, data_type], expr.line)
    return None

@specialize_function('format', str_rprimitive)
def translate_str_format(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(callee, MemberExpr) and isinstance(callee.expr, StrExpr) and (expr.arg_kinds.count(ARG_POS) == len(expr.arg_kinds)):
        format_str = callee.expr.value
        tokens = tokenizer_format_call(format_str)
        if tokens is None:
            return None
        (literals, format_ops) = tokens
        substitutions = convert_format_expr_to_str(builder, format_ops, expr.args, expr.line)
        if substitutions is None:
            return None
        return join_formatted_strings(builder, literals, substitutions, expr.line)
    return None

@specialize_function('join', str_rprimitive)
def translate_fstring(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        return 10
    "Special case for f-string, which is translated into str.join()\n    in mypy AST.\n\n    This specializer optimizes simplest f-strings which don't contain\n    any format operation.\n    "
    if isinstance(callee, MemberExpr) and isinstance(callee.expr, StrExpr) and (callee.expr.value == '') and (expr.arg_kinds == [ARG_POS]) and isinstance(expr.args[0], ListExpr):
        for item in expr.args[0].items:
            if isinstance(item, StrExpr):
                continue
            elif isinstance(item, CallExpr):
                if not isinstance(item.callee, MemberExpr) or item.callee.name != 'format':
                    return None
                elif not isinstance(item.callee.expr, StrExpr) or item.callee.expr.value != '{:{}}':
                    return None
                if not isinstance(item.args[1], StrExpr) or item.args[1].value != '':
                    return None
            else:
                return None
        format_ops = []
        exprs: list[Expression] = []
        for item in expr.args[0].items:
            if isinstance(item, StrExpr) and item.value != '':
                format_ops.append(FormatOp.STR)
                exprs.append(item)
            elif isinstance(item, CallExpr):
                format_ops.append(FormatOp.STR)
                exprs.append(item.args[0])
        substitutions = convert_format_expr_to_str(builder, format_ops, exprs, expr.line)
        if substitutions is None:
            return None
        return join_formatted_strings(builder, None, substitutions, expr.line)
    return None

@specialize_function('mypy_extensions.i64')
def translate_i64(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        return 10
    if len(expr.args) != 1 or expr.arg_kinds[0] != ARG_POS:
        return None
    arg = expr.args[0]
    arg_type = builder.node_type(arg)
    if is_int64_rprimitive(arg_type):
        return builder.accept(arg)
    elif is_int32_rprimitive(arg_type) or is_int16_rprimitive(arg_type):
        val = builder.accept(arg)
        return builder.add(Extend(val, int64_rprimitive, signed=True, line=expr.line))
    elif is_uint8_rprimitive(arg_type):
        val = builder.accept(arg)
        return builder.add(Extend(val, int64_rprimitive, signed=False, line=expr.line))
    elif is_int_rprimitive(arg_type) or is_bool_rprimitive(arg_type):
        val = builder.accept(arg)
        return builder.coerce(val, int64_rprimitive, expr.line)
    return None

@specialize_function('mypy_extensions.i32')
def translate_i32(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        print('Hello World!')
    if len(expr.args) != 1 or expr.arg_kinds[0] != ARG_POS:
        return None
    arg = expr.args[0]
    arg_type = builder.node_type(arg)
    if is_int32_rprimitive(arg_type):
        return builder.accept(arg)
    elif is_int64_rprimitive(arg_type):
        val = builder.accept(arg)
        return builder.add(Truncate(val, int32_rprimitive, line=expr.line))
    elif is_int16_rprimitive(arg_type):
        val = builder.accept(arg)
        return builder.add(Extend(val, int32_rprimitive, signed=True, line=expr.line))
    elif is_uint8_rprimitive(arg_type):
        val = builder.accept(arg)
        return builder.add(Extend(val, int32_rprimitive, signed=False, line=expr.line))
    elif is_int_rprimitive(arg_type) or is_bool_rprimitive(arg_type):
        val = builder.accept(arg)
        val = truncate_literal(val, int32_rprimitive)
        return builder.coerce(val, int32_rprimitive, expr.line)
    return None

@specialize_function('mypy_extensions.i16')
def translate_i16(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        return 10
    if len(expr.args) != 1 or expr.arg_kinds[0] != ARG_POS:
        return None
    arg = expr.args[0]
    arg_type = builder.node_type(arg)
    if is_int16_rprimitive(arg_type):
        return builder.accept(arg)
    elif is_int32_rprimitive(arg_type) or is_int64_rprimitive(arg_type):
        val = builder.accept(arg)
        return builder.add(Truncate(val, int16_rprimitive, line=expr.line))
    elif is_uint8_rprimitive(arg_type):
        val = builder.accept(arg)
        return builder.add(Extend(val, int16_rprimitive, signed=False, line=expr.line))
    elif is_int_rprimitive(arg_type) or is_bool_rprimitive(arg_type):
        val = builder.accept(arg)
        val = truncate_literal(val, int16_rprimitive)
        return builder.coerce(val, int16_rprimitive, expr.line)
    return None

@specialize_function('mypy_extensions.u8')
def translate_u8(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        return 10
    if len(expr.args) != 1 or expr.arg_kinds[0] != ARG_POS:
        return None
    arg = expr.args[0]
    arg_type = builder.node_type(arg)
    if is_uint8_rprimitive(arg_type):
        return builder.accept(arg)
    elif is_int16_rprimitive(arg_type) or is_int32_rprimitive(arg_type) or is_int64_rprimitive(arg_type):
        val = builder.accept(arg)
        return builder.add(Truncate(val, uint8_rprimitive, line=expr.line))
    elif is_int_rprimitive(arg_type) or is_bool_rprimitive(arg_type):
        val = builder.accept(arg)
        val = truncate_literal(val, uint8_rprimitive)
        return builder.coerce(val, uint8_rprimitive, expr.line)
    return None

def truncate_literal(value: Value, rtype: RPrimitive) -> Value:
    if False:
        while True:
            i = 10
    'If value is an integer literal value, truncate it to given native int rtype.\n\n    For example, truncate 256 into 0 if rtype is u8.\n    '
    if not isinstance(value, Integer):
        return value
    x = value.numeric_value()
    max_unsigned = (1 << rtype.size * 8) - 1
    x = x & max_unsigned
    if rtype.is_signed and x >= (max_unsigned + 1) // 2:
        x -= max_unsigned + 1
    return Integer(x, rtype)

@specialize_function('builtins.int')
def translate_int(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        i = 10
        return i + 15
    if len(expr.args) != 1 or expr.arg_kinds[0] != ARG_POS:
        return None
    arg = expr.args[0]
    arg_type = builder.node_type(arg)
    if is_bool_rprimitive(arg_type) or is_int_rprimitive(arg_type) or is_fixed_width_rtype(arg_type):
        src = builder.accept(arg)
        return builder.coerce(src, int_rprimitive, expr.line)
    return None

@specialize_function('builtins.bool')
def translate_bool(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        while True:
            i = 10
    if len(expr.args) != 1 or expr.arg_kinds[0] != ARG_POS:
        return None
    arg = expr.args[0]
    src = builder.accept(arg)
    return builder.builder.bool_value(src)

@specialize_function('builtins.float')
def translate_float(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if False:
        while True:
            i = 10
    if len(expr.args) != 1 or expr.arg_kinds[0] != ARG_POS:
        return None
    arg = expr.args[0]
    arg_type = builder.node_type(arg)
    if is_float_rprimitive(arg_type):
        return builder.accept(arg)
    return None