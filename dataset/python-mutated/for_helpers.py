"""Helpers for generating for loops and comprehensions.

We special case certain kinds for loops such as "for x in range(...)"
for better efficiency.  Each for loop generator class below deals one
such special case.
"""
from __future__ import annotations
from typing import Callable, ClassVar
from mypy.nodes import ARG_POS, CallExpr, Expression, GeneratorExpr, Lvalue, MemberExpr, RefExpr, SetExpr, TupleExpr, TypeAlias
from mypyc.ir.ops import BasicBlock, Branch, Integer, IntOp, LoadAddress, LoadMem, Register, TupleGet, TupleSet, Value
from mypyc.ir.rtypes import RTuple, RType, bool_rprimitive, int_rprimitive, is_dict_rprimitive, is_fixed_width_rtype, is_list_rprimitive, is_sequence_rprimitive, is_short_int_rprimitive, is_str_rprimitive, is_tuple_rprimitive, pointer_rprimitive, short_int_rprimitive
from mypyc.irbuild.builder import IRBuilder
from mypyc.irbuild.targets import AssignmentTarget, AssignmentTargetTuple
from mypyc.primitives.dict_ops import dict_check_size_op, dict_item_iter_op, dict_key_iter_op, dict_next_item_op, dict_next_key_op, dict_next_value_op, dict_value_iter_op
from mypyc.primitives.exc_ops import no_err_occurred_op
from mypyc.primitives.generic_ops import aiter_op, anext_op, iter_op, next_op
from mypyc.primitives.list_ops import list_append_op, list_get_item_unsafe_op, new_list_set_item_op
from mypyc.primitives.misc_ops import stop_async_iteration_op
from mypyc.primitives.registry import CFunctionDescription
from mypyc.primitives.set_ops import set_add_op
GenFunc = Callable[[], None]

def for_loop_helper(builder: IRBuilder, index: Lvalue, expr: Expression, body_insts: GenFunc, else_insts: GenFunc | None, is_async: bool, line: int) -> None:
    if False:
        i = 10
        return i + 15
    'Generate IR for a loop.\n\n    Args:\n        index: the loop index Lvalue\n        expr: the expression to iterate over\n        body_insts: a function that generates the body of the loop\n        else_insts: a function that generates the else block instructions\n    '
    body_block = BasicBlock()
    step_block = BasicBlock()
    else_block = BasicBlock()
    exit_block = BasicBlock()
    normal_loop_exit = else_block if else_insts is not None else exit_block
    for_gen = make_for_loop_generator(builder, index, expr, body_block, normal_loop_exit, line, is_async=is_async)
    builder.push_loop_stack(step_block, exit_block)
    condition_block = BasicBlock()
    builder.goto_and_activate(condition_block)
    for_gen.gen_condition()
    builder.activate_block(body_block)
    for_gen.begin_body()
    body_insts()
    builder.goto_and_activate(step_block)
    for_gen.gen_step()
    builder.goto(condition_block)
    for_gen.add_cleanup(normal_loop_exit)
    builder.pop_loop_stack()
    if else_insts is not None:
        builder.activate_block(else_block)
        else_insts()
        builder.goto(exit_block)
    builder.activate_block(exit_block)

def for_loop_helper_with_index(builder: IRBuilder, index: Lvalue, expr: Expression, expr_reg: Value, body_insts: Callable[[Value], None], line: int) -> None:
    if False:
        print('Hello World!')
    'Generate IR for a sequence iteration.\n\n    This function only works for sequence type. Compared to for_loop_helper,\n    it would feed iteration index to body_insts.\n\n    Args:\n        index: the loop index Lvalue\n        expr: the expression to iterate over\n        body_insts: a function that generates the body of the loop.\n                    It needs a index as parameter.\n    '
    assert is_sequence_rprimitive(expr_reg.type)
    target_type = builder.get_sequence_type(expr)
    body_block = BasicBlock()
    step_block = BasicBlock()
    exit_block = BasicBlock()
    condition_block = BasicBlock()
    for_gen = ForSequence(builder, index, body_block, exit_block, line, False)
    for_gen.init(expr_reg, target_type, reverse=False)
    builder.push_loop_stack(step_block, exit_block)
    builder.goto_and_activate(condition_block)
    for_gen.gen_condition()
    builder.activate_block(body_block)
    for_gen.begin_body()
    body_insts(builder.read(for_gen.index_target))
    builder.goto_and_activate(step_block)
    for_gen.gen_step()
    builder.goto(condition_block)
    for_gen.add_cleanup(exit_block)
    builder.pop_loop_stack()
    builder.activate_block(exit_block)

def sequence_from_generator_preallocate_helper(builder: IRBuilder, gen: GeneratorExpr, empty_op_llbuilder: Callable[[Value, int], Value], set_item_op: CFunctionDescription) -> Value | None:
    if False:
        for i in range(10):
            print('nop')
    'Generate a new tuple or list from a simple generator expression.\n\n    Currently we only optimize for simplest generator expression, which means that\n    there is no condition list in the generator and only one original sequence with\n    one index is allowed.\n\n    e.g.  (1) tuple(f(x) for x in a_list/a_tuple)\n          (2) list(f(x) for x in a_list/a_tuple)\n          (3) [f(x) for x in a_list/a_tuple]\n    RTuple as an original sequence is not supported yet.\n\n    Args:\n        empty_op_llbuilder: A function that can generate an empty sequence op when\n            passed in length. See `new_list_op_with_length` and `new_tuple_op_with_length`\n            for detailed implementation.\n        set_item_op: A primitive that can modify an arbitrary position of a sequence.\n            The op should have three arguments:\n                - Self\n                - Target position\n                - New Value\n            See `new_list_set_item_op` and `new_tuple_set_item_op` for detailed\n            implementation.\n    '
    if len(gen.sequences) == 1 and len(gen.indices) == 1 and (len(gen.condlists[0]) == 0):
        rtype = builder.node_type(gen.sequences[0])
        if is_list_rprimitive(rtype) or is_tuple_rprimitive(rtype) or is_str_rprimitive(rtype):
            sequence = builder.accept(gen.sequences[0])
            length = builder.builder.builtin_len(sequence, gen.line, use_pyssize_t=True)
            target_op = empty_op_llbuilder(length, gen.line)

            def set_item(item_index: Value) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                e = builder.accept(gen.left_expr)
                builder.call_c(set_item_op, [target_op, item_index, e], gen.line)
            for_loop_helper_with_index(builder, gen.indices[0], gen.sequences[0], sequence, set_item, gen.line)
            return target_op
    return None

def translate_list_comprehension(builder: IRBuilder, gen: GeneratorExpr) -> Value:
    if False:
        while True:
            i = 10
    val = sequence_from_generator_preallocate_helper(builder, gen, empty_op_llbuilder=builder.builder.new_list_op_with_length, set_item_op=new_list_set_item_op)
    if val is not None:
        return val
    list_ops = builder.maybe_spill(builder.new_list_op([], gen.line))
    loop_params = list(zip(gen.indices, gen.sequences, gen.condlists, gen.is_async))

    def gen_inner_stmts() -> None:
        if False:
            for i in range(10):
                print('nop')
        e = builder.accept(gen.left_expr)
        builder.call_c(list_append_op, [builder.read(list_ops), e], gen.line)
    comprehension_helper(builder, loop_params, gen_inner_stmts, gen.line)
    return builder.read(list_ops)

def translate_set_comprehension(builder: IRBuilder, gen: GeneratorExpr) -> Value:
    if False:
        return 10
    set_ops = builder.maybe_spill(builder.new_set_op([], gen.line))
    loop_params = list(zip(gen.indices, gen.sequences, gen.condlists, gen.is_async))

    def gen_inner_stmts() -> None:
        if False:
            for i in range(10):
                print('nop')
        e = builder.accept(gen.left_expr)
        builder.call_c(set_add_op, [builder.read(set_ops), e], gen.line)
    comprehension_helper(builder, loop_params, gen_inner_stmts, gen.line)
    return builder.read(set_ops)

def comprehension_helper(builder: IRBuilder, loop_params: list[tuple[Lvalue, Expression, list[Expression], bool]], gen_inner_stmts: Callable[[], None], line: int) -> None:
    if False:
        while True:
            i = 10
    'Helper function for list comprehensions.\n\n    Args:\n        loop_params: a list of (index, expr, [conditions]) tuples defining nested loops:\n            - "index" is the Lvalue indexing that loop;\n            - "expr" is the expression for the object to be iterated over;\n            - "conditions" is a list of conditions, evaluated in order with short-circuiting,\n                that must all be true for the loop body to be executed\n        gen_inner_stmts: function to generate the IR for the body of the innermost loop\n    '

    def handle_loop(loop_params: list[tuple[Lvalue, Expression, list[Expression], bool]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Generate IR for a loop.\n\n        Given a list of (index, expression, [conditions]) tuples, generate IR\n        for the nested loops the list defines.\n        '
        (index, expr, conds, is_async) = loop_params[0]
        for_loop_helper(builder, index, expr, lambda : loop_contents(conds, loop_params[1:]), None, is_async=is_async, line=line)

    def loop_contents(conds: list[Expression], remaining_loop_params: list[tuple[Lvalue, Expression, list[Expression], bool]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Generate the body of the loop.\n\n        Args:\n            conds: a list of conditions to be evaluated (in order, with short circuiting)\n                to gate the body of the loop\n            remaining_loop_params: the parameters for any further nested loops; if it\'s empty\n                we\'ll instead evaluate the "gen_inner_stmts" function\n        '
        for cond in conds:
            cond_val = builder.accept(cond)
            (cont_block, rest_block) = (BasicBlock(), BasicBlock())
            builder.add_bool_branch(cond_val, rest_block, cont_block)
            builder.activate_block(cont_block)
            builder.nonlocal_control[-1].gen_continue(builder, cond.line)
            builder.goto_and_activate(rest_block)
        if remaining_loop_params:
            return handle_loop(remaining_loop_params)
        else:
            gen_inner_stmts()
    handle_loop(loop_params)

def is_range_ref(expr: RefExpr) -> bool:
    if False:
        while True:
            i = 10
    return expr.fullname == 'builtins.range' or (isinstance(expr.node, TypeAlias) and expr.fullname == 'six.moves.xrange')

def make_for_loop_generator(builder: IRBuilder, index: Lvalue, expr: Expression, body_block: BasicBlock, loop_exit: BasicBlock, line: int, is_async: bool=False, nested: bool=False) -> ForGenerator:
    if False:
        print('Hello World!')
    'Return helper object for generating a for loop over an iterable.\n\n    If "nested" is True, this is a nested iterator such as "e" in "enumerate(e)".\n    '
    if is_async:
        expr_reg = builder.accept(expr)
        async_obj = ForAsyncIterable(builder, index, body_block, loop_exit, line, nested)
        item_type = builder._analyze_iterable_item_type(expr)
        item_rtype = builder.type_to_rtype(item_type)
        async_obj.init(expr_reg, item_rtype)
        return async_obj
    rtyp = builder.node_type(expr)
    if is_sequence_rprimitive(rtyp):
        expr_reg = builder.accept(expr)
        target_type = builder.get_sequence_type(expr)
        for_list = ForSequence(builder, index, body_block, loop_exit, line, nested)
        for_list.init(expr_reg, target_type, reverse=False)
        return for_list
    if is_dict_rprimitive(rtyp):
        expr_reg = builder.accept(expr)
        target_type = builder.get_dict_key_type(expr)
        for_dict = ForDictionaryKeys(builder, index, body_block, loop_exit, line, nested)
        for_dict.init(expr_reg, target_type)
        return for_dict
    if isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr):
        if is_range_ref(expr.callee) and (len(expr.args) <= 2 or (len(expr.args) == 3 and builder.extract_int(expr.args[2]) is not None)) and (set(expr.arg_kinds) == {ARG_POS}):
            if len(expr.args) == 1:
                start_reg: Value = Integer(0)
                end_reg = builder.accept(expr.args[0])
            else:
                start_reg = builder.accept(expr.args[0])
                end_reg = builder.accept(expr.args[1])
            if len(expr.args) == 3:
                step = builder.extract_int(expr.args[2])
                assert step is not None
                if step == 0:
                    builder.error("range() step can't be zero", expr.args[2].line)
            else:
                step = 1
            for_range = ForRange(builder, index, body_block, loop_exit, line, nested)
            for_range.init(start_reg, end_reg, step)
            return for_range
        elif expr.callee.fullname == 'builtins.enumerate' and len(expr.args) == 1 and (expr.arg_kinds == [ARG_POS]) and isinstance(index, TupleExpr) and (len(index.items) == 2):
            lvalue1 = index.items[0]
            lvalue2 = index.items[1]
            for_enumerate = ForEnumerate(builder, index, body_block, loop_exit, line, nested)
            for_enumerate.init(lvalue1, lvalue2, expr.args[0])
            return for_enumerate
        elif expr.callee.fullname == 'builtins.zip' and len(expr.args) >= 2 and (set(expr.arg_kinds) == {ARG_POS}) and isinstance(index, TupleExpr) and (len(index.items) == len(expr.args)):
            for_zip = ForZip(builder, index, body_block, loop_exit, line, nested)
            for_zip.init(index.items, expr.args)
            return for_zip
        if expr.callee.fullname == 'builtins.reversed' and len(expr.args) == 1 and (expr.arg_kinds == [ARG_POS]) and is_sequence_rprimitive(builder.node_type(expr.args[0])):
            expr_reg = builder.accept(expr.args[0])
            target_type = builder.get_sequence_type(expr)
            for_list = ForSequence(builder, index, body_block, loop_exit, line, nested)
            for_list.init(expr_reg, target_type, reverse=True)
            return for_list
    if isinstance(expr, CallExpr) and isinstance(expr.callee, MemberExpr) and (not expr.args):
        rtype = builder.node_type(expr.callee.expr)
        if is_dict_rprimitive(rtype) and expr.callee.name in ('keys', 'values', 'items'):
            expr_reg = builder.accept(expr.callee.expr)
            for_dict_type: type[ForGenerator] | None = None
            if expr.callee.name == 'keys':
                target_type = builder.get_dict_key_type(expr.callee.expr)
                for_dict_type = ForDictionaryKeys
            elif expr.callee.name == 'values':
                target_type = builder.get_dict_value_type(expr.callee.expr)
                for_dict_type = ForDictionaryValues
            else:
                target_type = builder.get_dict_item_type(expr.callee.expr)
                for_dict_type = ForDictionaryItems
            for_dict_gen = for_dict_type(builder, index, body_block, loop_exit, line, nested)
            for_dict_gen.init(expr_reg, target_type)
            return for_dict_gen
    iterable_expr_reg: Value | None = None
    if isinstance(expr, SetExpr):
        from mypyc.irbuild.expression import precompute_set_literal
        set_literal = precompute_set_literal(builder, expr)
        if set_literal is not None:
            iterable_expr_reg = set_literal
    if iterable_expr_reg is None:
        iterable_expr_reg = builder.accept(expr)
    for_obj = ForIterable(builder, index, body_block, loop_exit, line, nested)
    item_type = builder._analyze_iterable_item_type(expr)
    item_rtype = builder.type_to_rtype(item_type)
    for_obj.init(iterable_expr_reg, item_rtype)
    return for_obj

class ForGenerator:
    """Abstract base class for generating for loops."""

    def __init__(self, builder: IRBuilder, index: Lvalue, body_block: BasicBlock, loop_exit: BasicBlock, line: int, nested: bool) -> None:
        if False:
            i = 10
            return i + 15
        self.builder = builder
        self.index = index
        self.body_block = body_block
        self.line = line
        if self.need_cleanup() and (not nested):
            self.loop_exit = BasicBlock()
        else:
            self.loop_exit = loop_exit

    def need_cleanup(self) -> bool:
        if False:
            while True:
                i = 10
        'If this returns true, we need post-loop cleanup.'
        return False

    def add_cleanup(self, exit_block: BasicBlock) -> None:
        if False:
            i = 10
            return i + 15
        'Add post-loop cleanup, if needed.'
        if self.need_cleanup():
            self.builder.activate_block(self.loop_exit)
            self.gen_cleanup()
            self.builder.goto(exit_block)

    def gen_condition(self) -> None:
        if False:
            return 10
        'Generate check for loop exit (e.g. exhaustion of iteration).'

    def begin_body(self) -> None:
        if False:
            while True:
                i = 10
        'Generate ops at the beginning of the body (if needed).'

    def gen_step(self) -> None:
        if False:
            print('Hello World!')
        'Generate stepping to the next item (if needed).'

    def gen_cleanup(self) -> None:
        if False:
            return 10
        'Generate post-loop cleanup (if needed).'

    def load_len(self, expr: Value | AssignmentTarget) -> Value:
        if False:
            i = 10
            return i + 15
        'A helper to get collection length, used by several subclasses.'
        return self.builder.builder.builtin_len(self.builder.read(expr, self.line), self.line)

class ForIterable(ForGenerator):
    """Generate IR for a for loop over an arbitrary iterable (the general case)."""

    def need_cleanup(self) -> bool:
        if False:
            print('Hello World!')
        return True

    def init(self, expr_reg: Value, target_type: RType) -> None:
        if False:
            print('Hello World!')
        builder = self.builder
        iter_reg = builder.call_c(iter_op, [expr_reg], self.line)
        builder.maybe_spill(expr_reg)
        self.iter_target = builder.maybe_spill(iter_reg)
        self.target_type = target_type

    def gen_condition(self) -> None:
        if False:
            i = 10
            return i + 15
        builder = self.builder
        line = self.line
        self.next_reg = builder.call_c(next_op, [builder.read(self.iter_target, line)], line)
        builder.add(Branch(self.next_reg, self.loop_exit, self.body_block, Branch.IS_ERROR))

    def begin_body(self) -> None:
        if False:
            return 10
        builder = self.builder
        line = self.line
        next_reg = builder.coerce(self.next_reg, self.target_type, line)
        builder.assign(builder.get_assignment_target(self.index), next_reg, line)

    def gen_step(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def gen_cleanup(self) -> None:
        if False:
            while True:
                i = 10
        self.builder.call_c(no_err_occurred_op, [], self.line)

class ForAsyncIterable(ForGenerator):
    """Generate IR for an async for loop."""

    def init(self, expr_reg: Value, target_type: RType) -> None:
        if False:
            i = 10
            return i + 15
        builder = self.builder
        iter_reg = builder.call_c(aiter_op, [expr_reg], self.line)
        builder.maybe_spill(expr_reg)
        self.iter_target = builder.maybe_spill(iter_reg)
        self.target_type = target_type
        self.stop_reg = Register(bool_rprimitive)

    def gen_condition(self) -> None:
        if False:
            return 10
        from mypyc.irbuild.statement import emit_await, transform_try_except
        builder = self.builder
        line = self.line

        def except_match() -> Value:
            if False:
                i = 10
                return i + 15
            addr = builder.add(LoadAddress(pointer_rprimitive, stop_async_iteration_op.src, line))
            return builder.add(LoadMem(stop_async_iteration_op.type, addr))

        def try_body() -> None:
            if False:
                print('Hello World!')
            awaitable = builder.call_c(anext_op, [builder.read(self.iter_target)], line)
            self.next_reg = emit_await(builder, awaitable, line)
            builder.assign(self.stop_reg, builder.false(), -1)

        def except_body() -> None:
            if False:
                while True:
                    i = 10
            builder.assign(self.stop_reg, builder.true(), line)
        transform_try_except(builder, try_body, [((except_match, line), None, except_body)], None, line)
        builder.add(Branch(self.stop_reg, self.loop_exit, self.body_block, Branch.BOOL))

    def begin_body(self) -> None:
        if False:
            while True:
                i = 10
        builder = self.builder
        line = self.line
        next_reg = builder.coerce(self.next_reg, self.target_type, line)
        builder.assign(builder.get_assignment_target(self.index), next_reg, line)

    def gen_step(self) -> None:
        if False:
            print('Hello World!')
        pass

def unsafe_index(builder: IRBuilder, target: Value, index: Value, line: int) -> Value:
    if False:
        for i in range(10):
            print('nop')
    'Emit a potentially unsafe index into a target.'
    if is_list_rprimitive(target.type):
        return builder.call_c(list_get_item_unsafe_op, [target, index], line)
    else:
        return builder.gen_method_call(target, '__getitem__', [index], None, line)

class ForSequence(ForGenerator):
    """Generate optimized IR for a for loop over a sequence.

    Supports iterating in both forward and reverse.
    """

    def init(self, expr_reg: Value, target_type: RType, reverse: bool) -> None:
        if False:
            while True:
                i = 10
        builder = self.builder
        self.reverse = reverse
        self.expr_target = builder.maybe_spill(expr_reg)
        if not reverse:
            index_reg: Value = Integer(0)
        else:
            index_reg = builder.binary_op(self.load_len(self.expr_target), Integer(1), '-', self.line)
        self.index_target = builder.maybe_spill_assignable(index_reg)
        self.target_type = target_type

    def gen_condition(self) -> None:
        if False:
            return 10
        builder = self.builder
        line = self.line
        if self.reverse:
            comparison = builder.binary_op(builder.read(self.index_target, line), Integer(0), '>=', line)
            second_check = BasicBlock()
            builder.add_bool_branch(comparison, second_check, self.loop_exit)
            builder.activate_block(second_check)
        len_reg = self.load_len(self.expr_target)
        comparison = builder.binary_op(builder.read(self.index_target, line), len_reg, '<', line)
        builder.add_bool_branch(comparison, self.body_block, self.loop_exit)

    def begin_body(self) -> None:
        if False:
            i = 10
            return i + 15
        builder = self.builder
        line = self.line
        value_box = unsafe_index(builder, builder.read(self.expr_target, line), builder.read(self.index_target, line), line)
        assert value_box
        builder.assign(builder.get_assignment_target(self.index), builder.coerce(value_box, self.target_type, line), line)

    def gen_step(self) -> None:
        if False:
            i = 10
            return i + 15
        builder = self.builder
        line = self.line
        step = 1 if not self.reverse else -1
        add = builder.int_op(short_int_rprimitive, builder.read(self.index_target, line), Integer(step), IntOp.ADD, line)
        builder.assign(self.index_target, add, line)

class ForDictionaryCommon(ForGenerator):
    """Generate optimized IR for a for loop over dictionary keys/values.

    The logic is pretty straightforward, we use PyDict_Next() API wrapped in
    a tuple, so that we can modify only a single register. The layout of the tuple:
      * f0: are there more items (bool)
      * f1: current offset (int)
      * f2: next key (object)
      * f3: next value (object)
    For more info see https://docs.python.org/3/c-api/dict.html#c.PyDict_Next.

    Note that for subclasses we fall back to generic PyObject_GetIter() logic,
    since they may override some iteration methods in subtly incompatible manner.
    The fallback logic is implemented in CPy.h via dynamic type check.
    """
    dict_next_op: ClassVar[CFunctionDescription]
    dict_iter_op: ClassVar[CFunctionDescription]

    def need_cleanup(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def init(self, expr_reg: Value, target_type: RType) -> None:
        if False:
            print('Hello World!')
        builder = self.builder
        self.target_type = target_type
        self.expr_target = builder.maybe_spill(expr_reg)
        offset = Integer(0)
        self.offset_target = builder.maybe_spill_assignable(offset)
        self.size = builder.maybe_spill(self.load_len(self.expr_target))
        iter_reg = builder.call_c(self.dict_iter_op, [expr_reg], self.line)
        self.iter_target = builder.maybe_spill(iter_reg)

    def gen_condition(self) -> None:
        if False:
            print('Hello World!')
        'Get next key/value pair, set new offset, and check if we should continue.'
        builder = self.builder
        line = self.line
        self.next_tuple = self.builder.call_c(self.dict_next_op, [builder.read(self.iter_target, line), builder.read(self.offset_target, line)], line)
        new_offset = builder.add(TupleGet(self.next_tuple, 1, line))
        builder.assign(self.offset_target, new_offset, line)
        should_continue = builder.add(TupleGet(self.next_tuple, 0, line))
        builder.add(Branch(should_continue, self.body_block, self.loop_exit, Branch.BOOL))

    def gen_step(self) -> None:
        if False:
            print('Hello World!')
        "Check that dictionary didn't change size during iteration.\n\n        Raise RuntimeError if it is not the case to match CPython behavior.\n        "
        builder = self.builder
        line = self.line
        builder.call_c(dict_check_size_op, [builder.read(self.expr_target, line), builder.read(self.size, line)], line)

    def gen_cleanup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.builder.call_c(no_err_occurred_op, [], self.line)

class ForDictionaryKeys(ForDictionaryCommon):
    """Generate optimized IR for a for loop over dictionary keys."""
    dict_next_op = dict_next_key_op
    dict_iter_op = dict_key_iter_op

    def begin_body(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        builder = self.builder
        line = self.line
        key = builder.add(TupleGet(self.next_tuple, 2, line))
        builder.assign(builder.get_assignment_target(self.index), builder.coerce(key, self.target_type, line), line)

class ForDictionaryValues(ForDictionaryCommon):
    """Generate optimized IR for a for loop over dictionary values."""
    dict_next_op = dict_next_value_op
    dict_iter_op = dict_value_iter_op

    def begin_body(self) -> None:
        if False:
            print('Hello World!')
        builder = self.builder
        line = self.line
        value = builder.add(TupleGet(self.next_tuple, 2, line))
        builder.assign(builder.get_assignment_target(self.index), builder.coerce(value, self.target_type, line), line)

class ForDictionaryItems(ForDictionaryCommon):
    """Generate optimized IR for a for loop over dictionary items."""
    dict_next_op = dict_next_item_op
    dict_iter_op = dict_item_iter_op

    def begin_body(self) -> None:
        if False:
            return 10
        builder = self.builder
        line = self.line
        key = builder.add(TupleGet(self.next_tuple, 2, line))
        value = builder.add(TupleGet(self.next_tuple, 3, line))
        assert isinstance(self.target_type, RTuple)
        key = builder.coerce(key, self.target_type.types[0], line)
        value = builder.coerce(value, self.target_type.types[1], line)
        target = builder.get_assignment_target(self.index)
        if isinstance(target, AssignmentTargetTuple):
            if len(target.items) != 2:
                builder.error('Expected a pair for dict item iteration', line)
            builder.assign(target.items[0], key, line)
            builder.assign(target.items[1], value, line)
        else:
            rvalue = builder.add(TupleSet([key, value], line))
            builder.assign(target, rvalue, line)

class ForRange(ForGenerator):
    """Generate optimized IR for a for loop over an integer range."""

    def init(self, start_reg: Value, end_reg: Value, step: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        builder = self.builder
        self.start_reg = start_reg
        self.end_reg = end_reg
        self.step = step
        self.end_target = builder.maybe_spill(end_reg)
        if is_short_int_rprimitive(start_reg.type) and is_short_int_rprimitive(end_reg.type):
            index_type: RType = short_int_rprimitive
        elif is_fixed_width_rtype(end_reg.type):
            index_type = end_reg.type
        else:
            index_type = int_rprimitive
        index_reg = Register(index_type)
        builder.assign(index_reg, start_reg, -1)
        self.index_reg = builder.maybe_spill_assignable(index_reg)
        self.index_target: Register | AssignmentTarget = builder.get_assignment_target(self.index)
        builder.assign(self.index_target, builder.read(self.index_reg, self.line), self.line)

    def gen_condition(self) -> None:
        if False:
            i = 10
            return i + 15
        builder = self.builder
        line = self.line
        cmp = '<' if self.step > 0 else '>'
        comparison = builder.binary_op(builder.read(self.index_reg, line), builder.read(self.end_target, line), cmp, line)
        builder.add_bool_branch(comparison, self.body_block, self.loop_exit)

    def gen_step(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        builder = self.builder
        line = self.line
        if is_short_int_rprimitive(self.start_reg.type) and is_short_int_rprimitive(self.end_reg.type):
            new_val = builder.int_op(short_int_rprimitive, builder.read(self.index_reg, line), Integer(self.step), IntOp.ADD, line)
        else:
            new_val = builder.binary_op(builder.read(self.index_reg, line), Integer(self.step), '+', line)
        builder.assign(self.index_reg, new_val, line)
        builder.assign(self.index_target, new_val, line)

class ForInfiniteCounter(ForGenerator):
    """Generate optimized IR for a for loop counting from 0 to infinity."""

    def init(self) -> None:
        if False:
            i = 10
            return i + 15
        builder = self.builder
        zero = Integer(0)
        self.index_reg = builder.maybe_spill_assignable(zero)
        self.index_target: Register | AssignmentTarget = builder.get_assignment_target(self.index)
        builder.assign(self.index_target, zero, self.line)

    def gen_step(self) -> None:
        if False:
            i = 10
            return i + 15
        builder = self.builder
        line = self.line
        new_val = builder.int_op(short_int_rprimitive, builder.read(self.index_reg, line), Integer(1), IntOp.ADD, line)
        builder.assign(self.index_reg, new_val, line)
        builder.assign(self.index_target, new_val, line)

class ForEnumerate(ForGenerator):
    """Generate optimized IR for a for loop of form "for i, x in enumerate(it)"."""

    def need_cleanup(self) -> bool:
        if False:
            print('Hello World!')
        return True

    def init(self, index1: Lvalue, index2: Lvalue, expr: Expression) -> None:
        if False:
            while True:
                i = 10
        self.index_gen = ForInfiniteCounter(self.builder, index1, self.body_block, self.loop_exit, self.line, nested=True)
        self.index_gen.init()
        self.main_gen = make_for_loop_generator(self.builder, index2, expr, self.body_block, self.loop_exit, self.line, nested=True)

    def gen_condition(self) -> None:
        if False:
            while True:
                i = 10
        self.main_gen.gen_condition()

    def begin_body(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.index_gen.begin_body()
        self.main_gen.begin_body()

    def gen_step(self) -> None:
        if False:
            i = 10
            return i + 15
        self.index_gen.gen_step()
        self.main_gen.gen_step()

    def gen_cleanup(self) -> None:
        if False:
            print('Hello World!')
        self.index_gen.gen_cleanup()
        self.main_gen.gen_cleanup()

class ForZip(ForGenerator):
    """Generate IR for a for loop of form `for x, ... in zip(a, ...)`."""

    def need_cleanup(self) -> bool:
        if False:
            return 10
        return True

    def init(self, indexes: list[Lvalue], exprs: list[Expression]) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert len(indexes) == len(exprs)
        self.cond_blocks = [BasicBlock() for _ in range(len(indexes) - 1)] + [self.body_block]
        self.gens: list[ForGenerator] = []
        for (index, expr, next_block) in zip(indexes, exprs, self.cond_blocks):
            gen = make_for_loop_generator(self.builder, index, expr, next_block, self.loop_exit, self.line, nested=True)
            self.gens.append(gen)

    def gen_condition(self) -> None:
        if False:
            while True:
                i = 10
        for (i, gen) in enumerate(self.gens):
            gen.gen_condition()
            if i < len(self.gens) - 1:
                self.builder.activate_block(self.cond_blocks[i])

    def begin_body(self) -> None:
        if False:
            print('Hello World!')
        for gen in self.gens:
            gen.begin_body()

    def gen_step(self) -> None:
        if False:
            return 10
        for gen in self.gens:
            gen.gen_step()

    def gen_cleanup(self) -> None:
        if False:
            return 10
        for gen in self.gens:
            gen.gen_cleanup()