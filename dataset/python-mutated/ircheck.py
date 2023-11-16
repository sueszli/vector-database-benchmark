"""Utilities for checking that internal ir is valid and consistent."""
from __future__ import annotations
from mypyc.ir.func_ir import FUNC_STATICMETHOD, FuncIR
from mypyc.ir.ops import Assign, AssignMulti, BaseAssign, BasicBlock, Box, Branch, Call, CallC, Cast, ComparisonOp, ControlOp, DecRef, Extend, FloatComparisonOp, FloatNeg, FloatOp, GetAttr, GetElementPtr, Goto, IncRef, InitStatic, Integer, IntOp, KeepAlive, LoadAddress, LoadErrorValue, LoadGlobal, LoadLiteral, LoadMem, LoadStatic, MethodCall, Op, OpVisitor, RaiseStandardError, Register, Return, SetAttr, SetMem, Truncate, TupleGet, TupleSet, Unborrow, Unbox, Unreachable, Value
from mypyc.ir.pprint import format_func
from mypyc.ir.rtypes import RArray, RInstance, RPrimitive, RType, RUnion, bytes_rprimitive, dict_rprimitive, int_rprimitive, is_float_rprimitive, is_object_rprimitive, list_rprimitive, range_rprimitive, set_rprimitive, str_rprimitive, tuple_rprimitive

class FnError:

    def __init__(self, source: Op | BasicBlock, desc: str) -> None:
        if False:
            i = 10
            return i + 15
        self.source = source
        self.desc = desc

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(other, FnError) and self.source == other.source and (self.desc == other.desc)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'FnError(source={self.source}, desc={self.desc})'

def check_func_ir(fn: FuncIR) -> list[FnError]:
    if False:
        i = 10
        return i + 15
    'Applies validations to a given function ir and returns a list of errors found.'
    errors = []
    op_set = set()
    for block in fn.blocks:
        if not block.terminated:
            errors.append(FnError(source=block.ops[-1] if block.ops else block, desc='Block not terminated'))
        for op in block.ops[:-1]:
            if isinstance(op, ControlOp):
                errors.append(FnError(source=op, desc='Block has operations after control op'))
            if op in op_set:
                errors.append(FnError(source=op, desc='Func has a duplicate op'))
            op_set.add(op)
    errors.extend(check_op_sources_valid(fn))
    if errors:
        return errors
    op_checker = OpChecker(fn)
    for block in fn.blocks:
        for op in block.ops:
            op.accept(op_checker)
    return op_checker.errors

class IrCheckException(Exception):
    pass

def assert_func_ir_valid(fn: FuncIR) -> None:
    if False:
        return 10
    errors = check_func_ir(fn)
    if errors:
        raise IrCheckException('Internal error: Generated invalid IR: \n' + '\n'.join(format_func(fn, [(e.source, e.desc) for e in errors])))

def check_op_sources_valid(fn: FuncIR) -> list[FnError]:
    if False:
        while True:
            i = 10
    errors = []
    valid_ops: set[Op] = set()
    valid_registers: set[Register] = set()
    for block in fn.blocks:
        valid_ops.update(block.ops)
        for op in block.ops:
            if isinstance(op, BaseAssign):
                valid_registers.add(op.dest)
            elif isinstance(op, LoadAddress) and isinstance(op.src, Register):
                valid_registers.add(op.src)
    valid_registers.update(fn.arg_regs)
    for block in fn.blocks:
        for op in block.ops:
            for source in op.sources():
                if isinstance(source, Integer):
                    pass
                elif isinstance(source, Op):
                    if source not in valid_ops:
                        errors.append(FnError(source=op, desc=f'Invalid op reference to op of type {type(source).__name__}'))
                elif isinstance(source, Register):
                    if source not in valid_registers:
                        errors.append(FnError(source=op, desc=f'Invalid op reference to register {source.name!r}'))
    return errors
disjoint_types = {int_rprimitive.name, bytes_rprimitive.name, str_rprimitive.name, dict_rprimitive.name, list_rprimitive.name, set_rprimitive.name, tuple_rprimitive.name, range_rprimitive.name}

def can_coerce_to(src: RType, dest: RType) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check if src can be assigned to dest_rtype.\n\n    Currently okay to have false positives.\n    '
    if isinstance(dest, RUnion):
        return any((can_coerce_to(src, d) for d in dest.items))
    if isinstance(dest, RPrimitive):
        if isinstance(src, RPrimitive):
            if src.name in disjoint_types and dest.name in disjoint_types:
                return src.name == dest.name
            return src.size == dest.size
        if isinstance(src, RInstance):
            return is_object_rprimitive(dest)
        if isinstance(src, RUnion):
            return any((can_coerce_to(s, dest) for s in src.items))
        return False
    return True

class OpChecker(OpVisitor[None]):

    def __init__(self, parent_fn: FuncIR) -> None:
        if False:
            while True:
                i = 10
        self.parent_fn = parent_fn
        self.errors: list[FnError] = []

    def fail(self, source: Op, desc: str) -> None:
        if False:
            while True:
                i = 10
        self.errors.append(FnError(source=source, desc=desc))

    def check_control_op_targets(self, op: ControlOp) -> None:
        if False:
            for i in range(10):
                print('nop')
        for target in op.targets():
            if target not in self.parent_fn.blocks:
                self.fail(source=op, desc=f'Invalid control operation target: {target.label}')

    def check_type_coercion(self, op: Op, src: RType, dest: RType) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not can_coerce_to(src, dest):
            self.fail(source=op, desc=f'Cannot coerce source type {src.name} to dest type {dest.name}')

    def check_compatibility(self, op: Op, t: RType, s: RType) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not can_coerce_to(t, s) or not can_coerce_to(s, t):
            self.fail(source=op, desc=f'{t.name} and {s.name} are not compatible')

    def expect_float(self, op: Op, v: Value) -> None:
        if False:
            return 10
        if not is_float_rprimitive(v.type):
            self.fail(op, f'Float expected (actual type is {v.type})')

    def expect_non_float(self, op: Op, v: Value) -> None:
        if False:
            return 10
        if is_float_rprimitive(v.type):
            self.fail(op, 'Float not expected')

    def visit_goto(self, op: Goto) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.check_control_op_targets(op)

    def visit_branch(self, op: Branch) -> None:
        if False:
            while True:
                i = 10
        self.check_control_op_targets(op)

    def visit_return(self, op: Return) -> None:
        if False:
            return 10
        self.check_type_coercion(op, op.value.type, self.parent_fn.decl.sig.ret_type)

    def visit_unreachable(self, op: Unreachable) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_assign(self, op: Assign) -> None:
        if False:
            while True:
                i = 10
        self.check_type_coercion(op, op.src.type, op.dest.type)

    def visit_assign_multi(self, op: AssignMulti) -> None:
        if False:
            for i in range(10):
                print('nop')
        for src in op.src:
            assert isinstance(op.dest.type, RArray)
            self.check_type_coercion(op, src.type, op.dest.type.item_type)

    def visit_load_error_value(self, op: LoadErrorValue) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def check_tuple_items_valid_literals(self, op: LoadLiteral, t: tuple[object, ...]) -> None:
        if False:
            for i in range(10):
                print('nop')
        for x in t:
            if x is not None and (not isinstance(x, (str, bytes, bool, int, float, complex, tuple))):
                self.fail(op, f'Invalid type for item of tuple literal: {type(x)})')
            if isinstance(x, tuple):
                self.check_tuple_items_valid_literals(op, x)

    def check_frozenset_items_valid_literals(self, op: LoadLiteral, s: frozenset[object]) -> None:
        if False:
            return 10
        for x in s:
            if x is None or isinstance(x, (str, bytes, bool, int, float, complex)):
                pass
            elif isinstance(x, tuple):
                self.check_tuple_items_valid_literals(op, x)
            else:
                self.fail(op, f'Invalid type for item of frozenset literal: {type(x)})')

    def visit_load_literal(self, op: LoadLiteral) -> None:
        if False:
            return 10
        expected_type = None
        if op.value is None:
            expected_type = 'builtins.object'
        elif isinstance(op.value, int):
            expected_type = 'builtins.int'
        elif isinstance(op.value, str):
            expected_type = 'builtins.str'
        elif isinstance(op.value, bytes):
            expected_type = 'builtins.bytes'
        elif isinstance(op.value, bool):
            expected_type = 'builtins.object'
        elif isinstance(op.value, float):
            expected_type = 'builtins.float'
        elif isinstance(op.value, complex):
            expected_type = 'builtins.object'
        elif isinstance(op.value, tuple):
            expected_type = 'builtins.tuple'
            self.check_tuple_items_valid_literals(op, op.value)
        elif isinstance(op.value, frozenset):
            expected_type = 'builtins.set'
            self.check_frozenset_items_valid_literals(op, op.value)
        assert expected_type is not None, 'Missed a case for LoadLiteral check'
        if op.type.name not in [expected_type, 'builtins.object']:
            self.fail(op, f'Invalid literal value for type: value has type {expected_type}, but op has type {op.type.name}')

    def visit_get_attr(self, op: GetAttr) -> None:
        if False:
            print('Hello World!')
        pass

    def visit_set_attr(self, op: SetAttr) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_load_static(self, op: LoadStatic) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_init_static(self, op: InitStatic) -> None:
        if False:
            print('Hello World!')
        pass

    def visit_tuple_get(self, op: TupleGet) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_tuple_set(self, op: TupleSet) -> None:
        if False:
            print('Hello World!')
        pass

    def visit_inc_ref(self, op: IncRef) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_dec_ref(self, op: DecRef) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_call(self, op: Call) -> None:
        if False:
            for i in range(10):
                print('nop')
        for (arg_value, arg_runtime) in zip(op.args, op.fn.sig.args):
            self.check_type_coercion(op, arg_value.type, arg_runtime.type)

    def visit_method_call(self, op: MethodCall) -> None:
        if False:
            print('Hello World!')
        method_decl = op.receiver_type.class_ir.method_decl(op.method)
        if method_decl.kind == FUNC_STATICMETHOD:
            decl_index = 0
        else:
            decl_index = 1
        if len(op.args) + decl_index != len(method_decl.sig.args):
            self.fail(op, 'Incorrect number of args for method call.')
        for (arg_value, arg_runtime) in zip(op.args, method_decl.sig.args[decl_index:]):
            self.check_type_coercion(op, arg_value.type, arg_runtime.type)

    def visit_cast(self, op: Cast) -> None:
        if False:
            return 10
        pass

    def visit_box(self, op: Box) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_unbox(self, op: Unbox) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_raise_standard_error(self, op: RaiseStandardError) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_call_c(self, op: CallC) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_truncate(self, op: Truncate) -> None:
        if False:
            print('Hello World!')
        pass

    def visit_extend(self, op: Extend) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_load_global(self, op: LoadGlobal) -> None:
        if False:
            return 10
        pass

    def visit_int_op(self, op: IntOp) -> None:
        if False:
            i = 10
            return i + 15
        self.expect_non_float(op, op.lhs)
        self.expect_non_float(op, op.rhs)

    def visit_comparison_op(self, op: ComparisonOp) -> None:
        if False:
            while True:
                i = 10
        self.check_compatibility(op, op.lhs.type, op.rhs.type)
        self.expect_non_float(op, op.lhs)
        self.expect_non_float(op, op.rhs)

    def visit_float_op(self, op: FloatOp) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.expect_float(op, op.lhs)
        self.expect_float(op, op.rhs)

    def visit_float_neg(self, op: FloatNeg) -> None:
        if False:
            i = 10
            return i + 15
        self.expect_float(op, op.src)

    def visit_float_comparison_op(self, op: FloatComparisonOp) -> None:
        if False:
            print('Hello World!')
        self.expect_float(op, op.lhs)
        self.expect_float(op, op.rhs)

    def visit_load_mem(self, op: LoadMem) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_set_mem(self, op: SetMem) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_get_element_ptr(self, op: GetElementPtr) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_load_address(self, op: LoadAddress) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_keep_alive(self, op: KeepAlive) -> None:
        if False:
            return 10
        pass

    def visit_unborrow(self, op: Unborrow) -> None:
        if False:
            i = 10
            return i + 15
        pass