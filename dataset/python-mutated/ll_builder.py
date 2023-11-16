"""A "low-level" IR builder class.

LowLevelIRBuilder provides core abstractions we use for constructing
IR as well as a number of higher-level ones (accessing attributes,
calling functions and methods, and coercing between types, for
example). The core principle of the low-level IR builder is that all
of its facilities operate solely on the IR level and not the AST
level---it has *no knowledge* of mypy types or expressions.
"""
from __future__ import annotations
from typing import Callable, Final, Optional, Sequence, Tuple
from mypy.argmap import map_actuals_to_formals
from mypy.nodes import ARG_POS, ARG_STAR, ARG_STAR2, ArgKind
from mypy.operators import op_methods
from mypy.types import AnyType, TypeOfAny
from mypyc.common import BITMAP_BITS, FAST_ISINSTANCE_MAX_SUBCLASSES, MAX_LITERAL_SHORT_INT, MAX_SHORT_INT, MIN_LITERAL_SHORT_INT, MIN_SHORT_INT, PLATFORM_SIZE, use_method_vectorcall, use_vectorcall
from mypyc.errors import Errors
from mypyc.ir.class_ir import ClassIR, all_concrete_classes
from mypyc.ir.func_ir import FuncDecl, FuncSignature
from mypyc.ir.ops import ERR_FALSE, ERR_NEVER, NAMESPACE_MODULE, NAMESPACE_STATIC, NAMESPACE_TYPE, Assign, AssignMulti, BasicBlock, Box, Branch, Call, CallC, Cast, ComparisonOp, Extend, Float, FloatComparisonOp, FloatNeg, FloatOp, GetAttr, GetElementPtr, Goto, Integer, IntOp, KeepAlive, LoadAddress, LoadErrorValue, LoadLiteral, LoadMem, LoadStatic, MethodCall, Op, RaiseStandardError, Register, SetMem, Truncate, TupleGet, TupleSet, Unbox, Unreachable, Value, float_comparison_op_to_id, float_op_to_id, int_op_to_id
from mypyc.ir.rtypes import PyListObject, PyObject, PySetObject, PyVarObject, RArray, RInstance, RPrimitive, RTuple, RType, RUnion, bit_rprimitive, bitmap_rprimitive, bool_rprimitive, bytes_rprimitive, c_int_rprimitive, c_pointer_rprimitive, c_pyssize_t_rprimitive, c_size_t_rprimitive, check_native_int_range, dict_rprimitive, float_rprimitive, int_rprimitive, is_bit_rprimitive, is_bool_rprimitive, is_bytes_rprimitive, is_dict_rprimitive, is_fixed_width_rtype, is_float_rprimitive, is_int16_rprimitive, is_int32_rprimitive, is_int64_rprimitive, is_int_rprimitive, is_list_rprimitive, is_none_rprimitive, is_set_rprimitive, is_short_int_rprimitive, is_str_rprimitive, is_tagged, is_tuple_rprimitive, is_uint8_rprimitive, list_rprimitive, none_rprimitive, object_pointer_rprimitive, object_rprimitive, optional_value_type, pointer_rprimitive, short_int_rprimitive, str_rprimitive
from mypyc.irbuild.mapper import Mapper
from mypyc.irbuild.util import concrete_arg_kind
from mypyc.options import CompilerOptions
from mypyc.primitives.bytes_ops import bytes_compare
from mypyc.primitives.dict_ops import dict_build_op, dict_new_op, dict_ssize_t_size_op, dict_update_in_display_op
from mypyc.primitives.exc_ops import err_occurred_op, keep_propagating_op
from mypyc.primitives.float_ops import copysign_op, int_to_float_op
from mypyc.primitives.generic_ops import generic_len_op, generic_ssize_t_len_op, py_call_op, py_call_with_kwargs_op, py_getattr_op, py_method_call_op, py_vectorcall_method_op, py_vectorcall_op
from mypyc.primitives.int_ops import int16_divide_op, int16_mod_op, int16_overflow, int32_divide_op, int32_mod_op, int32_overflow, int64_divide_op, int64_mod_op, int64_to_int_op, int_comparison_op_mapping, int_to_int32_op, int_to_int64_op, ssize_t_to_int_op, uint8_overflow
from mypyc.primitives.list_ops import list_build_op, list_extend_op, new_list_op
from mypyc.primitives.misc_ops import bool_op, fast_isinstance_op, none_object_op
from mypyc.primitives.registry import ERR_NEG_INT, CFunctionDescription, binary_ops, method_call_ops, unary_ops
from mypyc.primitives.set_ops import new_set_op
from mypyc.primitives.str_ops import str_check_if_true, str_ssize_t_size_op, unicode_compare
from mypyc.primitives.tuple_ops import list_tuple_op, new_tuple_op, new_tuple_with_length_op
from mypyc.rt_subtype import is_runtime_subtype
from mypyc.sametype import is_same_type
from mypyc.subtype import is_subtype
DictEntry = Tuple[Optional[Value], Value]
LIST_BUILDING_EXPANSION_THRESHOLD = 10
PY_VECTORCALL_ARGUMENTS_OFFSET: Final = 1 << PLATFORM_SIZE * 8 - 1
FIXED_WIDTH_INT_BINARY_OPS: Final = {'+', '-', '*', '//', '%', '&', '|', '^', '<<', '>>', '+=', '-=', '*=', '//=', '%=', '&=', '|=', '^=', '<<=', '>>='}
BOOL_BINARY_OPS: Final = {'&', '&=', '|', '|=', '^', '^=', '==', '!=', '<', '<=', '>', '>='}

class LowLevelIRBuilder:

    def __init__(self, current_module: str, errors: Errors, mapper: Mapper, options: CompilerOptions) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.current_module = current_module
        self.errors = errors
        self.mapper = mapper
        self.options = options
        self.args: list[Register] = []
        self.blocks: list[BasicBlock] = []
        self.error_handlers: list[BasicBlock | None] = [None]
        self.keep_alives: list[Value] = []

    def set_module(self, module_name: str, module_path: str) -> None:
        if False:
            print('Hello World!')
        'Set the name and path of the current module.'
        self.module_name = module_name
        self.module_path = module_path

    def add(self, op: Op) -> Value:
        if False:
            while True:
                i = 10
        'Add an op.'
        assert not self.blocks[-1].terminated, "Can't add to finished block"
        self.blocks[-1].ops.append(op)
        return op

    def goto(self, target: BasicBlock) -> None:
        if False:
            print('Hello World!')
        'Add goto to a basic block.'
        if not self.blocks[-1].terminated:
            self.add(Goto(target))

    def activate_block(self, block: BasicBlock) -> None:
        if False:
            while True:
                i = 10
        'Add a basic block and make it the active one (target of adds).'
        if self.blocks:
            assert self.blocks[-1].terminated
        block.error_handler = self.error_handlers[-1]
        self.blocks.append(block)

    def goto_and_activate(self, block: BasicBlock) -> None:
        if False:
            print('Hello World!')
        'Add goto a block and make it the active block.'
        self.goto(block)
        self.activate_block(block)

    def keep_alive(self, values: list[Value], *, steal: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self.add(KeepAlive(values, steal=steal))

    def push_error_handler(self, handler: BasicBlock | None) -> None:
        if False:
            while True:
                i = 10
        self.error_handlers.append(handler)

    def pop_error_handler(self) -> BasicBlock | None:
        if False:
            i = 10
            return i + 15
        return self.error_handlers.pop()

    def self(self) -> Register:
        if False:
            while True:
                i = 10
        "Return reference to the 'self' argument.\n\n        This only works in a method.\n        "
        return self.args[0]

    def flush_keep_alives(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.keep_alives:
            self.add(KeepAlive(self.keep_alives.copy()))
            self.keep_alives = []

    def box(self, src: Value) -> Value:
        if False:
            i = 10
            return i + 15
        if src.type.is_unboxed:
            if isinstance(src, Integer) and is_tagged(src.type):
                return self.add(LoadLiteral(src.value >> 1, rtype=object_rprimitive))
            return self.add(Box(src))
        else:
            return src

    def unbox_or_cast(self, src: Value, target_type: RType, line: int, *, can_borrow: bool=False) -> Value:
        if False:
            i = 10
            return i + 15
        if target_type.is_unboxed:
            return self.add(Unbox(src, target_type, line))
        else:
            if can_borrow:
                self.keep_alives.append(src)
            return self.add(Cast(src, target_type, line, borrow=can_borrow))

    def coerce(self, src: Value, target_type: RType, line: int, force: bool=False, *, can_borrow: bool=False) -> Value:
        if False:
            print('Hello World!')
        'Generate a coercion/cast from one type to other (only if needed).\n\n        For example, int -> object boxes the source int; int -> int emits nothing;\n        object -> int unboxes the object. All conversions preserve object value.\n\n        If force is true, always generate an op (even if it is just an assignment) so\n        that the result will have exactly target_type as the type.\n\n        Returns the register with the converted value (may be same as src).\n        '
        src_type = src.type
        if src_type.is_unboxed and (not target_type.is_unboxed):
            return self.box(src)
        if (src_type.is_unboxed and target_type.is_unboxed) and (not is_runtime_subtype(src_type, target_type)):
            if isinstance(src, Integer) and is_short_int_rprimitive(src_type) and is_fixed_width_rtype(target_type):
                value = src.numeric_value()
                if not check_native_int_range(target_type, value):
                    self.error(f'Value {value} is out of range for "{target_type}"', line)
                return Integer(src.value >> 1, target_type)
            elif is_int_rprimitive(src_type) and is_fixed_width_rtype(target_type):
                return self.coerce_int_to_fixed_width(src, target_type, line)
            elif is_fixed_width_rtype(src_type) and is_int_rprimitive(target_type):
                return self.coerce_fixed_width_to_int(src, line)
            elif is_short_int_rprimitive(src_type) and is_fixed_width_rtype(target_type):
                return self.coerce_short_int_to_fixed_width(src, target_type, line)
            elif isinstance(src_type, RPrimitive) and isinstance(target_type, RPrimitive) and src_type.is_native_int and target_type.is_native_int and (src_type.size == target_type.size) and (src_type.is_signed == target_type.is_signed):
                return src
            elif (is_bool_rprimitive(src_type) or is_bit_rprimitive(src_type)) and is_tagged(target_type):
                shifted = self.int_op(bool_rprimitive, src, Integer(1, bool_rprimitive), IntOp.LEFT_SHIFT)
                return self.add(Extend(shifted, target_type, signed=False))
            elif (is_bool_rprimitive(src_type) or is_bit_rprimitive(src_type)) and is_fixed_width_rtype(target_type):
                return self.add(Extend(src, target_type, signed=False))
            elif isinstance(src, Integer) and is_float_rprimitive(target_type):
                if is_tagged(src_type):
                    return Float(float(src.value // 2))
                return Float(float(src.value))
            elif is_tagged(src_type) and is_float_rprimitive(target_type):
                return self.int_to_float(src, line)
            elif isinstance(src_type, RTuple) and isinstance(target_type, RTuple) and (len(src_type.types) == len(target_type.types)):
                values = []
                for i in range(len(src_type.types)):
                    v = None
                    if isinstance(src, TupleSet):
                        item = src.items[i]
                        if not isinstance(item, Register):
                            v = item
                    if v is None:
                        v = TupleGet(src, i)
                        self.add(v)
                    values.append(v)
                return self.add(TupleSet([self.coerce(v, t, line) for (v, t) in zip(values, target_type.types)], line))
            tmp = self.box(src)
            return self.unbox_or_cast(tmp, target_type, line)
        if not src_type.is_unboxed and target_type.is_unboxed or not is_subtype(src_type, target_type):
            return self.unbox_or_cast(src, target_type, line, can_borrow=can_borrow)
        elif force:
            tmp = Register(target_type)
            self.add(Assign(tmp, src))
            return tmp
        return src

    def coerce_int_to_fixed_width(self, src: Value, target_type: RType, line: int) -> Value:
        if False:
            for i in range(10):
                print('nop')
        assert is_fixed_width_rtype(target_type), target_type
        assert isinstance(target_type, RPrimitive)
        res = Register(target_type)
        (fast, slow, end) = (BasicBlock(), BasicBlock(), BasicBlock())
        check = self.check_tagged_short_int(src, line)
        self.add(Branch(check, fast, slow, Branch.BOOL))
        self.activate_block(fast)
        size = target_type.size
        if size < int_rprimitive.size:
            (fast2, fast3) = (BasicBlock(), BasicBlock())
            upper_bound = 1 << size * 8 - 1
            if not target_type.is_signed:
                upper_bound *= 2
            check2 = self.add(ComparisonOp(src, Integer(upper_bound, src.type), ComparisonOp.SLT))
            self.add(Branch(check2, fast2, slow, Branch.BOOL))
            self.activate_block(fast2)
            if target_type.is_signed:
                lower_bound = -upper_bound
            else:
                lower_bound = 0
            check3 = self.add(ComparisonOp(src, Integer(lower_bound, src.type), ComparisonOp.SGE))
            self.add(Branch(check3, fast3, slow, Branch.BOOL))
            self.activate_block(fast3)
            tmp = self.int_op(c_pyssize_t_rprimitive, src, Integer(1, c_pyssize_t_rprimitive), IntOp.RIGHT_SHIFT, line)
            tmp = self.add(Truncate(tmp, target_type))
        else:
            if size > int_rprimitive.size:
                tmp = self.add(Extend(src, target_type, signed=True))
            else:
                tmp = src
            tmp = self.int_op(target_type, tmp, Integer(1, target_type), IntOp.RIGHT_SHIFT, line)
        self.add(Assign(res, tmp))
        self.goto(end)
        self.activate_block(slow)
        if is_int64_rprimitive(target_type) or (is_int32_rprimitive(target_type) and size == int_rprimitive.size):
            ptr = self.int_op(pointer_rprimitive, src, Integer(1, pointer_rprimitive), IntOp.XOR, line)
            ptr2 = Register(c_pointer_rprimitive)
            self.add(Assign(ptr2, ptr))
            if is_int64_rprimitive(target_type):
                conv_op = int_to_int64_op
            else:
                conv_op = int_to_int32_op
            tmp = self.call_c(conv_op, [ptr2], line)
            self.add(Assign(res, tmp))
            self.add(KeepAlive([src]))
            self.goto(end)
        elif is_int32_rprimitive(target_type):
            self.call_c(int32_overflow, [], line)
            self.add(Unreachable())
        elif is_int16_rprimitive(target_type):
            self.call_c(int16_overflow, [], line)
            self.add(Unreachable())
        elif is_uint8_rprimitive(target_type):
            self.call_c(uint8_overflow, [], line)
            self.add(Unreachable())
        else:
            assert False, target_type
        self.activate_block(end)
        return res

    def coerce_short_int_to_fixed_width(self, src: Value, target_type: RType, line: int) -> Value:
        if False:
            return 10
        if is_int64_rprimitive(target_type):
            return self.int_op(target_type, src, Integer(1, target_type), IntOp.RIGHT_SHIFT, line)
        assert False, (src.type, target_type)

    def coerce_fixed_width_to_int(self, src: Value, line: int) -> Value:
        if False:
            print('Hello World!')
        if is_int32_rprimitive(src.type) and PLATFORM_SIZE == 8 or is_int16_rprimitive(src.type) or is_uint8_rprimitive(src.type):
            extended = self.add(Extend(src, c_pyssize_t_rprimitive, signed=src.type.is_signed))
            return self.int_op(int_rprimitive, extended, Integer(1, c_pyssize_t_rprimitive), IntOp.LEFT_SHIFT, line)
        assert is_fixed_width_rtype(src.type)
        assert isinstance(src.type, RPrimitive)
        src_type = src.type
        res = Register(int_rprimitive)
        (fast, fast2, slow, end) = (BasicBlock(), BasicBlock(), BasicBlock(), BasicBlock())
        c1 = self.add(ComparisonOp(src, Integer(MAX_SHORT_INT, src_type), ComparisonOp.SLE))
        self.add(Branch(c1, fast, slow, Branch.BOOL))
        self.activate_block(fast)
        c2 = self.add(ComparisonOp(src, Integer(MIN_SHORT_INT, src_type), ComparisonOp.SGE))
        self.add(Branch(c2, fast2, slow, Branch.BOOL))
        self.activate_block(slow)
        if is_int64_rprimitive(src_type):
            conv_op = int64_to_int_op
        elif is_int32_rprimitive(src_type):
            assert PLATFORM_SIZE == 4
            conv_op = ssize_t_to_int_op
        else:
            assert False, src_type
        x = self.call_c(conv_op, [src], line)
        self.add(Assign(res, x))
        self.goto(end)
        self.activate_block(fast2)
        if int_rprimitive.size < src_type.size:
            tmp = self.add(Truncate(src, c_pyssize_t_rprimitive))
        else:
            tmp = src
        s = self.int_op(int_rprimitive, tmp, Integer(1, tmp.type), IntOp.LEFT_SHIFT, line)
        self.add(Assign(res, s))
        self.goto(end)
        self.activate_block(end)
        return res

    def coerce_nullable(self, src: Value, target_type: RType, line: int) -> Value:
        if False:
            print('Hello World!')
        'Generate a coercion from a potentially null value.'
        if src.type.is_unboxed == target_type.is_unboxed and (target_type.is_unboxed and is_runtime_subtype(src.type, target_type) or (not target_type.is_unboxed and is_subtype(src.type, target_type))):
            return src
        target = Register(target_type)
        (valid, invalid, out) = (BasicBlock(), BasicBlock(), BasicBlock())
        self.add(Branch(src, invalid, valid, Branch.IS_ERROR))
        self.activate_block(valid)
        coerced = self.coerce(src, target_type, line)
        self.add(Assign(target, coerced, line))
        self.goto(out)
        self.activate_block(invalid)
        error = self.add(LoadErrorValue(target_type))
        self.add(Assign(target, error, line))
        self.goto_and_activate(out)
        return target

    def get_attr(self, obj: Value, attr: str, result_type: RType, line: int, *, borrow: bool=False) -> Value:
        if False:
            return 10
        'Get a native or Python attribute of an object.'
        if isinstance(obj.type, RInstance) and obj.type.class_ir.is_ext_class and obj.type.class_ir.has_attr(attr):
            op = GetAttr(obj, attr, line, borrow=borrow)
            if op.is_borrowed:
                self.keep_alives.append(obj)
            return self.add(op)
        elif isinstance(obj.type, RUnion):
            return self.union_get_attr(obj, obj.type, attr, result_type, line)
        else:
            return self.py_get_attr(obj, attr, line)

    def union_get_attr(self, obj: Value, rtype: RUnion, attr: str, result_type: RType, line: int) -> Value:
        if False:
            return 10
        'Get an attribute of an object with a union type.'

        def get_item_attr(value: Value) -> Value:
            if False:
                while True:
                    i = 10
            return self.get_attr(value, attr, result_type, line)
        return self.decompose_union_helper(obj, rtype, result_type, get_item_attr, line)

    def py_get_attr(self, obj: Value, attr: str, line: int) -> Value:
        if False:
            return 10
        'Get a Python attribute (slow).\n\n        Prefer get_attr() which generates optimized code for native classes.\n        '
        key = self.load_str(attr)
        return self.call_c(py_getattr_op, [obj, key], line)

    def isinstance_helper(self, obj: Value, class_irs: list[ClassIR], line: int) -> Value:
        if False:
            return 10
        'Fast path for isinstance() that checks against a list of native classes.'
        if not class_irs:
            return self.false()
        ret = self.isinstance_native(obj, class_irs[0], line)
        for class_ir in class_irs[1:]:

            def other() -> Value:
                if False:
                    print('Hello World!')
                return self.isinstance_native(obj, class_ir, line)
            ret = self.shortcircuit_helper('or', bool_rprimitive, lambda : ret, other, line)
        return ret

    def get_type_of_obj(self, obj: Value, line: int) -> Value:
        if False:
            i = 10
            return i + 15
        ob_type_address = self.add(GetElementPtr(obj, PyObject, 'ob_type', line))
        ob_type = self.add(LoadMem(object_rprimitive, ob_type_address))
        self.add(KeepAlive([obj]))
        return ob_type

    def type_is_op(self, obj: Value, type_obj: Value, line: int) -> Value:
        if False:
            return 10
        typ = self.get_type_of_obj(obj, line)
        return self.add(ComparisonOp(typ, type_obj, ComparisonOp.EQ, line))

    def isinstance_native(self, obj: Value, class_ir: ClassIR, line: int) -> Value:
        if False:
            i = 10
            return i + 15
        'Fast isinstance() check for a native class.\n\n        If there are three or fewer concrete (non-trait) classes among the class\n        and all its children, use even faster type comparison checks `type(obj)\n        is typ`.\n        '
        concrete = all_concrete_classes(class_ir)
        if concrete is None or len(concrete) > FAST_ISINSTANCE_MAX_SUBCLASSES + 1:
            return self.call_c(fast_isinstance_op, [obj, self.get_native_type(class_ir)], line)
        if not concrete:
            return self.false()
        type_obj = self.get_native_type(concrete[0])
        ret = self.type_is_op(obj, type_obj, line)
        for c in concrete[1:]:

            def other() -> Value:
                if False:
                    i = 10
                    return i + 15
                return self.type_is_op(obj, self.get_native_type(c), line)
            ret = self.shortcircuit_helper('or', bool_rprimitive, lambda : ret, other, line)
        return ret

    def _construct_varargs(self, args: Sequence[tuple[Value, ArgKind, str | None]], line: int, *, has_star: bool, has_star2: bool) -> tuple[Value | None, Value | None]:
        if False:
            while True:
                i = 10
        "Construct *args and **kwargs from a collection of arguments\n\n        This is pretty complicated, and almost all of the complication here stems from\n        one of two things (but mostly the second):\n          * The handling of ARG_STAR/ARG_STAR2. We want to create as much of the args/kwargs\n            values in one go as we can, so we collect values until our hand is forced, and\n            then we emit creation of the list/tuple, and expand it from there if needed.\n\n          * Support potentially nullable argument values. This has very narrow applicability,\n            as this will never be done by our compiled Python code, but is critically used\n            by gen_glue_method when generating glue methods to mediate between the function\n            signature of a parent class and its subclasses.\n\n            For named-only arguments, this is quite simple: if it is\n            null, don't put it in the dict.\n\n            For positional-or-named arguments, things are much more complicated.\n              * First, anything that was passed as a positional arg\n                must be forwarded along as a positional arg. It *must\n                not* be converted to a named arg. This is because mypy\n                does not enforce that positional-or-named arguments\n                have the same name in subclasses, and it is not\n                uncommon for code to have different names in\n                subclasses (a bunch of mypy's visitors do this, for\n                example!). This is arguably a bug in both mypy and code doing\n                this, and they ought to be using positional-only arguments, but\n                positional-only arguments are new and ugly.\n\n              * On the flip side, we're willing to accept the\n                infelicity of sometimes turning an argument that was\n                passed by keyword into a positional argument. It's wrong,\n                but it's very marginal, and avoiding it would require passing\n                a bitmask of which arguments were named with every function call,\n                or something similar.\n                (See some discussion of this in testComplicatedArgs)\n\n            Thus, our strategy for positional-or-named arguments is to\n            always pass them as positional, except in the one\n            situation where we can not, and where we can be absolutely\n            sure they were passed by name: when an *earlier*\n            positional argument was missing its value.\n\n            This means that if we have a method `f(self, x: int=..., y: object=...)`:\n              * x and y present:      args=(x, y), kwargs={}\n              * x present, y missing: args=(x,),   kwargs={}\n              * x missing, y present: args=(),     kwargs={'y': y}\n\n            To implement this, when we have multiple optional\n            positional arguments, we maintain a flag in a register\n            that tracks whether an argument has been missing, and for\n            each such optional argument (except the first), we check\n            the flag to determine whether to append the argument to\n            the *args list or add it to the **kwargs dict. What a\n            mess!\n\n            This is what really makes everything here such a tangle;\n            otherwise the *args and **kwargs code could be separated.\n\n        The arguments has_star and has_star2 indicate whether the target function\n        takes an ARG_STAR and ARG_STAR2 argument, respectively.\n        (These will always be true when making a pycall, and be based\n        on the actual target signature for a native call.)\n        "
        star_result: Value | None = None
        star2_result: Value | None = None
        star_values: list[Value] = []
        star2_keys: list[Value] = []
        star2_values: list[Value] = []
        seen_empty_reg: Register | None = None
        for (value, kind, name) in args:
            if kind == ARG_STAR:
                if star_result is None:
                    star_result = self.new_list_op(star_values, line)
                self.call_c(list_extend_op, [star_result, value], line)
            elif kind == ARG_STAR2:
                if star2_result is None:
                    star2_result = self._create_dict(star2_keys, star2_values, line)
                self.call_c(dict_update_in_display_op, [star2_result, value], line=line)
            else:
                nullable = kind.is_optional()
                maybe_pos = kind.is_positional() and has_star
                maybe_named = kind.is_named() or (kind.is_optional() and name and has_star2)
                if nullable:
                    if maybe_pos and star_result is None:
                        star_result = self.new_list_op(star_values, line)
                    if maybe_named and star2_result is None:
                        star2_result = self._create_dict(star2_keys, star2_values, line)
                if maybe_pos and star_result is None:
                    star_values.append(value)
                    continue
                if maybe_named and star2_result is None:
                    assert name is not None
                    key = self.load_str(name)
                    star2_keys.append(key)
                    star2_values.append(value)
                    continue
                new_seen_empty_reg = seen_empty_reg
                out = BasicBlock()
                if nullable:
                    if maybe_pos and (not seen_empty_reg):
                        new_seen_empty_reg = Register(bool_rprimitive)
                        self.add(Assign(new_seen_empty_reg, self.false(), line))
                    skip = BasicBlock() if maybe_pos else out
                    keep = BasicBlock()
                    self.add(Branch(value, skip, keep, Branch.IS_ERROR))
                    self.activate_block(keep)
                if maybe_pos and maybe_named and seen_empty_reg:
                    (pos_block, named_block) = (BasicBlock(), BasicBlock())
                    self.add(Branch(seen_empty_reg, named_block, pos_block, Branch.BOOL))
                else:
                    pos_block = named_block = BasicBlock()
                    self.goto(pos_block)
                if maybe_pos:
                    self.activate_block(pos_block)
                    assert star_result
                    self.translate_special_method_call(star_result, 'append', [value], result_type=None, line=line)
                    self.goto(out)
                if maybe_named and (not maybe_pos or seen_empty_reg):
                    self.activate_block(named_block)
                    assert name is not None
                    key = self.load_str(name)
                    assert star2_result
                    self.translate_special_method_call(star2_result, '__setitem__', [key, value], result_type=None, line=line)
                    self.goto(out)
                if nullable and maybe_pos and new_seen_empty_reg:
                    assert skip is not out
                    self.activate_block(skip)
                    self.add(Assign(new_seen_empty_reg, self.true(), line))
                    self.goto(out)
                self.activate_block(out)
                seen_empty_reg = new_seen_empty_reg
        assert not (star_result or star_values) or has_star
        assert not (star2_result or star2_values) or has_star2
        if has_star:
            if star_result is None:
                star_result = self.new_tuple(star_values, line)
            else:
                star_result = self.call_c(list_tuple_op, [star_result], line)
        if has_star2 and star2_result is None:
            star2_result = self._create_dict(star2_keys, star2_values, line)
        return (star_result, star2_result)

    def py_call(self, function: Value, arg_values: list[Value], line: int, arg_kinds: list[ArgKind] | None=None, arg_names: Sequence[str | None] | None=None) -> Value:
        if False:
            i = 10
            return i + 15
        'Call a Python function (non-native and slow).\n\n        Use py_call_op or py_call_with_kwargs_op for Python function call.\n        '
        if use_vectorcall(self.options.capi_version):
            result = self._py_vector_call(function, arg_values, line, arg_kinds, arg_names)
            if result is not None:
                return result
        if arg_kinds is None or all((kind == ARG_POS for kind in arg_kinds)):
            return self.call_c(py_call_op, [function] + arg_values, line)
        assert arg_names is not None
        (pos_args_tuple, kw_args_dict) = self._construct_varargs(list(zip(arg_values, arg_kinds, arg_names)), line, has_star=True, has_star2=True)
        assert pos_args_tuple and kw_args_dict
        return self.call_c(py_call_with_kwargs_op, [function, pos_args_tuple, kw_args_dict], line)

    def _py_vector_call(self, function: Value, arg_values: list[Value], line: int, arg_kinds: list[ArgKind] | None=None, arg_names: Sequence[str | None] | None=None) -> Value | None:
        if False:
            i = 10
            return i + 15
        'Call function using the vectorcall API if possible.\n\n        Return the return value if successful. Return None if a non-vectorcall\n        API should be used instead.\n        '
        if arg_kinds is None or all((not kind.is_star() and (not kind.is_optional()) for kind in arg_kinds)):
            if arg_values:
                coerced_args = [self.coerce(arg, object_rprimitive, line) for arg in arg_values]
                arg_ptr = self.setup_rarray(object_rprimitive, coerced_args, object_ptr=True)
            else:
                arg_ptr = Integer(0, object_pointer_rprimitive)
            num_pos = num_positional_args(arg_values, arg_kinds)
            keywords = self._vectorcall_keywords(arg_names)
            value = self.call_c(py_vectorcall_op, [function, arg_ptr, Integer(num_pos, c_size_t_rprimitive), keywords], line)
            if arg_values:
                self.add(KeepAlive(coerced_args))
            return value
        return None

    def _vectorcall_keywords(self, arg_names: Sequence[str | None] | None) -> Value:
        if False:
            print('Hello World!')
        'Return a reference to a tuple literal with keyword argument names.\n\n        Return null pointer if there are no keyword arguments.\n        '
        if arg_names:
            kw_list = [name for name in arg_names if name is not None]
            if kw_list:
                return self.add(LoadLiteral(tuple(kw_list), object_rprimitive))
        return Integer(0, object_rprimitive)

    def py_method_call(self, obj: Value, method_name: str, arg_values: list[Value], line: int, arg_kinds: list[ArgKind] | None, arg_names: Sequence[str | None] | None) -> Value:
        if False:
            print('Hello World!')
        'Call a Python method (non-native and slow).'
        if use_method_vectorcall(self.options.capi_version):
            result = self._py_vector_method_call(obj, method_name, arg_values, line, arg_kinds, arg_names)
            if result is not None:
                return result
        if arg_kinds is None or all((kind == ARG_POS for kind in arg_kinds)):
            method_name_reg = self.load_str(method_name)
            return self.call_c(py_method_call_op, [obj, method_name_reg] + arg_values, line)
        else:
            method = self.py_get_attr(obj, method_name, line)
            return self.py_call(method, arg_values, line, arg_kinds=arg_kinds, arg_names=arg_names)

    def _py_vector_method_call(self, obj: Value, method_name: str, arg_values: list[Value], line: int, arg_kinds: list[ArgKind] | None, arg_names: Sequence[str | None] | None) -> Value | None:
        if False:
            while True:
                i = 10
        'Call method using the vectorcall API if possible.\n\n        Return the return value if successful. Return None if a non-vectorcall\n        API should be used instead.\n        '
        if arg_kinds is None or all((not kind.is_star() and (not kind.is_optional()) for kind in arg_kinds)):
            method_name_reg = self.load_str(method_name)
            coerced_args = [self.coerce(arg, object_rprimitive, line) for arg in [obj] + arg_values]
            arg_ptr = self.setup_rarray(object_rprimitive, coerced_args, object_ptr=True)
            num_pos = num_positional_args(arg_values, arg_kinds)
            keywords = self._vectorcall_keywords(arg_names)
            value = self.call_c(py_vectorcall_method_op, [method_name_reg, arg_ptr, Integer(num_pos + 1 | PY_VECTORCALL_ARGUMENTS_OFFSET, c_size_t_rprimitive), keywords], line)
            self.add(KeepAlive(coerced_args))
            return value
        return None

    def call(self, decl: FuncDecl, args: Sequence[Value], arg_kinds: list[ArgKind], arg_names: Sequence[str | None], line: int, *, bitmap_args: list[Register] | None=None) -> Value:
        if False:
            i = 10
            return i + 15
        'Call a native function.\n\n        If bitmap_args is given, they override the values of (some) of the bitmap\n        arguments used to track the presence of values for certain arguments. By\n        default, the values of the bitmap arguments are inferred from args.\n        '
        args = self.native_args_to_positional(args, arg_kinds, arg_names, decl.sig, line, bitmap_args=bitmap_args)
        return self.add(Call(decl, args, line))

    def native_args_to_positional(self, args: Sequence[Value], arg_kinds: list[ArgKind], arg_names: Sequence[str | None], sig: FuncSignature, line: int, *, bitmap_args: list[Register] | None=None) -> list[Value]:
        if False:
            return 10
        'Prepare arguments for a native call.\n\n        Given args/kinds/names and a target signature for a native call, map\n        keyword arguments to their appropriate place in the argument list,\n        fill in error values for unspecified default arguments,\n        package arguments that will go into *args/**kwargs into a tuple/dict,\n        and coerce arguments to the appropriate type.\n        '
        sig_args = sig.args
        n = sig.num_bitmap_args
        if n:
            sig_args = sig_args[:-n]
        sig_arg_kinds = [arg.kind for arg in sig_args]
        sig_arg_names = [arg.name for arg in sig_args]
        concrete_kinds = [concrete_arg_kind(arg_kind) for arg_kind in arg_kinds]
        formal_to_actual = map_actuals_to_formals(concrete_kinds, arg_names, sig_arg_kinds, sig_arg_names, lambda n: AnyType(TypeOfAny.special_form))
        has_star = has_star2 = False
        star_arg_entries = []
        for (lst, arg) in zip(formal_to_actual, sig_args):
            if arg.kind.is_star():
                star_arg_entries.extend([(args[i], arg_kinds[i], arg_names[i]) for i in lst])
            has_star = has_star or arg.kind == ARG_STAR
            has_star2 = has_star2 or arg.kind == ARG_STAR2
        (star_arg, star2_arg) = self._construct_varargs(star_arg_entries, line, has_star=has_star, has_star2=has_star2)
        output_args: list[Value] = []
        for (lst, arg) in zip(formal_to_actual, sig_args):
            if arg.kind == ARG_STAR:
                assert star_arg
                output_arg = star_arg
            elif arg.kind == ARG_STAR2:
                assert star2_arg
                output_arg = star2_arg
            elif not lst:
                if is_fixed_width_rtype(arg.type):
                    output_arg = Integer(0, arg.type)
                elif is_float_rprimitive(arg.type):
                    output_arg = Float(0.0)
                else:
                    output_arg = self.add(LoadErrorValue(arg.type, is_borrowed=True))
            else:
                base_arg = args[lst[0]]
                if arg_kinds[lst[0]].is_optional():
                    output_arg = self.coerce_nullable(base_arg, arg.type, line)
                else:
                    output_arg = self.coerce(base_arg, arg.type, line)
            output_args.append(output_arg)
        for i in reversed(range(n)):
            if bitmap_args and i < len(bitmap_args):
                output_args.append(bitmap_args[i])
                continue
            bitmap = 0
            c = 0
            for (lst, arg) in zip(formal_to_actual, sig_args):
                if arg.kind.is_optional() and arg.type.error_overlap:
                    if i * BITMAP_BITS <= c < (i + 1) * BITMAP_BITS:
                        if lst:
                            bitmap |= 1 << (c & BITMAP_BITS - 1)
                    c += 1
            output_args.append(Integer(bitmap, bitmap_rprimitive))
        return output_args

    def gen_method_call(self, base: Value, name: str, arg_values: list[Value], result_type: RType | None, line: int, arg_kinds: list[ArgKind] | None=None, arg_names: list[str | None] | None=None, can_borrow: bool=False) -> Value:
        if False:
            i = 10
            return i + 15
        'Generate either a native or Python method call.'
        if arg_kinds is not None and any((kind.is_star() for kind in arg_kinds)):
            return self.py_method_call(base, name, arg_values, base.line, arg_kinds, arg_names)
        if isinstance(base.type, RInstance) and base.type.class_ir.is_ext_class and (not base.type.class_ir.builtin_base):
            if base.type.class_ir.has_method(name):
                decl = base.type.class_ir.method_decl(name)
                if arg_kinds is None:
                    assert arg_names is None, 'arg_kinds not present but arg_names is'
                    arg_kinds = [ARG_POS for _ in arg_values]
                    arg_names = [None for _ in arg_values]
                else:
                    assert arg_names is not None, 'arg_kinds present but arg_names is not'
                assert decl.bound_sig
                arg_values = self.native_args_to_positional(arg_values, arg_kinds, arg_names, decl.bound_sig, line)
                return self.add(MethodCall(base, name, arg_values, line))
            elif base.type.class_ir.has_attr(name):
                function = self.add(GetAttr(base, name, line))
                return self.py_call(function, arg_values, line, arg_kinds=arg_kinds, arg_names=arg_names)
        elif isinstance(base.type, RUnion):
            return self.union_method_call(base, base.type, name, arg_values, result_type, line, arg_kinds, arg_names)
        if not arg_kinds or arg_kinds == [ARG_POS] * len(arg_values):
            target = self.translate_special_method_call(base, name, arg_values, result_type, line, can_borrow=can_borrow)
            if target:
                return target
        return self.py_method_call(base, name, arg_values, line, arg_kinds, arg_names)

    def union_method_call(self, base: Value, obj_type: RUnion, name: str, arg_values: list[Value], return_rtype: RType | None, line: int, arg_kinds: list[ArgKind] | None, arg_names: list[str | None] | None) -> Value:
        if False:
            for i in range(10):
                print('nop')
        'Generate a method call with a union type for the object.'
        return_rtype = return_rtype or object_rprimitive

        def call_union_item(value: Value) -> Value:
            if False:
                print('Hello World!')
            return self.gen_method_call(value, name, arg_values, return_rtype, line, arg_kinds, arg_names)
        return self.decompose_union_helper(base, obj_type, return_rtype, call_union_item, line)

    def none(self) -> Value:
        if False:
            print('Hello World!')
        'Load unboxed None value (type: none_rprimitive).'
        return Integer(1, none_rprimitive)

    def true(self) -> Value:
        if False:
            while True:
                i = 10
        'Load unboxed True value (type: bool_rprimitive).'
        return Integer(1, bool_rprimitive)

    def false(self) -> Value:
        if False:
            return 10
        'Load unboxed False value (type: bool_rprimitive).'
        return Integer(0, bool_rprimitive)

    def none_object(self) -> Value:
        if False:
            i = 10
            return i + 15
        'Load Python None value (type: object_rprimitive).'
        return self.add(LoadAddress(none_object_op.type, none_object_op.src, line=-1))

    def load_int(self, value: int) -> Value:
        if False:
            while True:
                i = 10
        'Load a tagged (Python) integer literal value.'
        if value > MAX_LITERAL_SHORT_INT or value < MIN_LITERAL_SHORT_INT:
            return self.add(LoadLiteral(value, int_rprimitive))
        else:
            return Integer(value)

    def load_float(self, value: float) -> Value:
        if False:
            while True:
                i = 10
        'Load a float literal value.'
        return Float(value)

    def load_str(self, value: str) -> Value:
        if False:
            print('Hello World!')
        'Load a str literal value.\n\n        This is useful for more than just str literals; for example, method calls\n        also require a PyObject * form for the name of the method.\n        '
        return self.add(LoadLiteral(value, str_rprimitive))

    def load_bytes(self, value: bytes) -> Value:
        if False:
            for i in range(10):
                print('nop')
        'Load a bytes literal value.'
        return self.add(LoadLiteral(value, bytes_rprimitive))

    def load_complex(self, value: complex) -> Value:
        if False:
            for i in range(10):
                print('nop')
        'Load a complex literal value.'
        return self.add(LoadLiteral(value, object_rprimitive))

    def load_static_checked(self, typ: RType, identifier: str, module_name: str | None=None, namespace: str=NAMESPACE_STATIC, line: int=-1, error_msg: str | None=None) -> Value:
        if False:
            for i in range(10):
                print('nop')
        if error_msg is None:
            error_msg = f'name "{identifier}" is not defined'
        (ok_block, error_block) = (BasicBlock(), BasicBlock())
        value = self.add(LoadStatic(typ, identifier, module_name, namespace, line=line))
        self.add(Branch(value, error_block, ok_block, Branch.IS_ERROR, rare=True))
        self.activate_block(error_block)
        self.add(RaiseStandardError(RaiseStandardError.NAME_ERROR, error_msg, line))
        self.add(Unreachable())
        self.activate_block(ok_block)
        return value

    def load_module(self, name: str) -> Value:
        if False:
            print('Hello World!')
        return self.add(LoadStatic(object_rprimitive, name, namespace=NAMESPACE_MODULE))

    def get_native_type(self, cls: ClassIR) -> Value:
        if False:
            i = 10
            return i + 15
        'Load native type object.'
        fullname = f'{cls.module_name}.{cls.name}'
        return self.load_native_type_object(fullname)

    def load_native_type_object(self, fullname: str) -> Value:
        if False:
            return 10
        (module, name) = fullname.rsplit('.', 1)
        return self.add(LoadStatic(object_rprimitive, name, module, NAMESPACE_TYPE))

    def binary_op(self, lreg: Value, rreg: Value, op: str, line: int) -> Value:
        if False:
            while True:
                i = 10
        'Perform a binary operation.\n\n        Generate specialized operations based on operand types, with a fallback\n        to generic operations.\n        '
        ltype = lreg.type
        rtype = rreg.type
        if isinstance(ltype, RTuple) and isinstance(rtype, RTuple) and (op in ('==', '!=')):
            return self.compare_tuples(lreg, rreg, op, line)
        if op in ('==', '!='):
            value = self.translate_eq_cmp(lreg, rreg, op, line)
            if value is not None:
                return value
        if op in ('is', 'is not'):
            return self.translate_is_op(lreg, rreg, op, line)
        if is_str_rprimitive(ltype) and is_str_rprimitive(rtype) and (op in ('==', '!=')):
            return self.compare_strings(lreg, rreg, op, line)
        if is_bytes_rprimitive(ltype) and is_bytes_rprimitive(rtype) and (op in ('==', '!=')):
            return self.compare_bytes(lreg, rreg, op, line)
        if is_tagged(ltype) and is_tagged(rtype) and (op in int_comparison_op_mapping):
            return self.compare_tagged(lreg, rreg, op, line)
        if is_bool_rprimitive(ltype) and is_bool_rprimitive(rtype) and (op in BOOL_BINARY_OPS):
            if op in ComparisonOp.signed_ops:
                return self.bool_comparison_op(lreg, rreg, op, line)
            else:
                return self.bool_bitwise_op(lreg, rreg, op[0], line)
        if isinstance(rtype, RInstance) and op in ('in', 'not in'):
            return self.translate_instance_contains(rreg, lreg, op, line)
        if is_fixed_width_rtype(ltype):
            if op in FIXED_WIDTH_INT_BINARY_OPS:
                if op.endswith('='):
                    op = op[:-1]
                if op != '//':
                    op_id = int_op_to_id[op]
                else:
                    op_id = IntOp.DIV
                if is_bool_rprimitive(rtype) or is_bit_rprimitive(rtype):
                    rreg = self.coerce(rreg, ltype, line)
                    rtype = ltype
                if is_fixed_width_rtype(rtype) or is_tagged(rtype):
                    return self.fixed_width_int_op(ltype, lreg, rreg, op_id, line)
                if isinstance(rreg, Integer):
                    return self.fixed_width_int_op(ltype, lreg, self.coerce(rreg, ltype, line), op_id, line)
            elif op in ComparisonOp.signed_ops:
                if is_int_rprimitive(rtype):
                    rreg = self.coerce_int_to_fixed_width(rreg, ltype, line)
                elif is_bool_rprimitive(rtype) or is_bit_rprimitive(rtype):
                    rreg = self.coerce(rreg, ltype, line)
                op_id = ComparisonOp.signed_ops[op]
                if is_fixed_width_rtype(rreg.type):
                    return self.comparison_op(lreg, rreg, op_id, line)
                if isinstance(rreg, Integer):
                    return self.comparison_op(lreg, self.coerce(rreg, ltype, line), op_id, line)
        elif is_fixed_width_rtype(rtype):
            if op in FIXED_WIDTH_INT_BINARY_OPS:
                if op.endswith('='):
                    op = op[:-1]
                if op != '//':
                    op_id = int_op_to_id[op]
                else:
                    op_id = IntOp.DIV
                if isinstance(lreg, Integer):
                    return self.fixed_width_int_op(rtype, self.coerce(lreg, rtype, line), rreg, op_id, line)
                if is_tagged(ltype):
                    return self.fixed_width_int_op(rtype, lreg, rreg, op_id, line)
                if is_bool_rprimitive(ltype) or is_bit_rprimitive(ltype):
                    lreg = self.coerce(lreg, rtype, line)
                    return self.fixed_width_int_op(rtype, lreg, rreg, op_id, line)
            elif op in ComparisonOp.signed_ops:
                if is_int_rprimitive(ltype):
                    lreg = self.coerce_int_to_fixed_width(lreg, rtype, line)
                elif is_bool_rprimitive(ltype) or is_bit_rprimitive(ltype):
                    lreg = self.coerce(lreg, rtype, line)
                op_id = ComparisonOp.signed_ops[op]
                if isinstance(lreg, Integer):
                    return self.comparison_op(self.coerce(lreg, rtype, line), rreg, op_id, line)
                if is_fixed_width_rtype(lreg.type):
                    return self.comparison_op(lreg, rreg, op_id, line)
        if op in ('==', '!='):
            op_id = ComparisonOp.signed_ops[op]
            if is_tagged(ltype) and is_subtype(rtype, ltype):
                rreg = self.coerce(rreg, int_rprimitive, line)
                return self.comparison_op(lreg, rreg, op_id, line)
            if is_tagged(rtype) and is_subtype(ltype, rtype):
                lreg = self.coerce(lreg, int_rprimitive, line)
                return self.comparison_op(lreg, rreg, op_id, line)
        elif op in op in int_comparison_op_mapping:
            if is_tagged(ltype) and is_subtype(rtype, ltype):
                rreg = self.coerce(rreg, short_int_rprimitive, line)
                return self.compare_tagged(lreg, rreg, op, line)
            if is_tagged(rtype) and is_subtype(ltype, rtype):
                lreg = self.coerce(lreg, short_int_rprimitive, line)
                return self.compare_tagged(lreg, rreg, op, line)
        if is_float_rprimitive(ltype) or is_float_rprimitive(rtype):
            if isinstance(lreg, Integer):
                lreg = Float(float(lreg.numeric_value()))
            elif isinstance(rreg, Integer):
                rreg = Float(float(rreg.numeric_value()))
            elif is_int_rprimitive(lreg.type):
                lreg = self.int_to_float(lreg, line)
            elif is_int_rprimitive(rreg.type):
                rreg = self.int_to_float(rreg, line)
            if is_float_rprimitive(lreg.type) and is_float_rprimitive(rreg.type):
                if op in float_comparison_op_to_id:
                    return self.compare_floats(lreg, rreg, float_comparison_op_to_id[op], line)
                if op.endswith('='):
                    base_op = op[:-1]
                else:
                    base_op = op
                if base_op in float_op_to_id:
                    return self.float_op(lreg, rreg, base_op, line)
        call_c_ops_candidates = binary_ops.get(op, [])
        target = self.matching_call_c(call_c_ops_candidates, [lreg, rreg], line)
        assert target, 'Unsupported binary operation: %s' % op
        return target

    def check_tagged_short_int(self, val: Value, line: int, negated: bool=False) -> Value:
        if False:
            return 10
        "Check if a tagged integer is a short integer.\n\n        Return the result of the check (value of type 'bit').\n        "
        int_tag = Integer(1, c_pyssize_t_rprimitive, line)
        bitwise_and = self.int_op(c_pyssize_t_rprimitive, val, int_tag, IntOp.AND, line)
        zero = Integer(0, c_pyssize_t_rprimitive, line)
        op = ComparisonOp.NEQ if negated else ComparisonOp.EQ
        check = self.comparison_op(bitwise_and, zero, op, line)
        return check

    def compare_tagged(self, lhs: Value, rhs: Value, op: str, line: int) -> Value:
        if False:
            print('Hello World!')
        'Compare two tagged integers using given operator (value context).'
        if is_short_int_rprimitive(lhs.type) and is_short_int_rprimitive(rhs.type):
            return self.comparison_op(lhs, rhs, int_comparison_op_mapping[op][0], line)
        (op_type, c_func_desc, negate_result, swap_op) = int_comparison_op_mapping[op]
        result = Register(bool_rprimitive)
        (short_int_block, int_block, out) = (BasicBlock(), BasicBlock(), BasicBlock())
        check_lhs = self.check_tagged_short_int(lhs, line)
        if op in ('==', '!='):
            check = check_lhs
        else:
            check_rhs = self.check_tagged_short_int(rhs, line)
            check = self.int_op(bit_rprimitive, check_lhs, check_rhs, IntOp.AND, line)
        self.add(Branch(check, short_int_block, int_block, Branch.BOOL))
        self.activate_block(short_int_block)
        eq = self.comparison_op(lhs, rhs, op_type, line)
        self.add(Assign(result, eq, line))
        self.goto(out)
        self.activate_block(int_block)
        if swap_op:
            args = [rhs, lhs]
        else:
            args = [lhs, rhs]
        call = self.call_c(c_func_desc, args, line)
        if negate_result:
            call_result = self.unary_op(call, 'not', line)
        else:
            call_result = call
        self.add(Assign(result, call_result, line))
        self.goto_and_activate(out)
        return result

    def compare_tagged_condition(self, lhs: Value, rhs: Value, op: str, true: BasicBlock, false: BasicBlock, line: int) -> None:
        if False:
            print('Hello World!')
        "Compare two tagged integers using given operator (conditional context).\n\n        Assume lhs and rhs are tagged integers.\n\n        Args:\n            lhs: Left operand\n            rhs: Right operand\n            op: Operation, one of '==', '!=', '<', '<=', '>', '<='\n            true: Branch target if comparison is true\n            false: Branch target if comparison is false\n        "
        is_eq = op in ('==', '!=')
        if is_short_int_rprimitive(lhs.type) and is_short_int_rprimitive(rhs.type) or (is_eq and (is_short_int_rprimitive(lhs.type) or is_short_int_rprimitive(rhs.type))):
            check = self.comparison_op(lhs, rhs, int_comparison_op_mapping[op][0], line)
            self.flush_keep_alives()
            self.add(Branch(check, true, false, Branch.BOOL))
            return
        (op_type, c_func_desc, negate_result, swap_op) = int_comparison_op_mapping[op]
        (int_block, short_int_block) = (BasicBlock(), BasicBlock())
        check_lhs = self.check_tagged_short_int(lhs, line, negated=True)
        if is_eq or is_short_int_rprimitive(rhs.type):
            self.flush_keep_alives()
            self.add(Branch(check_lhs, int_block, short_int_block, Branch.BOOL))
        else:
            rhs_block = BasicBlock()
            self.add(Branch(check_lhs, int_block, rhs_block, Branch.BOOL))
            self.activate_block(rhs_block)
            check_rhs = self.check_tagged_short_int(rhs, line, negated=True)
            self.flush_keep_alives()
            self.add(Branch(check_rhs, int_block, short_int_block, Branch.BOOL))
        self.activate_block(int_block)
        if swap_op:
            args = [rhs, lhs]
        else:
            args = [lhs, rhs]
        call = self.call_c(c_func_desc, args, line)
        if negate_result:
            self.add(Branch(call, false, true, Branch.BOOL))
        else:
            self.flush_keep_alives()
            self.add(Branch(call, true, false, Branch.BOOL))
        self.activate_block(short_int_block)
        eq = self.comparison_op(lhs, rhs, op_type, line)
        self.add(Branch(eq, true, false, Branch.BOOL))

    def compare_strings(self, lhs: Value, rhs: Value, op: str, line: int) -> Value:
        if False:
            while True:
                i = 10
        'Compare two strings'
        compare_result = self.call_c(unicode_compare, [lhs, rhs], line)
        error_constant = Integer(-1, c_int_rprimitive, line)
        compare_error_check = self.add(ComparisonOp(compare_result, error_constant, ComparisonOp.EQ, line))
        (exception_check, propagate, final_compare) = (BasicBlock(), BasicBlock(), BasicBlock())
        branch = Branch(compare_error_check, exception_check, final_compare, Branch.BOOL)
        branch.negated = False
        self.add(branch)
        self.activate_block(exception_check)
        check_error_result = self.call_c(err_occurred_op, [], line)
        null = Integer(0, pointer_rprimitive, line)
        compare_error_check = self.add(ComparisonOp(check_error_result, null, ComparisonOp.NEQ, line))
        branch = Branch(compare_error_check, propagate, final_compare, Branch.BOOL)
        branch.negated = False
        self.add(branch)
        self.activate_block(propagate)
        self.call_c(keep_propagating_op, [], line)
        self.goto(final_compare)
        self.activate_block(final_compare)
        op_type = ComparisonOp.EQ if op == '==' else ComparisonOp.NEQ
        return self.add(ComparisonOp(compare_result, Integer(0, c_int_rprimitive), op_type, line))

    def compare_bytes(self, lhs: Value, rhs: Value, op: str, line: int) -> Value:
        if False:
            while True:
                i = 10
        compare_result = self.call_c(bytes_compare, [lhs, rhs], line)
        op_type = ComparisonOp.EQ if op == '==' else ComparisonOp.NEQ
        return self.add(ComparisonOp(compare_result, Integer(1, c_int_rprimitive), op_type, line))

    def compare_tuples(self, lhs: Value, rhs: Value, op: str, line: int=-1) -> Value:
        if False:
            print('Hello World!')
        'Compare two tuples item by item'
        assert isinstance(lhs.type, RTuple) and isinstance(rhs.type, RTuple)
        equal = True if op == '==' else False
        result = Register(bool_rprimitive)
        if len(lhs.type.types) == 0 and len(rhs.type.types) == 0:
            self.add(Assign(result, self.true() if equal else self.false(), line))
            return result
        length = len(lhs.type.types)
        (false_assign, true_assign, out) = (BasicBlock(), BasicBlock(), BasicBlock())
        check_blocks = [BasicBlock() for _ in range(length)]
        lhs_items = [self.add(TupleGet(lhs, i, line)) for i in range(length)]
        rhs_items = [self.add(TupleGet(rhs, i, line)) for i in range(length)]
        if equal:
            (early_stop, final) = (false_assign, true_assign)
        else:
            (early_stop, final) = (true_assign, false_assign)
        for i in range(len(lhs.type.types)):
            if i != 0:
                self.activate_block(check_blocks[i])
            lhs_item = lhs_items[i]
            rhs_item = rhs_items[i]
            compare = self.binary_op(lhs_item, rhs_item, op, line)
            if not is_bool_rprimitive(compare.type):
                compare = self.call_c(bool_op, [compare], line)
            if i < len(lhs.type.types) - 1:
                branch = Branch(compare, early_stop, check_blocks[i + 1], Branch.BOOL)
            else:
                branch = Branch(compare, early_stop, final, Branch.BOOL)
            branch.negated = equal
            self.add(branch)
        self.activate_block(false_assign)
        self.add(Assign(result, self.false(), line))
        self.goto(out)
        self.activate_block(true_assign)
        self.add(Assign(result, self.true(), line))
        self.goto_and_activate(out)
        return result

    def translate_instance_contains(self, inst: Value, item: Value, op: str, line: int) -> Value:
        if False:
            return 10
        res = self.gen_method_call(inst, '__contains__', [item], None, line)
        if not is_bool_rprimitive(res.type):
            res = self.call_c(bool_op, [res], line)
        if op == 'not in':
            res = self.bool_bitwise_op(res, Integer(1, rtype=bool_rprimitive), '^', line)
        return res

    def bool_bitwise_op(self, lreg: Value, rreg: Value, op: str, line: int) -> Value:
        if False:
            return 10
        if op == '&':
            code = IntOp.AND
        elif op == '|':
            code = IntOp.OR
        elif op == '^':
            code = IntOp.XOR
        else:
            assert False, op
        return self.add(IntOp(bool_rprimitive, lreg, rreg, code, line))

    def bool_comparison_op(self, lreg: Value, rreg: Value, op: str, line: int) -> Value:
        if False:
            return 10
        op_id = ComparisonOp.signed_ops[op]
        return self.comparison_op(lreg, rreg, op_id, line)

    def unary_not(self, value: Value, line: int) -> Value:
        if False:
            print('Hello World!')
        mask = Integer(1, value.type, line)
        return self.int_op(value.type, value, mask, IntOp.XOR, line)

    def unary_op(self, value: Value, expr_op: str, line: int) -> Value:
        if False:
            while True:
                i = 10
        typ = value.type
        if is_bool_rprimitive(typ) or is_bit_rprimitive(typ):
            if expr_op == 'not':
                return self.unary_not(value, line)
            if expr_op == '+':
                return value
        if is_fixed_width_rtype(typ):
            if expr_op == '-':
                return self.int_op(typ, Integer(0, typ), value, IntOp.SUB, line)
            elif expr_op == '~':
                if typ.is_signed:
                    return self.int_op(typ, value, Integer(-1, typ), IntOp.XOR, line)
                else:
                    mask = (1 << typ.size * 8) - 1
                    return self.int_op(typ, value, Integer(mask, typ), IntOp.XOR, line)
            elif expr_op == '+':
                return value
        if is_float_rprimitive(typ):
            if expr_op == '-':
                return self.add(FloatNeg(value, line))
            elif expr_op == '+':
                return value
        if isinstance(value, Integer):
            num = value.value
            if is_short_int_rprimitive(typ):
                num >>= 1
            return Integer(-num, typ, value.line)
        if is_tagged(typ) and expr_op == '+':
            return value
        if isinstance(value, Float):
            return Float(-value.value, value.line)
        if isinstance(typ, RInstance):
            if expr_op == '-':
                method = '__neg__'
            elif expr_op == '+':
                method = '__pos__'
            elif expr_op == '~':
                method = '__invert__'
            else:
                method = ''
            if method and typ.class_ir.has_method(method):
                return self.gen_method_call(value, method, [], None, line)
        call_c_ops_candidates = unary_ops.get(expr_op, [])
        target = self.matching_call_c(call_c_ops_candidates, [value], line)
        assert target, 'Unsupported unary operation: %s' % expr_op
        return target

    def make_dict(self, key_value_pairs: Sequence[DictEntry], line: int) -> Value:
        if False:
            return 10
        result: Value | None = None
        keys: list[Value] = []
        values: list[Value] = []
        for (key, value) in key_value_pairs:
            if key is not None:
                if result is None:
                    keys.append(key)
                    values.append(value)
                    continue
                self.translate_special_method_call(result, '__setitem__', [key, value], result_type=None, line=line)
            else:
                if result is None:
                    result = self._create_dict(keys, values, line)
                self.call_c(dict_update_in_display_op, [result, value], line=line)
        if result is None:
            result = self._create_dict(keys, values, line)
        return result

    def new_list_op_with_length(self, length: Value, line: int) -> Value:
        if False:
            return 10
        'This function returns an uninitialized list.\n\n        If the length is non-zero, the caller must initialize the list, before\n        it can be made visible to user code -- otherwise the list object is broken.\n        You might need further initialization with `new_list_set_item_op` op.\n\n        Args:\n            length: desired length of the new list. The rtype should be\n                    c_pyssize_t_rprimitive\n            line: line number\n        '
        return self.call_c(new_list_op, [length], line)

    def new_list_op(self, values: list[Value], line: int) -> Value:
        if False:
            for i in range(10):
                print('nop')
        length: list[Value] = [Integer(len(values), c_pyssize_t_rprimitive, line)]
        if len(values) >= LIST_BUILDING_EXPANSION_THRESHOLD:
            return self.call_c(list_build_op, length + values, line)
        result_list = self.call_c(new_list_op, length, line)
        if not values:
            return result_list
        args = [self.coerce(item, object_rprimitive, line) for item in values]
        ob_item_ptr = self.add(GetElementPtr(result_list, PyListObject, 'ob_item', line))
        ob_item_base = self.add(LoadMem(pointer_rprimitive, ob_item_ptr, line))
        for i in range(len(values)):
            if i == 0:
                item_address = ob_item_base
            else:
                offset = Integer(PLATFORM_SIZE * i, c_pyssize_t_rprimitive, line)
                item_address = self.add(IntOp(pointer_rprimitive, ob_item_base, offset, IntOp.ADD, line))
            self.add(SetMem(object_rprimitive, item_address, args[i], line))
        self.add(KeepAlive([result_list]))
        return result_list

    def new_set_op(self, values: list[Value], line: int) -> Value:
        if False:
            print('Hello World!')
        return self.call_c(new_set_op, values, line)

    def setup_rarray(self, item_type: RType, values: Sequence[Value], *, object_ptr: bool=False) -> Value:
        if False:
            while True:
                i = 10
        'Declare and initialize a new RArray, returning its address.'
        array = Register(RArray(item_type, len(values)))
        self.add(AssignMulti(array, list(values)))
        return self.add(LoadAddress(object_pointer_rprimitive if object_ptr else c_pointer_rprimitive, array))

    def shortcircuit_helper(self, op: str, expr_type: RType, left: Callable[[], Value], right: Callable[[], Value], line: int) -> Value:
        if False:
            i = 10
            return i + 15
        target = Register(expr_type)
        (left_body, right_body, next_block) = (BasicBlock(), BasicBlock(), BasicBlock())
        (true_body, false_body) = (right_body, left_body) if op == 'and' else (left_body, right_body)
        left_value = left()
        self.add_bool_branch(left_value, true_body, false_body)
        self.activate_block(left_body)
        left_coerced = self.coerce(left_value, expr_type, line)
        self.add(Assign(target, left_coerced))
        self.goto(next_block)
        self.activate_block(right_body)
        right_value = right()
        right_coerced = self.coerce(right_value, expr_type, line)
        self.add(Assign(target, right_coerced))
        self.goto(next_block)
        self.activate_block(next_block)
        return target

    def bool_value(self, value: Value) -> Value:
        if False:
            while True:
                i = 10
        'Return bool(value).\n\n        The result type can be bit_rprimitive or bool_rprimitive.\n        '
        if is_bool_rprimitive(value.type) or is_bit_rprimitive(value.type):
            result = value
        elif is_runtime_subtype(value.type, int_rprimitive):
            zero = Integer(0, short_int_rprimitive)
            result = self.comparison_op(value, zero, ComparisonOp.NEQ, value.line)
        elif is_fixed_width_rtype(value.type):
            zero = Integer(0, value.type)
            result = self.add(ComparisonOp(value, zero, ComparisonOp.NEQ))
        elif is_same_type(value.type, str_rprimitive):
            result = self.call_c(str_check_if_true, [value], value.line)
        elif is_same_type(value.type, list_rprimitive) or is_same_type(value.type, dict_rprimitive):
            length = self.builtin_len(value, value.line)
            zero = Integer(0)
            result = self.binary_op(length, zero, '!=', value.line)
        elif isinstance(value.type, RInstance) and value.type.class_ir.is_ext_class and value.type.class_ir.has_method('__bool__'):
            result = self.gen_method_call(value, '__bool__', [], bool_rprimitive, value.line)
        elif is_float_rprimitive(value.type):
            result = self.compare_floats(value, Float(0.0), FloatComparisonOp.NEQ, value.line)
        else:
            value_type = optional_value_type(value.type)
            if value_type is not None:
                not_none = self.translate_is_op(value, self.none_object(), 'is not', value.line)
                always_truthy = False
                if isinstance(value_type, RInstance):
                    if not value_type.class_ir.has_method('__bool__') and value_type.class_ir.is_method_final('__bool__'):
                        always_truthy = True
                if always_truthy:
                    result = not_none
                else:
                    result = Register(bit_rprimitive)
                    (true, false, end) = (BasicBlock(), BasicBlock(), BasicBlock())
                    branch = Branch(not_none, true, false, Branch.BOOL)
                    self.add(branch)
                    self.activate_block(true)
                    remaining = self.unbox_or_cast(value, value_type, value.line)
                    as_bool = self.bool_value(remaining)
                    self.add(Assign(result, as_bool))
                    self.goto(end)
                    self.activate_block(false)
                    self.add(Assign(result, Integer(0, bit_rprimitive)))
                    self.goto(end)
                    self.activate_block(end)
            else:
                result = self.call_c(bool_op, [value], value.line)
        return result

    def add_bool_branch(self, value: Value, true: BasicBlock, false: BasicBlock) -> None:
        if False:
            for i in range(10):
                print('nop')
        opt_value_type = optional_value_type(value.type)
        if opt_value_type is None:
            bool_value = self.bool_value(value)
            self.add(Branch(bool_value, true, false, Branch.BOOL))
        else:
            is_none = self.translate_is_op(value, self.none_object(), 'is not', value.line)
            branch = Branch(is_none, true, false, Branch.BOOL)
            self.add(branch)
            always_truthy = False
            if isinstance(opt_value_type, RInstance):
                if not opt_value_type.class_ir.has_method('__bool__') and opt_value_type.class_ir.is_method_final('__bool__'):
                    always_truthy = True
            if not always_truthy:
                branch.true = BasicBlock()
                self.activate_block(branch.true)
                remaining = self.unbox_or_cast(value, opt_value_type, value.line)
                self.add_bool_branch(remaining, true, false)

    def call_c(self, desc: CFunctionDescription, args: list[Value], line: int, result_type: RType | None=None) -> Value:
        if False:
            while True:
                i = 10
        'Call function using C/native calling convention (not a Python callable).'
        coerced = []
        for i in range(min(len(args), len(desc.arg_types))):
            formal_type = desc.arg_types[i]
            arg = args[i]
            arg = self.coerce(arg, formal_type, line)
            coerced.append(arg)
        if desc.ordering is not None:
            assert desc.var_arg_type is None
            coerced = [coerced[i] for i in desc.ordering]
        var_arg_idx = -1
        if desc.var_arg_type is not None:
            var_arg_idx = len(desc.arg_types)
            for i in range(len(desc.arg_types), len(args)):
                arg = args[i]
                arg = self.coerce(arg, desc.var_arg_type, line)
                coerced.append(arg)
        for item in desc.extra_int_constants:
            (val, typ) = item
            extra_int_constant = Integer(val, typ, line)
            coerced.append(extra_int_constant)
        error_kind = desc.error_kind
        if error_kind == ERR_NEG_INT:
            error_kind = ERR_NEVER
        target = self.add(CallC(desc.c_function_name, coerced, desc.return_type, desc.steals, desc.is_borrowed, error_kind, line, var_arg_idx))
        if desc.is_borrowed:
            for arg in coerced:
                if not isinstance(arg, (Integer, LoadLiteral)):
                    self.keep_alives.append(arg)
        if desc.error_kind == ERR_NEG_INT:
            comp = ComparisonOp(target, Integer(0, desc.return_type, line), ComparisonOp.SGE, line)
            comp.error_kind = ERR_FALSE
            self.add(comp)
        if desc.truncated_type is None:
            result = target
        else:
            truncate = self.add(Truncate(target, desc.truncated_type))
            result = truncate
        if result_type and (not is_runtime_subtype(result.type, result_type)):
            if is_none_rprimitive(result_type):
                result = self.none()
            else:
                result = self.coerce(target, result_type, line, can_borrow=desc.is_borrowed)
        return result

    def matching_call_c(self, candidates: list[CFunctionDescription], args: list[Value], line: int, result_type: RType | None=None, can_borrow: bool=False) -> Value | None:
        if False:
            for i in range(10):
                print('nop')
        matching: CFunctionDescription | None = None
        for desc in candidates:
            if len(desc.arg_types) != len(args):
                continue
            if all((is_subtype(actual.type, formal) for (actual, formal) in zip(args, desc.arg_types))) and (not desc.is_borrowed or can_borrow):
                if matching:
                    assert matching.priority != desc.priority, 'Ambiguous:\n1) {}\n2) {}'.format(matching, desc)
                    if desc.priority > matching.priority:
                        matching = desc
                else:
                    matching = desc
        if matching:
            target = self.call_c(matching, args, line, result_type)
            return target
        return None

    def int_op(self, type: RType, lhs: Value, rhs: Value, op: int, line: int=-1) -> Value:
        if False:
            print('Hello World!')
        'Generate a native integer binary op.\n\n        Use native/C semantics, which sometimes differ from Python\n        semantics.\n\n        Args:\n            type: Either int64_rprimitive or int32_rprimitive\n            op: IntOp.* constant (e.g. IntOp.ADD)\n        '
        return self.add(IntOp(type, lhs, rhs, op, line))

    def float_op(self, lhs: Value, rhs: Value, op: str, line: int) -> Value:
        if False:
            return 10
        "Generate a native float binary arithmetic operation.\n\n        This follows Python semantics (e.g. raise exception on division by zero).\n        Add a FloatOp directly if you want low-level semantics.\n\n        Args:\n            op: Binary operator (e.g. '+' or '*')\n        "
        op_id = float_op_to_id[op]
        if op_id in (FloatOp.DIV, FloatOp.MOD):
            if not (isinstance(rhs, Float) and rhs.value != 0.0):
                c = self.compare_floats(rhs, Float(0.0), FloatComparisonOp.EQ, line)
                (err, ok) = (BasicBlock(), BasicBlock())
                self.add(Branch(c, err, ok, Branch.BOOL, rare=True))
                self.activate_block(err)
                if op_id == FloatOp.DIV:
                    msg = 'float division by zero'
                else:
                    msg = 'float modulo'
                self.add(RaiseStandardError(RaiseStandardError.ZERO_DIVISION_ERROR, msg, line))
                self.add(Unreachable())
                self.activate_block(ok)
        if op_id == FloatOp.MOD:
            return self.float_mod(lhs, rhs, line)
        else:
            return self.add(FloatOp(lhs, rhs, op_id, line))

    def float_mod(self, lhs: Value, rhs: Value, line: int) -> Value:
        if False:
            for i in range(10):
                print('nop')
        'Perform x % y on floats using Python semantics.'
        mod = self.add(FloatOp(lhs, rhs, FloatOp.MOD, line))
        res = Register(float_rprimitive)
        self.add(Assign(res, mod))
        (tricky, adjust, copysign, done) = (BasicBlock(), BasicBlock(), BasicBlock(), BasicBlock())
        is_zero = self.add(FloatComparisonOp(res, Float(0.0), FloatComparisonOp.EQ, line))
        self.add(Branch(is_zero, copysign, tricky, Branch.BOOL))
        self.activate_block(tricky)
        same_signs = self.is_same_float_signs(lhs, rhs, line)
        self.add(Branch(same_signs, done, adjust, Branch.BOOL))
        self.activate_block(adjust)
        adj = self.float_op(res, rhs, '+', line)
        self.add(Assign(res, adj))
        self.add(Goto(done))
        self.activate_block(copysign)
        adj = self.call_c(copysign_op, [Float(0.0), rhs], line)
        self.add(Assign(res, adj))
        self.add(Goto(done))
        self.activate_block(done)
        return res

    def compare_floats(self, lhs: Value, rhs: Value, op: int, line: int) -> Value:
        if False:
            i = 10
            return i + 15
        return self.add(FloatComparisonOp(lhs, rhs, op, line))

    def fixed_width_int_op(self, type: RPrimitive, lhs: Value, rhs: Value, op: int, line: int) -> Value:
        if False:
            return 10
        'Generate a binary op using Python fixed-width integer semantics.\n\n        These may differ in overflow/rounding behavior from native/C ops.\n\n        Args:\n            type: Either int64_rprimitive or int32_rprimitive\n            op: IntOp.* constant (e.g. IntOp.ADD)\n        '
        lhs = self.coerce(lhs, type, line)
        rhs = self.coerce(rhs, type, line)
        if op == IntOp.DIV:
            if isinstance(rhs, Integer) and rhs.value not in (-1, 0):
                if not type.is_signed:
                    return self.int_op(type, lhs, rhs, IntOp.DIV, line)
                else:
                    return self.inline_fixed_width_divide(type, lhs, rhs, line)
            if is_int64_rprimitive(type):
                prim = int64_divide_op
            elif is_int32_rprimitive(type):
                prim = int32_divide_op
            elif is_int16_rprimitive(type):
                prim = int16_divide_op
            elif is_uint8_rprimitive(type):
                self.check_for_zero_division(rhs, type, line)
                return self.int_op(type, lhs, rhs, op, line)
            else:
                assert False, type
            return self.call_c(prim, [lhs, rhs], line)
        if op == IntOp.MOD:
            if isinstance(rhs, Integer) and rhs.value not in (-1, 0):
                if not type.is_signed:
                    return self.int_op(type, lhs, rhs, IntOp.MOD, line)
                else:
                    return self.inline_fixed_width_mod(type, lhs, rhs, line)
            if is_int64_rprimitive(type):
                prim = int64_mod_op
            elif is_int32_rprimitive(type):
                prim = int32_mod_op
            elif is_int16_rprimitive(type):
                prim = int16_mod_op
            elif is_uint8_rprimitive(type):
                self.check_for_zero_division(rhs, type, line)
                return self.int_op(type, lhs, rhs, op, line)
            else:
                assert False, type
            return self.call_c(prim, [lhs, rhs], line)
        return self.int_op(type, lhs, rhs, op, line)

    def check_for_zero_division(self, rhs: Value, type: RType, line: int) -> None:
        if False:
            i = 10
            return i + 15
        (err, ok) = (BasicBlock(), BasicBlock())
        is_zero = self.binary_op(rhs, Integer(0, type), '==', line)
        self.add(Branch(is_zero, err, ok, Branch.BOOL))
        self.activate_block(err)
        self.add(RaiseStandardError(RaiseStandardError.ZERO_DIVISION_ERROR, 'integer division or modulo by zero', line))
        self.add(Unreachable())
        self.activate_block(ok)

    def inline_fixed_width_divide(self, type: RType, lhs: Value, rhs: Value, line: int) -> Value:
        if False:
            print('Hello World!')
        res = Register(type)
        div = self.int_op(type, lhs, rhs, IntOp.DIV, line)
        self.add(Assign(res, div))
        same_signs = self.is_same_native_int_signs(type, lhs, rhs, line)
        (tricky, adjust, done) = (BasicBlock(), BasicBlock(), BasicBlock())
        self.add(Branch(same_signs, done, tricky, Branch.BOOL))
        self.activate_block(tricky)
        mul = self.int_op(type, res, rhs, IntOp.MUL, line)
        mul_eq = self.add(ComparisonOp(mul, lhs, ComparisonOp.EQ, line))
        self.add(Branch(mul_eq, done, adjust, Branch.BOOL))
        self.activate_block(adjust)
        adj = self.int_op(type, res, Integer(1, type), IntOp.SUB, line)
        self.add(Assign(res, adj))
        self.add(Goto(done))
        self.activate_block(done)
        return res

    def inline_fixed_width_mod(self, type: RType, lhs: Value, rhs: Value, line: int) -> Value:
        if False:
            for i in range(10):
                print('nop')
        res = Register(type)
        mod = self.int_op(type, lhs, rhs, IntOp.MOD, line)
        self.add(Assign(res, mod))
        same_signs = self.is_same_native_int_signs(type, lhs, rhs, line)
        (tricky, adjust, done) = (BasicBlock(), BasicBlock(), BasicBlock())
        self.add(Branch(same_signs, done, tricky, Branch.BOOL))
        self.activate_block(tricky)
        is_zero = self.add(ComparisonOp(res, Integer(0, type), ComparisonOp.EQ, line))
        self.add(Branch(is_zero, done, adjust, Branch.BOOL))
        self.activate_block(adjust)
        adj = self.int_op(type, res, rhs, IntOp.ADD, line)
        self.add(Assign(res, adj))
        self.add(Goto(done))
        self.activate_block(done)
        return res

    def is_same_native_int_signs(self, type: RType, a: Value, b: Value, line: int) -> Value:
        if False:
            while True:
                i = 10
        neg1 = self.add(ComparisonOp(a, Integer(0, type), ComparisonOp.SLT, line))
        neg2 = self.add(ComparisonOp(b, Integer(0, type), ComparisonOp.SLT, line))
        return self.add(ComparisonOp(neg1, neg2, ComparisonOp.EQ, line))

    def is_same_float_signs(self, a: Value, b: Value, line: int) -> Value:
        if False:
            while True:
                i = 10
        neg1 = self.add(FloatComparisonOp(a, Float(0.0), FloatComparisonOp.LT, line))
        neg2 = self.add(FloatComparisonOp(b, Float(0.0), FloatComparisonOp.LT, line))
        return self.add(ComparisonOp(neg1, neg2, ComparisonOp.EQ, line))

    def comparison_op(self, lhs: Value, rhs: Value, op: int, line: int) -> Value:
        if False:
            return 10
        return self.add(ComparisonOp(lhs, rhs, op, line))

    def builtin_len(self, val: Value, line: int, use_pyssize_t: bool=False) -> Value:
        if False:
            return 10
        'Generate len(val).\n\n        Return short_int_rprimitive by default.\n        Return c_pyssize_t if use_pyssize_t is true (unshifted).\n        '
        typ = val.type
        size_value = None
        if is_list_rprimitive(typ) or is_tuple_rprimitive(typ) or is_bytes_rprimitive(typ):
            elem_address = self.add(GetElementPtr(val, PyVarObject, 'ob_size'))
            size_value = self.add(LoadMem(c_pyssize_t_rprimitive, elem_address))
            self.add(KeepAlive([val]))
        elif is_set_rprimitive(typ):
            elem_address = self.add(GetElementPtr(val, PySetObject, 'used'))
            size_value = self.add(LoadMem(c_pyssize_t_rprimitive, elem_address))
            self.add(KeepAlive([val]))
        elif is_dict_rprimitive(typ):
            size_value = self.call_c(dict_ssize_t_size_op, [val], line)
        elif is_str_rprimitive(typ):
            size_value = self.call_c(str_ssize_t_size_op, [val], line)
        if size_value is not None:
            if use_pyssize_t:
                return size_value
            offset = Integer(1, c_pyssize_t_rprimitive, line)
            return self.int_op(short_int_rprimitive, size_value, offset, IntOp.LEFT_SHIFT, line)
        if isinstance(typ, RInstance):
            assert not use_pyssize_t
            length = self.gen_method_call(val, '__len__', [], int_rprimitive, line)
            length = self.coerce(length, int_rprimitive, line)
            (ok, fail) = (BasicBlock(), BasicBlock())
            self.compare_tagged_condition(length, Integer(0), '>=', ok, fail, line)
            self.activate_block(fail)
            self.add(RaiseStandardError(RaiseStandardError.VALUE_ERROR, '__len__() should return >= 0', line))
            self.add(Unreachable())
            self.activate_block(ok)
            return length
        if use_pyssize_t:
            return self.call_c(generic_ssize_t_len_op, [val], line)
        else:
            return self.call_c(generic_len_op, [val], line)

    def new_tuple(self, items: list[Value], line: int) -> Value:
        if False:
            print('Hello World!')
        size: Value = Integer(len(items), c_pyssize_t_rprimitive)
        return self.call_c(new_tuple_op, [size] + items, line)

    def new_tuple_with_length(self, length: Value, line: int) -> Value:
        if False:
            return 10
        'This function returns an uninitialized tuple.\n\n        If the length is non-zero, the caller must initialize the tuple, before\n        it can be made visible to user code -- otherwise the tuple object is broken.\n        You might need further initialization with `new_tuple_set_item_op` op.\n\n        Args:\n            length: desired length of the new tuple. The rtype should be\n                    c_pyssize_t_rprimitive\n            line: line number\n        '
        return self.call_c(new_tuple_with_length_op, [length], line)

    def int_to_float(self, n: Value, line: int) -> Value:
        if False:
            for i in range(10):
                print('nop')
        return self.call_c(int_to_float_op, [n], line)

    def decompose_union_helper(self, obj: Value, rtype: RUnion, result_type: RType, process_item: Callable[[Value], Value], line: int) -> Value:
        if False:
            i = 10
            return i + 15
        'Generate isinstance() + specialized operations for union items.\n\n        Say, for Union[A, B] generate ops resembling this (pseudocode):\n\n            if isinstance(obj, A):\n                result = <result of process_item(cast(A, obj)>\n            else:\n                result = <result of process_item(cast(B, obj)>\n\n        Args:\n            obj: value with a union type\n            rtype: the union type\n            result_type: result of the operation\n            process_item: callback to generate op for a single union item (arg is coerced\n                to union item type)\n            line: line number\n        '
        fast_items = []
        rest_items = []
        for item in rtype.items:
            if isinstance(item, RInstance):
                fast_items.append(item)
            else:
                rest_items.append(item)
        exit_block = BasicBlock()
        result = Register(result_type)
        for (i, item) in enumerate(fast_items):
            more_types = i < len(fast_items) - 1 or rest_items
            if more_types:
                op = self.isinstance_native(obj, item.class_ir, line)
                (true_block, false_block) = (BasicBlock(), BasicBlock())
                self.add_bool_branch(op, true_block, false_block)
                self.activate_block(true_block)
            coerced = self.coerce(obj, item, line)
            temp = process_item(coerced)
            temp2 = self.coerce(temp, result_type, line)
            self.add(Assign(result, temp2))
            self.goto(exit_block)
            if more_types:
                self.activate_block(false_block)
        if rest_items:
            coerced = self.coerce(obj, object_rprimitive, line, force=True)
            temp = process_item(coerced)
            temp2 = self.coerce(temp, result_type, line)
            self.add(Assign(result, temp2))
            self.goto(exit_block)
        self.activate_block(exit_block)
        return result

    def translate_special_method_call(self, base_reg: Value, name: str, args: list[Value], result_type: RType | None, line: int, can_borrow: bool=False) -> Value | None:
        if False:
            i = 10
            return i + 15
        'Translate a method call which is handled nongenerically.\n\n        These are special in the sense that we have code generated specifically for them.\n        They tend to be method calls which have equivalents in C that are more direct\n        than calling with the PyObject api.\n\n        Return None if no translation found; otherwise return the target register.\n        '
        call_c_ops_candidates = method_call_ops.get(name, [])
        call_c_op = self.matching_call_c(call_c_ops_candidates, [base_reg] + args, line, result_type, can_borrow=can_borrow)
        return call_c_op

    def translate_eq_cmp(self, lreg: Value, rreg: Value, expr_op: str, line: int) -> Value | None:
        if False:
            print('Hello World!')
        "Add a equality comparison operation.\n\n        Args:\n            expr_op: either '==' or '!='\n        "
        ltype = lreg.type
        rtype = rreg.type
        if not (isinstance(ltype, RInstance) and ltype == rtype):
            return None
        class_ir = ltype.class_ir
        cmp_varies_at_runtime = not class_ir.is_method_final('__eq__') or not class_ir.is_method_final('__ne__') or class_ir.inherits_python or class_ir.is_augmented
        if cmp_varies_at_runtime:
            return None
        if not class_ir.has_method('__eq__'):
            identity_ref_op = 'is' if expr_op == '==' else 'is not'
            return self.translate_is_op(lreg, rreg, identity_ref_op, line)
        return self.gen_method_call(lreg, op_methods[expr_op], [rreg], ltype, line)

    def translate_is_op(self, lreg: Value, rreg: Value, expr_op: str, line: int) -> Value:
        if False:
            for i in range(10):
                print('nop')
        "Create equality comparison operation between object identities\n\n        Args:\n            expr_op: either 'is' or 'is not'\n        "
        op = ComparisonOp.EQ if expr_op == 'is' else ComparisonOp.NEQ
        lhs = self.coerce(lreg, object_rprimitive, line)
        rhs = self.coerce(rreg, object_rprimitive, line)
        return self.add(ComparisonOp(lhs, rhs, op, line))

    def _create_dict(self, keys: list[Value], values: list[Value], line: int) -> Value:
        if False:
            for i in range(10):
                print('nop')
        'Create a dictionary(possibly empty) using keys and values'
        size = len(keys)
        if size > 0:
            size_value: Value = Integer(size, c_pyssize_t_rprimitive)
            items = [i for t in list(zip(keys, values)) for i in t]
            return self.call_c(dict_build_op, [size_value] + items, line)
        else:
            return self.call_c(dict_new_op, [], line)

    def error(self, msg: str, line: int) -> None:
        if False:
            while True:
                i = 10
        self.errors.error(msg, self.module_path, line)

def num_positional_args(arg_values: list[Value], arg_kinds: list[ArgKind] | None) -> int:
    if False:
        print('Hello World!')
    if arg_kinds is None:
        return len(arg_values)
    num_pos = 0
    for kind in arg_kinds:
        if kind == ARG_POS:
            num_pos += 1
    return num_pos