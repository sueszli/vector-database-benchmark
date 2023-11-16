"""Code generation for native function bodies."""
from __future__ import annotations
from typing import Final
from mypyc.analysis.blockfreq import frequently_executed_blocks
from mypyc.codegen.emit import DEBUG_ERRORS, Emitter, TracebackAndGotoHandler, c_array_initializer
from mypyc.common import MODULE_PREFIX, NATIVE_PREFIX, REG_PREFIX, STATIC_PREFIX, TYPE_PREFIX
from mypyc.ir.class_ir import ClassIR
from mypyc.ir.func_ir import FUNC_CLASSMETHOD, FUNC_STATICMETHOD, FuncDecl, FuncIR, all_values
from mypyc.ir.ops import ERR_FALSE, NAMESPACE_MODULE, NAMESPACE_STATIC, NAMESPACE_TYPE, Assign, AssignMulti, BasicBlock, Box, Branch, Call, CallC, Cast, ComparisonOp, ControlOp, DecRef, Extend, Float, FloatComparisonOp, FloatNeg, FloatOp, GetAttr, GetElementPtr, Goto, IncRef, InitStatic, Integer, IntOp, KeepAlive, LoadAddress, LoadErrorValue, LoadGlobal, LoadLiteral, LoadMem, LoadStatic, MethodCall, Op, OpVisitor, RaiseStandardError, Register, Return, SetAttr, SetMem, Truncate, TupleGet, TupleSet, Unborrow, Unbox, Unreachable, Value
from mypyc.ir.pprint import generate_names_for_ir
from mypyc.ir.rtypes import RArray, RStruct, RTuple, RType, is_int32_rprimitive, is_int64_rprimitive, is_int_rprimitive, is_pointer_rprimitive, is_tagged

def native_function_type(fn: FuncIR, emitter: Emitter) -> str:
    if False:
        i = 10
        return i + 15
    args = ', '.join((emitter.ctype(arg.type) for arg in fn.args)) or 'void'
    ret = emitter.ctype(fn.ret_type)
    return f'{ret} (*)({args})'

def native_function_header(fn: FuncDecl, emitter: Emitter) -> str:
    if False:
        i = 10
        return i + 15
    args = []
    for arg in fn.sig.args:
        args.append(f'{emitter.ctype_spaced(arg.type)}{REG_PREFIX}{arg.name}')
    return '{ret_type}{name}({args})'.format(ret_type=emitter.ctype_spaced(fn.sig.ret_type), name=emitter.native_function_name(fn), args=', '.join(args) or 'void')

def generate_native_function(fn: FuncIR, emitter: Emitter, source_path: str, module_name: str) -> None:
    if False:
        while True:
            i = 10
    declarations = Emitter(emitter.context)
    names = generate_names_for_ir(fn.arg_regs, fn.blocks)
    body = Emitter(emitter.context, names)
    visitor = FunctionEmitterVisitor(body, declarations, source_path, module_name)
    declarations.emit_line(f'{native_function_header(fn.decl, emitter)} {{')
    body.indent()
    for r in all_values(fn.arg_regs, fn.blocks):
        if isinstance(r.type, RTuple):
            emitter.declare_tuple_struct(r.type)
        if isinstance(r.type, RArray):
            continue
        if r in fn.arg_regs:
            continue
        ctype = emitter.ctype_spaced(r.type)
        init = ''
        declarations.emit_line('{ctype}{prefix}{name}{init};'.format(ctype=ctype, prefix=REG_PREFIX, name=names[r], init=init))
    blocks = fn.blocks
    for (i, block) in enumerate(blocks):
        block.label = i
    for block in fn.blocks:
        terminator = block.terminator
        assert isinstance(terminator, ControlOp)
        for target in terminator.targets():
            is_next_block = target.label == block.label + 1
            is_problematic_op = isinstance(terminator, Branch) and any((isinstance(s, GetAttr) for s in terminator.sources()))
            if not is_next_block or is_problematic_op:
                fn.blocks[target.label].referenced = True
    common = frequently_executed_blocks(fn.blocks[0])
    for i in range(len(blocks)):
        block = blocks[i]
        visitor.rare = block not in common
        next_block = None
        if i + 1 < len(blocks):
            next_block = blocks[i + 1]
        body.emit_label(block)
        visitor.next_block = next_block
        ops = block.ops
        visitor.ops = ops
        visitor.op_index = 0
        while visitor.op_index < len(ops):
            ops[visitor.op_index].accept(visitor)
            visitor.op_index += 1
    body.emit_line('}')
    emitter.emit_from_emitter(declarations)
    emitter.emit_from_emitter(body)

class FunctionEmitterVisitor(OpVisitor[None]):

    def __init__(self, emitter: Emitter, declarations: Emitter, source_path: str, module_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.emitter = emitter
        self.names = emitter.names
        self.declarations = declarations
        self.source_path = source_path
        self.module_name = module_name
        self.literals = emitter.context.literals
        self.rare = False
        self.next_block: BasicBlock | None = None
        self.ops: list[Op] = []
        self.op_index = 0

    def temp_name(self) -> str:
        if False:
            return 10
        return self.emitter.temp_name()

    def visit_goto(self, op: Goto) -> None:
        if False:
            for i in range(10):
                print('nop')
        if op.label is not self.next_block:
            self.emit_line('goto %s;' % self.label(op.label))

    def visit_branch(self, op: Branch) -> None:
        if False:
            return 10
        (true, false) = (op.true, op.false)
        negated = op.negated
        negated_rare = False
        if true is self.next_block and op.traceback_entry is None:
            (true, false) = (false, true)
            negated = not negated
            negated_rare = True
        neg = '!' if negated else ''
        cond = ''
        if op.op == Branch.BOOL:
            expr_result = self.reg(op.value)
            cond = f'{neg}{expr_result}'
        elif op.op == Branch.IS_ERROR:
            typ = op.value.type
            compare = '!=' if negated else '=='
            if isinstance(typ, RTuple):
                cond = self.emitter.tuple_undefined_check_cond(typ, self.reg(op.value), self.c_error_value, compare)
            else:
                cond = f'{self.reg(op.value)} {compare} {self.c_error_value(typ)}'
        else:
            assert False, 'Invalid branch'
        if op.traceback_entry is not None or op.rare:
            if not negated_rare:
                cond = f'unlikely({cond})'
            else:
                cond = f'likely({cond})'
        if false is self.next_block:
            if op.traceback_entry is None:
                if true is not self.next_block:
                    self.emit_line(f'if ({cond}) goto {self.label(true)};')
            else:
                self.emit_line(f'if ({cond}) {{')
                self.emit_traceback(op)
                self.emit_lines('goto %s;' % self.label(true), '}')
        else:
            self.emit_line(f'if ({cond}) {{')
            self.emit_traceback(op)
            if true is not self.next_block:
                self.emit_line('goto %s;' % self.label(true))
            self.emit_lines('} else', '    goto %s;' % self.label(false))

    def visit_return(self, op: Return) -> None:
        if False:
            for i in range(10):
                print('nop')
        value_str = self.reg(op.value)
        self.emit_line('return %s;' % value_str)

    def visit_tuple_set(self, op: TupleSet) -> None:
        if False:
            i = 10
            return i + 15
        dest = self.reg(op)
        tuple_type = op.tuple_type
        self.emitter.declare_tuple_struct(tuple_type)
        if len(op.items) == 0:
            self.emit_line(f'{dest}.empty_struct_error_flag = 0;')
        else:
            for (i, item) in enumerate(op.items):
                self.emit_line(f'{dest}.f{i} = {self.reg(item)};')

    def visit_assign(self, op: Assign) -> None:
        if False:
            print('Hello World!')
        dest = self.reg(op.dest)
        src = self.reg(op.src)
        if dest != src:
            if op.src.type.is_unboxed and (not op.dest.type.is_unboxed):
                src = f'(void *){src}'
            self.emit_line(f'{dest} = {src};')

    def visit_assign_multi(self, op: AssignMulti) -> None:
        if False:
            for i in range(10):
                print('nop')
        typ = op.dest.type
        assert isinstance(typ, RArray)
        dest = self.reg(op.dest)
        self.emit_line('%s%s[%d] = %s;' % (self.emitter.ctype_spaced(typ.item_type), dest, len(op.src), c_array_initializer([self.reg(s) for s in op.src], indented=True)))

    def visit_load_error_value(self, op: LoadErrorValue) -> None:
        if False:
            while True:
                i = 10
        if isinstance(op.type, RTuple):
            values = [self.c_undefined_value(item) for item in op.type.types]
            tmp = self.temp_name()
            self.emit_line('{} {} = {{ {} }};'.format(self.ctype(op.type), tmp, ', '.join(values)))
            self.emit_line(f'{self.reg(op)} = {tmp};')
        else:
            self.emit_line(f'{self.reg(op)} = {self.c_error_value(op.type)};')

    def visit_load_literal(self, op: LoadLiteral) -> None:
        if False:
            while True:
                i = 10
        index = self.literals.literal_index(op.value)
        if not is_int_rprimitive(op.type):
            self.emit_line('%s = CPyStatics[%d];' % (self.reg(op), index), ann=op.value)
        else:
            self.emit_line('%s = (CPyTagged)CPyStatics[%d] | 1;' % (self.reg(op), index), ann=op.value)

    def get_attr_expr(self, obj: str, op: GetAttr | SetAttr, decl_cl: ClassIR) -> str:
        if False:
            i = 10
            return i + 15
        'Generate attribute accessor for normal (non-property) access.\n\n        This either has a form like obj->attr_name for attributes defined in non-trait\n        classes, and *(obj + attr_offset) for attributes defined by traits. We also\n        insert all necessary C casts here.\n        '
        cast = f'({op.class_type.struct_name(self.emitter.names)} *)'
        if decl_cl.is_trait and op.class_type.class_ir.is_trait:
            trait_attr_index = list(decl_cl.attributes).index(op.attr)
            offset = self.emitter.temp_name()
            self.declarations.emit_line(f'size_t {offset};')
            self.emitter.emit_line('{} = {};'.format(offset, 'CPy_FindAttrOffset({}, {}, {})'.format(self.emitter.type_struct_name(decl_cl), f'({cast}{obj})->vtable', trait_attr_index)))
            attr_cast = f'({self.ctype(op.class_type.attr_type(op.attr))} *)'
            return f'*{attr_cast}((char *){obj} + {offset})'
        else:
            if op.class_type.class_ir.is_trait:
                assert not decl_cl.is_trait
                cast = f'({decl_cl.struct_name(self.emitter.names)} *)'
            return f'({cast}{obj})->{self.emitter.attr(op.attr)}'

    def visit_get_attr(self, op: GetAttr) -> None:
        if False:
            print('Hello World!')
        dest = self.reg(op)
        obj = self.reg(op.obj)
        rtype = op.class_type
        cl = rtype.class_ir
        (attr_rtype, decl_cl) = cl.attr_details(op.attr)
        prefer_method = cl.is_trait and attr_rtype.error_overlap
        if cl.get_method(op.attr, prefer_method=prefer_method):
            version = '_TRAIT' if cl.is_trait else ''
            self.emit_line('%s = CPY_GET_ATTR%s(%s, %s, %d, %s, %s); /* %s */' % (dest, version, obj, self.emitter.type_struct_name(rtype.class_ir), rtype.getter_index(op.attr), rtype.struct_name(self.names), self.ctype(rtype.attr_type(op.attr)), op.attr))
        else:
            attr_expr = self.get_attr_expr(obj, op, decl_cl)
            self.emitter.emit_line(f'{dest} = {attr_expr};')
            always_defined = cl.is_always_defined(op.attr)
            merged_branch = None
            if not always_defined:
                self.emitter.emit_undefined_attr_check(attr_rtype, dest, '==', obj, op.attr, cl, unlikely=True)
                branch = self.next_branch()
                if branch is not None:
                    if branch.value is op and branch.op == Branch.IS_ERROR and (branch.traceback_entry is not None) and (not branch.negated):
                        self.emit_attribute_error(branch, cl.name, op.attr)
                        self.emit_line('goto %s;' % self.label(branch.true))
                        merged_branch = branch
                        self.emitter.emit_line('}')
                if not merged_branch:
                    exc_class = 'PyExc_AttributeError'
                    self.emitter.emit_line('PyErr_SetString({}, "attribute {} of {} undefined");'.format(exc_class, repr(op.attr), repr(cl.name)))
            if attr_rtype.is_refcounted and (not op.is_borrowed):
                if not merged_branch and (not always_defined):
                    self.emitter.emit_line('} else {')
                self.emitter.emit_inc_ref(dest, attr_rtype)
            if merged_branch:
                if merged_branch.false is not self.next_block:
                    self.emit_line('goto %s;' % self.label(merged_branch.false))
                self.op_index += 1
            elif not always_defined:
                self.emitter.emit_line('}')

    def next_branch(self) -> Branch | None:
        if False:
            return 10
        if self.op_index + 1 < len(self.ops):
            next_op = self.ops[self.op_index + 1]
            if isinstance(next_op, Branch):
                return next_op
        return None

    def visit_set_attr(self, op: SetAttr) -> None:
        if False:
            for i in range(10):
                print('nop')
        if op.error_kind == ERR_FALSE:
            dest = self.reg(op)
        obj = self.reg(op.obj)
        src = self.reg(op.src)
        rtype = op.class_type
        cl = rtype.class_ir
        (attr_rtype, decl_cl) = cl.attr_details(op.attr)
        if cl.get_method(op.attr):
            assert not op.is_init and op.error_kind == ERR_FALSE, '%s %d %d %s' % (op.attr, op.is_init, op.error_kind, rtype)
            version = '_TRAIT' if cl.is_trait else ''
            self.emit_line('%s = CPY_SET_ATTR%s(%s, %s, %d, %s, %s, %s); /* %s */' % (dest, version, obj, self.emitter.type_struct_name(rtype.class_ir), rtype.setter_index(op.attr), src, rtype.struct_name(self.names), self.ctype(rtype.attr_type(op.attr)), op.attr))
        else:
            attr_expr = self.get_attr_expr(obj, op, decl_cl)
            if not op.is_init and attr_rtype.is_refcounted:
                always_defined = cl.is_always_defined(op.attr)
                if not always_defined:
                    self.emitter.emit_undefined_attr_check(attr_rtype, attr_expr, '!=', obj, op.attr, cl)
                self.emitter.emit_dec_ref(attr_expr, attr_rtype)
                if not always_defined:
                    self.emitter.emit_line('}')
            elif attr_rtype.error_overlap and (not cl.is_always_defined(op.attr)):
                self.emitter.emit_attr_bitmap_set(src, obj, attr_rtype, cl, op.attr)
            self.emitter.emit_line(f'{attr_expr} = {src};')
            if op.error_kind == ERR_FALSE:
                self.emitter.emit_line(f'{dest} = 1;')
    PREFIX_MAP: Final = {NAMESPACE_STATIC: STATIC_PREFIX, NAMESPACE_TYPE: TYPE_PREFIX, NAMESPACE_MODULE: MODULE_PREFIX}

    def visit_load_static(self, op: LoadStatic) -> None:
        if False:
            print('Hello World!')
        dest = self.reg(op)
        prefix = self.PREFIX_MAP[op.namespace]
        name = self.emitter.static_name(op.identifier, op.module_name, prefix)
        if op.namespace == NAMESPACE_TYPE:
            name = '(PyObject *)%s' % name
        self.emit_line(f'{dest} = {name};', ann=op.ann)

    def visit_init_static(self, op: InitStatic) -> None:
        if False:
            return 10
        value = self.reg(op.value)
        prefix = self.PREFIX_MAP[op.namespace]
        name = self.emitter.static_name(op.identifier, op.module_name, prefix)
        if op.namespace == NAMESPACE_TYPE:
            value = '(PyTypeObject *)%s' % value
        self.emit_line(f'{name} = {value};')
        self.emit_inc_ref(name, op.value.type)

    def visit_tuple_get(self, op: TupleGet) -> None:
        if False:
            while True:
                i = 10
        dest = self.reg(op)
        src = self.reg(op.src)
        self.emit_line(f'{dest} = {src}.f{op.index};')
        if not op.is_borrowed:
            self.emit_inc_ref(dest, op.type)

    def get_dest_assign(self, dest: Value) -> str:
        if False:
            i = 10
            return i + 15
        if not dest.is_void:
            return self.reg(dest) + ' = '
        else:
            return ''

    def visit_call(self, op: Call) -> None:
        if False:
            print('Hello World!')
        'Call native function.'
        dest = self.get_dest_assign(op)
        args = ', '.join((self.reg(arg) for arg in op.args))
        lib = self.emitter.get_group_prefix(op.fn)
        cname = op.fn.cname(self.names)
        self.emit_line(f'{dest}{lib}{NATIVE_PREFIX}{cname}({args});')

    def visit_method_call(self, op: MethodCall) -> None:
        if False:
            i = 10
            return i + 15
        'Call native method.'
        dest = self.get_dest_assign(op)
        obj = self.reg(op.obj)
        rtype = op.receiver_type
        class_ir = rtype.class_ir
        name = op.method
        method = rtype.class_ir.get_method(name)
        assert method is not None
        is_direct = class_ir.is_method_final(name)
        obj_args = [] if method.decl.kind == FUNC_STATICMETHOD else [f'(PyObject *)Py_TYPE({obj})'] if method.decl.kind == FUNC_CLASSMETHOD else [obj]
        args = ', '.join(obj_args + [self.reg(arg) for arg in op.args])
        mtype = native_function_type(method, self.emitter)
        version = '_TRAIT' if rtype.class_ir.is_trait else ''
        if is_direct:
            lib = self.emitter.get_group_prefix(method.decl)
            self.emit_line(f'{dest}{lib}{NATIVE_PREFIX}{method.cname(self.names)}({args});')
        else:
            method_idx = rtype.method_index(name)
            self.emit_line('{}CPY_GET_METHOD{}({}, {}, {}, {}, {})({}); /* {} */'.format(dest, version, obj, self.emitter.type_struct_name(rtype.class_ir), method_idx, rtype.struct_name(self.names), mtype, args, op.method))

    def visit_inc_ref(self, op: IncRef) -> None:
        if False:
            i = 10
            return i + 15
        src = self.reg(op.src)
        self.emit_inc_ref(src, op.src.type)

    def visit_dec_ref(self, op: DecRef) -> None:
        if False:
            print('Hello World!')
        src = self.reg(op.src)
        self.emit_dec_ref(src, op.src.type, is_xdec=op.is_xdec)

    def visit_box(self, op: Box) -> None:
        if False:
            i = 10
            return i + 15
        self.emitter.emit_box(self.reg(op.src), self.reg(op), op.src.type, can_borrow=True)

    def visit_cast(self, op: Cast) -> None:
        if False:
            while True:
                i = 10
        branch = self.next_branch()
        handler = None
        if branch is not None:
            if branch.value is op and branch.op == Branch.IS_ERROR and (branch.traceback_entry is not None) and (not branch.negated) and (branch.false is self.next_block):
                handler = TracebackAndGotoHandler(self.label(branch.true), self.source_path, self.module_name, branch.traceback_entry)
                self.op_index += 1
        self.emitter.emit_cast(self.reg(op.src), self.reg(op), op.type, src_type=op.src.type, error=handler)

    def visit_unbox(self, op: Unbox) -> None:
        if False:
            i = 10
            return i + 15
        self.emitter.emit_unbox(self.reg(op.src), self.reg(op), op.type)

    def visit_unreachable(self, op: Unreachable) -> None:
        if False:
            while True:
                i = 10
        self.emitter.emit_line('CPy_Unreachable();')

    def visit_raise_standard_error(self, op: RaiseStandardError) -> None:
        if False:
            return 10
        if op.value is not None:
            if isinstance(op.value, str):
                message = op.value.replace('"', '\\"')
                self.emitter.emit_line(f'PyErr_SetString(PyExc_{op.class_name}, "{message}");')
            elif isinstance(op.value, Value):
                self.emitter.emit_line('PyErr_SetObject(PyExc_{}, {});'.format(op.class_name, self.emitter.reg(op.value)))
            else:
                assert False, 'op value type must be either str or Value'
        else:
            self.emitter.emit_line(f'PyErr_SetNone(PyExc_{op.class_name});')
        self.emitter.emit_line(f'{self.reg(op)} = 0;')

    def visit_call_c(self, op: CallC) -> None:
        if False:
            while True:
                i = 10
        if op.is_void:
            dest = ''
        else:
            dest = self.get_dest_assign(op)
        args = ', '.join((self.reg(arg) for arg in op.args))
        self.emitter.emit_line(f'{dest}{op.function_name}({args});')

    def visit_truncate(self, op: Truncate) -> None:
        if False:
            while True:
                i = 10
        dest = self.reg(op)
        value = self.reg(op.src)
        self.emit_line(f'{dest} = {value};')

    def visit_extend(self, op: Extend) -> None:
        if False:
            return 10
        dest = self.reg(op)
        value = self.reg(op.src)
        if op.signed:
            src_cast = self.emit_signed_int_cast(op.src.type)
        else:
            src_cast = self.emit_unsigned_int_cast(op.src.type)
        self.emit_line(f'{dest} = {src_cast}{value};')

    def visit_load_global(self, op: LoadGlobal) -> None:
        if False:
            i = 10
            return i + 15
        dest = self.reg(op)
        self.emit_line(f'{dest} = {op.identifier};', ann=op.ann)

    def visit_int_op(self, op: IntOp) -> None:
        if False:
            for i in range(10):
                print('nop')
        dest = self.reg(op)
        lhs = self.reg(op.lhs)
        rhs = self.reg(op.rhs)
        if op.op == IntOp.RIGHT_SHIFT:
            lhs = self.emit_signed_int_cast(op.lhs.type) + lhs
            rhs = self.emit_signed_int_cast(op.rhs.type) + rhs
        self.emit_line(f'{dest} = {lhs} {op.op_str[op.op]} {rhs};')

    def visit_comparison_op(self, op: ComparisonOp) -> None:
        if False:
            for i in range(10):
                print('nop')
        dest = self.reg(op)
        lhs = self.reg(op.lhs)
        rhs = self.reg(op.rhs)
        lhs_cast = ''
        rhs_cast = ''
        if op.op in (ComparisonOp.SLT, ComparisonOp.SGT, ComparisonOp.SLE, ComparisonOp.SGE):
            lhs_cast = self.emit_signed_int_cast(op.lhs.type)
            rhs_cast = self.emit_signed_int_cast(op.rhs.type)
        elif op.op in (ComparisonOp.ULT, ComparisonOp.UGT, ComparisonOp.ULE, ComparisonOp.UGE):
            lhs_cast = self.emit_unsigned_int_cast(op.lhs.type)
            rhs_cast = self.emit_unsigned_int_cast(op.rhs.type)
        elif isinstance(op.lhs, Integer) and op.lhs.value < 0:
            rhs_cast = self.emit_signed_int_cast(op.rhs.type)
        elif isinstance(op.rhs, Integer) and op.rhs.value < 0:
            lhs_cast = self.emit_signed_int_cast(op.lhs.type)
        self.emit_line(f'{dest} = {lhs_cast}{lhs} {op.op_str[op.op]} {rhs_cast}{rhs};')

    def visit_float_op(self, op: FloatOp) -> None:
        if False:
            i = 10
            return i + 15
        dest = self.reg(op)
        lhs = self.reg(op.lhs)
        rhs = self.reg(op.rhs)
        if op.op != FloatOp.MOD:
            self.emit_line(f'{dest} = {lhs} {op.op_str[op.op]} {rhs};')
        else:
            self.emit_line(f'{dest} = fmod({lhs}, {rhs});')

    def visit_float_neg(self, op: FloatNeg) -> None:
        if False:
            print('Hello World!')
        dest = self.reg(op)
        src = self.reg(op.src)
        self.emit_line(f'{dest} = -{src};')

    def visit_float_comparison_op(self, op: FloatComparisonOp) -> None:
        if False:
            return 10
        dest = self.reg(op)
        lhs = self.reg(op.lhs)
        rhs = self.reg(op.rhs)
        self.emit_line(f'{dest} = {lhs} {op.op_str[op.op]} {rhs};')

    def visit_load_mem(self, op: LoadMem) -> None:
        if False:
            print('Hello World!')
        dest = self.reg(op)
        src = self.reg(op.src)
        type = self.ctype(op.type)
        self.emit_line(f'{dest} = *({type} *){src};')

    def visit_set_mem(self, op: SetMem) -> None:
        if False:
            print('Hello World!')
        dest = self.reg(op.dest)
        src = self.reg(op.src)
        dest_type = self.ctype(op.dest_type)
        if dest != src:
            self.emit_line(f'*({dest_type} *){dest} = {src};')

    def visit_get_element_ptr(self, op: GetElementPtr) -> None:
        if False:
            for i in range(10):
                print('nop')
        dest = self.reg(op)
        src = self.reg(op.src)
        assert isinstance(op.src_type, RStruct)
        assert op.field in op.src_type.names, 'Invalid field name.'
        self.emit_line('{} = ({})&(({} *){})->{};'.format(dest, op.type._ctype, op.src_type.name, src, op.field))

    def visit_load_address(self, op: LoadAddress) -> None:
        if False:
            i = 10
            return i + 15
        typ = op.type
        dest = self.reg(op)
        if isinstance(op.src, Register):
            src = self.reg(op.src)
        elif isinstance(op.src, LoadStatic):
            prefix = self.PREFIX_MAP[op.src.namespace]
            src = self.emitter.static_name(op.src.identifier, op.src.module_name, prefix)
        else:
            src = op.src
        self.emit_line(f'{dest} = ({typ._ctype})&{src};')

    def visit_keep_alive(self, op: KeepAlive) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_unborrow(self, op: Unborrow) -> None:
        if False:
            print('Hello World!')
        dest = self.reg(op)
        src = self.reg(op.src)
        self.emit_line(f'{dest} = {src};')

    def label(self, label: BasicBlock) -> str:
        if False:
            while True:
                i = 10
        return self.emitter.label(label)

    def reg(self, reg: Value) -> str:
        if False:
            while True:
                i = 10
        if isinstance(reg, Integer):
            val = reg.value
            if val == 0 and is_pointer_rprimitive(reg.type):
                return 'NULL'
            s = str(val)
            if val >= 1 << 31:
                if val >= 1 << 63:
                    s += 'ULL'
                else:
                    s += 'LL'
            elif val == -(1 << 63):
                s = '(-9223372036854775807LL - 1)'
            elif val <= -(1 << 31):
                s += 'LL'
            return s
        elif isinstance(reg, Float):
            r = repr(reg.value)
            if r == 'inf':
                return 'INFINITY'
            elif r == '-inf':
                return '-INFINITY'
            elif r == 'nan':
                return 'NAN'
            return r
        else:
            return self.emitter.reg(reg)

    def ctype(self, rtype: RType) -> str:
        if False:
            return 10
        return self.emitter.ctype(rtype)

    def c_error_value(self, rtype: RType) -> str:
        if False:
            return 10
        return self.emitter.c_error_value(rtype)

    def c_undefined_value(self, rtype: RType) -> str:
        if False:
            print('Hello World!')
        return self.emitter.c_undefined_value(rtype)

    def emit_line(self, line: str, *, ann: object=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.emitter.emit_line(line, ann=ann)

    def emit_lines(self, *lines: str) -> None:
        if False:
            while True:
                i = 10
        self.emitter.emit_lines(*lines)

    def emit_inc_ref(self, dest: str, rtype: RType) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.emitter.emit_inc_ref(dest, rtype, rare=self.rare)

    def emit_dec_ref(self, dest: str, rtype: RType, is_xdec: bool) -> None:
        if False:
            return 10
        self.emitter.emit_dec_ref(dest, rtype, is_xdec=is_xdec, rare=self.rare)

    def emit_declaration(self, line: str) -> None:
        if False:
            return 10
        self.declarations.emit_line(line)

    def emit_traceback(self, op: Branch) -> None:
        if False:
            for i in range(10):
                print('nop')
        if op.traceback_entry is not None:
            self.emitter.emit_traceback(self.source_path, self.module_name, op.traceback_entry)

    def emit_attribute_error(self, op: Branch, class_name: str, attr: str) -> None:
        if False:
            i = 10
            return i + 15
        assert op.traceback_entry is not None
        globals_static = self.emitter.static_name('globals', self.module_name)
        self.emit_line('CPy_AttributeError("%s", "%s", "%s", "%s", %d, %s);' % (self.source_path.replace('\\', '\\\\'), op.traceback_entry[0], class_name, attr, op.traceback_entry[1], globals_static))
        if DEBUG_ERRORS:
            self.emit_line('assert(PyErr_Occurred() != NULL && "failure w/o err!");')

    def emit_signed_int_cast(self, type: RType) -> str:
        if False:
            return 10
        if is_tagged(type):
            return '(Py_ssize_t)'
        else:
            return ''

    def emit_unsigned_int_cast(self, type: RType) -> str:
        if False:
            return 10
        if is_int32_rprimitive(type):
            return '(uint32_t)'
        elif is_int64_rprimitive(type):
            return '(uint64_t)'
        else:
            return ''