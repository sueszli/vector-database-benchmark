"""Utilities for emitting C code."""
from __future__ import annotations
import pprint
import sys
import textwrap
from typing import Callable, Final
from mypyc.codegen.literals import Literals
from mypyc.common import ATTR_PREFIX, BITMAP_BITS, FAST_ISINSTANCE_MAX_SUBCLASSES, NATIVE_PREFIX, REG_PREFIX, STATIC_PREFIX, TYPE_PREFIX, use_vectorcall
from mypyc.ir.class_ir import ClassIR, all_concrete_classes
from mypyc.ir.func_ir import FuncDecl
from mypyc.ir.ops import BasicBlock, Value
from mypyc.ir.rtypes import RInstance, RPrimitive, RTuple, RType, RUnion, int_rprimitive, is_bit_rprimitive, is_bool_rprimitive, is_bytes_rprimitive, is_dict_rprimitive, is_fixed_width_rtype, is_float_rprimitive, is_int16_rprimitive, is_int32_rprimitive, is_int64_rprimitive, is_int_rprimitive, is_list_rprimitive, is_none_rprimitive, is_object_rprimitive, is_optional_type, is_range_rprimitive, is_set_rprimitive, is_short_int_rprimitive, is_str_rprimitive, is_tuple_rprimitive, is_uint8_rprimitive, object_rprimitive, optional_value_type
from mypyc.namegen import NameGenerator, exported_name
from mypyc.sametype import is_same_type
DEBUG_ERRORS: Final = False

class HeaderDeclaration:
    """A representation of a declaration in C.

    This is used to generate declarations in header files and
    (optionally) definitions in source files.

    Attributes:
      decl: C source code for the declaration.
      defn: Optionally, C source code for a definition.
      dependencies: The names of any objects that must be declared prior.
      is_type: Whether the declaration is of a C type. (C types will be declared in
               external header files and not marked 'extern'.)
      needs_export: Whether the declared object needs to be exported to
                    other modules in the linking table.
    """

    def __init__(self, decl: str | list[str], defn: list[str] | None=None, *, dependencies: set[str] | None=None, is_type: bool=False, needs_export: bool=False) -> None:
        if False:
            print('Hello World!')
        self.decl = [decl] if isinstance(decl, str) else decl
        self.defn = defn
        self.dependencies = dependencies or set()
        self.is_type = is_type
        self.needs_export = needs_export

class EmitterContext:
    """Shared emitter state for a compilation group."""

    def __init__(self, names: NameGenerator, group_name: str | None=None, group_map: dict[str, str | None] | None=None) -> None:
        if False:
            print('Hello World!')
        'Setup shared emitter state.\n\n        Args:\n            names: The name generator to use\n            group_map: Map from module names to group name\n            group_name: Current group name\n        '
        self.temp_counter = 0
        self.names = names
        self.group_name = group_name
        self.group_map = group_map or {}
        self.group_deps: set[str] = set()
        self.declarations: dict[str, HeaderDeclaration] = {}
        self.literals = Literals()

class ErrorHandler:
    """Describes handling errors in unbox/cast operations."""

class AssignHandler(ErrorHandler):
    """Assign an error value on error."""

class GotoHandler(ErrorHandler):
    """Goto label on error."""

    def __init__(self, label: str) -> None:
        if False:
            i = 10
            return i + 15
        self.label = label

class TracebackAndGotoHandler(ErrorHandler):
    """Add traceback item and goto label on error."""

    def __init__(self, label: str, source_path: str, module_name: str, traceback_entry: tuple[str, int]) -> None:
        if False:
            while True:
                i = 10
        self.label = label
        self.source_path = source_path
        self.module_name = module_name
        self.traceback_entry = traceback_entry

class ReturnHandler(ErrorHandler):
    """Return a constant value on error."""

    def __init__(self, value: str) -> None:
        if False:
            i = 10
            return i + 15
        self.value = value

class Emitter:
    """Helper for C code generation."""

    def __init__(self, context: EmitterContext, value_names: dict[Value, str] | None=None, capi_version: tuple[int, int] | None=None) -> None:
        if False:
            return 10
        self.context = context
        self.capi_version = capi_version or sys.version_info[:2]
        self.names = context.names
        self.value_names = value_names or {}
        self.fragments: list[str] = []
        self._indent = 0

    def indent(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._indent += 4

    def dedent(self) -> None:
        if False:
            print('Hello World!')
        self._indent -= 4
        assert self._indent >= 0

    def label(self, label: BasicBlock) -> str:
        if False:
            i = 10
            return i + 15
        return 'CPyL%s' % label.label

    def reg(self, reg: Value) -> str:
        if False:
            for i in range(10):
                print('nop')
        return REG_PREFIX + self.value_names[reg]

    def attr(self, name: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ATTR_PREFIX + name

    def object_annotation(self, obj: object, line: str) -> str:
        if False:
            while True:
                i = 10
        "Build a C comment with an object's string represention.\n\n        If the comment exceeds the line length limit, it's wrapped into a\n        multiline string (with the extra lines indented to be aligned with\n        the first line's comment).\n\n        If it contains illegal characters, an empty string is returned."
        line_width = self._indent + len(line)
        formatted = pprint.pformat(obj, compact=True, width=max(90 - line_width, 20))
        if any((x in formatted for x in ('/*', '*/', '\x00'))):
            return ''
        if '\n' in formatted:
            (first_line, rest) = formatted.split('\n', maxsplit=1)
            comment_continued = textwrap.indent(rest, (line_width + 3) * ' ')
            return f' /* {first_line}\n{comment_continued} */'
        else:
            return f' /* {formatted} */'

    def emit_line(self, line: str='', *, ann: object=None) -> None:
        if False:
            print('Hello World!')
        if line.startswith('}'):
            self.dedent()
        comment = self.object_annotation(ann, line) if ann is not None else ''
        self.fragments.append(self._indent * ' ' + line + comment + '\n')
        if line.endswith('{'):
            self.indent()

    def emit_lines(self, *lines: str) -> None:
        if False:
            i = 10
            return i + 15
        for line in lines:
            self.emit_line(line)

    def emit_label(self, label: BasicBlock | str) -> None:
        if False:
            return 10
        if isinstance(label, str):
            text = label
        else:
            if label.label == 0 or not label.referenced:
                return
            text = self.label(label)
        self.fragments.append(f'{text}: ;\n')

    def emit_from_emitter(self, emitter: Emitter) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.fragments.extend(emitter.fragments)

    def emit_printf(self, fmt: str, *args: str) -> None:
        if False:
            while True:
                i = 10
        fmt = fmt.replace('\n', '\\n')
        self.emit_line('printf(%s);' % ', '.join(['"%s"' % fmt] + list(args)))
        self.emit_line('fflush(stdout);')

    def temp_name(self) -> str:
        if False:
            return 10
        self.context.temp_counter += 1
        return '__tmp%d' % self.context.temp_counter

    def new_label(self) -> str:
        if False:
            print('Hello World!')
        self.context.temp_counter += 1
        return '__LL%d' % self.context.temp_counter

    def get_module_group_prefix(self, module_name: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Get the group prefix for a module (relative to the current group).\n\n        The prefix should be prepended to the object name whenever\n        accessing an object from this module.\n\n        If the module lives is in the current compilation group, there is\n        no prefix.  But if it lives in a different group (and hence a separate\n        extension module), we need to access objects from it indirectly via an\n        export table.\n\n        For example, for code in group `a` to call a function `bar` in group `b`,\n        it would need to do `exports_b.CPyDef_bar(...)`, while code that is\n        also in group `b` can simply do `CPyDef_bar(...)`.\n\n        Thus the prefix for a module in group `b` is 'exports_b.' if the current\n        group is *not* b and just '' if it is.\n        "
        groups = self.context.group_map
        target_group_name = groups.get(module_name)
        if target_group_name and target_group_name != self.context.group_name:
            self.context.group_deps.add(target_group_name)
            return f'exports_{exported_name(target_group_name)}.'
        else:
            return ''

    def get_group_prefix(self, obj: ClassIR | FuncDecl) -> str:
        if False:
            return 10
        'Get the group prefix for an object.'
        return self.get_module_group_prefix(obj.module_name)

    def static_name(self, id: str, module: str | None, prefix: str=STATIC_PREFIX) -> str:
        if False:
            return 10
        'Create name of a C static variable.\n\n        These are used for literals and imported modules, among other\n        things.\n\n        The caller should ensure that the (id, module) pair cannot\n        overlap with other calls to this method within a compilation\n        group.\n        '
        lib_prefix = '' if not module else self.get_module_group_prefix(module)
        star_maybe = '*' if lib_prefix else ''
        suffix = self.names.private_name(module or '', id)
        return f'{star_maybe}{lib_prefix}{prefix}{suffix}'

    def type_struct_name(self, cl: ClassIR) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.static_name(cl.name, cl.module_name, prefix=TYPE_PREFIX)

    def ctype(self, rtype: RType) -> str:
        if False:
            for i in range(10):
                print('nop')
        return rtype._ctype

    def ctype_spaced(self, rtype: RType) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Adds a space after ctype for non-pointers.'
        ctype = self.ctype(rtype)
        if ctype[-1] == '*':
            return ctype
        else:
            return ctype + ' '

    def c_undefined_value(self, rtype: RType) -> str:
        if False:
            i = 10
            return i + 15
        if not rtype.is_unboxed:
            return 'NULL'
        elif isinstance(rtype, RPrimitive):
            return rtype.c_undefined
        elif isinstance(rtype, RTuple):
            return self.tuple_undefined_value(rtype)
        assert False, rtype

    def c_error_value(self, rtype: RType) -> str:
        if False:
            i = 10
            return i + 15
        return self.c_undefined_value(rtype)

    def native_function_name(self, fn: FuncDecl) -> str:
        if False:
            return 10
        return f'{NATIVE_PREFIX}{fn.cname(self.names)}'

    def tuple_c_declaration(self, rtuple: RTuple) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        result = [f'#ifndef MYPYC_DECLARED_{rtuple.struct_name}', f'#define MYPYC_DECLARED_{rtuple.struct_name}', f'typedef struct {rtuple.struct_name} {{']
        if len(rtuple.types) == 0:
            result.append('int empty_struct_error_flag;')
        else:
            i = 0
            for typ in rtuple.types:
                result.append(f'{self.ctype_spaced(typ)}f{i};')
                i += 1
        result.append(f'}} {rtuple.struct_name};')
        result.append('#endif')
        result.append('')
        return result

    def bitmap_field(self, index: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return C field name used for attribute bitmap.'
        n = index // BITMAP_BITS
        if n == 0:
            return 'bitmap'
        return f'bitmap{n + 1}'

    def attr_bitmap_expr(self, obj: str, cl: ClassIR, index: int) -> str:
        if False:
            return 10
        'Return reference to the attribute definedness bitmap.'
        cast = f'({cl.struct_name(self.names)} *)'
        attr = self.bitmap_field(index)
        return f'({cast}{obj})->{attr}'

    def emit_attr_bitmap_set(self, value: str, obj: str, rtype: RType, cl: ClassIR, attr: str) -> None:
        if False:
            print('Hello World!')
        "Mark an attribute as defined in the attribute bitmap.\n\n        Assumes that the attribute is tracked in the bitmap (only some attributes\n        use the bitmap). If 'value' is not equal to the error value, do nothing.\n        "
        self._emit_attr_bitmap_update(value, obj, rtype, cl, attr, clear=False)

    def emit_attr_bitmap_clear(self, obj: str, rtype: RType, cl: ClassIR, attr: str) -> None:
        if False:
            return 10
        'Mark an attribute as undefined in the attribute bitmap.\n\n        Unlike emit_attr_bitmap_set, clear unconditionally.\n        '
        self._emit_attr_bitmap_update('', obj, rtype, cl, attr, clear=True)

    def _emit_attr_bitmap_update(self, value: str, obj: str, rtype: RType, cl: ClassIR, attr: str, clear: bool) -> None:
        if False:
            return 10
        if value:
            check = self.error_value_check(rtype, value, '==')
            self.emit_line(f'if (unlikely({check})) {{')
        index = cl.bitmap_attrs.index(attr)
        mask = 1 << (index & BITMAP_BITS - 1)
        bitmap = self.attr_bitmap_expr(obj, cl, index)
        if clear:
            self.emit_line(f'{bitmap} &= ~{mask};')
        else:
            self.emit_line(f'{bitmap} |= {mask};')
        if value:
            self.emit_line('}')

    def use_vectorcall(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return use_vectorcall(self.capi_version)

    def emit_undefined_attr_check(self, rtype: RType, attr_expr: str, compare: str, obj: str, attr: str, cl: ClassIR, *, unlikely: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        check = self.error_value_check(rtype, attr_expr, compare)
        if unlikely:
            check = f'unlikely({check})'
        if rtype.error_overlap:
            index = cl.bitmap_attrs.index(attr)
            bit = 1 << (index & BITMAP_BITS - 1)
            attr = self.bitmap_field(index)
            obj_expr = f'({cl.struct_name(self.names)} *){obj}'
            check = f'{check} && !(({obj_expr})->{attr} & {bit})'
        self.emit_line(f'if ({check}) {{')

    def error_value_check(self, rtype: RType, value: str, compare: str) -> str:
        if False:
            return 10
        if isinstance(rtype, RTuple):
            return self.tuple_undefined_check_cond(rtype, value, self.c_error_value, compare, check_exception=False)
        else:
            return f'{value} {compare} {self.c_error_value(rtype)}'

    def tuple_undefined_check_cond(self, rtuple: RTuple, tuple_expr_in_c: str, c_type_compare_val: Callable[[RType], str], compare: str, *, check_exception: bool=True) -> str:
        if False:
            i = 10
            return i + 15
        if len(rtuple.types) == 0:
            return '{}.empty_struct_error_flag {} {}'.format(tuple_expr_in_c, compare, c_type_compare_val(int_rprimitive))
        if rtuple.error_overlap:
            i = 0
            item_type = rtuple.types[0]
        else:
            for (i, typ) in enumerate(rtuple.types):
                if not typ.error_overlap:
                    item_type = rtuple.types[i]
                    break
            else:
                assert False, 'not expecting tuple with error overlap'
        if isinstance(item_type, RTuple):
            return self.tuple_undefined_check_cond(item_type, tuple_expr_in_c + f'.f{i}', c_type_compare_val, compare)
        else:
            check = f'{tuple_expr_in_c}.f{i} {compare} {c_type_compare_val(item_type)}'
            if rtuple.error_overlap and check_exception:
                check += ' && PyErr_Occurred()'
            return check

    def tuple_undefined_value(self, rtuple: RTuple) -> str:
        if False:
            while True:
                i = 10
        'Undefined tuple value suitable in an expression.'
        return f'({rtuple.struct_name}) {self.c_initializer_undefined_value(rtuple)}'

    def c_initializer_undefined_value(self, rtype: RType) -> str:
        if False:
            i = 10
            return i + 15
        'Undefined value represented in a form suitable for variable initialization.'
        if isinstance(rtype, RTuple):
            if not rtype.types:
                return f'{{ {int_rprimitive.c_undefined} }}'
            items = ', '.join([self.c_initializer_undefined_value(t) for t in rtype.types])
            return f'{{ {items} }}'
        else:
            return self.c_undefined_value(rtype)

    def declare_tuple_struct(self, tuple_type: RTuple) -> None:
        if False:
            while True:
                i = 10
        if tuple_type.struct_name not in self.context.declarations:
            dependencies = set()
            for typ in tuple_type.types:
                if isinstance(typ, RTuple):
                    dependencies.add(typ.struct_name)
            self.context.declarations[tuple_type.struct_name] = HeaderDeclaration(self.tuple_c_declaration(tuple_type), dependencies=dependencies, is_type=True)

    def emit_inc_ref(self, dest: str, rtype: RType, *, rare: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Increment reference count of C expression `dest`.\n\n        For composite unboxed structures (e.g. tuples) recursively\n        increment reference counts for each component.\n\n        If rare is True, optimize for code size and compilation speed.\n        '
        if is_int_rprimitive(rtype):
            if rare:
                self.emit_line('CPyTagged_IncRef(%s);' % dest)
            else:
                self.emit_line('CPyTagged_INCREF(%s);' % dest)
        elif isinstance(rtype, RTuple):
            for (i, item_type) in enumerate(rtype.types):
                self.emit_inc_ref(f'{dest}.f{i}', item_type)
        elif not rtype.is_unboxed:
            self.emit_line('CPy_INCREF(%s);' % dest)

    def emit_dec_ref(self, dest: str, rtype: RType, *, is_xdec: bool=False, rare: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Decrement reference count of C expression `dest`.\n\n        For composite unboxed structures (e.g. tuples) recursively\n        decrement reference counts for each component.\n\n        If rare is True, optimize for code size and compilation speed.\n        '
        x = 'X' if is_xdec else ''
        if is_int_rprimitive(rtype):
            if rare:
                self.emit_line(f'CPyTagged_{x}DecRef({dest});')
            else:
                self.emit_line(f'CPyTagged_{x}DECREF({dest});')
        elif isinstance(rtype, RTuple):
            for (i, item_type) in enumerate(rtype.types):
                self.emit_dec_ref(f'{dest}.f{i}', item_type, is_xdec=is_xdec, rare=rare)
        elif not rtype.is_unboxed:
            if rare:
                self.emit_line(f'CPy_{x}DecRef({dest});')
            else:
                self.emit_line(f'CPy_{x}DECREF({dest});')

    def pretty_name(self, typ: RType) -> str:
        if False:
            return 10
        value_type = optional_value_type(typ)
        if value_type is not None:
            return '%s or None' % self.pretty_name(value_type)
        return str(typ)

    def emit_cast(self, src: str, dest: str, typ: RType, *, declare_dest: bool=False, error: ErrorHandler | None=None, raise_exception: bool=True, optional: bool=False, src_type: RType | None=None, likely: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Emit code for casting a value of given type.\n\n        Somewhat strangely, this supports unboxed types but only\n        operates on boxed versions.  This is necessary to properly\n        handle types such as Optional[int] in compatibility glue.\n\n        By default, assign NULL (error value) to dest if the value has\n        an incompatible type and raise TypeError. These can be customized\n        using 'error' and 'raise_exception'.\n\n        Always copy/steal the reference in 'src'.\n\n        Args:\n            src: Name of source C variable\n            dest: Name of target C variable\n            typ: Type of value\n            declare_dest: If True, also declare the variable 'dest'\n            error: What happens on error\n            raise_exception: If True, also raise TypeError on failure\n            likely: If the cast is likely to succeed (can be False for unions)\n        "
        error = error or AssignHandler()
        if src_type and is_optional_type(src_type) and (not is_object_rprimitive(typ)):
            value_type = optional_value_type(src_type)
            assert value_type is not None
            if is_same_type(value_type, typ):
                if declare_dest:
                    self.emit_line(f'PyObject *{dest};')
                check = '({} != Py_None)'
                if likely:
                    check = f'(likely{check})'
                self.emit_arg_check(src, dest, typ, check.format(src), optional)
                self.emit_lines(f'    {dest} = {src};', 'else {')
                self.emit_cast_error_handler(error, src, dest, typ, raise_exception)
                self.emit_line('}')
                return
        if is_list_rprimitive(typ) or is_dict_rprimitive(typ) or is_set_rprimitive(typ) or is_str_rprimitive(typ) or is_range_rprimitive(typ) or is_float_rprimitive(typ) or is_int_rprimitive(typ) or is_bool_rprimitive(typ) or is_bit_rprimitive(typ) or is_fixed_width_rtype(typ):
            if declare_dest:
                self.emit_line(f'PyObject *{dest};')
            if is_list_rprimitive(typ):
                prefix = 'PyList'
            elif is_dict_rprimitive(typ):
                prefix = 'PyDict'
            elif is_set_rprimitive(typ):
                prefix = 'PySet'
            elif is_str_rprimitive(typ):
                prefix = 'PyUnicode'
            elif is_range_rprimitive(typ):
                prefix = 'PyRange'
            elif is_float_rprimitive(typ):
                prefix = 'CPyFloat'
            elif is_int_rprimitive(typ) or is_fixed_width_rtype(typ):
                prefix = 'PyLong'
            elif is_bool_rprimitive(typ) or is_bit_rprimitive(typ):
                prefix = 'PyBool'
            else:
                assert False, f'unexpected primitive type: {typ}'
            check = '({}_Check({}))'
            if likely:
                check = f'(likely{check})'
            self.emit_arg_check(src, dest, typ, check.format(prefix, src), optional)
            self.emit_lines(f'    {dest} = {src};', 'else {')
            self.emit_cast_error_handler(error, src, dest, typ, raise_exception)
            self.emit_line('}')
        elif is_bytes_rprimitive(typ):
            if declare_dest:
                self.emit_line(f'PyObject *{dest};')
            check = '(PyBytes_Check({}) || PyByteArray_Check({}))'
            if likely:
                check = f'(likely{check})'
            self.emit_arg_check(src, dest, typ, check.format(src, src), optional)
            self.emit_lines(f'    {dest} = {src};', 'else {')
            self.emit_cast_error_handler(error, src, dest, typ, raise_exception)
            self.emit_line('}')
        elif is_tuple_rprimitive(typ):
            if declare_dest:
                self.emit_line(f'{self.ctype(typ)} {dest};')
            check = '(PyTuple_Check({}))'
            if likely:
                check = f'(likely{check})'
            self.emit_arg_check(src, dest, typ, check.format(src), optional)
            self.emit_lines(f'    {dest} = {src};', 'else {')
            self.emit_cast_error_handler(error, src, dest, typ, raise_exception)
            self.emit_line('}')
        elif isinstance(typ, RInstance):
            if declare_dest:
                self.emit_line(f'PyObject *{dest};')
            concrete = all_concrete_classes(typ.class_ir)
            if not concrete or len(concrete) > FAST_ISINSTANCE_MAX_SUBCLASSES + 1:
                check = '(PyObject_TypeCheck({}, {}))'.format(src, self.type_struct_name(typ.class_ir))
            else:
                full_str = '(Py_TYPE({src}) == {targets[0]})'
                for i in range(1, len(concrete)):
                    full_str += ' || (Py_TYPE({src}) == {targets[%d]})' % i
                if len(concrete) > 1:
                    full_str = '(%s)' % full_str
                check = full_str.format(src=src, targets=[self.type_struct_name(ir) for ir in concrete])
            if likely:
                check = f'(likely{check})'
            self.emit_arg_check(src, dest, typ, check, optional)
            self.emit_lines(f'    {dest} = {src};', 'else {')
            self.emit_cast_error_handler(error, src, dest, typ, raise_exception)
            self.emit_line('}')
        elif is_none_rprimitive(typ):
            if declare_dest:
                self.emit_line(f'PyObject *{dest};')
            check = '({} == Py_None)'
            if likely:
                check = f'(likely{check})'
            self.emit_arg_check(src, dest, typ, check.format(src), optional)
            self.emit_lines(f'    {dest} = {src};', 'else {')
            self.emit_cast_error_handler(error, src, dest, typ, raise_exception)
            self.emit_line('}')
        elif is_object_rprimitive(typ):
            if declare_dest:
                self.emit_line(f'PyObject *{dest};')
            self.emit_arg_check(src, dest, typ, '', optional)
            self.emit_line(f'{dest} = {src};')
            if optional:
                self.emit_line('}')
        elif isinstance(typ, RUnion):
            self.emit_union_cast(src, dest, typ, declare_dest, error, optional, src_type, raise_exception)
        elif isinstance(typ, RTuple):
            assert not optional
            self.emit_tuple_cast(src, dest, typ, declare_dest, error, src_type)
        else:
            assert False, 'Cast not implemented: %s' % typ

    def emit_cast_error_handler(self, error: ErrorHandler, src: str, dest: str, typ: RType, raise_exception: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        if raise_exception:
            if isinstance(error, TracebackAndGotoHandler):
                self.emit_type_error_traceback(error.source_path, error.module_name, error.traceback_entry, typ=typ, src=src)
                self.emit_line('goto %s;' % error.label)
                return
            self.emit_line(f'CPy_TypeError("{self.pretty_name(typ)}", {src}); ')
        if isinstance(error, AssignHandler):
            self.emit_line('%s = NULL;' % dest)
        elif isinstance(error, GotoHandler):
            self.emit_line('goto %s;' % error.label)
        elif isinstance(error, TracebackAndGotoHandler):
            self.emit_line('%s = NULL;' % dest)
            self.emit_traceback(error.source_path, error.module_name, error.traceback_entry)
            self.emit_line('goto %s;' % error.label)
        else:
            assert isinstance(error, ReturnHandler)
            self.emit_line('return %s;' % error.value)

    def emit_union_cast(self, src: str, dest: str, typ: RUnion, declare_dest: bool, error: ErrorHandler, optional: bool, src_type: RType | None, raise_exception: bool) -> None:
        if False:
            print('Hello World!')
        'Emit cast to a union type.\n\n        The arguments are similar to emit_cast.\n        '
        if declare_dest:
            self.emit_line(f'PyObject *{dest};')
        good_label = self.new_label()
        if optional:
            self.emit_line(f'if ({src} == NULL) {{')
            self.emit_line(f'{dest} = {self.c_error_value(typ)};')
            self.emit_line(f'goto {good_label};')
            self.emit_line('}')
        for item in typ.items:
            self.emit_cast(src, dest, item, declare_dest=False, raise_exception=False, optional=False, likely=False)
            self.emit_line(f'if ({dest} != NULL) goto {good_label};')
        self.emit_cast_error_handler(error, src, dest, typ, raise_exception)
        self.emit_label(good_label)

    def emit_tuple_cast(self, src: str, dest: str, typ: RTuple, declare_dest: bool, error: ErrorHandler, src_type: RType | None) -> None:
        if False:
            print('Hello World!')
        'Emit cast to a tuple type.\n\n        The arguments are similar to emit_cast.\n        '
        if declare_dest:
            self.emit_line(f'PyObject *{dest};')
        out_label = self.new_label()
        self.emit_lines('if (unlikely(!(PyTuple_Check({r}) && PyTuple_GET_SIZE({r}) == {size}))) {{'.format(r=src, size=len(typ.types)), f'{dest} = NULL;', f'goto {out_label};', '}')
        for (i, item) in enumerate(typ.types):
            self.emit_cast(f'PyTuple_GET_ITEM({src}, {i})', dest, item, declare_dest=False, raise_exception=False, optional=False)
            self.emit_line(f'if ({dest} == NULL) goto {out_label};')
        self.emit_line(f'{dest} = {src};')
        self.emit_label(out_label)

    def emit_arg_check(self, src: str, dest: str, typ: RType, check: str, optional: bool) -> None:
        if False:
            while True:
                i = 10
        if optional:
            self.emit_line(f'if ({src} == NULL) {{')
            self.emit_line(f'{dest} = {self.c_error_value(typ)};')
        if check != '':
            self.emit_line('{}if {}'.format('} else ' if optional else '', check))
        elif optional:
            self.emit_line('else {')

    def emit_unbox(self, src: str, dest: str, typ: RType, *, declare_dest: bool=False, error: ErrorHandler | None=None, raise_exception: bool=True, optional: bool=False, borrow: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        "Emit code for unboxing a value of given type (from PyObject *).\n\n        By default, assign error value to dest if the value has an\n        incompatible type and raise TypeError. These can be customized\n        using 'error' and 'raise_exception'.\n\n        Generate a new reference unless 'borrow' is True.\n\n        Args:\n            src: Name of source C variable\n            dest: Name of target C variable\n            typ: Type of value\n            declare_dest: If True, also declare the variable 'dest'\n            error: What happens on error\n            raise_exception: If True, also raise TypeError on failure\n            borrow: If True, create a borrowed reference\n\n        "
        error = error or AssignHandler()
        if isinstance(error, AssignHandler):
            failure = f'{dest} = {self.c_error_value(typ)};'
        elif isinstance(error, GotoHandler):
            failure = 'goto %s;' % error.label
        else:
            assert isinstance(error, ReturnHandler)
            failure = 'return %s;' % error.value
        if raise_exception:
            raise_exc = f'CPy_TypeError("{self.pretty_name(typ)}", {src}); '
            failure = raise_exc + failure
        if is_int_rprimitive(typ) or is_short_int_rprimitive(typ):
            if declare_dest:
                self.emit_line(f'CPyTagged {dest};')
            self.emit_arg_check(src, dest, typ, f'(likely(PyLong_Check({src})))', optional)
            if borrow:
                self.emit_line(f'    {dest} = CPyTagged_BorrowFromObject({src});')
            else:
                self.emit_line(f'    {dest} = CPyTagged_FromObject({src});')
            self.emit_line('else {')
            self.emit_line(failure)
            self.emit_line('}')
        elif is_bool_rprimitive(typ) or is_bit_rprimitive(typ):
            if declare_dest:
                self.emit_line(f'char {dest};')
            self.emit_arg_check(src, dest, typ, f'(unlikely(!PyBool_Check({src}))) {{', optional)
            self.emit_line(failure)
            self.emit_line('} else')
            conversion = f'{src} == Py_True'
            self.emit_line(f'    {dest} = {conversion};')
        elif is_none_rprimitive(typ):
            if declare_dest:
                self.emit_line(f'char {dest};')
            self.emit_arg_check(src, dest, typ, f'(unlikely({src} != Py_None)) {{', optional)
            self.emit_line(failure)
            self.emit_line('} else')
            self.emit_line(f'    {dest} = 1;')
        elif is_int64_rprimitive(typ):
            assert not optional
            if declare_dest:
                self.emit_line(f'int64_t {dest};')
            self.emit_line(f'{dest} = CPyLong_AsInt64({src});')
            if not isinstance(error, AssignHandler):
                self.emit_unbox_failure_with_overlapping_error_value(dest, typ, failure)
        elif is_int32_rprimitive(typ):
            assert not optional
            if declare_dest:
                self.emit_line(f'int32_t {dest};')
            self.emit_line(f'{dest} = CPyLong_AsInt32({src});')
            if not isinstance(error, AssignHandler):
                self.emit_unbox_failure_with_overlapping_error_value(dest, typ, failure)
        elif is_int16_rprimitive(typ):
            assert not optional
            if declare_dest:
                self.emit_line(f'int16_t {dest};')
            self.emit_line(f'{dest} = CPyLong_AsInt16({src});')
            if not isinstance(error, AssignHandler):
                self.emit_unbox_failure_with_overlapping_error_value(dest, typ, failure)
        elif is_uint8_rprimitive(typ):
            assert not optional
            if declare_dest:
                self.emit_line(f'uint8_t {dest};')
            self.emit_line(f'{dest} = CPyLong_AsUInt8({src});')
            if not isinstance(error, AssignHandler):
                self.emit_unbox_failure_with_overlapping_error_value(dest, typ, failure)
        elif is_float_rprimitive(typ):
            assert not optional
            if declare_dest:
                self.emit_line(f'double {dest};')
            self.emit_line(f'{dest} = PyFloat_AsDouble({src});')
            self.emit_lines(f'if ({dest} == -1.0 && PyErr_Occurred()) {{', failure, '}')
        elif isinstance(typ, RTuple):
            self.declare_tuple_struct(typ)
            if declare_dest:
                self.emit_line(f'{self.ctype(typ)} {dest};')
            if optional:
                self.emit_line(f'if ({src} == NULL) {{')
                self.emit_line(f'{dest} = {self.c_error_value(typ)};')
                self.emit_line('} else {')
            cast_temp = self.temp_name()
            self.emit_tuple_cast(src, cast_temp, typ, declare_dest=True, error=error, src_type=None)
            self.emit_line(f'if (unlikely({cast_temp} == NULL)) {{')
            self.emit_line(failure)
            self.emit_line('} else {')
            if not typ.types:
                self.emit_line(f'{dest}.empty_struct_error_flag = 0;')
            for (i, item_type) in enumerate(typ.types):
                temp = self.temp_name()
                self.emit_line(f'PyObject *{temp} = PyTuple_GET_ITEM({src}, {i});')
                temp2 = self.temp_name()
                if item_type.is_unboxed:
                    self.emit_unbox(temp, temp2, item_type, raise_exception=raise_exception, error=error, declare_dest=True, borrow=borrow)
                else:
                    if not borrow:
                        self.emit_inc_ref(temp, object_rprimitive)
                    self.emit_cast(temp, temp2, item_type, declare_dest=True)
                self.emit_line(f'{dest}.f{i} = {temp2};')
            self.emit_line('}')
            if optional:
                self.emit_line('}')
        else:
            assert False, 'Unboxing not implemented: %s' % typ

    def emit_box(self, src: str, dest: str, typ: RType, declare_dest: bool=False, can_borrow: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Emit code for boxing a value of given type.\n\n        Generate a simple assignment if no boxing is needed.\n\n        The source reference count is stolen for the result (no need to decref afterwards).\n        '
        if declare_dest:
            declaration = 'PyObject *'
        else:
            declaration = ''
        if is_int_rprimitive(typ) or is_short_int_rprimitive(typ):
            self.emit_line(f'{declaration}{dest} = CPyTagged_StealAsObject({src});')
        elif is_bool_rprimitive(typ) or is_bit_rprimitive(typ):
            self.emit_lines(f'{declaration}{dest} = {src} ? Py_True : Py_False;')
            if not can_borrow:
                self.emit_inc_ref(dest, object_rprimitive)
        elif is_none_rprimitive(typ):
            self.emit_lines(f'{declaration}{dest} = Py_None;')
            if not can_borrow:
                self.emit_inc_ref(dest, object_rprimitive)
        elif is_int32_rprimitive(typ) or is_int16_rprimitive(typ) or is_uint8_rprimitive(typ):
            self.emit_line(f'{declaration}{dest} = PyLong_FromLong({src});')
        elif is_int64_rprimitive(typ):
            self.emit_line(f'{declaration}{dest} = PyLong_FromLongLong({src});')
        elif is_float_rprimitive(typ):
            self.emit_line(f'{declaration}{dest} = PyFloat_FromDouble({src});')
        elif isinstance(typ, RTuple):
            self.declare_tuple_struct(typ)
            self.emit_line(f'{declaration}{dest} = PyTuple_New({len(typ.types)});')
            self.emit_line(f'if (unlikely({dest} == NULL))')
            self.emit_line('    CPyError_OutOfMemory();')
            for i in range(0, len(typ.types)):
                if not typ.is_unboxed:
                    self.emit_line(f'PyTuple_SET_ITEM({dest}, {i}, {src}.f{i}')
                else:
                    inner_name = self.temp_name()
                    self.emit_box(f'{src}.f{i}', inner_name, typ.types[i], declare_dest=True)
                    self.emit_line(f'PyTuple_SET_ITEM({dest}, {i}, {inner_name});')
        else:
            assert not typ.is_unboxed
            self.emit_line(f'{declaration}{dest} = {src};')

    def emit_error_check(self, value: str, rtype: RType, failure: str) -> None:
        if False:
            while True:
                i = 10
        'Emit code for checking a native function return value for uncaught exception.'
        if isinstance(rtype, RTuple):
            if len(rtype.types) == 0:
                return
            else:
                cond = self.tuple_undefined_check_cond(rtype, value, self.c_error_value, '==')
                self.emit_line(f'if ({cond}) {{')
        elif rtype.error_overlap:
            self.emit_line(f'if ({value} == {self.c_error_value(rtype)} && PyErr_Occurred()) {{')
        else:
            self.emit_line(f'if ({value} == {self.c_error_value(rtype)}) {{')
        self.emit_lines(failure, '}')

    def emit_gc_visit(self, target: str, rtype: RType) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Emit code for GC visiting a C variable reference.\n\n        Assume that 'target' represents a C expression that refers to a\n        struct member, such as 'self->x'.\n        "
        if not rtype.is_refcounted:
            return
        elif isinstance(rtype, RPrimitive) and rtype.name == 'builtins.int':
            self.emit_line(f'if (CPyTagged_CheckLong({target})) {{')
            self.emit_line(f'Py_VISIT(CPyTagged_LongAsObject({target}));')
            self.emit_line('}')
        elif isinstance(rtype, RTuple):
            for (i, item_type) in enumerate(rtype.types):
                self.emit_gc_visit(f'{target}.f{i}', item_type)
        elif self.ctype(rtype) == 'PyObject *':
            self.emit_line(f'Py_VISIT({target});')
        else:
            assert False, 'emit_gc_visit() not implemented for %s' % repr(rtype)

    def emit_gc_clear(self, target: str, rtype: RType) -> None:
        if False:
            while True:
                i = 10
        "Emit code for clearing a C attribute reference for GC.\n\n        Assume that 'target' represents a C expression that refers to a\n        struct member, such as 'self->x'.\n        "
        if not rtype.is_refcounted:
            return
        elif isinstance(rtype, RPrimitive) and rtype.name == 'builtins.int':
            self.emit_line(f'if (CPyTagged_CheckLong({target})) {{')
            self.emit_line(f'CPyTagged __tmp = {target};')
            self.emit_line(f'{target} = {self.c_undefined_value(rtype)};')
            self.emit_line('Py_XDECREF(CPyTagged_LongAsObject(__tmp));')
            self.emit_line('}')
        elif isinstance(rtype, RTuple):
            for (i, item_type) in enumerate(rtype.types):
                self.emit_gc_clear(f'{target}.f{i}', item_type)
        elif self.ctype(rtype) == 'PyObject *' and self.c_undefined_value(rtype) == 'NULL':
            self.emit_line(f'Py_CLEAR({target});')
        else:
            assert False, 'emit_gc_clear() not implemented for %s' % repr(rtype)

    def emit_traceback(self, source_path: str, module_name: str, traceback_entry: tuple[str, int]) -> None:
        if False:
            i = 10
            return i + 15
        return self._emit_traceback('CPy_AddTraceback', source_path, module_name, traceback_entry)

    def emit_type_error_traceback(self, source_path: str, module_name: str, traceback_entry: tuple[str, int], *, typ: RType, src: str) -> None:
        if False:
            print('Hello World!')
        func = 'CPy_TypeErrorTraceback'
        type_str = f'"{self.pretty_name(typ)}"'
        return self._emit_traceback(func, source_path, module_name, traceback_entry, type_str=type_str, src=src)

    def _emit_traceback(self, func: str, source_path: str, module_name: str, traceback_entry: tuple[str, int], type_str: str='', src: str='') -> None:
        if False:
            i = 10
            return i + 15
        globals_static = self.static_name('globals', module_name)
        line = '%s("%s", "%s", %d, %s' % (func, source_path.replace('\\', '\\\\'), traceback_entry[0], traceback_entry[1], globals_static)
        if type_str:
            assert src
            line += f', {type_str}, {src}'
        line += ');'
        self.emit_line(line)
        if DEBUG_ERRORS:
            self.emit_line('assert(PyErr_Occurred() != NULL && "failure w/o err!");')

    def emit_unbox_failure_with_overlapping_error_value(self, dest: str, typ: RType, failure: str) -> None:
        if False:
            return 10
        self.emit_line(f'if ({dest} == {self.c_error_value(typ)} && PyErr_Occurred()) {{')
        self.emit_line(failure)
        self.emit_line('}')

def c_array_initializer(components: list[str], *, indented: bool=False) -> str:
    if False:
        while True:
            i = 10
    'Construct an initializer for a C array variable.\n\n    Components are C expressions valid in an initializer.\n\n    For example, if components are ["1", "2"], the result\n    would be "{1, 2}", which can be used like this:\n\n        int a[] = {1, 2};\n\n    If the result is long, split it into multiple lines.\n    '
    indent = ' ' * 4 if indented else ''
    res = []
    current: list[str] = []
    cur_len = 0
    for c in components:
        if not current or cur_len + 2 + len(indent) + len(c) < 70:
            current.append(c)
            cur_len += len(c) + 2
        else:
            res.append(indent + ', '.join(current))
            current = [c]
            cur_len = len(c)
    if not res:
        return '{%s}' % ', '.join(current)
    res.append(indent + ', '.join(current))
    return '{\n    ' + ',\n    '.join(res) + '\n' + indent + '}'