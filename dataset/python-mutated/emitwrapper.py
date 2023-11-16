"""Generate CPython API wrapper functions for native functions.

The wrapper functions are used by the CPython runtime when calling
native functions from interpreted code, and when the called function
can't be determined statically in compiled code. They validate, match,
unbox and type check function arguments, and box return values as
needed. All wrappers accept and return 'PyObject *' (boxed) values.

The wrappers aren't used for most calls between two native functions
or methods in a single compilation unit.
"""
from __future__ import annotations
from typing import Sequence
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR, ARG_STAR2, ArgKind
from mypy.operators import op_methods_to_symbols, reverse_op_method_names, reverse_op_methods
from mypyc.codegen.emit import AssignHandler, Emitter, ErrorHandler, GotoHandler, ReturnHandler
from mypyc.common import BITMAP_BITS, BITMAP_TYPE, DUNDER_PREFIX, NATIVE_PREFIX, PREFIX, bitmap_name, use_vectorcall
from mypyc.ir.class_ir import ClassIR
from mypyc.ir.func_ir import FUNC_STATICMETHOD, FuncIR, RuntimeArg
from mypyc.ir.rtypes import RInstance, RType, is_bool_rprimitive, is_int_rprimitive, is_object_rprimitive, object_rprimitive
from mypyc.namegen import NameGenerator

def wrapper_function_header(fn: FuncIR, names: NameGenerator) -> str:
    if False:
        i = 10
        return i + 15
    'Return header of a vectorcall wrapper function.\n\n    See comment above for a summary of the arguments.\n    '
    return 'PyObject *{prefix}{name}(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames)'.format(prefix=PREFIX, name=fn.cname(names))

def generate_traceback_code(fn: FuncIR, emitter: Emitter, source_path: str, module_name: str) -> str:
    if False:
        print('Hello World!')
    globals_static = emitter.static_name('globals', module_name)
    traceback_code = 'CPy_AddTraceback("%s", "%s", %d, %s);' % (source_path.replace('\\', '\\\\'), fn.traceback_name or fn.name, fn.line, globals_static)
    return traceback_code

def make_arg_groups(args: list[RuntimeArg]) -> dict[ArgKind, list[RuntimeArg]]:
    if False:
        print('Hello World!')
    'Group arguments by kind.'
    return {k: [arg for arg in args if arg.kind == k] for k in ArgKind}

def reorder_arg_groups(groups: dict[ArgKind, list[RuntimeArg]]) -> list[RuntimeArg]:
    if False:
        return 10
    'Reorder argument groups to match their order in a format string.'
    return groups[ARG_POS] + groups[ARG_OPT] + groups[ARG_NAMED_OPT] + groups[ARG_NAMED]

def make_static_kwlist(args: list[RuntimeArg]) -> str:
    if False:
        return 10
    arg_names = ''.join((f'"{arg.name}", ' for arg in args))
    return f'static const char * const kwlist[] = {{{arg_names}0}};'

def make_format_string(func_name: str | None, groups: dict[ArgKind, list[RuntimeArg]]) -> str:
    if False:
        i = 10
        return i + 15
    "Return a format string that specifies the accepted arguments.\n\n    The format string is an extended subset of what is supported by\n    PyArg_ParseTupleAndKeywords(). Only the type 'O' is used, and we\n    also support some extensions:\n\n    - Required keyword-only arguments are introduced after '@'\n    - If the function receives *args or **kwargs, we add a '%' prefix\n\n    Each group requires the previous groups' delimiters to be present\n    first.\n\n    These are used by both vectorcall and legacy wrapper functions.\n    "
    format = ''
    if groups[ARG_STAR] or groups[ARG_STAR2]:
        format += '%'
    format += 'O' * len(groups[ARG_POS])
    if groups[ARG_OPT] or groups[ARG_NAMED_OPT] or groups[ARG_NAMED]:
        format += '|' + 'O' * len(groups[ARG_OPT])
    if groups[ARG_NAMED_OPT] or groups[ARG_NAMED]:
        format += '$' + 'O' * len(groups[ARG_NAMED_OPT])
    if groups[ARG_NAMED]:
        format += '@' + 'O' * len(groups[ARG_NAMED])
    if func_name is not None:
        format += f':{func_name}'
    return format

def generate_wrapper_function(fn: FuncIR, emitter: Emitter, source_path: str, module_name: str) -> None:
    if False:
        return 10
    'Generate a CPython-compatible vectorcall wrapper for a native function.\n\n    In particular, this handles unboxing the arguments, calling the native function, and\n    then boxing the return value.\n    '
    emitter.emit_line(f'{wrapper_function_header(fn, emitter.names)} {{')
    real_args = list(fn.args)
    if fn.sig.num_bitmap_args:
        real_args = real_args[:-fn.sig.num_bitmap_args]
    if fn.class_name and (not fn.decl.kind == FUNC_STATICMETHOD):
        arg = real_args.pop(0)
        emitter.emit_line(f'PyObject *obj_{arg.name} = self;')
    groups = make_arg_groups(real_args)
    reordered_args = reorder_arg_groups(groups)
    emitter.emit_line(make_static_kwlist(reordered_args))
    fmt = make_format_string(fn.name, groups)
    emitter.emit_line(f'static CPyArg_Parser parser = {{"{fmt}", kwlist, 0}};')
    for arg in real_args:
        emitter.emit_line('PyObject *obj_{}{};'.format(arg.name, ' = NULL' if arg.optional else ''))
    cleanups = [f'CPy_DECREF(obj_{arg.name});' for arg in groups[ARG_STAR] + groups[ARG_STAR2]]
    arg_ptrs: list[str] = []
    if groups[ARG_STAR] or groups[ARG_STAR2]:
        arg_ptrs += [f'&obj_{groups[ARG_STAR][0].name}' if groups[ARG_STAR] else 'NULL']
        arg_ptrs += [f'&obj_{groups[ARG_STAR2][0].name}' if groups[ARG_STAR2] else 'NULL']
    arg_ptrs += [f'&obj_{arg.name}' for arg in reordered_args]
    if fn.name == '__call__' and use_vectorcall(emitter.capi_version):
        nargs = 'PyVectorcall_NARGS(nargs)'
    else:
        nargs = 'nargs'
    parse_fn = 'CPyArg_ParseStackAndKeywords'
    if not real_args:
        parse_fn = 'CPyArg_ParseStackAndKeywordsNoArgs'
    elif len(real_args) == 1 and len(groups[ARG_POS]) == 1:
        parse_fn = 'CPyArg_ParseStackAndKeywordsOneArg'
    elif len(real_args) == len(groups[ARG_POS]) + len(groups[ARG_OPT]):
        parse_fn = 'CPyArg_ParseStackAndKeywordsSimple'
    emitter.emit_lines('if (!{}(args, {}, kwnames, &parser{})) {{'.format(parse_fn, nargs, ''.join((', ' + n for n in arg_ptrs))), 'return NULL;', '}')
    for i in range(fn.sig.num_bitmap_args):
        name = bitmap_name(i)
        emitter.emit_line(f'{BITMAP_TYPE} {name} = 0;')
    traceback_code = generate_traceback_code(fn, emitter, source_path, module_name)
    generate_wrapper_core(fn, emitter, groups[ARG_OPT] + groups[ARG_NAMED_OPT], cleanups=cleanups, traceback_code=traceback_code)
    emitter.emit_line('}')

def legacy_wrapper_function_header(fn: FuncIR, names: NameGenerator) -> str:
    if False:
        print('Hello World!')
    return 'PyObject *{prefix}{name}(PyObject *self, PyObject *args, PyObject *kw)'.format(prefix=PREFIX, name=fn.cname(names))

def generate_legacy_wrapper_function(fn: FuncIR, emitter: Emitter, source_path: str, module_name: str) -> None:
    if False:
        return 10
    'Generates a CPython-compatible legacy wrapper for a native function.\n\n    In particular, this handles unboxing the arguments, calling the native function, and\n    then boxing the return value.\n    '
    emitter.emit_line(f'{legacy_wrapper_function_header(fn, emitter.names)} {{')
    real_args = list(fn.args)
    if fn.sig.num_bitmap_args:
        real_args = real_args[:-fn.sig.num_bitmap_args]
    if fn.class_name and (not fn.decl.kind == FUNC_STATICMETHOD):
        arg = real_args.pop(0)
        emitter.emit_line(f'PyObject *obj_{arg.name} = self;')
    groups = make_arg_groups(real_args)
    reordered_args = reorder_arg_groups(groups)
    emitter.emit_line(make_static_kwlist(reordered_args))
    for arg in real_args:
        emitter.emit_line('PyObject *obj_{}{};'.format(arg.name, ' = NULL' if arg.optional else ''))
    cleanups = [f'CPy_DECREF(obj_{arg.name});' for arg in groups[ARG_STAR] + groups[ARG_STAR2]]
    arg_ptrs: list[str] = []
    if groups[ARG_STAR] or groups[ARG_STAR2]:
        arg_ptrs += [f'&obj_{groups[ARG_STAR][0].name}' if groups[ARG_STAR] else 'NULL']
        arg_ptrs += [f'&obj_{groups[ARG_STAR2][0].name}' if groups[ARG_STAR2] else 'NULL']
    arg_ptrs += [f'&obj_{arg.name}' for arg in reordered_args]
    emitter.emit_lines('if (!CPyArg_ParseTupleAndKeywords(args, kw, "{}", "{}", kwlist{})) {{'.format(make_format_string(None, groups), fn.name, ''.join((', ' + n for n in arg_ptrs))), 'return NULL;', '}')
    for i in range(fn.sig.num_bitmap_args):
        name = bitmap_name(i)
        emitter.emit_line(f'{BITMAP_TYPE} {name} = 0;')
    traceback_code = generate_traceback_code(fn, emitter, source_path, module_name)
    generate_wrapper_core(fn, emitter, groups[ARG_OPT] + groups[ARG_NAMED_OPT], cleanups=cleanups, traceback_code=traceback_code)
    emitter.emit_line('}')

def generate_dunder_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        print('Hello World!')
    'Generates a wrapper for native __dunder__ methods to be able to fit into the mapping\n    protocol slot. This specifically means that the arguments are taken as *PyObjects and returned\n    as *PyObjects.\n    '
    gen = WrapperGenerator(cl, emitter)
    gen.set_target(fn)
    gen.emit_header()
    gen.emit_arg_processing()
    gen.emit_call()
    gen.finish()
    return gen.wrapper_name()

def generate_ipow_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        print('Hello World!')
    'Generate a wrapper for native __ipow__.\n\n    Since __ipow__ fills a ternary slot, but almost no one defines __ipow__ to take three\n    arguments, the wrapper needs to tweaked to force it to accept three arguments.\n    '
    gen = WrapperGenerator(cl, emitter)
    gen.set_target(fn)
    assert len(fn.args) in (2, 3), '__ipow__ should only take 2 or 3 arguments'
    gen.arg_names = ['self', 'exp', 'mod']
    gen.emit_header()
    gen.emit_arg_processing()
    handle_third_pow_argument(fn, emitter, gen, if_unsupported=['PyErr_SetString(PyExc_TypeError, "__ipow__ takes 2 positional arguments but 3 were given");', 'return NULL;'])
    gen.emit_call()
    gen.finish()
    return gen.wrapper_name()

def generate_bin_op_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        print('Hello World!')
    'Generates a wrapper for a native binary dunder method.\n\n    The same wrapper that handles the forward method (e.g. __add__) also handles\n    the corresponding reverse method (e.g. __radd__), if defined.\n\n    Both arguments and the return value are PyObject *.\n    '
    gen = WrapperGenerator(cl, emitter)
    gen.set_target(fn)
    if fn.name in ('__pow__', '__rpow__'):
        gen.arg_names = ['left', 'right', 'mod']
    else:
        gen.arg_names = ['left', 'right']
    wrapper_name = gen.wrapper_name()
    gen.emit_header()
    if fn.name not in reverse_op_methods and fn.name in reverse_op_method_names:
        generate_bin_op_reverse_only_wrapper(fn, emitter, gen)
    else:
        rmethod = reverse_op_methods[fn.name]
        fn_rev = cl.get_method(rmethod)
        if fn_rev is None:
            generate_bin_op_forward_only_wrapper(fn, emitter, gen)
        else:
            generate_bin_op_both_wrappers(cl, fn, fn_rev, emitter, gen)
    return wrapper_name

def generate_bin_op_forward_only_wrapper(fn: FuncIR, emitter: Emitter, gen: WrapperGenerator) -> None:
    if False:
        i = 10
        return i + 15
    gen.emit_arg_processing(error=GotoHandler('typefail'), raise_exception=False)
    handle_third_pow_argument(fn, emitter, gen, if_unsupported=['goto typefail;'])
    gen.emit_call(not_implemented_handler='goto typefail;')
    gen.emit_error_handling()
    emitter.emit_label('typefail')
    generate_bin_op_reverse_dunder_call(fn, emitter, reverse_op_methods[fn.name])
    gen.finish()

def generate_bin_op_reverse_only_wrapper(fn: FuncIR, emitter: Emitter, gen: WrapperGenerator) -> None:
    if False:
        i = 10
        return i + 15
    gen.arg_names = ['right', 'left']
    gen.emit_arg_processing(error=GotoHandler('typefail'), raise_exception=False)
    handle_third_pow_argument(fn, emitter, gen, if_unsupported=['goto typefail;'])
    gen.emit_call()
    gen.emit_error_handling()
    emitter.emit_label('typefail')
    emitter.emit_line('Py_INCREF(Py_NotImplemented);')
    emitter.emit_line('return Py_NotImplemented;')
    gen.finish()

def generate_bin_op_both_wrappers(cl: ClassIR, fn: FuncIR, fn_rev: FuncIR, emitter: Emitter, gen: WrapperGenerator) -> None:
    if False:
        for i in range(10):
            print('nop')
    emitter.emit_line('if (PyObject_IsInstance(obj_left, (PyObject *){})) {{'.format(emitter.type_struct_name(cl)))
    gen.emit_arg_processing(error=GotoHandler('typefail'), raise_exception=False)
    handle_third_pow_argument(fn, emitter, gen, if_unsupported=['goto typefail2;'])
    if fn.name == '__pow__' and len(fn.args) == 3:
        fwd_not_implemented_handler = 'goto typefail2;'
    else:
        fwd_not_implemented_handler = 'goto typefail;'
    gen.emit_call(not_implemented_handler=fwd_not_implemented_handler)
    gen.emit_error_handling()
    emitter.emit_line('}')
    emitter.emit_label('typefail')
    emitter.emit_line('if (PyObject_IsInstance(obj_right, (PyObject *){})) {{'.format(emitter.type_struct_name(cl)))
    gen.set_target(fn_rev)
    gen.arg_names = ['right', 'left']
    gen.emit_arg_processing(error=GotoHandler('typefail2'), raise_exception=False)
    handle_third_pow_argument(fn_rev, emitter, gen, if_unsupported=['goto typefail2;'])
    gen.emit_call()
    gen.emit_error_handling()
    emitter.emit_line('} else {')
    generate_bin_op_reverse_dunder_call(fn, emitter, fn_rev.name)
    emitter.emit_line('}')
    emitter.emit_label('typefail2')
    emitter.emit_line('Py_INCREF(Py_NotImplemented);')
    emitter.emit_line('return Py_NotImplemented;')
    gen.finish()

def generate_bin_op_reverse_dunder_call(fn: FuncIR, emitter: Emitter, rmethod: str) -> None:
    if False:
        return 10
    if fn.name in ('__pow__', '__rpow__'):
        emitter.emit_line('if (obj_mod == Py_None) {')
    emitter.emit_line(f'_Py_IDENTIFIER({rmethod});')
    emitter.emit_line('return CPy_CallReverseOpMethod(obj_left, obj_right, "{}", &PyId_{});'.format(op_methods_to_symbols[fn.name], rmethod))
    if fn.name in ('__pow__', '__rpow__'):
        emitter.emit_line('} else {')
        emitter.emit_line('Py_INCREF(Py_NotImplemented);')
        emitter.emit_line('return Py_NotImplemented;')
        emitter.emit_line('}')

def handle_third_pow_argument(fn: FuncIR, emitter: Emitter, gen: WrapperGenerator, *, if_unsupported: list[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    if fn.name not in ('__pow__', '__rpow__', '__ipow__'):
        return
    if fn.name in ('__pow__', '__ipow__') and len(fn.args) == 2 or fn.name == '__rpow__':
        emitter.emit_line('if (obj_mod != Py_None) {')
        for line in if_unsupported:
            emitter.emit_line(line)
        emitter.emit_line('}')
        if len(gen.arg_names) == 3:
            gen.arg_names.pop()
RICHCOMPARE_OPS = {'__lt__': 'Py_LT', '__gt__': 'Py_GT', '__le__': 'Py_LE', '__ge__': 'Py_GE', '__eq__': 'Py_EQ', '__ne__': 'Py_NE'}

def generate_richcompare_wrapper(cl: ClassIR, emitter: Emitter) -> str | None:
    if False:
        i = 10
        return i + 15
    'Generates a wrapper for richcompare dunder methods.'
    matches = sorted((name for name in RICHCOMPARE_OPS if cl.has_method(name)))
    if not matches:
        return None
    name = f'{DUNDER_PREFIX}_RichCompare_{cl.name_prefix(emitter.names)}'
    emitter.emit_line('static PyObject *{name}(PyObject *obj_lhs, PyObject *obj_rhs, int op) {{'.format(name=name))
    emitter.emit_line('switch (op) {')
    for func in matches:
        emitter.emit_line(f'case {RICHCOMPARE_OPS[func]}: {{')
        method = cl.get_method(func)
        assert method is not None
        generate_wrapper_core(method, emitter, arg_names=['lhs', 'rhs'])
        emitter.emit_line('}')
    emitter.emit_line('}')
    emitter.emit_line('Py_INCREF(Py_NotImplemented);')
    emitter.emit_line('return Py_NotImplemented;')
    emitter.emit_line('}')
    return name

def generate_get_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        i = 10
        return i + 15
    'Generates a wrapper for native __get__ methods.'
    name = f'{DUNDER_PREFIX}{fn.name}{cl.name_prefix(emitter.names)}'
    emitter.emit_line('static PyObject *{name}(PyObject *self, PyObject *instance, PyObject *owner) {{'.format(name=name))
    emitter.emit_line('instance = instance ? instance : Py_None;')
    emitter.emit_line(f'return {NATIVE_PREFIX}{fn.cname(emitter.names)}(self, instance, owner);')
    emitter.emit_line('}')
    return name

def generate_hash_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        i = 10
        return i + 15
    'Generates a wrapper for native __hash__ methods.'
    name = f'{DUNDER_PREFIX}{fn.name}{cl.name_prefix(emitter.names)}'
    emitter.emit_line(f'static Py_ssize_t {name}(PyObject *self) {{')
    emitter.emit_line('{}retval = {}{}{}(self);'.format(emitter.ctype_spaced(fn.ret_type), emitter.get_group_prefix(fn.decl), NATIVE_PREFIX, fn.cname(emitter.names)))
    emitter.emit_error_check('retval', fn.ret_type, 'return -1;')
    if is_int_rprimitive(fn.ret_type):
        emitter.emit_line('Py_ssize_t val = CPyTagged_AsSsize_t(retval);')
    else:
        emitter.emit_line('Py_ssize_t val = PyLong_AsSsize_t(retval);')
    emitter.emit_dec_ref('retval', fn.ret_type)
    emitter.emit_line('if (PyErr_Occurred()) return -1;')
    emitter.emit_line('if (val == -1) return -2;')
    emitter.emit_line('return val;')
    emitter.emit_line('}')
    return name

def generate_len_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        while True:
            i = 10
    'Generates a wrapper for native __len__ methods.'
    name = f'{DUNDER_PREFIX}{fn.name}{cl.name_prefix(emitter.names)}'
    emitter.emit_line(f'static Py_ssize_t {name}(PyObject *self) {{')
    emitter.emit_line('{}retval = {}{}{}(self);'.format(emitter.ctype_spaced(fn.ret_type), emitter.get_group_prefix(fn.decl), NATIVE_PREFIX, fn.cname(emitter.names)))
    emitter.emit_error_check('retval', fn.ret_type, 'return -1;')
    if is_int_rprimitive(fn.ret_type):
        emitter.emit_line('Py_ssize_t val = CPyTagged_AsSsize_t(retval);')
    else:
        emitter.emit_line('Py_ssize_t val = PyLong_AsSsize_t(retval);')
    emitter.emit_dec_ref('retval', fn.ret_type)
    emitter.emit_line('if (PyErr_Occurred()) return -1;')
    emitter.emit_line('return val;')
    emitter.emit_line('}')
    return name

def generate_bool_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        print('Hello World!')
    'Generates a wrapper for native __bool__ methods.'
    name = f'{DUNDER_PREFIX}{fn.name}{cl.name_prefix(emitter.names)}'
    emitter.emit_line(f'static int {name}(PyObject *self) {{')
    emitter.emit_line('{}val = {}{}(self);'.format(emitter.ctype_spaced(fn.ret_type), NATIVE_PREFIX, fn.cname(emitter.names)))
    emitter.emit_error_check('val', fn.ret_type, 'return -1;')
    assert is_bool_rprimitive(fn.ret_type), 'Only bool return supported for __bool__'
    emitter.emit_line('return val;')
    emitter.emit_line('}')
    return name

def generate_del_item_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        print('Hello World!')
    'Generates a wrapper for native __delitem__.\n\n    This is only called from a combined __delitem__/__setitem__ wrapper.\n    '
    name = '{}{}{}'.format(DUNDER_PREFIX, '__delitem__', cl.name_prefix(emitter.names))
    input_args = ', '.join((f'PyObject *obj_{arg.name}' for arg in fn.args))
    emitter.emit_line(f'static int {name}({input_args}) {{')
    generate_set_del_item_wrapper_inner(fn, emitter, fn.args)
    return name

def generate_set_del_item_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        print('Hello World!')
    'Generates a wrapper for native __setitem__ method (also works for __delitem__).\n\n    This is used with the mapping protocol slot. Arguments are taken as *PyObjects and we\n    return a negative C int on error.\n\n    Create a separate wrapper function for __delitem__ as needed and have the\n    __setitem__ wrapper call it if the value is NULL. Return the name\n    of the outer (__setitem__) wrapper.\n    '
    method_cls = cl.get_method_and_class('__delitem__')
    del_name = None
    if method_cls and method_cls[1] == cl:
        del_name = generate_del_item_wrapper(cl, method_cls[0], emitter)
    args = fn.args
    if fn.name == '__delitem__':
        args = list(args) + [RuntimeArg('___value', object_rprimitive, ARG_POS)]
    name = '{}{}{}'.format(DUNDER_PREFIX, '__setitem__', cl.name_prefix(emitter.names))
    input_args = ', '.join((f'PyObject *obj_{arg.name}' for arg in args))
    emitter.emit_line(f'static int {name}({input_args}) {{')
    emitter.emit_line(f'if (obj_{args[2].name} == NULL) {{')
    if del_name is not None:
        emitter.emit_line(f'return {del_name}(obj_{args[0].name}, obj_{args[1].name});')
    else:
        emitter.emit_line(f'PyObject *super = CPy_Super(CPyModule_builtins, obj_{args[0].name});')
        emitter.emit_line('if (super == NULL) return -1;')
        emitter.emit_line('PyObject *result = PyObject_CallMethod(super, "__delitem__", "O", obj_{});'.format(args[1].name))
        emitter.emit_line('Py_DECREF(super);')
        emitter.emit_line('Py_XDECREF(result);')
        emitter.emit_line('return result == NULL ? -1 : 0;')
    emitter.emit_line('}')
    method_cls = cl.get_method_and_class('__setitem__')
    if method_cls and method_cls[1] == cl:
        generate_set_del_item_wrapper_inner(fn, emitter, args)
    else:
        emitter.emit_line(f'PyObject *super = CPy_Super(CPyModule_builtins, obj_{args[0].name});')
        emitter.emit_line('if (super == NULL) return -1;')
        emitter.emit_line('PyObject *result;')
        if method_cls is None and cl.builtin_base is None:
            msg = f"'{cl.name}' object does not support item assignment"
            emitter.emit_line(f'PyErr_SetString(PyExc_TypeError, "{msg}");')
            emitter.emit_line('result = NULL;')
        else:
            emitter.emit_line('result = PyObject_CallMethod(super, "__setitem__", "OO", obj_{}, obj_{});'.format(args[1].name, args[2].name))
        emitter.emit_line('Py_DECREF(super);')
        emitter.emit_line('Py_XDECREF(result);')
        emitter.emit_line('return result == NULL ? -1 : 0;')
        emitter.emit_line('}')
    return name

def generate_set_del_item_wrapper_inner(fn: FuncIR, emitter: Emitter, args: Sequence[RuntimeArg]) -> None:
    if False:
        for i in range(10):
            print('nop')
    for arg in args:
        generate_arg_check(arg.name, arg.type, emitter, GotoHandler('fail'))
    native_args = ', '.join((f'arg_{arg.name}' for arg in args))
    emitter.emit_line('{}val = {}{}({});'.format(emitter.ctype_spaced(fn.ret_type), NATIVE_PREFIX, fn.cname(emitter.names), native_args))
    emitter.emit_error_check('val', fn.ret_type, 'goto fail;')
    emitter.emit_dec_ref('val', fn.ret_type)
    emitter.emit_line('return 0;')
    emitter.emit_label('fail')
    emitter.emit_line('return -1;')
    emitter.emit_line('}')

def generate_contains_wrapper(cl: ClassIR, fn: FuncIR, emitter: Emitter) -> str:
    if False:
        return 10
    'Generates a wrapper for a native __contains__ method.'
    name = f'{DUNDER_PREFIX}{fn.name}{cl.name_prefix(emitter.names)}'
    emitter.emit_line(f'static int {name}(PyObject *self, PyObject *obj_item) {{')
    generate_arg_check('item', fn.args[1].type, emitter, ReturnHandler('-1'))
    emitter.emit_line('{}val = {}{}(self, arg_item);'.format(emitter.ctype_spaced(fn.ret_type), NATIVE_PREFIX, fn.cname(emitter.names)))
    emitter.emit_error_check('val', fn.ret_type, 'return -1;')
    if is_bool_rprimitive(fn.ret_type):
        emitter.emit_line('return val;')
    else:
        emitter.emit_line('int boolval = PyObject_IsTrue(val);')
        emitter.emit_dec_ref('val', fn.ret_type)
        emitter.emit_line('return boolval;')
    emitter.emit_line('}')
    return name

def generate_wrapper_core(fn: FuncIR, emitter: Emitter, optional_args: list[RuntimeArg] | None=None, arg_names: list[str] | None=None, cleanups: list[str] | None=None, traceback_code: str | None=None) -> None:
    if False:
        while True:
            i = 10
    'Generates the core part of a wrapper function for a native function.\n\n    This expects each argument as a PyObject * named obj_{arg} as a precondition.\n    It converts the PyObject *s to the necessary types, checking and unboxing if necessary,\n    makes the call, then boxes the result if necessary and returns it.\n    '
    gen = WrapperGenerator(None, emitter)
    gen.set_target(fn)
    if arg_names:
        gen.arg_names = arg_names
    gen.cleanups = cleanups or []
    gen.optional_args = optional_args or []
    gen.traceback_code = traceback_code or ''
    error = ReturnHandler('NULL') if not gen.use_goto() else GotoHandler('fail')
    gen.emit_arg_processing(error=error)
    gen.emit_call()
    gen.emit_error_handling()

def generate_arg_check(name: str, typ: RType, emitter: Emitter, error: ErrorHandler | None=None, *, optional: bool=False, raise_exception: bool=True, bitmap_arg_index: int=0) -> None:
    if False:
        i = 10
        return i + 15
    'Insert a runtime check for argument and unbox if necessary.\n\n    The object is named PyObject *obj_{}. This is expected to generate\n    a value of name arg_{} (unboxed if necessary). For each primitive a runtime\n    check ensures the correct type.\n    '
    error = error or AssignHandler()
    if typ.is_unboxed:
        if typ.error_overlap and optional:
            init = emitter.c_undefined_value(typ)
            emitter.emit_line(f'{emitter.ctype(typ)} arg_{name} = {init};')
            emitter.emit_line(f'if (obj_{name} != NULL) {{')
            bitmap = bitmap_name(bitmap_arg_index // BITMAP_BITS)
            emitter.emit_line(f'{bitmap} |= 1 << {bitmap_arg_index & BITMAP_BITS - 1};')
            emitter.emit_unbox(f'obj_{name}', f'arg_{name}', typ, declare_dest=False, raise_exception=raise_exception, error=error, borrow=True)
            emitter.emit_line('}')
        else:
            emitter.emit_unbox(f'obj_{name}', f'arg_{name}', typ, declare_dest=True, raise_exception=raise_exception, error=error, borrow=True, optional=optional)
    elif is_object_rprimitive(typ):
        if optional:
            emitter.emit_line(f'PyObject *arg_{name};')
            emitter.emit_line(f'if (obj_{name} == NULL) {{')
            emitter.emit_line(f'arg_{name} = {emitter.c_error_value(typ)};')
            emitter.emit_lines('} else {', f'arg_{name} = obj_{name}; ', '}')
        else:
            emitter.emit_line(f'PyObject *arg_{name} = obj_{name};')
    else:
        emitter.emit_cast(f'obj_{name}', f'arg_{name}', typ, declare_dest=True, raise_exception=raise_exception, error=error, optional=optional)

class WrapperGenerator:
    """Helper that simplifies the generation of wrapper functions."""

    def __init__(self, cl: ClassIR | None, emitter: Emitter) -> None:
        if False:
            i = 10
            return i + 15
        self.cl = cl
        self.emitter = emitter
        self.cleanups: list[str] = []
        self.optional_args: list[RuntimeArg] = []
        self.traceback_code = ''

    def set_target(self, fn: FuncIR) -> None:
        if False:
            while True:
                i = 10
        "Set the wrapped function.\n\n        It's fine to modify the attributes initialized here later to customize\n        the wrapper function.\n        "
        self.target_name = fn.name
        self.target_cname = fn.cname(self.emitter.names)
        self.num_bitmap_args = fn.sig.num_bitmap_args
        if self.num_bitmap_args:
            self.args = fn.args[:-self.num_bitmap_args]
        else:
            self.args = fn.args
        self.arg_names = [arg.name for arg in self.args]
        self.ret_type = fn.ret_type

    def wrapper_name(self) -> str:
        if False:
            print('Hello World!')
        'Return the name of the wrapper function.'
        return '{}{}{}'.format(DUNDER_PREFIX, self.target_name, self.cl.name_prefix(self.emitter.names) if self.cl else '')

    def use_goto(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Do we use a goto for error handling (instead of straight return)?'
        return bool(self.cleanups or self.traceback_code)

    def emit_header(self) -> None:
        if False:
            while True:
                i = 10
        'Emit the function header of the wrapper implementation.'
        input_args = ', '.join((f'PyObject *obj_{arg}' for arg in self.arg_names))
        self.emitter.emit_line('static PyObject *{name}({input_args}) {{'.format(name=self.wrapper_name(), input_args=input_args))

    def emit_arg_processing(self, error: ErrorHandler | None=None, raise_exception: bool=True) -> None:
        if False:
            return 10
        'Emit validation and unboxing of arguments.'
        error = error or self.error()
        bitmap_arg_index = 0
        for (arg_name, arg) in zip(self.arg_names, self.args):
            typ = arg.type if arg.kind not in (ARG_STAR, ARG_STAR2) else object_rprimitive
            optional = arg in self.optional_args
            generate_arg_check(arg_name, typ, self.emitter, error, raise_exception=raise_exception, optional=optional, bitmap_arg_index=bitmap_arg_index)
            if optional and typ.error_overlap:
                bitmap_arg_index += 1

    def emit_call(self, not_implemented_handler: str='') -> None:
        if False:
            i = 10
            return i + 15
        "Emit call to the wrapper function.\n\n        If not_implemented_handler is non-empty, use this C code to handle\n        a NotImplemented return value (if it's possible based on the return type).\n        "
        native_args = ', '.join((f'arg_{arg}' for arg in self.arg_names))
        if self.num_bitmap_args:
            bitmap_args = ', '.join([bitmap_name(i) for i in reversed(range(self.num_bitmap_args))])
            native_args = f'{native_args}, {bitmap_args}'
        ret_type = self.ret_type
        emitter = self.emitter
        if ret_type.is_unboxed or self.use_goto():
            emitter.emit_line('{}retval = {}{}({});'.format(emitter.ctype_spaced(ret_type), NATIVE_PREFIX, self.target_cname, native_args))
            emitter.emit_lines(*self.cleanups)
            if ret_type.is_unboxed:
                emitter.emit_error_check('retval', ret_type, 'return NULL;')
                emitter.emit_box('retval', 'retbox', ret_type, declare_dest=True)
            emitter.emit_line('return {};'.format('retbox' if ret_type.is_unboxed else 'retval'))
        elif not_implemented_handler and (not isinstance(ret_type, RInstance)):
            emitter.emit_line('PyObject *retbox = {}{}({});'.format(NATIVE_PREFIX, self.target_cname, native_args))
            emitter.emit_lines('if (retbox == Py_NotImplemented) {', not_implemented_handler, '}', 'return retbox;')
        else:
            emitter.emit_line(f'return {NATIVE_PREFIX}{self.target_cname}({native_args});')

    def error(self) -> ErrorHandler:
        if False:
            while True:
                i = 10
        'Figure out how to deal with errors in the wrapper.'
        if self.cleanups or self.traceback_code:
            return GotoHandler('fail')
        else:
            return ReturnHandler('NULL')

    def emit_error_handling(self) -> None:
        if False:
            while True:
                i = 10
        'Emit error handling block at the end of the wrapper, if needed.'
        emitter = self.emitter
        if self.use_goto():
            emitter.emit_label('fail')
            emitter.emit_lines(*self.cleanups)
            if self.traceback_code:
                emitter.emit_line(self.traceback_code)
            emitter.emit_line('return NULL;')

    def finish(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.emitter.emit_line('}')