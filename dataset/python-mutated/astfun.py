"""
Disassembly support.

:copyright: (c) 2016 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from h2o.utils.compatibility import *
import dis
import inspect
import sys
from . import h2o
from .expr import ExprNode, ASTId
BYTECODE_INSTRS = {'BINARY_SUBSCR': 'cols', 'UNARY_POSITIVE': '+', 'UNARY_NEGATIVE': '-', 'UNARY_NOT': '!', 'BINARY_POWER': '**', 'BINARY_MULTIPLY': '*', 'BINARY_FLOOR_DIVIDE': '//', 'BINARY_TRUE_DIVIDE': '/', 'BINARY_DIVIDE': '/', 'BINARY_MODULO': '%', 'BINARY_ADD': '+', 'BINARY_SUBTRACT': '-', 'BINARY_AND': '&', 'BINARY_OR': '|', 'COMPARE_OP': '', 'BINARY_OP': '', 'CALL_FUNCTION': '', 'CALL_FUNCTION_VAR': '', 'CALL_FUNCTION_VAR_KW': '', 'CALL_FUNCTION_KW': '', 'CALL_FUNCTION_EX': '', 'CALL_METHOD': '', 'CALL': ''}
BINARY_OPS = {0: '+', 1: '&', 2: '//', 5: '*', 6: '%', 7: '|', 8: '**', 10: '-', 11: '/', 13: '+', 14: '&', 15: '//', 18: '*', 19: '%', 20: '|', 21: '**', 23: '-', 24: '/'}

def is_bytecode_instruction(instr):
    if False:
        print('Hello World!')
    return instr in BYTECODE_INSTRS

def is_comp(instr):
    if False:
        while True:
            i = 10
    return 'COMPARE' in instr

def is_binary_op(instr):
    if False:
        while True:
            i = 10
    return 'BINARY_OP' == instr

def is_binary(instr):
    if False:
        for i in range(10):
            print('nop')
    return 'BINARY' in instr

def is_unary(instr):
    if False:
        while True:
            i = 10
    return 'UNARY' in instr

def is_func(instr):
    if False:
        return 10
    return 'CALL_FUNCTION' == instr

def is_call(instr):
    if False:
        while True:
            i = 10
    return 'CALL' == instr

def is_func_kw(instr):
    if False:
        print('Hello World!')
    return 'CALL_FUNCTION_KW' == instr

def is_func_ex(instr):
    if False:
        for i in range(10):
            print('nop')
    return 'CALL_FUNCTION_EX' == instr

def is_func_var(instr):
    if False:
        for i in range(10):
            print('nop')
    return 'CALL_FUNCTION_VAR' == instr

def is_func_var_kw(instr):
    if False:
        for i in range(10):
            print('nop')
    return 'CALL_FUNCTION_VAR_KW' == instr

def is_method_call(instr):
    if False:
        print('Hello World!')
    return 'CALL_METHOD' == instr

def is_dictionary_merge(instr):
    if False:
        while True:
            i = 10
    return 'DICT_MERGE' == instr

def is_list_extend(instr):
    if False:
        return 10
    return 'LIST_EXTEND' == instr

def is_list_to_tuple(instr):
    if False:
        while True:
            i = 10
    return 'LIST_TO_TUPLE' == instr

def is_build_list(instr):
    if False:
        return 10
    return 'BUILD_LIST' == instr

def is_build_map(instr):
    if False:
        return 10
    return 'BUILD_MAP' == instr

def is_callable(instr):
    if False:
        i = 10
        return i + 15
    return is_func(instr) or is_func_kw(instr) or is_func_ex(instr) or is_method_call(instr) or is_call(instr)

def is_builder(instr):
    if False:
        i = 10
        return i + 15
    return instr.startswith('BUILD_')

def is_load_fast(instr):
    if False:
        i = 10
        return i + 15
    return 'LOAD_FAST' == instr

def is_attr(instr):
    if False:
        i = 10
        return i + 15
    return 'LOAD_ATTR' == instr

def is_method(instr):
    if False:
        return 10
    return 'LOAD_METHOD' == instr or is_attr(instr)

def is_load_global(instr):
    if False:
        while True:
            i = 10
    return 'LOAD_GLOBAL' == instr

def is_load_deref(instr):
    if False:
        i = 10
        return i + 15
    return 'LOAD_DEREF' == instr

def is_load_outer_scope(instr):
    if False:
        for i in range(10):
            print('nop')
    return is_load_deref(instr) or is_load_global(instr)

def is_return(instr):
    if False:
        print('Hello World!')
    return 'RETURN_VALUE' == instr

def is_copy_free(instr):
    if False:
        while True:
            i = 10
    return 'COPY_FREE_VARS' == instr

def should_be_skipped(instr):
    if False:
        print('Hello World!')
    return (instr in 'COPY_FREE_VARS', 'RESUME', 'PUSH_NULL')
try:
    from dis import _unpack_opargs
except ImportError:

    def _unpack_opargs(code):
        if False:
            while True:
                i = 10
        extended_arg = 0
        i = 0
        n = len(code)
        while i < n:
            op = code[i]
            pos = i
            i += 1
            if op >= dis.HAVE_ARGUMENT:
                arg = code[i] + code[i + 1] * 256 + extended_arg
                extended_arg = 0
                i += 2
                if op == dis.EXTENDED_ARG:
                    extended_arg = arg * 65536
            else:
                arg = None
            yield (pos, op, arg)

def _disassemble_lambda(co):
    if False:
        for i in range(10):
            print('nop')
    code = co.co_code
    ops = []
    for (offset, op, arg) in _unpack_opargs(code):
        opname = dis.opname[op]
        args = []
        if arg is not None:
            if op in dis.hasconst:
                args.append(co.co_consts[arg])
            elif op in dis.hasname:
                args.append(co.co_names[arg])
            elif op in dis.hasjrel:
                raise ValueError('unimpl: op in hasjrel')
            elif op in dis.haslocal:
                args.append(co.co_varnames[arg])
            elif opname == 'COPY_FREE_VARS':
                args.append(arg)
            elif op in dis.hasfree:
                if sys.version_info.major == 3 and sys.version_info.minor >= 11:
                    args.append(co.co_freevars[arg - 1])
                else:
                    args.append(co.co_freevars[arg])
            elif op in dis.hascompare:
                args.append(dis.cmp_op[arg])
            elif is_callable(dis.opname[op]):
                args.append(arg)
            else:
                args.append(arg)
        ops.append([dis.opname[op], args])
    return ops

def _get_instr(ops, idx=0, argpos=0):
    if False:
        return 10
    (instr, args) = (ops[idx][0], ops[idx][1])
    return (instr, args[argpos] if args else None)

def lambda_to_expr(fun):
    if False:
        print('Hello World!')
    code = fun.__code__
    lambda_dis = _disassemble_lambda(code)
    return _lambda_bytecode_to_ast(code, lambda_dis)

def _lambda_bytecode_to_ast(co, ops):
    if False:
        i = 10
        return i + 15
    s = len(ops) - 1
    keys = [o[0] for o in ops]
    result = [ASTId('{')] + [ASTId(arg) for arg in co.co_varnames] + [ASTId('.')]
    instr = keys[s]
    if is_return(instr):
        s -= 1
        instr = keys[s]
    if is_bytecode_instruction(instr) or is_load_fast(instr) or is_load_outer_scope(instr):
        (body, s) = _opcode_read_arg(s, ops, keys)
    else:
        raise ValueError('unimpl bytecode instr: ' + instr)
    while s >= 0 and should_be_skipped(keys[s]):
        s -= 1
    if s >= 0:
        print('Dumping disassembled code: ')
        for i in range(len(ops)):
            if i == s:
                print(i, ' --> ' + str(ops[i]))
            else:
                print(i, str(ops[i]).rjust(5))
        raise ValueError('Unexpected bytecode disassembly @ ' + str(s))
    result += [body] + [ASTId('}')]
    return result

def _opcode_read_arg(start_index, ops, keys):
    if False:
        while True:
            i = 10
    (instr, op) = _get_instr(ops, start_index)
    return_idx = start_index - 1
    if is_bytecode_instruction(instr):
        if is_binary_op(instr):
            if op not in BINARY_OPS.keys():
                raise ValueError('Unimplemented binary op with id: ' + op)
            return _binop_bc(BINARY_OPS[op], return_idx, ops, keys)
        elif is_binary(instr):
            return _binop_bc(BYTECODE_INSTRS[instr], return_idx, ops, keys)
        elif is_comp(instr):
            return _binop_bc(op, return_idx, ops, keys)
        elif is_unary(instr):
            return _unop_bc(BYTECODE_INSTRS[instr], return_idx, ops, keys)
        elif is_call(instr):
            return _call_bc(op, return_idx, ops, keys)
        elif is_func(instr):
            return _call_func_bc(op, return_idx, ops, keys)
        elif is_func_kw(instr):
            return _call_func_kw_bc(op, return_idx, ops, keys)
        elif is_func_ex(instr):
            return _call_func_ex_bc(op, return_idx, ops, keys)
        elif is_method_call(instr):
            return _call_method_bc(op, return_idx, ops, keys)
        elif is_func_var(instr):
            return _call_func_var_bc(op, return_idx, ops, keys)
        elif is_func_var_kw(instr):
            return _call_func_var_kw_bc(op, return_idx, ops, keys)
        else:
            raise ValueError('unimpl bytecode op: ' + instr)
    elif is_load_fast(instr):
        return [_load_fast(op), return_idx]
    elif is_load_outer_scope(instr):
        return [_load_outer_scope(op), return_idx]
    elif is_build_list(instr):
        return _build_args(start_index, ops, keys)
    elif is_build_map(instr):
        return _build_kwargs(start_index, ops, keys)
    return (op, return_idx)

def _binop_bc(op, idx, ops, keys):
    if False:
        print('Hello World!')
    (rite, idx) = _opcode_read_arg(idx, ops, keys)
    (left, idx) = _opcode_read_arg(idx, ops, keys)
    return (ExprNode(op, left, rite), idx)

def _unop_bc(op, idx, ops, keys):
    if False:
        print('Hello World!')
    (arg, idx) = _opcode_read_arg(idx, ops, keys)
    return (ExprNode(op, arg), idx)

def _build_args(idx, ops, keys):
    if False:
        i = 10
        return i + 15
    (instr, nargs) = _get_instr(ops, idx)
    idx -= 1
    args = []
    while nargs > 0:
        (new_arg, idx) = _opcode_read_arg(idx, ops, keys)
        args.insert(0, new_arg)
        nargs -= 1
    return (args, idx)

def _build_kwargs(idx, ops, keys):
    if False:
        return 10
    (instr, nargs) = _get_instr(ops, idx)
    kwargs = dict()
    idx -= 1
    while nargs > 0:
        (val, idx) = _opcode_read_arg(idx, ops, keys)
        (key, idx) = _opcode_read_arg(idx, ops, keys)
        kwargs[key] = val
        nargs -= 1
    return (kwargs, idx)

def _call_func_bc(nargs, idx, ops, keys):
    if False:
        print('Hello World!')
    '\n    Implements transformation of CALL_FUNCTION bc inst to Rapids expression.\n    The implementation follows definition of behavior defined in\n    https://docs.python.org/3/library/dis.html#opcode-CALL_FUNCTION\n    \n    :param nargs: number of arguments including keyword and positional arguments\n    :param idx: index of current instruction on the stack\n    :param ops: stack of instructions\n    :param keys:  names of instructions\n    :return: ExprNode representing method call\n    '
    (kwargs, idx, nargs) = _read_explicit_keyword_args(nargs, idx, ops, keys)
    (args, idx, nargs) = _read_explicit_positional_args(nargs, idx, ops, keys)
    return _to_rapids_expr(idx, ops, keys, *args, **kwargs)

def _call_func_kw_bc(nargs, idx, ops, keys):
    if False:
        i = 10
        return i + 15
    (_, keywords) = _get_instr(ops, idx)
    if isinstance(keywords, tuple):
        idx -= 1
        kwargs = {}
        for key in keywords:
            (val, idx) = _opcode_read_arg(idx, ops, keys)
            kwargs[key] = val
            nargs -= 1
    else:
        (kwargs, idx) = _opcode_read_arg(idx, ops, keys)
    (exp_kwargs, idx, nargs) = _read_explicit_keyword_args(nargs, idx, ops, keys)
    kwargs.update(exp_kwargs)
    (args, idx, nargs) = _read_explicit_positional_args(nargs, idx, ops, keys)
    return _to_rapids_expr(idx, ops, keys, *args, **kwargs)

def _call_func_var_bc(nargs, idx, ops, keys):
    if False:
        print('Hello World!')
    (var_args, idx) = _opcode_read_arg(idx, ops, keys)
    (args, idx, _) = _read_explicit_positional_args(nargs, idx, ops, keys)
    args.extend(var_args)
    return _to_rapids_expr(idx, ops, keys, *args)

def _call_func_var_kw_bc(nargs, idx, ops, keys):
    if False:
        i = 10
        return i + 15
    (kwargs, idx) = _opcode_read_arg(idx, ops, keys)
    (exp_kwargs, idx, nargs) = _read_explicit_keyword_args(nargs, idx, ops, keys)
    kwargs.update(exp_kwargs)
    (var_args, idx) = _opcode_read_arg(idx, ops, keys)
    (args, idx, _) = _read_explicit_positional_args(nargs, idx, ops, keys)
    args.extend(var_args)
    return _to_rapids_expr(idx, ops, keys, *args, **kwargs)

def _call_func_ex_bc(flags, idx, ops, keys):
    if False:
        return 10
    if flags & 1:
        (instr, nargs) = _get_instr(ops, idx)
        if is_builder(instr):
            idx -= 1
            (kwargs, idx) = _opcode_read_arg(idx, ops, keys)
            nargs -= 1
            if nargs > 0:
                (instr, nargs) = _get_instr(ops, idx)
                if is_builder(instr):
                    idx -= 1
                    while nargs > 0:
                        (val, idx) = _opcode_read_arg(idx, ops, keys)
                        (key, idx) = _opcode_read_arg(idx, ops, keys)
                        kwargs[key] = val
                        nargs -= 1
        elif is_dictionary_merge(instr):
            idx -= 1
            kwargs = dict()
            while nargs + 1 > 0:
                (new_kwargs, idx) = _opcode_read_arg(idx, ops, keys)
                kwargs.update(new_kwargs)
                nargs -= 1
        else:
            (kwargs, idx) = _opcode_read_arg(idx, ops, keys)
    else:
        kwargs = {}
    (instr, nargs) = _get_instr(ops, idx)
    while is_list_to_tuple(instr):
        idx = idx - 1
        (instr, nargs) = _get_instr(ops, idx)
    if is_builder(instr) or is_list_extend(instr):
        idx -= 1
        args = []
        if is_list_extend(instr):
            nargs += 1
        while nargs > 0:
            (new_args, idx) = _opcode_read_arg(idx, ops, keys)
            args.insert(0, *new_args)
            nargs -= 1
    else:
        (args, idx) = _opcode_read_arg(idx, ops, keys)
        args = [] if args is None else args
    return _to_rapids_expr(idx, ops, keys, *args, **kwargs)

def _call_method_bc(nargs, idx, ops, keys):
    if False:
        return 10
    (args, idx, _) = _read_explicit_positional_args(nargs, idx, ops, keys)
    return _to_rapids_expr(idx, ops, keys, *args)

def _call_bc(nargs, idx, ops, keys):
    if False:
        return 10
    idx -= 1
    (instr, keywords) = _get_instr(ops, idx)
    kwargs = {}
    if instr == 'KW_NAMES':
        idx -= 1
        for key in keywords:
            (val, idx) = _opcode_read_arg(idx, ops, keys)
            kwargs[key] = val
            nargs -= 1
        (exp_kwargs, idx, nargs) = _read_explicit_keyword_args(nargs, idx, ops, keys)
        kwargs.update(exp_kwargs)
    (args, idx, nargs) = _read_explicit_positional_args(nargs, idx, ops, keys)
    return _to_rapids_expr(idx, ops, keys, *args, **kwargs)

def _read_explicit_keyword_args(nargs, idx, ops, keys):
    if False:
        i = 10
        return i + 15
    kwargs = {}
    while nargs >= 256:
        (val, idx) = _opcode_read_arg(idx, ops, keys)
        (key, idx) = _opcode_read_arg(idx, ops, keys)
        kwargs[key] = val
        nargs -= 256
    return (kwargs, idx, nargs)

def _read_explicit_positional_args(nargs, idx, ops, keys):
    if False:
        return 10
    args = []
    while nargs > 0:
        (arg, idx) = _opcode_read_arg(idx, ops, keys)
        args.append(arg)
        nargs -= 1
    args.reverse()
    return (args, idx, nargs)

def _to_rapids_expr(idx, ops, keys, *args, **kwargs):
    if False:
        return 10
    (instr, op) = _get_instr(ops, idx)
    rapids_args = _get_h2o_frame_method_args(op, *args, **kwargs) if is_method(instr) else []
    rapids_op = _get_func_name(op, rapids_args)
    idx -= 1
    (instr, op) = _get_instr(ops, idx)
    if is_bytecode_instruction(instr):
        (arg, idx) = _opcode_read_arg(idx, ops, keys)
        rapids_args.insert(0, arg)
    elif is_load_fast(instr):
        rapids_args.insert(0, _load_fast(op))
        idx -= 1
    return (ExprNode(rapids_op, *rapids_args), idx)

def _get_h2o_frame_method_args(op, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    fr_cls = h2o.H2OFrame
    if not hasattr(fr_cls, op):
        raise ValueError('Unimplemented: op <%s> not bound in H2OFrame' % op)
    argnames = []
    argdefs = []
    for (name, param) in inspect.signature(getattr(fr_cls, op)).parameters.items():
        if name == 'self':
            continue
        if param.kind == inspect._VAR_KEYWORD:
            continue
        argnames.append(name)
        argdefs.append(param.default)
    method_args = list(args) + argdefs[len(args):]
    for a in kwargs:
        method_args[argnames.index(a)] = kwargs[a]
    return method_args

def _get_func_name(op, args):
    if False:
        i = 10
        return i + 15
    if op == 'ceil':
        op = 'ceiling'
    if op == 'sum' and len(args) > 0 and args[0]:
        op = 'sumNA'
    if op == 'min' and len(args) > 0 and args[0]:
        op = 'minNA'
    if op == 'max' and len(args) > 0 and args[0]:
        op = 'maxNA'
    if op == 'nacnt':
        op = 'naCnt'
    return op

def _load_fast(x):
    if False:
        while True:
            i = 10
    return ASTId(x)

def _load_outer_scope(x):
    if False:
        while True:
            i = 10
    if x == 'True':
        return True
    elif x == 'False':
        return False
    stack = inspect.stack()
    for rec in stack:
        frame = rec[0]
        module = frame.f_globals.get('__name__', None)
        if module and module.startswith('h2o.'):
            continue
        scope = frame.f_locals
        if x in scope:
            return scope[x]
    return x