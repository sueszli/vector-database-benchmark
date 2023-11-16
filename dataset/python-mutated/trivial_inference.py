"""Trivial type inference for simple functions.

For internal use only; no backwards-compatibility guarantees.
"""
import builtins
import collections
import dis
import inspect
import pprint
import sys
import traceback
import types
from functools import reduce
from apache_beam import pvalue
from apache_beam.typehints import Any
from apache_beam.typehints import row_type
from apache_beam.typehints import typehints
from apache_beam.utils import python_callable

class TypeInferenceError(ValueError):
    """Error to raise when type inference failed."""
    pass

def instance_to_type(o):
    if False:
        print('Hello World!')
    'Given a Python object o, return the corresponding type hint.\n  '
    t = type(o)
    if o is None:
        return type(None)
    elif t == pvalue.Row:
        return row_type.RowTypeConstraint.from_fields([(name, instance_to_type(value)) for (name, value) in o.as_dict().items()])
    elif t not in typehints.DISALLOWED_PRIMITIVE_TYPES:
        if t == BoundMethod:
            return types.MethodType
        return t
    elif t == tuple:
        return typehints.Tuple[[instance_to_type(item) for item in o]]
    elif t == list:
        if len(o) > 0:
            return typehints.List[typehints.Union[[instance_to_type(item) for item in o]]]
        else:
            return typehints.List[typehints.Any]
    elif t == set:
        if len(o) > 0:
            return typehints.Set[typehints.Union[[instance_to_type(item) for item in o]]]
        else:
            return typehints.Set[typehints.Any]
    elif t == frozenset:
        if len(o) > 0:
            return typehints.FrozenSet[typehints.Union[[instance_to_type(item) for item in o]]]
        else:
            return typehints.FrozenSet[typehints.Any]
    elif t == dict:
        if len(o) > 0:
            return typehints.Dict[typehints.Union[[instance_to_type(k) for (k, v) in o.items()]], typehints.Union[[instance_to_type(v) for (k, v) in o.items()]]]
        else:
            return typehints.Dict[typehints.Any, typehints.Any]
    else:
        raise TypeInferenceError('Unknown forbidden type: %s' % t)

def union_list(xs, ys):
    if False:
        print('Hello World!')
    assert len(xs) == len(ys)
    return [union(x, y) for (x, y) in zip(xs, ys)]

class Const(object):

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value
        self.type = instance_to_type(value)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, Const) and self.value == other.value

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.value)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Const[%s]' % str(self.value)[:100]

    @staticmethod
    def unwrap(x):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(x, Const):
            return x.type
        return x

    @staticmethod
    def unwrap_all(xs):
        if False:
            while True:
                i = 10
        return [Const.unwrap(x) for x in xs]

class FrameState(object):
    """Stores the state of the frame at a particular point of execution.
  """

    def __init__(self, f, local_vars=None, stack=(), kw_names=None):
        if False:
            return 10
        self.f = f
        self.co = f.__code__
        self.vars = list(local_vars)
        self.stack = list(stack)
        self.kw_names = kw_names

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, FrameState) and self.__dict__ == other.__dict__

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(tuple(sorted(self.__dict__.items())))

    def copy(self):
        if False:
            i = 10
            return i + 15
        return FrameState(self.f, self.vars, self.stack, self.kw_names)

    def const_type(self, i):
        if False:
            print('Hello World!')
        return Const(self.co.co_consts[i])

    def get_closure(self, i):
        if False:
            for i in range(10):
                print('nop')
        num_cellvars = len(self.co.co_cellvars)
        if i < num_cellvars:
            return self.vars[i]
        else:
            return self.f.__closure__[i - num_cellvars].cell_contents

    def closure_type(self, i):
        if False:
            while True:
                i = 10
        'Returns a TypeConstraint or Const.'
        val = self.get_closure(i)
        if isinstance(val, typehints.TypeConstraint):
            return val
        else:
            return Const(val)

    def get_global(self, i):
        if False:
            return 10
        name = self.get_name(i)
        if name in self.f.__globals__:
            return Const(self.f.__globals__[name])
        if name in builtins.__dict__:
            return Const(builtins.__dict__[name])
        return Any

    def get_name(self, i):
        if False:
            return 10
        return self.co.co_names[i]

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'Stack: %s Vars: %s' % (self.stack, self.vars)

    def __or__(self, other):
        if False:
            return 10
        if self is None:
            return other.copy()
        elif other is None:
            return self.copy()
        return FrameState(self.f, union_list(self.vars, other.vars), union_list(self.stack, other.stack))

    def __ror__(self, left):
        if False:
            i = 10
            return i + 15
        return self | left

def union(a, b):
    if False:
        return 10
    'Returns the union of two types or Const values.\n  '
    if a == b:
        return a
    elif not a:
        return b
    elif not b:
        return a
    a = Const.unwrap(a)
    b = Const.unwrap(b)
    if type(a) == type(b) and element_type(a) == typehints.Union[()]:
        return b
    elif type(a) == type(b) and element_type(b) == typehints.Union[()]:
        return a
    return typehints.Union[a, b]

def finalize_hints(type_hint):
    if False:
        return 10
    'Sets type hint for empty data structures to Any.'

    def visitor(tc, unused_arg):
        if False:
            return 10
        if isinstance(tc, typehints.DictConstraint):
            empty_union = typehints.Union[()]
            if tc.key_type == empty_union:
                tc.key_type = Any
            if tc.value_type == empty_union:
                tc.value_type = Any
    if isinstance(type_hint, typehints.TypeConstraint):
        type_hint.visit(visitor, None)

def element_type(hint):
    if False:
        while True:
            i = 10
    'Returns the element type of a composite type.\n  '
    hint = Const.unwrap(hint)
    if isinstance(hint, typehints.SequenceTypeConstraint):
        return hint.inner_type
    elif isinstance(hint, typehints.TupleHint.TupleConstraint):
        return typehints.Union[hint.tuple_types]
    elif isinstance(hint, typehints.UnionHint.UnionConstraint) and (not hint.union_types):
        return hint
    return Any

def key_value_types(kv_type):
    if False:
        for i in range(10):
            print('nop')
    'Returns the key and value type of a KV type.\n  '
    if isinstance(kv_type, typehints.TupleHint.TupleConstraint) and len(kv_type.tuple_types) == 2:
        return kv_type.tuple_types
    elif isinstance(kv_type, typehints.UnionHint.UnionConstraint) and (not kv_type.union_types):
        return (kv_type, kv_type)
    return (Any, Any)
known_return_types = {len: int, hash: int}

class BoundMethod(object):
    """Used to create a bound method when we only know the type of the instance.
  """

    def __init__(self, func, type):
        if False:
            for i in range(10):
                print('nop')
        "Instantiates a bound method object.\n\n    Args:\n      func (types.FunctionType): The method's underlying function\n      type (type): The class of the method.\n    "
        self.func = func
        self.type = type

def hashable(c):
    if False:
        for i in range(10):
            print('nop')
    try:
        hash(c)
        return True
    except TypeError:
        return False

def infer_return_type(c, input_types, debug=False, depth=5):
    if False:
        for i in range(10):
            print('nop')
    'Analyses a callable to deduce its return type.\n\n  Args:\n    c: A Python callable to infer the return type of.\n    input_types: A sequence of inputs corresponding to the input types.\n    debug: Whether to print verbose debugging information.\n    depth: Maximum inspection depth during type inference.\n\n  Returns:\n    A TypeConstraint that that the return value of this function will (likely)\n    satisfy given the specified inputs.\n  '
    try:
        if hashable(c) and c in known_return_types:
            return known_return_types[c]
        elif isinstance(c, types.FunctionType):
            return infer_return_type_func(c, input_types, debug, depth)
        elif isinstance(c, types.MethodType):
            if c.__self__ is not None:
                input_types = [Const(c.__self__)] + input_types
            return infer_return_type_func(c.__func__, input_types, debug, depth)
        elif isinstance(c, BoundMethod):
            input_types = [c.type] + input_types
            return infer_return_type_func(c.func, input_types, debug, depth)
        elif inspect.isclass(c):
            if c in typehints.DISALLOWED_PRIMITIVE_TYPES:
                return {list: typehints.List[Any], set: typehints.Set[Any], frozenset: typehints.FrozenSet[Any], tuple: typehints.Tuple[Any, ...], dict: typehints.Dict[Any, Any]}[c]
            return c
        elif c == getattr and len(input_types) == 2 and isinstance(input_types[1], Const):
            from apache_beam.typehints import opcodes
            return opcodes._getattr(input_types[0], input_types[1].value)
        elif isinstance(c, python_callable.PythonCallableWithSource):
            return infer_return_type(c._callable, input_types, debug, depth)
        else:
            return Any
    except TypeInferenceError:
        if debug:
            traceback.print_exc()
        return Any
    except Exception:
        if debug:
            sys.stdout.flush()
            raise
        else:
            return Any

def infer_return_type_func(f, input_types, debug=False, depth=0):
    if False:
        for i in range(10):
            print('nop')
    'Analyses a function to deduce its return type.\n\n  Args:\n    f: A Python function object to infer the return type of.\n    input_types: A sequence of inputs corresponding to the input types.\n    debug: Whether to print verbose debugging information.\n    depth: Maximum inspection depth during type inference.\n\n  Returns:\n    A TypeConstraint that that the return value of this function will (likely)\n    satisfy given the specified inputs.\n\n  Raises:\n    TypeInferenceError: if no type can be inferred.\n  '
    if debug:
        print()
        print(f, id(f), input_types)
        if (sys.version_info.major, sys.version_info.minor) >= (3, 11):
            dis.dis(f, show_caches=True)
        else:
            dis.dis(f)
    from . import opcodes
    simple_ops = dict(((k.upper(), v) for (k, v) in opcodes.__dict__.items()))
    co = f.__code__
    code = co.co_code
    end = len(code)
    pc = 0
    free = None
    yields = set()
    returns = set()
    local_vars = list(input_types) + [typehints.Union[()]] * (len(co.co_varnames) - len(input_types))
    state = FrameState(f, local_vars)
    states = collections.defaultdict(lambda : None)
    jumps = collections.defaultdict(int)
    ofs_table = {}
    if (sys.version_info.major, sys.version_info.minor) >= (3, 11):
        dis_ints = dis.get_instructions(f, show_caches=True)
    else:
        dis_ints = dis.get_instructions(f)
    for instruction in dis_ints:
        ofs_table[instruction.offset] = instruction
    inst_size = 2
    opt_arg_size = 0
    if (sys.version_info.major, sys.version_info.minor) >= (3, 10):
        jump_multiplier = 2
    else:
        jump_multiplier = 1
    last_pc = -1
    last_real_opname = opname = None
    while pc < end:
        if opname not in ('PRECALL', 'CACHE'):
            last_real_opname = opname
        start = pc
        instruction = ofs_table[pc]
        op = instruction.opcode
        if debug:
            print('-->' if pc == last_pc else '    ', end=' ')
            print(repr(pc).rjust(4), end=' ')
            print(dis.opname[op].ljust(20), end=' ')
        pc += inst_size
        arg = None
        if op >= dis.HAVE_ARGUMENT:
            arg = instruction.arg
            pc += opt_arg_size
            if debug:
                print(str(arg).rjust(5), end=' ')
                if op in dis.hasconst:
                    print('(' + repr(co.co_consts[arg]) + ')', end=' ')
                elif op in dis.hasname:
                    if (sys.version_info.major, sys.version_info.minor) >= (3, 11):
                        print_arg = arg >> 1
                    else:
                        print_arg = arg
                    print('(' + co.co_names[print_arg] + ')', end=' ')
                elif op in dis.hasjrel:
                    print('(to ' + repr(pc + arg * jump_multiplier) + ')', end=' ')
                elif op in dis.haslocal:
                    print('(' + co.co_varnames[arg] + ')', end=' ')
                elif op in dis.hascompare:
                    print('(' + dis.cmp_op[arg] + ')', end=' ')
                elif op in dis.hasfree:
                    if free is None:
                        free = co.co_cellvars + co.co_freevars
                    print_arg = arg
                    if (sys.version_info.major, sys.version_info.minor) >= (3, 11):
                        print_arg = arg - len(co.co_varnames)
                    print('(' + free[print_arg] + ')', end=' ')
        if state is None and states[start] is None:
            if debug:
                print()
            continue
        state |= states[start]
        opname = dis.opname[op]
        jmp = jmp_state = None
        if opname.startswith('CALL_FUNCTION'):
            if opname == 'CALL_FUNCTION':
                pop_count = arg + 1
                if depth <= 0:
                    return_type = Any
                elif isinstance(state.stack[-pop_count], Const):
                    return_type = infer_return_type(state.stack[-pop_count].value, state.stack[1 - pop_count:], debug=debug, depth=depth - 1)
                else:
                    return_type = Any
            elif opname == 'CALL_FUNCTION_KW':
                pop_count = arg + 2
                if isinstance(state.stack[-pop_count], Const):
                    from apache_beam.pvalue import Row
                    if state.stack[-pop_count].value == Row:
                        fields = state.stack[-1].value
                        return_type = row_type.RowTypeConstraint.from_fields(list(zip(fields, Const.unwrap_all(state.stack[-pop_count + 1:-1]))))
                    else:
                        return_type = Any
                else:
                    return_type = Any
            elif opname == 'CALL_FUNCTION_EX':
                has_kwargs = arg & 1
                pop_count = has_kwargs + 2
                if has_kwargs:
                    return_type = Any
                else:
                    args = state.stack[-1]
                    _callable = state.stack[-2]
                    if isinstance(args, typehints.ListConstraint):
                        args = [args]
                    elif isinstance(args, typehints.TupleConstraint):
                        args = list(args._inner_types())
                    elif isinstance(args, typehints.SequenceTypeConstraint):
                        args = [element_type(args)] * len(inspect.getfullargspec(_callable.value).args)
                    return_type = infer_return_type(_callable.value, args, debug=debug, depth=depth - 1)
            else:
                raise TypeInferenceError('unable to handle %s' % opname)
            state.stack[-pop_count:] = [return_type]
        elif opname == 'CALL_METHOD':
            pop_count = 1 + arg
            if isinstance(state.stack[-pop_count], Const) and depth > 0:
                return_type = infer_return_type(state.stack[-pop_count].value, state.stack[1 - pop_count:], debug=debug, depth=depth - 1)
            else:
                return_type = typehints.Any
            state.stack[-pop_count:] = [return_type]
        elif opname == 'CALL':
            pop_count = 1 + arg
            if state.kw_names is not None:
                if isinstance(state.stack[-pop_count], Const):
                    from apache_beam.pvalue import Row
                    if state.stack[-pop_count].value == Row:
                        fields = state.kw_names
                        return_type = row_type.RowTypeConstraint.from_fields(list(zip(fields, Const.unwrap_all(state.stack[-pop_count + 1:]))))
                    else:
                        return_type = Any
                state.kw_names = None
            else:
                if pop_count == 1 and last_real_opname == 'GET_ITER' and (len(state.stack) > 1) and isinstance(state.stack[-2], Const) and (getattr(state.stack[-2].value, '__name__', None) in ('<listcomp>', '<dictcomp>', '<setcomp>', '<genexpr>')):
                    pop_count += 1
                if depth <= 0 or pop_count > len(state.stack):
                    return_type = Any
                elif isinstance(state.stack[-pop_count], Const):
                    return_type = infer_return_type(state.stack[-pop_count].value, state.stack[1 - pop_count:], debug=debug, depth=depth - 1)
                else:
                    return_type = Any
            state.stack[-pop_count:] = [return_type]
        elif opname in simple_ops:
            if debug:
                print('Executing simple op ' + opname)
            simple_ops[opname](state, arg)
        elif opname == 'RETURN_VALUE':
            returns.add(state.stack[-1])
            state = None
        elif opname == 'YIELD_VALUE':
            yields.add(state.stack[-1])
        elif opname == 'JUMP_FORWARD':
            jmp = pc + arg * jump_multiplier
            jmp_state = state
            state = None
        elif opname in ('JUMP_BACKWARD', 'JUMP_BACKWARD_NO_INTERRUPT'):
            jmp = pc - arg * jump_multiplier
            jmp_state = state
            state = None
        elif opname == 'JUMP_ABSOLUTE':
            jmp = arg * jump_multiplier
            jmp_state = state
            state = None
        elif opname in ('POP_JUMP_IF_TRUE', 'POP_JUMP_IF_FALSE'):
            state.stack.pop()
            jmp = arg * jump_multiplier
            jmp_state = state.copy()
        elif opname in ('POP_JUMP_FORWARD_IF_TRUE', 'POP_JUMP_FORWARD_IF_FALSE'):
            state.stack.pop()
            jmp = pc + arg * jump_multiplier
            jmp_state = state.copy()
        elif opname in ('POP_JUMP_BACKWARD_IF_TRUE', 'POP_JUMP_BACKWARD_IF_FALSE'):
            state.stack.pop()
            jmp = pc - arg * jump_multiplier
            jmp_state = state.copy()
        elif opname in ('POP_JUMP_FORWARD_IF_NONE', 'POP_JUMP_FORWARD_IF_NOT_NONE'):
            state.stack.pop()
            jmp = pc + arg * jump_multiplier
            jmp_state = state.copy()
        elif opname in ('POP_JUMP_BACKWARD_IF_NONE', 'POP_JUMP_BACKWARD_IF_NOT_NONE'):
            state.stack.pop()
            jmp = pc - arg * jump_multiplier
            jmp_state = state.copy()
        elif opname in ('JUMP_IF_TRUE_OR_POP', 'JUMP_IF_FALSE_OR_POP'):
            if (sys.version_info.major, sys.version_info.minor) >= (3, 11):
                jmp = pc + arg * jump_multiplier
            else:
                jmp = arg * jump_multiplier
            jmp_state = state.copy()
            state.stack.pop()
        elif opname == 'FOR_ITER':
            jmp = pc + arg * jump_multiplier
            jmp_state = state.copy()
            jmp_state.stack.pop()
            state.stack.append(element_type(state.stack[-1]))
        elif opname == 'COPY_FREE_VARS':
            pass
        elif opname == 'KW_NAMES':
            tup = co.co_consts[arg]
            state.kw_names = tup
        elif opname == 'RESUME':
            pass
        elif opname == 'PUSH_NULL':
            pass
        elif opname == 'PRECALL':
            pass
        elif opname == 'MAKE_CELL':
            pass
        elif opname == 'RETURN_GENERATOR':
            state.stack.append(None)
            pass
        elif opname == 'CACHE':
            pass
        else:
            raise TypeInferenceError('unable to handle %s' % opname)
        if jmp is not None:
            new_state = states[jmp] | jmp_state
            if jmp < pc and new_state != states[jmp] and (jumps[pc] < 5):
                jumps[pc] += 1
                pc = jmp
            states[jmp] = new_state
        if debug:
            print()
            print(state)
            pprint.pprint(dict((item for item in states.items() if item[1])))
    if yields:
        result = typehints.Iterable[reduce(union, Const.unwrap_all(yields))]
    else:
        result = reduce(union, Const.unwrap_all(returns))
    finalize_hints(result)
    if debug:
        print(f, id(f), input_types, '->', result)
    return result