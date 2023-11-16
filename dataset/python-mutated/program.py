from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import logging as _logging
import numpy as _np
import sympy as _sm
from . import types
from .block import Function
from .var import Var
from .types.symbolic import k_used_symbols, k_num_internal_syms
from coremltools.converters.mil.input_types import InputType

class Program(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.main_input_types = {}
        self.functions = {}
        self.parameters = {}

    def add_function(self, name, ssa_func):
        if False:
            i = 10
            return i + 15
        if not isinstance(ssa_func, Function):
            raise ValueError('Only Function can be added to Program.')
        self.functions[name] = ssa_func

    def add_parameters(self, name, ssa_val):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def set_main_input_types(self, inputs):
        if False:
            print('Hello World!')
        if not isinstance(inputs, tuple):
            raise ValueError('main inputs should be tuple of TensorType or ImageType')
        elif not all([isinstance(inp, InputType) for inp in inputs]):
            raise ValueError('main inputs should be tuple of InputSpec')
        self.main_input_types = inputs

    def find_ops(self, prefix=None, op_type=None, exactly_one=False):
        if False:
            i = 10
            return i + 15
        '\n        Return list of ops with name matching `prefix` if specified, and\n        op_type, if specified. At least one of {prefix, op_type} must be\n        specified.\n\n        If `exactly_one` == True, raise ValueError if we find <1 or >1 ops satisfying\n        the criteria.\n\n        prefix: str\n\n        Return list[Operation]. Empty list if no op satisfies.\n        '
        found_ops = []
        for (f_name, f) in self.functions.items():
            found_ops.extend(f.find_ops(prefix=prefix, op_type=op_type))
        if exactly_one and len(found_ops) != 1:
            msg = 'Found matching ops not exactly one. Found ops: {}'
            raise ValueError(msg.format(found_ops))
        return found_ops

    def validate(self):
        if False:
            while True:
                i = 10
        for (f_name, f) in self.functions.items():
            f.validate()

    def __getitem__(self, func_name):
        if False:
            return 10
        if func_name not in self.functions:
            msg = 'Function {} not found in among functions {}.'
            raise KeyError(msg.format(func_name, self.functions.keys()))
        return self.functions[func_name]

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__str__()

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        s = ''
        for (f_name, f) in self.functions.items():
            s += f.to_str(f_name)
        return s

class Placeholder(object):
    counter = 0

    def __init__(self, sym_shape, dtype=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        sym_shape: () or [] for scalar. list, tuple, np.ndarray for tensor. May\n        contain Symbol as symbolic shape (but not string).\n\n        dtype: types.float or other scalar builtin types.\n        '
        if not isinstance(sym_shape, (list, tuple, _np.generic, _np.ndarray)):
            raise ValueError('Illegal shape for Placeholder: {}'.format(sym_shape))
        self.sym_shape = sym_shape
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = types.float
        sym_type = self.type_inference()
        name = 'placeholder_' + str(self.__class__.counter)
        self.__class__.counter += 1
        self.outputs = [Var(name, sym_type)]

    def set_name(self, name):
        if False:
            while True:
                i = 10
        self.name = name
        self.outputs[0].name = name

    def type_inference(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.sym_shape) == 0:
            return self.dtype
        return types.tensor(self.dtype, self.sym_shape)

    def __str__(self):
        if False:
            return 10
        return str(self.outputs[0])

def get_new_variadic_symbol():
    if False:
        print('Hello World!')
    global k_num_internal_syms
    s = Symbol('*is' + str(k_num_internal_syms))
    k_num_internal_syms += 1
    return s

def get_new_symbol(name=None):
    if False:
        while True:
            i = 10
    '\n    Returns a new symbol, optionally named.\n\n    name: str (optional)\n        Optional name that provides more readability. If the name specified is\n        not available, an extra integer will be appended.\n    '
    global k_used_symbols
    global k_num_internal_syms
    if name is not None:
        s = Symbol(name)
        if s in k_used_symbols:
            new_name = name + k_num_internal_syms
            msg = 'Symbol name "{}" already occupied. Renaming to {}'
            _logging.warning(msg.format(name, new_name))
            s = Symbol(new_name)
    else:
        s = Symbol('is' + str(k_num_internal_syms))
    k_num_internal_syms += 1
    return s

class Symbol(_sm.Symbol):

    def __init__(self, sym_name):
        if False:
            print('Hello World!')
        "\n        Essentially sympy.Symbol representing an i32 value in shape.\n\n        sym_name: str. If first character is *, then this symbol represents\n        variadic rank. Otherwise the symbol name should start with a alpha\n        character. `sym_name` must be unique if specified, or it'd be auto\n        generated (to a non-variadic symbol). Furthermore, sym_name may not\n        start with 'is' (internal symbol)\n        "
        if not (sym_name[0].isalpha() or sym_name[0] == '*'):
            msg = 'Symbol name must start with a letter or *. Got {}'
            raise ValueError(msg.format(sym_name))
        global k_used_symbols
        if sym_name in k_used_symbols:
            msg = 'Symbol `{}` is used already.'
            raise ValueError(msg.format(sym_name))
        k_used_symbols.add(sym_name)
        self.name = sym_name