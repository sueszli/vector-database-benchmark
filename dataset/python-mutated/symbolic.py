import sympy as sm
import numpy as np
import six
k_used_symbols = set()
k_num_internal_syms = 0

def is_compatible_symbolic_vector(val_a, val_b):
    if False:
        return 10
    '\n    compare two vector and check if they are compatible.\n    ([is0, 4], [9, 4]), ([is0, 1],[is1, is2]) are twp compatible examples.\n    '
    val_a = tuple(val_a)
    val_b = tuple(val_b)
    if len(val_a) != len(val_b):
        return False
    for (a, b) in zip(val_a, val_b):
        if not is_symbolic(a) and (not is_symbolic(b)):
            if a != b:
                return False
    return True

def is_symbolic(val):
    if False:
        print('Hello World!')
    return issubclass(type(val), sm.Basic)

def is_variadic(val):
    if False:
        while True:
            i = 10
    return issubclass(type(val), sm.Symbol) and val.name[0] == '*'

def num_symbolic(val):
    if False:
        i = 10
        return i + 15
    '\n    Return the number of symbols in val\n    '
    if is_symbolic(val):
        return 1
    elif isinstance(val, np.ndarray) and np.issctype(val.dtype):
        return 0
    elif hasattr(val, '__iter__'):
        return sum((any_symbolic(i) for i in val))
    return 0

def any_symbolic(val):
    if False:
        print('Hello World!')
    if is_symbolic(val):
        return True
    if isinstance(val, np.ndarray) and val.ndim == 0:
        return is_symbolic(val[()])
    elif isinstance(val, np.ndarray) and np.issctype(val.dtype):
        return False
    elif isinstance(val, six.string_types):
        return False
    elif hasattr(val, '__iter__'):
        return any((any_symbolic(i) for i in val))
    return False

def any_variadic(val):
    if False:
        for i in range(10):
            print('nop')
    if is_variadic(val):
        return True
    elif isinstance(val, np.ndarray) and np.issctype(val.dtype):
        return False
    elif isinstance(val, six.string_types):
        return False
    elif hasattr(val, '__iter__'):
        return any((any_variadic(i) for i in val))
    return False

def isscalar(val):
    if False:
        print('Hello World!')
    return np.isscalar(val) or issubclass(type(val), sm.Basic)