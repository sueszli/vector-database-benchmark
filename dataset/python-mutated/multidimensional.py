"""
Provides functionality for multidimensional usage of scalar-functions.

Read the vectorize docstring for more details.
"""
from functools import wraps

def apply_on_element(f, args, kwargs, n):
    if False:
        i = 10
        return i + 15
    '\n    Returns a structure with the same dimension as the specified argument,\n    where each basic element is replaced by the function f applied on it. All\n    other arguments stay the same.\n    '
    if isinstance(n, int):
        structure = args[n]
        is_arg = True
    elif isinstance(n, str):
        structure = kwargs[n]
        is_arg = False

    def f_reduced(x):
        if False:
            return 10
        if hasattr(x, '__iter__'):
            return list(map(f_reduced, x))
        else:
            if is_arg:
                args[n] = x
            else:
                kwargs[n] = x
            return f(*args, **kwargs)
    return list(map(f_reduced, structure))

def iter_copy(structure):
    if False:
        i = 10
        return i + 15
    '\n    Returns a copy of an iterable object (also copying all embedded iterables).\n    '
    return [iter_copy(i) if hasattr(i, '__iter__') else i for i in structure]

def structure_copy(structure):
    if False:
        print('Hello World!')
    '\n    Returns a copy of the given structure (numpy-array, list, iterable, ..).\n    '
    if hasattr(structure, 'copy'):
        return structure.copy()
    return iter_copy(structure)

class vectorize:
    """
    Generalizes a function taking scalars to accept multidimensional arguments.

    Examples
    ========

    >>> from sympy import vectorize, diff, sin, symbols, Function
    >>> x, y, z = symbols('x y z')
    >>> f, g, h = list(map(Function, 'fgh'))

    >>> @vectorize(0)
    ... def vsin(x):
    ...     return sin(x)

    >>> vsin([1, x, y])
    [sin(1), sin(x), sin(y)]

    >>> @vectorize(0, 1)
    ... def vdiff(f, y):
    ...     return diff(f, y)

    >>> vdiff([f(x, y, z), g(x, y, z), h(x, y, z)], [x, y, z])
    [[Derivative(f(x, y, z), x), Derivative(f(x, y, z), y), Derivative(f(x, y, z), z)], [Derivative(g(x, y, z), x), Derivative(g(x, y, z), y), Derivative(g(x, y, z), z)], [Derivative(h(x, y, z), x), Derivative(h(x, y, z), y), Derivative(h(x, y, z), z)]]
    """

    def __init__(self, *mdargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        The given numbers and strings characterize the arguments that will be\n        treated as data structures, where the decorated function will be applied\n        to every single element.\n        If no argument is given, everything is treated multidimensional.\n        '
        for a in mdargs:
            if not isinstance(a, (int, str)):
                raise TypeError('a is of invalid type')
        self.mdargs = mdargs

    def __call__(self, f):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a wrapper for the one-dimensional function that can handle\n        multidimensional arguments.\n        '

        @wraps(f)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if self.mdargs:
                mdargs = self.mdargs
            else:
                mdargs = range(len(args)) + kwargs.keys()
            arglength = len(args)
            for n in mdargs:
                if isinstance(n, int):
                    if n >= arglength:
                        continue
                    entry = args[n]
                    is_arg = True
                elif isinstance(n, str):
                    try:
                        entry = kwargs[n]
                    except KeyError:
                        continue
                    is_arg = False
                if hasattr(entry, '__iter__'):
                    if is_arg:
                        args = list(args)
                        args[n] = structure_copy(entry)
                    else:
                        kwargs[n] = structure_copy(entry)
                    result = apply_on_element(wrapper, args, kwargs, n)
                    return result
            return f(*args, **kwargs)
        return wrapper