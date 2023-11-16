__all__ = ['get_flinalg_funcs']
try:
    from . import _flinalg
except ImportError:
    _flinalg = None

def has_column_major_storage(arr):
    if False:
        for i in range(10):
            print('nop')
    return arr.flags['FORTRAN']
_type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z'}

def get_flinalg_funcs(names, arrays=(), debug=0):
    if False:
        return 10
    'Return optimal available _flinalg function objects with\n    names. Arrays are used to determine optimal prefix.'
    ordering = []
    for (i, ar) in enumerate(arrays):
        t = ar.dtype.char
        if t not in _type_conv:
            t = 'd'
        ordering.append((t, i))
    if ordering:
        ordering.sort()
        required_prefix = _type_conv[ordering[0][0]]
    else:
        required_prefix = 'd'
    if ordering and has_column_major_storage(arrays[ordering[0][1]]):
        (suffix1, suffix2) = ('_c', '_r')
    else:
        (suffix1, suffix2) = ('_r', '_c')
    funcs = []
    for name in names:
        func_name = required_prefix + name
        func = getattr(_flinalg, func_name + suffix1, getattr(_flinalg, func_name + suffix2, None))
        funcs.append(func)
    return tuple(funcs)