import cupy

def _none_to_empty_2d(arg):
    if False:
        return 10
    if arg is None:
        return cupy.zeros((0, 0))
    else:
        return arg

def _atleast_2d_or_none(arg):
    if False:
        while True:
            i = 10
    if arg is not None:
        return cupy.atleast_2d(arg)

def _shape_or_none(M):
    if False:
        for i in range(10):
            print('nop')
    if M is not None:
        return M.shape
    else:
        return (None,) * 2

def _choice_not_none(*args):
    if False:
        while True:
            i = 10
    for arg in args:
        if arg is not None:
            return arg

def _restore(M, shape):
    if False:
        print('Hello World!')
    if M.shape == (0, 0):
        return cupy.zeros(shape)
    else:
        if M.shape != shape:
            raise ValueError('The input arrays have incompatible shapes.')
        return M

def abcd_normalize(A=None, B=None, C=None, D=None):
    if False:
        print('Hello World!')
    'Check state-space matrices and ensure they are 2-D.\n\n    If enough information on the system is provided, that is, enough\n    properly-shaped arrays are passed to the function, the missing ones\n    are built from this information, ensuring the correct number of\n    rows and columns. Otherwise a ValueError is raised.\n\n    Parameters\n    ----------\n    A, B, C, D : array_like, optional\n        State-space matrices. All of them are None (missing) by default.\n        See `ss2tf` for format.\n\n    Returns\n    -------\n    A, B, C, D : array\n        Properly shaped state-space matrices.\n\n    Raises\n    ------\n    ValueError\n        If not enough information on the system was provided.\n\n    '
    (A, B, C, D) = map(_atleast_2d_or_none, (A, B, C, D))
    (MA, NA) = _shape_or_none(A)
    (MB, NB) = _shape_or_none(B)
    (MC, NC) = _shape_or_none(C)
    (MD, ND) = _shape_or_none(D)
    p = _choice_not_none(MA, MB, NC)
    q = _choice_not_none(NB, ND)
    r = _choice_not_none(MC, MD)
    if p is None or q is None or r is None:
        raise ValueError('Not enough information on the system.')
    (A, B, C, D) = map(_none_to_empty_2d, (A, B, C, D))
    A = _restore(A, (p, p))
    B = _restore(B, (p, q))
    C = _restore(C, (r, p))
    D = _restore(D, (r, q))
    return (A, B, C, D)